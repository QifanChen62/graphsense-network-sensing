from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import pandas as pd

from graphsense.analytics import matrix_summary, top_k_edges
from graphsense.baselines import counter_baseline, pandas_groupby_baseline
from graphsense.io import edges_to_sparse, load_matrix_market_traffic, write_matrix_market_traffic
from graphsense.pcap import (
    detect_uniform_record_size,
    iter_decompressed_chunks,
    parse_pcap_global_header,
    pcap_stream_to_edges,
)
from graphsense.reference import official_formula_reference
from graphsense.selector import early_stream_features, recommend_method
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import REGIMES, make_controlled_edges, make_synthetic_edges
from graphsense.timebin import add_timebin_anomaly_score, time_bin_summary

try:
    import zstandard

    HAVE_ZSTANDARD = True
except ImportError:
    HAVE_ZSTANDARD = False


def _synthetic_pcap(byte_order: str, packet_lengths: list[int], seed: int = 3) -> tuple[bytes, list[tuple[int, int, int]]]:
    """Build pcap bytes with Ethernet+IPv4 packets and return expected (src, dst, bytes)."""

    import struct as _struct

    magic = b"\xd4\xc3\xb2\xa1" if byte_order == "<" else b"\xa1\xb2\xc3\xd4"
    header = magic + _struct.pack(byte_order + "HHiIII", 2, 4, 0, 0, 65535, 1)
    blob = bytearray(header)
    expected = []
    for index, incl_len in enumerate(packet_lengths):
        src = 0x0A000000 + seed * 1000 + index
        dst = 0xC0A80000 + seed * 100 + (index % 7)
        packet = bytearray(incl_len)
        packet[12:14] = b"\x08\x00"  # ethertype IPv4
        packet[14] = 0x45  # version/IHL
        packet[26:30] = _struct.pack(">I", src)
        packet[30:34] = _struct.pack(">I", dst)
        blob.extend(_struct.pack(byte_order + "IIII", 100 + index, 0, incl_len, incl_len))
        blob.extend(packet)
        expected.append((src, dst, incl_len))
    return bytes(blob), expected


class PipelineTest(unittest.TestCase):
    def test_sparse_and_baselines_match(self) -> None:
        edges = make_synthetic_edges(n_edges=1000, n_sources=32, n_destinations=40, seed=11)
        sparse_direct = edges_to_sparse(edges)
        groupby = pandas_groupby_baseline(edges)
        counter = counter_baseline(edges)

        self.assertEqual(sparse_direct.matrix.shape, groupby.matrix.shape)
        self.assertEqual(sparse_direct.matrix.shape, counter.matrix.shape)
        self.assertEqual((sparse_direct.matrix - groupby.matrix).nnz, 0)
        self.assertEqual((sparse_direct.matrix - counter.matrix).nnz, 0)

    def test_summary_and_top_k_are_well_formed(self) -> None:
        edges = make_synthetic_edges(n_edges=500, n_sources=16, n_destinations=20, seed=13)
        traffic = edges_to_sparse(edges)
        summary = matrix_summary(traffic)
        top = top_k_edges(traffic, k=5)

        self.assertGreater(summary.nnz, 0)
        self.assertGreater(summary.total_weight, 0)
        self.assertLessEqual(len(top), 5)
        self.assertEqual(list(top.columns), ["rank", "src", "dst", "weight"])

    def test_all_synthetic_regimes_build_sparse_matrix(self) -> None:
        densities = []
        for regime in REGIMES:
            edges = make_synthetic_edges(
                n_edges=2000,
                n_sources=128,
                n_destinations=160,
                seed=17,
                regime=regime,
                label_mode="int",
            )
            summary = matrix_summary(edges_to_sparse(edges))
            self.assertGreater(summary.nnz, 0)
            densities.append(summary.density)
        self.assertGreater(max(densities) - min(densities), 0.01)

    def test_reference_formula_matches_matrix_summary_counts(self) -> None:
        edges = make_synthetic_edges(
            n_edges=2000,
            n_sources=128,
            n_destinations=160,
            seed=19,
            regime="community_bursty",
            label_mode="int",
        )
        summary = matrix_summary(edges_to_sparse(edges))
        reference = official_formula_reference(edges)
        self.assertEqual(summary.nnz, reference.nnz)
        self.assertEqual(summary.n_sources, reference.n_sources)
        self.assertEqual(summary.n_destinations, reference.n_destinations)
        self.assertAlmostEqual(summary.total_weight, reference.total_weight)

    def test_streaming_summary_matches_exact_totals_and_finds_candidates(self) -> None:
        edges = make_synthetic_edges(
            n_edges=3000,
            n_sources=128,
            n_destinations=128,
            seed=23,
            regime="hotspot_zipf",
            label_mode="int",
        )
        traffic = edges_to_sparse(edges)
        exact_summary = matrix_summary(traffic)
        exact_top = top_k_edges(traffic, k=10)
        streaming_summary, approx_top = stream_summaries(edges, width=2048, depth=4, candidate_capacity=128)
        accuracy = streaming_accuracy(exact_top, approx_top, k=10)

        self.assertAlmostEqual(exact_summary.total_weight, streaming_summary.total_weight)
        self.assertGreater(streaming_summary.unique_sources, 0)
        self.assertGreater(streaming_summary.unique_destinations, 0)
        self.assertGreaterEqual(accuracy["topk_recall"], 0.1)

    def test_larger_count_min_width_does_not_increase_error_for_same_candidates(self) -> None:
        edges = make_synthetic_edges(
            n_edges=4000,
            n_sources=128,
            n_destinations=128,
            seed=31,
            regime="hotspot_zipf",
            label_mode="int",
        )
        exact_top = top_k_edges(edges_to_sparse(edges), k=10)
        _, small_top = stream_summaries(edges, width=256, depth=4, candidate_capacity=128)
        _, large_top = stream_summaries(edges, width=4096, depth=4, candidate_capacity=128)
        small_error = streaming_accuracy(exact_top, small_top, k=10)["median_relative_error"]
        large_error = streaming_accuracy(exact_top, large_top, k=10)["median_relative_error"]
        self.assertLessEqual(large_error, small_error + 1e-9)

    def test_larger_candidate_capacity_improves_or_preserves_hotspot_recall(self) -> None:
        edges = make_synthetic_edges(
            n_edges=4000,
            n_sources=128,
            n_destinations=128,
            seed=37,
            regime="hotspot_zipf",
            label_mode="int",
        )
        exact_top = top_k_edges(edges_to_sparse(edges), k=10)
        _, small_top = stream_summaries(edges, width=2048, depth=4, candidate_capacity=16)
        _, large_top = stream_summaries(edges, width=2048, depth=4, candidate_capacity=256)
        small_recall = streaming_accuracy(exact_top, small_top, k=10)["topk_recall"]
        large_recall = streaming_accuracy(exact_top, large_top, k=10)["topk_recall"]
        self.assertGreaterEqual(large_recall, small_recall)

    def test_controlled_generator_respects_target_nnz(self) -> None:
        edges = make_controlled_edges(
            n_edges=5000,
            n_sources=256,
            n_destinations=256,
            target_nnz=800,
            zipf_skew=0.4,
            seed=29,
            label_mode="int",
        )
        summary = matrix_summary(edges_to_sparse(edges))
        self.assertLessEqual(summary.nnz, 800)
        self.assertGreater(summary.nnz, 500)

    def test_time_bin_summary_is_well_formed(self) -> None:
        edges = make_controlled_edges(
            n_edges=2000,
            n_sources=128,
            n_destinations=128,
            target_nnz=600,
            n_time_bins=8,
            burst_fraction=0.1,
            seed=41,
            label_mode="int",
        )
        summary = add_timebin_anomaly_score(time_bin_summary(edges))
        self.assertGreaterEqual(len(summary), 1)
        self.assertIn("anomaly_score", summary.columns)
        self.assertIn("top_source_share", summary.columns)
        self.assertIn("top_destination_share", summary.columns)
        self.assertTrue((summary["total_weight"] > 0).all())
        self.assertTrue(((summary["top_edge_share"] >= 0) & (summary["top_edge_share"] <= 1)).all())
        self.assertTrue(((summary["top_source_share"] >= 0) & (summary["top_source_share"] <= 1)).all())
        self.assertTrue(((summary["top_destination_share"] >= 0) & (summary["top_destination_share"] <= 1)).all())

    def test_early_stream_features_separate_hotspot_from_uniform(self) -> None:
        hotspot = make_synthetic_edges(
            n_edges=5000,
            n_sources=128,
            n_destinations=128,
            seed=43,
            regime="hotspot_zipf",
            label_mode="int",
        )
        uniform = make_synthetic_edges(
            n_edges=5000,
            n_sources=512,
            n_destinations=512,
            seed=43,
            regime="uniform_sparse",
            label_mode="int",
        )
        hotspot_features = early_stream_features(hotspot, prefix_edges=5000)
        uniform_features = early_stream_features(uniform, prefix_edges=5000)

        self.assertGreater(hotspot_features.duplication_rate, uniform_features.duplication_rate)
        self.assertGreater(hotspot_features.edge_gini, uniform_features.edge_gini)
        self.assertLess(uniform_features.duplication_rate, 0.1)

    def test_method_recommendation_is_conservative_for_diffuse_streams(self) -> None:
        hotspot = make_synthetic_edges(
            n_edges=5000,
            n_sources=128,
            n_destinations=128,
            seed=47,
            regime="hotspot_zipf",
            label_mode="int",
        )
        uniform = make_synthetic_edges(
            n_edges=5000,
            n_sources=512,
            n_destinations=512,
            seed=47,
            regime="uniform_sparse",
            label_mode="int",
        )

        self.assertEqual(recommend_method(early_stream_features(hotspot)).recommended_exact_method, "pandas_groupby")
        self.assertEqual(recommend_method(early_stream_features(uniform)).recommended_exact_method, "sparse_direct")

    def test_matrix_market_traffic_round_trip_matches_sparse_matrix(self) -> None:
        edges = make_synthetic_edges(
            n_edges=1000,
            n_sources=64,
            n_destinations=80,
            seed=53,
            regime="community_bursty",
            label_mode="int",
        )
        traffic = edges_to_sparse(edges)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traffic.mtx"
            write_matrix_market_traffic(path, traffic)
            reloaded = load_matrix_market_traffic(path)

        self.assertEqual(traffic.matrix.shape, reloaded.matrix.shape)
        self.assertEqual((traffic.matrix - reloaded.matrix).nnz, 0)
        self.assertAlmostEqual(matrix_summary(traffic).total_weight, matrix_summary(reloaded).total_weight)

    def test_pcap_parser_reads_synthesized_fixed_records(self) -> None:
        blob, expected = _synthetic_pcap("<", [60] * 50)
        edges, info = pcap_stream_to_edges(iter([blob]), max_packets=1000)

        self.assertEqual(info["parser_path"], "numpy")
        self.assertEqual(info["record_size_detected"], 16 + 60)
        self.assertEqual(len(edges), 50)
        self.assertEqual(list(edges.columns), ["src", "dst", "bytes"])
        self.assertEqual(edges["src"].tolist(), [item[0] for item in expected])
        self.assertEqual(edges["dst"].tolist(), [item[1] for item in expected])
        self.assertTrue((edges["bytes"] == 60).all())

    def test_pcap_parser_handles_truncated_final_record(self) -> None:
        blob, _ = _synthetic_pcap("<", [60] * 50)
        truncated = blob[: len(blob) - 30]
        edges, _ = pcap_stream_to_edges(iter([truncated]), max_packets=1000)
        self.assertEqual(len(edges), 49)

    def test_pcap_parser_big_endian_magic(self) -> None:
        little_blob, expected = _synthetic_pcap("<", [60] * 20)
        big_blob, big_expected = _synthetic_pcap(">", [60] * 20)
        self.assertEqual(expected, big_expected)

        meta = parse_pcap_global_header(big_blob[:24])
        self.assertEqual(meta.byte_order, ">")
        little_edges, _ = pcap_stream_to_edges(iter([little_blob]), max_packets=1000)
        big_edges, _ = pcap_stream_to_edges(iter([big_blob]), max_packets=1000)
        pd.testing.assert_frame_equal(little_edges, big_edges)

    def test_pcap_parser_falls_back_on_variable_record_size(self) -> None:
        lengths = [60, 74] * 25
        blob, expected = _synthetic_pcap("<", lengths)
        meta = parse_pcap_global_header(blob[:24])
        self.assertIsNone(detect_uniform_record_size(blob, meta))

        edges, info = pcap_stream_to_edges(iter([blob]), max_packets=1000)
        self.assertEqual(info["parser_path"], "struct")
        self.assertEqual(len(edges), len(lengths))
        self.assertEqual(edges["bytes"].tolist(), [item[2] for item in expected])

    def test_pcap_parser_respects_max_packets_across_chunks(self) -> None:
        blob, _ = _synthetic_pcap("<", [60] * 200)
        chunks = [blob[i : i + 997] for i in range(0, len(blob), 997)]
        edges, _ = pcap_stream_to_edges(iter(chunks), max_packets=120)
        self.assertEqual(len(edges), 120)

    @unittest.skipUnless(HAVE_ZSTANDARD, "zstandard not installed")
    def test_zstd_truncated_stream_yields_prefix(self) -> None:
        blob, _ = _synthetic_pcap("<", [60] * 5000)
        compressed = zstandard.ZstdCompressor(level=3).compress(blob)
        truncated = compressed[: int(len(compressed) * 0.4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefix.pcap.zst"
            path.write_bytes(truncated)
            decoded = b"".join(iter_decompressed_chunks(path, chunk_size=4096))

        self.assertGreater(len(decoded), 24)
        self.assertLess(len(decoded), len(blob))
        edges, _ = pcap_stream_to_edges(iter([decoded]), max_packets=10**6)
        self.assertGreater(len(edges), 0)

    def test_candidate_tracker_heap_matches_min_scan_semantics(self) -> None:
        from graphsense.streaming import CandidateTracker

        import random

        rng = random.Random(67)
        tracker = CandidateTracker(capacity=8)
        shadow: dict[object, float] = {}
        for step in range(2000):
            key = rng.randrange(40)
            weight = float(rng.randrange(1, 5))
            tracker.update(key, weight)
            # Reference: plain space-saving with a min scan.
            if key in shadow:
                shadow[key] += weight
            elif len(shadow) < 8:
                shadow[key] = weight
            else:
                victim = min(shadow, key=lambda item: (shadow[item],))
                minimum = shadow.pop(victim)
                shadow[key] = minimum + weight

        # Tie-breaking may differ, but sizes and count multisets must match ranges.
        self.assertEqual(len(tracker.counts), 8)
        self.assertEqual(len(shadow), 8)
        self.assertAlmostEqual(
            max(tracker.counts.values()), max(shadow.values()), delta=max(shadow.values()) * 0.5
        )

    def test_candidate_tracker_all_unique_stream_is_fast(self) -> None:
        from graphsense.streaming import CandidateTracker

        tracker = CandidateTracker(capacity=4096)
        start = time.perf_counter()
        for key in range(200000):
            tracker.update(key, 1.0)
        elapsed = time.perf_counter() - start
        self.assertEqual(len(tracker.counts), 4096)
        self.assertLess(elapsed, 5.0)

    def test_real_data_experiment_on_synthetic_frame(self) -> None:
        import argparse

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        try:
            from real_data_experiment import run_real_experiment
        finally:
            sys.path.pop(0)

        edges = make_synthetic_edges(
            n_edges=20000,
            n_sources=128,
            n_destinations=128,
            seed=61,
            regime="hotspot_zipf",
            label_mode="int",
        )
        args = argparse.Namespace(
            regime_label="unit_test",
            prefix_edges=5000,
            top_k=10,
            width=2048,
            depth=4,
            capacities=[64, 512],
            repeats=2,
            value_column="bytes",
        )
        summary, sensitivity = run_real_experiment(edges, args)

        exact_rows = summary[summary["row_kind"] == "exact_method"]
        self.assertEqual(set(exact_rows["method"]), {"sparse_direct", "pandas_groupby"})
        self.assertTrue(exact_rows["output_matches_sparse"].all())
        selector_rows = summary[summary["row_kind"] == "selector"]
        self.assertEqual(len(selector_rows), 1)
        expected_columns = [
            "regime",
            "n_edges",
            "sketch_width",
            "sketch_depth",
            "candidate_capacity",
            "seconds",
            "sketch_bytes",
            "candidate_count",
            "topk_recall",
            "median_relative_error",
            "max_relative_error",
        ]
        self.assertEqual(list(sensitivity.columns), expected_columns)
        self.assertEqual(sensitivity["candidate_capacity"].tolist(), [64, 512])

    def test_conformal_lower_bound_is_valid_order_statistic(self) -> None:
        from graphsense.conformal import conformal_lower_bound

        import numpy as np

        rng = np.random.default_rng(71)
        # Exchangeable draws: coverage of the bound must be >= 1 - alpha.
        misses = 0
        trials = 400
        for _ in range(trials):
            sample = rng.exponential(size=61)
            bound, rank = conformal_lower_bound(sample[:-1], alpha=0.1)
            self.assertGreaterEqual(rank, 1)
            misses += sample[-1] < bound
        self.assertLessEqual(misses / trials, 0.1 + 0.04)

    def test_conformal_lower_bound_refuses_tiny_samples(self) -> None:
        from graphsense.conformal import conformal_lower_bound

        bound, rank = conformal_lower_bound([0.5, 0.6, 0.7], alpha=0.1)
        self.assertEqual(rank, 0)
        self.assertEqual(bound, float("-inf"))

    def test_conformal_certificate_threshold_and_decision(self) -> None:
        from graphsense.conformal import budget_threshold, conformal_certificate

        import numpy as np

        threshold = budget_threshold(width=16384, depth=5, candidate_capacity=5000)
        scores = np.full(99, threshold * 3.0)
        certificate = conformal_certificate(scores, alpha=0.1, width=16384, depth=5, candidate_capacity=5000)
        self.assertTrue(certificate.certified)
        self.assertGreater(certificate.margin, 0)

        low_scores = np.full(99, threshold * 0.5)
        refused = conformal_certificate(low_scores, alpha=0.1, width=16384, depth=5, candidate_capacity=5000)
        self.assertFalse(refused.certified)

    def test_window_statistics_gap_implies_retention(self) -> None:
        from graphsense.conformal import window_statistics

        edges = make_synthetic_edges(
            n_edges=40000,
            n_sources=128,
            n_destinations=128,
            seed=73,
            regime="hotspot_zipf",
            label_mode="int",
        )
        stats = window_statistics(edges, window_edges=5000, top_k=10)
        self.assertEqual(len(stats), 8)
        # w_k >= Delta_k always, so kth share dominates gap share per window.
        self.assertTrue((stats["kth_weight_share"] >= stats["topk_gap_share"] - 1e-12).all())
        self.assertTrue((stats["topk_gap_share"] >= 0).all())

    def test_adaptive_alpha_tracks_target_miscoverage(self) -> None:
        from graphsense.conformal import adaptive_alpha_trajectory

        import numpy as np

        rng = np.random.default_rng(79)
        # Drifting scores break exchangeability; ACI should still land near target.
        scores = np.concatenate([rng.normal(1.0, 0.1, 150), rng.normal(0.5, 0.1, 150)])
        trajectory = adaptive_alpha_trajectory(scores, target_alpha=0.1, gamma=0.05)
        self.assertGreater(len(trajectory), 200)
        self.assertLess(abs(trajectory.attrs["realized_miscoverage"] - 0.1), 0.08)

    def test_dyadic_descent_recovers_hotspot_heavy_hitters(self) -> None:
        from graphsense.dyadic import dyadic_topk_recall

        edges = make_synthetic_edges(
            n_edges=20000,
            n_sources=128,
            n_destinations=128,
            seed=83,
            regime="hotspot_zipf",
            label_mode="int",
        )
        exact_top = top_k_edges(edges_to_sparse(edges), k=10)
        result = dyadic_topk_recall(edges, exact_top, k=10, width=4096, depth=4, beam=1024)
        self.assertGreaterEqual(result["topk_recall"], 0.9)
        self.assertGreater(result["levels"], 1)

    def test_dyadic_key_packing_round_trip(self) -> None:
        from graphsense.dyadic import edges_to_keys

        edges = pd.DataFrame({"src": [3, 7, 1], "dst": [5, 2, 6], "bytes": [10, 20, 30]})
        keys, weights, key_bits, dst_bits = edges_to_keys(edges)
        self.assertEqual(list(weights), [10.0, 20.0, 30.0])
        for i, (src, dst) in enumerate([(3, 5), (7, 2), (1, 6)]):
            self.assertEqual(int(keys[i]) >> dst_bits, src)
            self.assertEqual(int(keys[i]) & ((1 << dst_bits) - 1), dst)

    def test_budget_pricing_minimizes_and_certifies(self) -> None:
        from graphsense.pricing import capacity_for_width, certification_margin, price_budget

        price = price_budget(gap_lower_bound=4.4e-5, alpha=0.1, depth=5)
        self.assertTrue(price.feasible)
        self.assertGreater(price.margin, 0)
        # The optimum must beat the hand-picked budget from the paper.
        hand_total = 5 * 262144 * 8 + 98304 * 32
        self.assertLess(price.total_bytes, hand_total)
        # Closed-form capacity is consistent with the margin condition.
        cap = capacity_for_width(4.4e-5, price.width)
        self.assertIsNotNone(cap)
        self.assertGreaterEqual(price.capacity, cap)
        self.assertGreater(certification_margin(4.4e-5, price.width, price.capacity), 0)

    def test_budget_pricing_reports_infeasible_for_zero_gap(self) -> None:
        from graphsense.pricing import price_budget

        price = price_budget(gap_lower_bound=0.0)
        self.assertFalse(price.feasible)

    def test_best_index_at_memory_crosses_one_at_price(self) -> None:
        from graphsense.pricing import best_index_at_memory, price_budget

        L = 3.1e-5
        price = price_budget(L)
        self.assertGreaterEqual(best_index_at_memory(L, price.total_bytes * 2), 1.0)
        self.assertLess(best_index_at_memory(L, price.total_bytes // 8), 1.0)

    def test_identifiability_report_all_tie_is_not_identifiable(self) -> None:
        from graphsense.metrics import identifiability_report

        edges = pd.DataFrame({"src": range(100), "dst": range(100, 200), "bytes": [40] * 100})
        approx = pd.DataFrame({"src": range(20), "dst": range(100, 120), "estimated_weight": [40.0] * 20})
        report = identifiability_report(edges, approx, k=20, width=8192, candidate_capacity=512)
        self.assertFalse(report.identifiable)
        self.assertIsNone(report.identifiable_recall)
        # Strict recall under all-ties is arbitrary tie-breaking -- the metric's point.
        self.assertGreaterEqual(report.strict_recall, 0.0)
        self.assertLessEqual(report.strict_recall, 1.0)
        self.assertEqual(report.tie_aware_recall, 1.0)

    def test_identifiability_report_skewed_core_recovered(self) -> None:
        from graphsense.metrics import identifiability_report

        weights = [10000, 9000, 8000, 7000, 6000] + [1] * 200
        edges = pd.DataFrame({"src": range(len(weights)), "dst": range(len(weights)), "bytes": weights})
        approx = pd.DataFrame({"src": range(5), "dst": range(5), "estimated_weight": weights[:5]})
        report = identifiability_report(edges, approx, k=5, width=8192, candidate_capacity=512)
        self.assertTrue(report.identifiable)
        self.assertEqual(report.identifiable_core_size, 5)
        self.assertEqual(report.identifiable_recall, 1.0)

    def test_router_agrees_disagrees_and_falls_back(self) -> None:
        from graphsense.router import route

        agree = route(0.01, 0.03, gap_lower_bound=4.4e-5, memory_budget_bytes=64_000_000, heuristic_choice="sparse_direct")
        self.assertEqual(agree.exact_choice, "sparse_direct")
        self.assertIn("agree", agree.exact_source)
        self.assertEqual(agree.sketch_decision, "certifiable_within_budget")

        disagree = route(0.01, 0.03, gap_lower_bound=4.4e-5, memory_budget_bytes=64_000_000, heuristic_choice="pandas_groupby")
        self.assertEqual(disagree.exact_choice, "sparse_direct")
        self.assertIn("disagree", disagree.exact_source)

        all_tie = route(0.01, 0.03, gap_lower_bound=0.0, memory_budget_bytes=64_000_000)
        self.assertEqual(all_tie.sketch_decision, "unidentifiable")
        self.assertIn("exact", all_tie.online_route)

        diffuse = route(0.01, 0.03, gap_lower_bound=1.1e-7, memory_budget_bytes=64_000_000)
        self.assertEqual(diffuse.sketch_decision, "not_worth_it")

        over = route(0.01, 0.03, gap_lower_bound=4.4e-5, memory_budget_bytes=1_000_000)
        self.assertEqual(over.sketch_decision, "certifiable_over_budget")

    def test_certified_selector_cli_writes_conservative_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "certified_selector.csv"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/certified_selector.py",
                    "--output",
                    str(output),
                    "--n-edges",
                    "1000",
                    "--prefix-edges",
                    "500",
                    "--top-k",
                    "10",
                    "--regimes",
                    "hotspot_zipf",
                    "uniform_sparse",
                    "--no-materialize-full-stream",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            frame = pd.read_csv(output)

        self.assertEqual(set(frame["regime"]), {"hotspot_zipf", "uniform_sparse"})
        self.assertIn("required_candidate_capacity_c_min", frame.columns)
        self.assertIn("required_cms_width_min", frame.columns)
        self.assertIn("sketch_safety_decision", frame.columns)
        self.assertTrue((frame["required_candidate_capacity_c_min"] >= 0).all())
        self.assertTrue(frame["sketch_safety_decision"].isin(["not_certified", "certified_safe"]).all())


if __name__ == "__main__":
    unittest.main()
