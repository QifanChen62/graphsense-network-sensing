from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from graphsense.analytics import matrix_summary, top_k_edges
from graphsense.baselines import counter_baseline, pandas_groupby_baseline
from graphsense.io import edges_to_sparse, load_matrix_market_traffic, write_matrix_market_traffic
from graphsense.reference import official_formula_reference
from graphsense.selector import early_stream_features, recommend_method
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import REGIMES, make_controlled_edges, make_synthetic_edges
from graphsense.timebin import add_timebin_anomaly_score, time_bin_summary


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
