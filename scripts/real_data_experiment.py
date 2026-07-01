#!/usr/bin/env python3
"""Validate exact methods, the early-stream selector, and sketch limits on real data.

Runs three stages on an anonymized src,dst,bytes edge table (e.g. the official
capture prefix): exact construction timing for sparse_direct vs pandas_groupby,
early-stream selector validation against the measured winner, and a streaming
candidate-capacity sweep against exact top-k heavy hitters.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import matrix_summary, top_k_edges
from graphsense.benchmark import METHODS, _equivalent, sparse_direct
from graphsense.io import read_edges
from graphsense.selector import early_stream_features, recommend_method
from graphsense.streaming import stream_summaries, streaming_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/real/official_prefix_edges.csv")
    parser.add_argument("--regime-label", default="official_prefix")
    parser.add_argument("--summary-output", default="results/real_data_summary.csv")
    parser.add_argument("--sensitivity-output", default="results/real_data_candidate_sensitivity.csv")
    parser.add_argument("--prefix-edges", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--capacities", type=int, nargs="+", default=[64, 512, 2048, 8192])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--value-column", default="bytes")
    return parser.parse_args()


def run_real_experiment(edges: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_frame, sensitivity_frame) for one real edge table."""

    value_column = args.value_column
    reference = sparse_direct(edges, value_column)
    summary = matrix_summary(reference)

    # Stage (a): exact construction timing with equivalence checks.
    summary_rows = []
    timings: dict[str, float] = {}
    for method in ("sparse_direct", "pandas_groupby"):
        builder = METHODS[method]
        seconds = []
        matches = True
        for _ in range(args.repeats):
            start = time.perf_counter()
            built = builder(edges, value_column)
            seconds.append(time.perf_counter() - start)
            matches = matches and _equivalent(reference, built)
        timings[method] = min(seconds)
        summary_rows.append(
            {
                "source": "real",
                "regime": args.regime_label,
                "row_kind": "exact_method",
                "method": method,
                "n_edges": len(edges),
                "seconds_mean": sum(seconds) / len(seconds),
                "seconds_min": min(seconds),
                "edges_per_second": len(edges) / min(seconds),
                "nnz": summary.nnz,
                "nnz_per_edge": summary.nnz / len(edges),
                "density": summary.density,
                "total_weight": summary.total_weight,
                "gini_edge_weight": summary.gini_edge_weight,
                "output_matches_sparse": matches,
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
        )
    winner = min(timings, key=timings.get)

    # Stage (b): early-stream selector versus the measured winner.
    features = early_stream_features(edges, prefix_edges=args.prefix_edges, value_column=value_column)
    recommendation = recommend_method(features)
    summary_rows.append(
        {
            "source": "real",
            "regime": args.regime_label,
            "row_kind": "selector",
            "method": recommendation.recommended_exact_method,
            "n_edges": len(edges),
            "prefix_edges": features.observed_edges,
            "duplication_rate": features.duplication_rate,
            "nnz_growth_rate": features.nnz_growth_rate,
            "edge_gini": features.edge_gini,
            "top_edge_share": features.top_edge_share,
            "streaming_heavy_hitter_advice": recommendation.streaming_heavy_hitter_advice,
            "selector_reason": recommendation.reason,
            "observed_fastest_exact_method": winner,
            "recommendation_matches": recommendation.recommended_exact_method == winner,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )

    # Stage (c): candidate-capacity sweep against exact top-k.
    exact_top = top_k_edges(reference, k=args.top_k)
    sensitivity_rows = []
    for capacity in args.capacities:
        start = time.perf_counter()
        streaming_summary, approx_top = stream_summaries(
            edges,
            value_column=value_column,
            width=args.width,
            depth=args.depth,
            candidate_capacity=capacity,
        )
        elapsed = time.perf_counter() - start
        accuracy = streaming_accuracy(exact_top, approx_top, k=args.top_k)
        sensitivity_rows.append(
            {
                "regime": args.regime_label,
                "n_edges": len(edges),
                "sketch_width": args.width,
                "sketch_depth": args.depth,
                "candidate_capacity": capacity,
                "seconds": elapsed,
                "sketch_bytes": streaming_summary.sketch_bytes,
                "candidate_count": streaming_summary.candidate_count,
                "topk_recall": accuracy["topk_recall"],
                "median_relative_error": accuracy["median_relative_error"],
                "max_relative_error": accuracy["max_relative_error"],
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(sensitivity_rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found. Run `make fetch-official-prefix` (or "
            "scripts/fetch_official_prefix.py) first to build the real edge table."
        )
    edges = read_edges(input_path, value_column=args.value_column)
    summary_frame, sensitivity_frame = run_real_experiment(edges, args)

    for path, frame in ((args.summary_output, summary_frame), (args.sensitivity_output, sensitivity_frame)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"wrote {output} ({len(frame)} rows)")

    selector_row = summary_frame[summary_frame["row_kind"] == "selector"].iloc[0]
    print(f"measured winner: {selector_row['observed_fastest_exact_method']}")
    print(f"selector recommendation: {selector_row['method']} (match={selector_row['recommendation_matches']})")
    print(sensitivity_frame[["candidate_capacity", "seconds", "topk_recall", "median_relative_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
