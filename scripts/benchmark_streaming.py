#!/usr/bin/env python3
"""Benchmark exact sparse analytics against streaming/sketch analytics."""

from __future__ import annotations

import argparse
import platform
import resource
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import matrix_summary, top_k_edges
from graphsense.benchmark import regime_shape
from graphsense.io import edges_to_sparse
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import make_synthetic_edges


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/streaming_benchmark.csv")
    parser.add_argument("--sizes", nargs="+", type=int, default=[25000, 100000, 500000])
    parser.add_argument("--regimes", nargs="+", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--sketch-width", type=int, default=8192)
    parser.add_argument("--sketch-depth", type=int, default=5)
    parser.add_argument("--candidate-capacity", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for regime in args.regimes:
        for n_edges in args.sizes:
            n_sources, n_destinations = regime_shape(regime, n_edges)
            edges = make_synthetic_edges(
                n_edges=n_edges,
                n_sources=n_sources,
                n_destinations=n_destinations,
                seed=args.seed + n_edges + len(regime),
                regime=regime,
                label_mode="int",
            )

            start = time.perf_counter()
            traffic = edges_to_sparse(edges)
            exact_summary = matrix_summary(traffic)
            exact_top = top_k_edges(traffic, k=args.top_k)
            exact_seconds = time.perf_counter() - start
            rows.append(
                {
                    "regime": regime,
                    "n_edges": n_edges,
                    "method": "exact_sparse",
                    "seconds": exact_seconds,
                    "max_rss_mb": _rss_mb(),
                    "nnz": exact_summary.nnz,
                    "density": exact_summary.density,
                    "total_weight": exact_summary.total_weight,
                    "row_entropy_bits": exact_summary.row_entropy_bits,
                    "column_entropy_bits": exact_summary.column_entropy_bits,
                    "topk_recall": 1.0,
                    "median_relative_error": 0.0,
                    "sketch_bytes": 0,
                }
            )

            start = time.perf_counter()
            streaming_summary, approx_top = stream_summaries(
                edges,
                width=args.sketch_width,
                depth=args.sketch_depth,
                candidate_capacity=args.candidate_capacity,
            )
            streaming_seconds = time.perf_counter() - start
            accuracy = streaming_accuracy(exact_top, approx_top, k=args.top_k)
            rows.append(
                {
                    "regime": regime,
                    "n_edges": n_edges,
                    "method": "streaming_sketch",
                    "seconds": streaming_seconds,
                    "max_rss_mb": _rss_mb(),
                    "nnz": "",
                    "density": "",
                    "total_weight": streaming_summary.total_weight,
                    "row_entropy_bits": streaming_summary.row_entropy_bits,
                    "column_entropy_bits": streaming_summary.column_entropy_bits,
                    "topk_recall": accuracy["topk_recall"],
                    "median_relative_error": accuracy["median_relative_error"],
                    "max_relative_error": accuracy["max_relative_error"],
                    "sketch_bytes": streaming_summary.sketch_bytes,
                    "candidate_count": streaming_summary.candidate_count,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(frame.groupby(["regime", "method", "n_edges"])["seconds"].mean().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
