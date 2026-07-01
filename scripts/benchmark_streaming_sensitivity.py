#!/usr/bin/env python3
"""Sensitivity study for streaming/sketch width and heavy-hitter accuracy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import top_k_edges
from graphsense.benchmark import regime_shape
from graphsense.io import edges_to_sparse
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/streaming_sensitivity.csv")
    parser.add_argument("--n-edges", type=int, default=100000)
    parser.add_argument("--regimes", nargs="+", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--widths", nargs="+", type=int, default=[512, 2048, 8192, 32768])
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--candidate-capacity", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for regime in args.regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        edges = make_synthetic_edges(
            n_edges=args.n_edges,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )
        exact_top = top_k_edges(edges_to_sparse(edges), k=args.top_k)

        for width in args.widths:
            start = time.perf_counter()
            summary, approx_top = stream_summaries(
                edges,
                width=width,
                depth=args.depth,
                candidate_capacity=args.candidate_capacity,
            )
            elapsed = time.perf_counter() - start
            accuracy = streaming_accuracy(exact_top, approx_top, k=args.top_k)
            rows.append(
                {
                    "regime": regime,
                    "n_edges": args.n_edges,
                    "sketch_width": width,
                    "sketch_depth": args.depth,
                    "candidate_capacity": args.candidate_capacity,
                    "seconds": elapsed,
                    "sketch_bytes": summary.sketch_bytes,
                    "topk_recall": accuracy["topk_recall"],
                    "median_relative_error": accuracy["median_relative_error"],
                    "max_relative_error": accuracy["max_relative_error"],
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
