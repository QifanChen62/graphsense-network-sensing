#!/usr/bin/env python3
"""Identifiability-aware Recall@k across all sources at the default budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import regime_shape
from graphsense.io import read_edges
from graphsense.metrics import identifiability_report
from graphsense.streaming import stream_summaries
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=["data/real/ctu13_edges.csv", "data/real/official_prefix_edges.csv"])
    parser.add_argument("--regimes", nargs="*", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--n-edges", type=int, default=500_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", default="results/identifiable_recall.csv")
    return parser.parse_args()


def _sources(args: argparse.Namespace):
    for input_path in args.inputs:
        yield Path(input_path).stem, read_edges(input_path)
    for regime in args.regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        yield regime, make_synthetic_edges(
            n_edges=args.n_edges,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )


def main() -> None:
    args = parse_args()
    rows = []
    for name, edges in _sources(args):
        _, approx_top = stream_summaries(
            edges, width=args.width, depth=args.depth, candidate_capacity=args.capacity
        )
        report = identifiability_report(
            edges, approx_top, k=args.top_k, width=args.width, candidate_capacity=args.capacity
        )
        rows.append({"source": name, "n_edges": len(edges), **report.to_dict()})
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
