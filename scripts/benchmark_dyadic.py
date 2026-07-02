#!/usr/bin/env python3
"""Compare dyadic Count-Min enumeration against SpaceSaving candidate tracking.

Two decoder families for the same sparse-recovery problem: per-record candidate
tracking (SpaceSaving) versus query-time hierarchical descent (dyadic Count-Min
tree). If diffuse-regime failure were an artifact of one data structure, the
other family would behave differently; matched failure supports the
identifiability explanation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import top_k_edges
from graphsense.benchmark import regime_shape
from graphsense.dyadic import dyadic_topk_recall
from graphsense.io import edges_to_sparse, read_edges
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--regimes", nargs="*", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--n-edges", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam", type=int, default=4096)
    parser.add_argument("--tracker-capacity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", default="results/dyadic_comparison.csv")
    parser.add_argument("--value-column", default="bytes")
    return parser.parse_args()


def _sources(args: argparse.Namespace):
    for input_path in args.inputs:
        edges = read_edges(input_path, value_column=args.value_column).head(args.n_edges)
        yield Path(input_path).stem, edges
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
        exact_top = top_k_edges(edges_to_sparse(edges, value_column=args.value_column), k=args.top_k)

        start = time.perf_counter()
        _, tracker_top = stream_summaries(
            edges, value_column=args.value_column, width=args.width, depth=args.depth, candidate_capacity=args.tracker_capacity
        )
        tracker_seconds = time.perf_counter() - start
        tracker_recall = streaming_accuracy(exact_top, tracker_top, k=args.top_k)["topk_recall"]

        start = time.perf_counter()
        dyadic = dyadic_topk_recall(
            edges, exact_top, k=args.top_k, width=args.width, depth=args.depth, beam=args.beam, value_column=args.value_column
        )
        dyadic_seconds = time.perf_counter() - start

        rows.append(
            {
                "source": name,
                "n_edges": len(edges),
                "top_k": args.top_k,
                "tracker_capacity": args.tracker_capacity,
                "tracker_recall": tracker_recall,
                "tracker_seconds": tracker_seconds,
                "dyadic_beam": args.beam,
                "dyadic_levels": dyadic["levels"],
                "dyadic_recall": dyadic["topk_recall"],
                "dyadic_sketch_bytes": dyadic["sketch_bytes"],
                "dyadic_seconds": dyadic_seconds,
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(frame[["source", "tracker_recall", "dyadic_recall", "dyadic_sketch_bytes", "tracker_seconds", "dyadic_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
