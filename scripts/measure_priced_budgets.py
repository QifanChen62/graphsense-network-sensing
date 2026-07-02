#!/usr/bin/env python3
"""Measure Recall@20 at the conformally priced budgets (certified region).

Runs the SpaceSaving tracker at each feasible source's memory-minimal
certified budget from Algorithm 2, so the phase diagram contains measured
points at and beyond I_k = 1 and the one-sided soundness claim is testable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import top_k_edges
from graphsense.benchmark import regime_shape
from graphsense.io import edges_to_sparse, read_edges
from graphsense.streaming import stream_summaries, streaming_accuracy
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="results/conformal_budget_prices.csv")
    parser.add_argument("--ctu-edges", default="data/real/ctu13_edges.csv")
    parser.add_argument("--output", default="results/priced_budget_recall.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-edges", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--sources", nargs="*", default=["hotspot_zipf", "ctu13_edges"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = pd.read_csv(args.prices).set_index("source")
    rows = []
    for source in args.sources:
        price = prices.loc[source]
        if not bool(price["feasible"]):
            continue
        if source == "ctu13_edges":
            edges = read_edges(args.ctu_edges)
        else:
            n_sources, n_destinations = regime_shape(source, args.n_edges)
            edges = make_synthetic_edges(
                n_edges=args.n_edges,
                n_sources=n_sources,
                n_destinations=n_destinations,
                seed=args.seed + args.n_edges + len(source),
                regime=source,
                label_mode="int",
            )
        exact_top = top_k_edges(edges_to_sparse(edges), k=args.top_k)
        _, approx_top = stream_summaries(
            edges, width=int(price["width"]), depth=5, candidate_capacity=int(price["capacity"])
        )
        accuracy = streaming_accuracy(exact_top, approx_top, k=args.top_k)
        rows.append(
            {
                "source": source,
                "n_edges": len(edges),
                "width": int(price["width"]),
                "capacity": int(price["capacity"]),
                "total_bytes": int(price["total_bytes"]),
                "topk_recall": accuracy["topk_recall"],
                "median_relative_error": accuracy["median_relative_error"],
            }
        )
        print(f"{source}: recall {accuracy['topk_recall']} at priced budget {price['total_bytes']/1e6:.1f} MB")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
