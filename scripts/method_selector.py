#!/usr/bin/env python3
"""Evaluate early-stream method recommendations against saved benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import regime_shape
from graphsense.selector import early_stream_features, recommend_method
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/method_selector_summary.csv")
    parser.add_argument("--benchmark-csv", default="results/benchmark_large_summary.csv")
    parser.add_argument("--n-edges", type=int, default=5_000_000)
    parser.add_argument("--prefix-edges", type=int, default=50_000)
    parser.add_argument("--regimes", nargs="+", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--materialize-full-stream",
        action="store_true",
        help="Generate the full target stream and take the first prefix records. This is slower but matches online-prefix wording exactly.",
    )
    return parser.parse_args()


def fastest_methods(path: str | Path, n_edges: int) -> dict[str, str]:
    frame = pd.read_csv(path)
    frame = frame[(frame["n_edges"] == n_edges) & frame["method"].isin(["sparse_direct", "pandas_groupby"])]
    if frame.empty:
        raise ValueError(f"no exact-method benchmark rows for n_edges={n_edges} in {path}")
    grouped = frame.groupby(["regime", "method"], as_index=False)["seconds"].mean()
    winners = grouped.loc[grouped.groupby("regime")["seconds"].idxmin()]
    return dict(zip(winners["regime"], winners["method"]))


def main() -> None:
    args = parse_args()
    winners = fastest_methods(args.benchmark_csv, args.n_edges)
    rows = []
    for regime in args.regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        n_generated = args.n_edges if args.materialize_full_stream else args.prefix_edges
        edges = make_synthetic_edges(
            n_edges=n_generated,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )
        prefix = edges.head(args.prefix_edges)
        features = early_stream_features(prefix, prefix_edges=args.prefix_edges)
        recommendation = recommend_method(features)
        observed = winners.get(regime, "missing")
        rows.append(
            {
                "regime": regime,
                "target_n_edges": args.n_edges,
                "prefix_edges": args.prefix_edges,
                **features.to_dict(),
                **recommendation.to_dict(),
                "observed_fastest_exact_method": observed,
                "recommendation_matches": recommendation.recommended_exact_method == observed,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    columns = [
        "regime",
        "duplication_rate",
        "nnz_growth_rate",
        "edge_gini",
        "top_edge_share",
        "recommended_exact_method",
        "observed_fastest_exact_method",
        "recommendation_matches",
    ]
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
