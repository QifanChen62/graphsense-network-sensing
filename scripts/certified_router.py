#!/usr/bin/env python3
"""Algorithm 2: certified router over all sources.

Profiles both exact methods on each stream's prefix, reads the conformal gap
lower bound, prices sketch certification, and emits one operational decision
per source: exact path, sketch certification and price, backend, online and
batch routes, and the reason.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.baselines import pandas_groupby_baseline
from graphsense.benchmark import regime_shape, sparse_direct
from graphsense.io import read_edges
from graphsense.router import route
from graphsense.selector import early_stream_features, recommend_method
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", default="results/conformal_certificates.csv")
    parser.add_argument("--inputs", nargs="*", default=["data/real/ctu13_edges.csv", "data/real/official_prefix_edges.csv"])
    parser.add_argument("--regimes", nargs="*", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--n-edges", type=int, default=5_000_000)
    parser.add_argument("--prefix-edges", type=int, default=50_000)
    parser.add_argument("--memory-budget-mb", type=float, default=64.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/router_summary.csv")
    return parser.parse_args()


def _sources(args: argparse.Namespace):
    for input_path in args.inputs:
        name = Path(input_path).stem
        yield name, read_edges(input_path), name == "official_prefix_edges"
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
        yield regime, edges, False


def _profile(edges: pd.DataFrame, repeats: int) -> tuple[float, float]:
    sparse_seconds = min(
        _timed(lambda: sparse_direct(edges)) for _ in range(repeats)
    )
    pandas_seconds = min(
        _timed(lambda: pandas_groupby_baseline(edges)) for _ in range(repeats)
    )
    return sparse_seconds, pandas_seconds


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    bounds = pd.read_csv(args.certificates).groupby("source")["conformal_gap_lower_bound"].first()
    budget_bytes = int(args.memory_budget_mb * 1e6)
    rows = []
    for name, edges, native in _sources(args):
        prefix = edges.head(args.prefix_edges)
        sparse_seconds, pandas_seconds = _profile(prefix, args.repeats)
        heuristic = recommend_method(early_stream_features(prefix, prefix_edges=args.prefix_edges)).recommended_exact_method
        bound = max(float(bounds.get(name, bounds.get(name.replace("_edges", ""), 0.0))), 0.0)
        decision = route(
            prefix_sparse_seconds=sparse_seconds,
            prefix_pandas_seconds=pandas_seconds,
            gap_lower_bound=bound,
            memory_budget_bytes=budget_bytes,
            heuristic_choice=heuristic,
            native_address_space=native,
        )
        rows.append(
            {
                "source": name,
                "prefix_sparse_seconds": sparse_seconds,
                "prefix_pandas_seconds": pandas_seconds,
                "heuristic_choice": heuristic,
                "gap_lower_bound": bound,
                "memory_budget_mb": args.memory_budget_mb,
                **decision.to_dict(),
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(
        frame[
            ["source", "exact_choice", "exact_source", "sketch_decision", "certified_price_bytes", "batch_route"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
