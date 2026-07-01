#!/usr/bin/env python3
"""Run local benchmark suite and save CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark_summary.csv")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 25000, 100000])
    parser.add_argument("--regimes", nargs="+", default=["hotspot_zipf"])
    parser.add_argument("--methods", nargs="+", default=["sparse_direct", "pandas_groupby", "python_counter"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = run_benchmark(
        sizes=args.sizes,
        repeats=args.repeats,
        methods=tuple(args.methods),
        regimes=tuple(args.regimes),
        seed=args.seed,
    )
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    print(frame.groupby(["regime", "method", "n_edges"])["seconds"].mean().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
