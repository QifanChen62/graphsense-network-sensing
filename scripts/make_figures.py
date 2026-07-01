#!/usr/bin/env python3
"""Generate figures from saved benchmark CSVs."""

from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from graphsense.plotting import make_benchmark_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/benchmark_summary.csv")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in make_benchmark_plots(args.input, args.outdir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
