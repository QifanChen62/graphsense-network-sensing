#!/usr/bin/env python3
"""Run the sparse analytics pipeline on edge-record CSV/Parquet input."""

from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import save_analysis
from graphsense.io import load_sparse_from_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/Parquet edge records with src,dst,bytes columns")
    parser.add_argument("--value-column", default="bytes")
    parser.add_argument("--out-prefix", default="results/run")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traffic = load_sparse_from_edges(args.input, value_column=args.value_column)
    save_analysis(args.out_prefix, traffic, top_k=args.top_k)
    print(f"wrote {args.out_prefix}_summary.csv")
    print(f"wrote {args.out_prefix}_top_edges.csv")
    print(f"wrote {args.out_prefix}_src_strength.csv")
    print(f"wrote {args.out_prefix}_dst_strength.csv")


if __name__ == "__main__":
    main()
