#!/usr/bin/env python3
"""Algorithm 2: conformal budget pricing per stream.

Reads the conformal gap lower bounds produced by conformal_certificate.py and
computes, for each source, the memory-minimal (width, capacity) that certifies
future-window top-k identifiability at level 1-alpha, or reports impossible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.pricing import price_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", default="results/conformal_certificates.csv")
    parser.add_argument("--output", default="results/conformal_budget_prices.csv")
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    certificates = pd.read_csv(args.certificates)
    bounds = (
        certificates.groupby("source")
        .agg(alpha=("alpha", "first"), gap_lower_bound=("conformal_gap_lower_bound", "first"))
        .reset_index()
    )
    rows = []
    for row in bounds.itertuples():
        bound = max(float(row.gap_lower_bound), 0.0)
        price = price_budget(bound, alpha=float(row.alpha), depth=args.depth)
        rows.append({"source": row.source, **price.to_dict()})
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    display = frame[["source", "feasible", "width", "capacity", "total_bytes"]].copy()
    display["total_mb"] = (display["total_bytes"] / 1e6).round(2)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
