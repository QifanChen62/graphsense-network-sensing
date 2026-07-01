#!/usr/bin/env python3
"""Create small reproducible sample data for local smoke tests.

The official challenge data is referenced in README.md and docs/challenge_notes.md.
This script intentionally does not download multi-GB traffic captures by default.
"""

from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from graphsense.synthetic import make_controlled_edges, write_synthetic_edges


OFFICIAL_URLS = [
    "https://graphchallenge.mit.edu/data-sets",
    "https://graphchallenge.mit.edu/challenges",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/sample/tiny_edges.csv")
    parser.add_argument("--n-edges", type=int, default=5000)
    parser.add_argument("--n-sources", type=int, default=512)
    parser.add_argument("--n-destinations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--regime", default="hotspot_zipf")
    parser.add_argument("--target-nnz", type=int)
    parser.add_argument("--density", type=float)
    parser.add_argument("--duplication-rate", type=float)
    parser.add_argument("--zipf-skew", type=float, default=0.0)
    parser.add_argument("--print-official-urls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_official_urls:
        for url in OFFICIAL_URLS:
            print(url)
    if args.target_nnz is not None or args.density is not None or args.duplication_rate is not None:
        frame = make_controlled_edges(
            n_edges=args.n_edges,
            n_sources=args.n_sources,
            n_destinations=args.n_destinations,
            target_nnz=args.target_nnz,
            density=args.density,
            duplication_rate=args.duplication_rate,
            zipf_skew=args.zipf_skew,
            seed=args.seed,
        )
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    else:
        path = write_synthetic_edges(
            args.output,
            n_edges=args.n_edges,
            n_sources=args.n_sources,
            n_destinations=args.n_destinations,
            seed=args.seed,
            regime=args.regime,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
