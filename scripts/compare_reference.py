#!/usr/bin/env python3
"""Compare our sparse pipeline with a reference-formula implementation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import matrix_summary
from graphsense.io import edges_to_sparse
from graphsense.reference import official_formula_reference
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/reference_comparison.csv")
    parser.add_argument("--n-edges", type=int, default=100000)
    parser.add_argument("--regime", default="community_bursty")
    parser.add_argument("--n-sources", type=int, default=2048)
    parser.add_argument("--n-destinations", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = make_synthetic_edges(
        n_edges=args.n_edges,
        n_sources=args.n_sources,
        n_destinations=args.n_destinations,
        regime=args.regime,
        seed=args.seed,
        label_mode="int",
    )

    start = time.perf_counter()
    ours = matrix_summary(edges_to_sparse(edges))
    ours_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference = official_formula_reference(edges)
    reference_seconds = time.perf_counter() - start

    rows = [
        {
            "implementation": "ours_sparse_pipeline",
            "source": "this_repo_scipy_sparse",
            "regime": args.regime,
            "n_edges": args.n_edges,
            "seconds": ours_seconds,
            "n_sources": ours.n_sources,
            "n_destinations": ours.n_destinations,
            "nnz": ours.nnz,
            "total_weight": ours.total_weight,
            "density": ours.density,
            "extra_reference_similarity": False,
        },
        {
            "implementation": "official_formula_reference",
            "source": "graphchallenge_paper_table_i_style",
            "regime": args.regime,
            "n_edges": args.n_edges,
            "seconds": reference_seconds,
            "n_sources": reference.n_sources,
            "n_destinations": reference.n_destinations,
            "nnz": reference.nnz,
            "total_weight": reference.total_weight,
            "density": reference.nnz / max(reference.n_sources * reference.n_destinations, 1),
            "extra_reference_similarity": True,
            "source_similarity_nnz": reference.source_similarity_nnz,
            "destination_similarity_nnz": reference.destination_similarity_nnz,
        },
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
