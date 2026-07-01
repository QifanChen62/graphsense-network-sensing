#!/usr/bin/env python3
"""Small official-format compatibility smoke test.

This does not download official GraphChallenge data. It verifies that the
pipeline can round-trip a sparse traffic matrix through Matrix Market, a common
sparse coordinate exchange format used around GraphBLAS workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.analytics import matrix_summary
from graphsense.io import edges_to_sparse, load_matrix_market_traffic, write_matrix_market_traffic
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-output", default="data/official_format/tiny_traffic_matrix.mtx")
    parser.add_argument("--summary-output", default="results/official_format_smoke.csv")
    parser.add_argument("--n-edges", type=int, default=10_000)
    parser.add_argument("--n-sources", type=int, default=256)
    parser.add_argument("--n-destinations", type=int, default=256)
    parser.add_argument("--regime", default="community_bursty")
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = make_synthetic_edges(
        n_edges=args.n_edges,
        n_sources=args.n_sources,
        n_destinations=args.n_destinations,
        seed=args.seed,
        regime=args.regime,
        label_mode="int",
    )
    from_edges = edges_to_sparse(edges)
    matrix_path = write_matrix_market_traffic(args.matrix_output, from_edges)
    from_matrix_market = load_matrix_market_traffic(matrix_path)

    edge_summary = matrix_summary(from_edges)
    mtx_summary = matrix_summary(from_matrix_market)
    matrix_difference = (from_edges.matrix - from_matrix_market.matrix).tocoo()
    exact_match = bool(
        from_edges.matrix.shape == from_matrix_market.matrix.shape
        and matrix_difference.nnz == 0
        and abs(edge_summary.total_weight - mtx_summary.total_weight) < 1e-9
    )

    rows = [
        {
            "check": "edge_table_sparse",
            "input": "synthetic_edge_table",
            "format": "src_dst_weight_records",
            "official_data_used": False,
            "n_edges": args.n_edges,
            "n_sources": edge_summary.n_sources,
            "n_destinations": edge_summary.n_destinations,
            "nnz": edge_summary.nnz,
            "density": edge_summary.density,
            "total_weight": edge_summary.total_weight,
            "matches_edge_table_sparse": True,
        },
        {
            "check": "matrix_market_reload",
            "input": str(matrix_path),
            "format": "graphblas_style_sparse_coordinate_matrix",
            "official_data_used": False,
            "n_edges": args.n_edges,
            "n_sources": mtx_summary.n_sources,
            "n_destinations": mtx_summary.n_destinations,
            "nnz": mtx_summary.nnz,
            "density": mtx_summary.density,
            "total_weight": mtx_summary.total_weight,
            "matches_edge_table_sparse": exact_match,
        },
    ]

    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"wrote {matrix_path}")
    print(f"wrote {output}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
