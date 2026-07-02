#!/usr/bin/env python3
"""Compare SciPy sparse construction against a SuiteSparse:GraphBLAS baseline.

Times three paths on the same src,dst,bytes edge table: (1) pandas factorize
label mapping (shared preprocessing for compact matrices), (2) SciPy
coo->csr duplicate-summing construction on factorized codes, (3) GraphBLAS
Matrix.from_coo on factorized codes, and (4) GraphBLAS from_coo directly in
the native 2^32 x 2^32 anonymized-address space (hypersparse), which needs no
factorization and which SciPy CSR cannot represent (the indptr array alone
would be tens of GB).

Requires python-graphblas with suitesparse (numpy<2 wheels on Python 3.9; use
a side venv if the main environment runs numpy>=2).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/real/official_prefix_edges.csv")
    parser.add_argument("--output", default="results/graphblas_comparison.csv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--value-column", default="bytes")
    return parser.parse_args()


def time_min(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    args = parse_args()
    import graphblas as gb
    import scipy.sparse as sp

    edges = pd.read_csv(args.input)
    weights = edges[args.value_column].to_numpy(dtype=np.float64)

    start = time.perf_counter()
    src_codes, src_unique = pd.factorize(edges["src"], sort=True)
    dst_codes, dst_unique = pd.factorize(edges["dst"], sort=True)
    factorize_seconds = time.perf_counter() - start
    shape = (len(src_unique), len(dst_unique))

    scipy_seconds = time_min(
        lambda: sp.coo_matrix((weights, (src_codes, dst_codes)), shape=shape).tocsr(), args.repeats
    )
    grb_seconds = time_min(
        lambda: gb.Matrix.from_coo(src_codes, dst_codes, weights, nrows=shape[0], ncols=shape[1], dup_op=gb.binary.plus),
        args.repeats,
    )
    src_native = edges["src"].to_numpy(dtype=np.uint64)
    dst_native = edges["dst"].to_numpy(dtype=np.uint64)
    grb_native_seconds = time_min(
        lambda: gb.Matrix.from_coo(src_native, dst_native, weights, nrows=2**32, ncols=2**32, dup_op=gb.binary.plus),
        args.repeats,
    )

    scipy_matrix = sp.coo_matrix((weights, (src_codes, dst_codes)), shape=shape).tocsr()
    grb_matrix = gb.Matrix.from_coo(src_codes, dst_codes, weights, nrows=shape[0], ncols=shape[1], dup_op=gb.binary.plus)
    matches = bool(
        scipy_matrix.nnz == grb_matrix.nvals
        and abs(float(scipy_matrix.sum()) - float(grb_matrix.reduce_scalar(gb.monoid.plus).value)) < 1e-6
    )

    frame = pd.DataFrame(
        [
            {
                "input": args.input,
                "n_edges": len(edges),
                "nnz": int(scipy_matrix.nnz),
                "factorize_seconds": factorize_seconds,
                "scipy_csr_seconds": scipy_seconds,
                "graphblas_from_coo_seconds": grb_seconds,
                "graphblas_native_2e32_seconds": grb_native_seconds,
                "outputs_match": matches,
                "graphblas_backend": gb.backend,
            }
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output}")
    print(frame.T.to_string())


if __name__ == "__main__":
    main()
