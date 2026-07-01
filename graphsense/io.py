"""Input/output helpers for anonymized edge records and sparse matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


REQUIRED_COLUMNS = ("src", "dst")


@dataclass(frozen=True)
class TrafficMatrix:
    """Sparse source-destination traffic matrix plus label maps."""

    matrix: sparse.csr_matrix
    src_labels: np.ndarray
    dst_labels: np.ndarray
    value_column: str


def read_edges(path: str | Path, value_column: str = "bytes") -> pd.DataFrame:
    """Read edge records with columns src, dst, and a numeric value column."""

    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if value_column not in frame.columns:
        if "packets" in frame.columns:
            value_column = "packets"
        else:
            frame[value_column] = 1.0

    frame = frame.loc[:, ["src", "dst", value_column]].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
    frame = frame[frame[value_column] > 0]
    return frame


def edges_to_sparse(
    edges: pd.DataFrame,
    value_column: str = "bytes",
    src_labels: Optional[Iterable[object]] = None,
    dst_labels: Optional[Iterable[object]] = None,
) -> TrafficMatrix:
    """Aggregate edge records into a CSR traffic matrix."""

    if value_column not in edges.columns:
        raise ValueError(f"value column {value_column!r} not present")

    if src_labels is None:
        src_codes, src_unique = pd.factorize(edges["src"], sort=True)
        src_labels_arr = np.asarray(src_unique, dtype=object)
    else:
        src_labels_arr = np.asarray(list(src_labels), dtype=object)
        src_index = {label: idx for idx, label in enumerate(src_labels_arr)}
        src_codes = edges["src"].map(src_index).to_numpy()

    if dst_labels is None:
        dst_codes, dst_unique = pd.factorize(edges["dst"], sort=True)
        dst_labels_arr = np.asarray(dst_unique, dtype=object)
    else:
        dst_labels_arr = np.asarray(list(dst_labels), dtype=object)
        dst_index = {label: idx for idx, label in enumerate(dst_labels_arr)}
        dst_codes = edges["dst"].map(dst_index).to_numpy()

    if np.any(pd.isna(src_codes)) or np.any(pd.isna(dst_codes)):
        raise ValueError("edges contain labels outside the supplied label maps")

    values = edges[value_column].to_numpy(dtype=np.float64)
    matrix = sparse.coo_matrix(
        (values, (src_codes.astype(np.int64), dst_codes.astype(np.int64))),
        shape=(len(src_labels_arr), len(dst_labels_arr)),
    ).tocsr()
    matrix.sum_duplicates()
    return TrafficMatrix(matrix=matrix, src_labels=src_labels_arr, dst_labels=dst_labels_arr, value_column=value_column)


def load_sparse_from_edges(path: str | Path, value_column: str = "bytes") -> TrafficMatrix:
    """Read edge records from disk and return an aggregated sparse matrix."""

    return edges_to_sparse(read_edges(path, value_column=value_column), value_column=value_column)


def load_matrix_market_traffic(path: str | Path, value_column: str = "bytes") -> TrafficMatrix:
    """Read a sparse source-destination matrix from Matrix Market coordinate format."""

    matrix = mmread(Path(path)).tocsr().astype(np.float64)
    matrix.sum_duplicates()
    src_labels = np.arange(matrix.shape[0], dtype=np.int64)
    dst_labels = np.arange(matrix.shape[1], dtype=np.int64)
    return TrafficMatrix(matrix=matrix, src_labels=src_labels, dst_labels=dst_labels, value_column=value_column)


def write_matrix_market_traffic(path: str | Path, traffic: TrafficMatrix) -> Path:
    """Write a traffic matrix in Matrix Market coordinate format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mmwrite(output, traffic.matrix.tocoo())
    return output
