"""Sparse traffic-matrix analytics used by the benchmark and paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from .io import TrafficMatrix


@dataclass(frozen=True)
class MatrixSummary:
    n_sources: int
    n_destinations: int
    nnz: int
    density: float
    total_weight: float
    row_entropy_bits: float
    column_entropy_bits: float
    edge_entropy_bits: float
    top_singular_value: float
    gini_edge_weight: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.astype(np.float64))
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1)
    return float((2 * np.dot(index, sorted_values) / (sorted_values.size * total)) - ((sorted_values.size + 1) / sorted_values.size))


def _power_iteration_top_singular_value(matrix: sparse.csr_matrix, n_iter: int = 4) -> float:
    rng = np.random.default_rng(123)
    vector = rng.normal(size=matrix.shape[1])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0.0
    vector = vector / norm
    for _ in range(n_iter):
        left = matrix @ vector
        left_norm = np.linalg.norm(left)
        if left_norm == 0:
            return 0.0
        right = matrix.T @ (left / left_norm)
        right_norm = np.linalg.norm(right)
        if right_norm == 0:
            return 0.0
        vector = right / right_norm
    return float(np.linalg.norm(matrix @ vector))


def matrix_summary(traffic: TrafficMatrix) -> MatrixSummary:
    """Compute scalar graph/statistical summaries from a sparse matrix."""

    matrix = traffic.matrix.tocsr()
    rows, cols = matrix.shape
    total_cells = max(rows * cols, 1)
    row_strength = np.asarray(matrix.sum(axis=1)).ravel()
    col_strength = np.asarray(matrix.sum(axis=0)).ravel()
    data = matrix.data

    if min(matrix.shape) > 1 and matrix.nnz > 0 and matrix.nnz <= 200_000:
        try:
            top_sv = float(svds(matrix.astype(np.float64), k=1, return_singular_vectors=False)[0])
        except Exception:
            top_sv = float(np.linalg.norm(data))
    elif min(matrix.shape) > 1 and matrix.nnz > 0:
        top_sv = _power_iteration_top_singular_value(matrix.astype(np.float64))
    else:
        top_sv = float(np.linalg.norm(data))

    return MatrixSummary(
        n_sources=rows,
        n_destinations=cols,
        nnz=int(matrix.nnz),
        density=float(matrix.nnz / total_cells),
        total_weight=float(data.sum()),
        row_entropy_bits=_entropy(row_strength),
        column_entropy_bits=_entropy(col_strength),
        edge_entropy_bits=_entropy(data),
        top_singular_value=top_sv,
        gini_edge_weight=_gini(data),
    )


def top_k_edges(traffic: TrafficMatrix, k: int = 20) -> pd.DataFrame:
    """Return the heaviest source-destination pairs in descending order."""

    matrix = traffic.matrix.tocoo()
    if matrix.nnz == 0:
        return pd.DataFrame(columns=["rank", "src", "dst", "weight"])

    order = np.argsort(matrix.data)[::-1][:k]
    rows = matrix.row[order]
    cols = matrix.col[order]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "src": traffic.src_labels[rows],
            "dst": traffic.dst_labels[cols],
            "weight": matrix.data[order],
        }
    )


def strength_distribution(traffic: TrafficMatrix, axis: str = "src") -> pd.DataFrame:
    """Return source or destination strength and degree distributions."""

    matrix = traffic.matrix.tocsr()
    if axis == "src":
        labels: Sequence[object] = traffic.src_labels
        strength = np.asarray(matrix.sum(axis=1)).ravel()
        degree = np.diff(matrix.indptr)
        label_col = "src"
    elif axis == "dst":
        csc = matrix.tocsc()
        labels = traffic.dst_labels
        strength = np.asarray(csc.sum(axis=0)).ravel()
        degree = np.diff(csc.indptr)
        label_col = "dst"
    else:
        raise ValueError("axis must be 'src' or 'dst'")

    return pd.DataFrame({label_col: labels, "strength": strength, "degree": degree}).sort_values(
        ["strength", "degree"], ascending=False
    )


def save_analysis(prefix: str, traffic: TrafficMatrix, top_k: int = 20) -> None:
    """Write analysis CSV files using a common output prefix."""

    pd.DataFrame([matrix_summary(traffic).to_dict()]).to_csv(f"{prefix}_summary.csv", index=False)
    top_k_edges(traffic, k=top_k).to_csv(f"{prefix}_top_edges.csv", index=False)
    strength_distribution(traffic, axis="src").to_csv(f"{prefix}_src_strength.csv", index=False)
    strength_distribution(traffic, axis="dst").to_csv(f"{prefix}_dst_strength.csv", index=False)
