"""Reference-style sparse traffic-matrix computations.

The GraphChallenge Anonymized Network Sensing paper describes sparse matrix
construction and sensing quantities in terms of a traffic matrix T. This module
keeps a small local reference implementation of those formulas for reproducible
small-sample comparison when the full official code/data path is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .io import edges_to_sparse


@dataclass(frozen=True)
class ReferenceSummary:
    n_sources: int
    n_destinations: int
    nnz: int
    total_weight: float
    source_activity_nnz: int
    destination_activity_nnz: int
    pair_activity_nnz: int
    source_similarity_nnz: int
    destination_similarity_nnz: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def official_formula_reference(edges: pd.DataFrame, value_column: str = "bytes") -> ReferenceSummary:
    """Compute Table-I-style sparse sensing quantities from edge records."""

    traffic = edges_to_sparse(edges, value_column=value_column)
    matrix = traffic.matrix.tocsr()
    source_activity = matrix.sum(axis=1)
    destination_activity = matrix.sum(axis=0)
    pair_activity = matrix.copy()
    pair_activity.data = np.ones_like(pair_activity.data)
    source_similarity = pair_activity @ pair_activity.T
    destination_similarity = pair_activity.T @ pair_activity
    return ReferenceSummary(
        n_sources=matrix.shape[0],
        n_destinations=matrix.shape[1],
        nnz=matrix.nnz,
        total_weight=float(matrix.data.sum()),
        source_activity_nnz=int(np.count_nonzero(np.asarray(source_activity).ravel())),
        destination_activity_nnz=int(np.count_nonzero(np.asarray(destination_activity).ravel())),
        pair_activity_nnz=int(pair_activity.nnz),
        source_similarity_nnz=int(source_similarity.nnz),
        destination_similarity_nnz=int(destination_similarity.nnz),
    )
