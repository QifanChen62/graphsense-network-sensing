"""Transparent local baselines for sparse traffic-matrix benchmarks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .io import TrafficMatrix, edges_to_sparse


def counter_baseline(edges: pd.DataFrame, value_column: str = "bytes") -> TrafficMatrix:
    """Aggregate edges with Python dictionaries before building a sparse matrix."""

    totals: dict[tuple[object, object], float] = defaultdict(float)
    for row in edges.itertuples(index=False):
        row_dict = row._asdict()
        totals[(row_dict["src"], row_dict["dst"])] += float(row_dict[value_column])

    if not totals:
        return edges_to_sparse(pd.DataFrame(columns=["src", "dst", value_column]), value_column=value_column)

    src, dst, values = zip(*((pair[0], pair[1], value) for pair, value in totals.items()))
    aggregated = pd.DataFrame({"src": src, "dst": dst, value_column: np.asarray(values)})
    return edges_to_sparse(aggregated, value_column=value_column)


def pandas_groupby_baseline(edges: pd.DataFrame, value_column: str = "bytes") -> TrafficMatrix:
    """Aggregate edges with pandas groupby before building a sparse matrix."""

    grouped = edges.groupby(["src", "dst"], as_index=False)[value_column].sum()
    return edges_to_sparse(grouped, value_column=value_column)
