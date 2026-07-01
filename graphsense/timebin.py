"""Time-bin sensing summaries for anonymized traffic edge records."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def time_bin_summary(edges: pd.DataFrame, value_column: str = "bytes") -> pd.DataFrame:
    """Compute interpretable sensing summaries for each time bin."""

    rows = []
    for time_bin, part in edges.groupby("time_bin", sort=True):
        grouped_edges = part.groupby(["src", "dst"], as_index=False)[value_column].sum()
        row_strength = part.groupby("src")[value_column].sum().to_numpy(dtype=np.float64)
        col_strength = part.groupby("dst")[value_column].sum().to_numpy(dtype=np.float64)
        weights = grouped_edges[value_column].to_numpy(dtype=np.float64)
        total = float(weights.sum())
        rows.append(
            {
                "time_bin": int(time_bin),
                "n_edges": int(len(part)),
                "nnz": int(len(grouped_edges)),
                "total_weight": total,
                "source_entropy_bits": _entropy(row_strength),
                "destination_entropy_bits": _entropy(col_strength),
                "edge_gini": _gini(weights),
                "top_edge_share": float(weights.max() / total) if total > 0 and weights.size else 0.0,
                "top_source_share": float(row_strength.max() / total) if total > 0 and row_strength.size else 0.0,
                "top_destination_share": float(col_strength.max() / total) if total > 0 and col_strength.size else 0.0,
                "unique_sources": int(part["src"].nunique()),
                "unique_destinations": int(part["dst"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def add_timebin_anomaly_score(summary: pd.DataFrame) -> pd.DataFrame:
    """Add robust z-score style anomaly columns for plotting."""

    frame = summary.copy()
    anomaly_columns = [
        "total_weight",
        "nnz",
        "source_entropy_bits",
        "destination_entropy_bits",
        "edge_gini",
        "top_edge_share",
        "top_source_share",
        "top_destination_share",
        "unique_sources",
        "unique_destinations",
    ]
    for column in anomaly_columns:
        median = float(frame[column].median())
        mad = float((frame[column] - median).abs().median())
        scale = mad if mad > 0 else float(frame[column].std(ddof=0) or 1.0)
        frame[f"{column}_robust_z"] = (frame[column] - median) / scale
    frame["anomaly_score"] = sum(
        frame[f"{column}_robust_z"].clip(lower=0)
        for column in [
            "total_weight",
            "nnz",
            "source_entropy_bits",
            "destination_entropy_bits",
            "top_edge_share",
            "top_source_share",
            "top_destination_share",
            "unique_sources",
            "unique_destinations",
        ]
    )
    return frame
