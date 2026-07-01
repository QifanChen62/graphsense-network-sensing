"""Early-stream regime features and conservative method recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    total = float(sorted_values.sum())
    if total <= 0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1)
    return float((2 * np.dot(index, sorted_values) / (sorted_values.size * total)) - ((sorted_values.size + 1) / sorted_values.size))


@dataclass(frozen=True)
class EarlyStreamFeatures:
    observed_edges: int
    unique_pairs: int
    unique_sources: int
    unique_destinations: int
    duplication_rate: float
    nnz_growth_rate: float
    source_entropy_bits: float
    destination_entropy_bits: float
    edge_entropy_bits: float
    edge_gini: float
    top_edge_share: float
    top_source_share: float
    top_destination_share: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodRecommendation:
    recommended_exact_method: str
    streaming_heavy_hitter_advice: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def early_stream_features(edges: pd.DataFrame, prefix_edges: int = 50_000, value_column: str = "bytes") -> EarlyStreamFeatures:
    """Summarize cheap online features from the first records in an edge stream."""

    if prefix_edges <= 0:
        raise ValueError("prefix_edges must be positive")
    if value_column not in edges.columns:
        raise ValueError(f"missing value column {value_column!r}")

    prefix = edges.head(prefix_edges)
    observed_edges = int(len(prefix))
    if observed_edges == 0:
        return EarlyStreamFeatures(
            observed_edges=0,
            unique_pairs=0,
            unique_sources=0,
            unique_destinations=0,
            duplication_rate=0.0,
            nnz_growth_rate=0.0,
            source_entropy_bits=0.0,
            destination_entropy_bits=0.0,
            edge_entropy_bits=0.0,
            edge_gini=0.0,
            top_edge_share=0.0,
            top_source_share=0.0,
            top_destination_share=0.0,
        )

    grouped_edges = prefix.groupby(["src", "dst"], sort=False)[value_column].sum()
    source_strength = prefix.groupby("src", sort=False)[value_column].sum()
    destination_strength = prefix.groupby("dst", sort=False)[value_column].sum()
    total_weight = float(grouped_edges.sum())
    unique_pairs = int(len(grouped_edges))

    return EarlyStreamFeatures(
        observed_edges=observed_edges,
        unique_pairs=unique_pairs,
        unique_sources=int(source_strength.size),
        unique_destinations=int(destination_strength.size),
        duplication_rate=float(1.0 - (unique_pairs / observed_edges)),
        nnz_growth_rate=float(unique_pairs / observed_edges),
        source_entropy_bits=_entropy(source_strength.to_numpy(dtype=np.float64)),
        destination_entropy_bits=_entropy(destination_strength.to_numpy(dtype=np.float64)),
        edge_entropy_bits=_entropy(grouped_edges.to_numpy(dtype=np.float64)),
        edge_gini=_gini(grouped_edges.to_numpy(dtype=np.float64)),
        top_edge_share=float(grouped_edges.max() / total_weight) if total_weight > 0 else 0.0,
        top_source_share=float(source_strength.max() / total_weight) if total_weight > 0 else 0.0,
        top_destination_share=float(destination_strength.max() / total_weight) if total_weight > 0 else 0.0,
    )


def recommend_method(features: EarlyStreamFeatures) -> MethodRecommendation:
    """Recommend a construction path from online traffic-shape features.

    The thresholds are deliberately simple and conservative. They are meant to
    make the benchmark's shape dependence explicit, not to replace profiling.
    """

    concentrated = (
        features.duplication_rate >= 0.55
        or features.edge_gini >= 0.82
        or features.top_edge_share >= 0.01
        or max(features.top_source_share, features.top_destination_share) >= 0.05
    )
    diffuse = features.duplication_rate <= 0.20 and features.nnz_growth_rate >= 0.80

    if concentrated:
        return MethodRecommendation(
            recommended_exact_method="pandas_groupby",
            streaming_heavy_hitter_advice="viable_for_concentrated_heavy_hitters",
            reason=(
                "early prefix is concentrated: duplicate pairs, high edge/source concentration, "
                "or both make pre-aggregation and sketch candidates plausible"
            ),
        )
    if diffuse:
        return MethodRecommendation(
            recommended_exact_method="sparse_direct",
            streaming_heavy_hitter_advice="not_recommended_for_topk_without_large_candidates",
            reason="early prefix is diffuse: nnz grows nearly with the stream and candidate discovery is likely limiting",
        )
    return MethodRecommendation(
        recommended_exact_method="sparse_direct",
        streaming_heavy_hitter_advice="profile_before_using_for_topk",
        reason="early prefix is mixed; sparse direct is the safer default until profiling shows enough concentration",
    )
