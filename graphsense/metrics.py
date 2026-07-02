"""Identifiability-aware top-k evaluation.

Strict Recall@k treats top-k membership as ground truth even when weights tie
at the boundary, where membership is arbitrary. The identifiable core ties the
tolerance to the sketch budget's error resolution, gamma = 2(eps + 1/(c+1)):
only pairs separated from the (k+1)-st weight by more than gamma * W are
required to be recovered. An empty core means top-k identity is not defined at
this budget, and the metric reports that instead of a recall of zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class IdentifiabilityReport:
    strict_recall: float
    tie_aware_recall: float
    identifiable_core_size: int
    identifiable_recall: float | None
    identifiable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def budget_gamma(width: int, candidate_capacity: int) -> float:
    """Budget-matched tolerance: the error resolution of the sketch pair."""

    return 2.0 * (math.e / width + 1.0 / (candidate_capacity + 1))


def identifiability_report(
    edges: pd.DataFrame,
    approx_top: pd.DataFrame,
    k: int,
    width: int,
    candidate_capacity: int,
    value_column: str = "bytes",
) -> IdentifiabilityReport:
    """Strict, tie-aware, and identifiable Recall@k against exact aggregation."""

    grouped = edges.groupby(["src", "dst"], sort=False)[value_column].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    exact_pairs = list(grouped.index[:k])
    w_k = float(grouped.iloc[k - 1]) if len(grouped) >= k else 0.0
    w_next = float(grouped.iloc[k]) if len(grouped) > k else 0.0

    found = set(zip(approx_top["src"].tolist(), approx_top["dst"].tolist())) if len(approx_top) else set()
    found = set(list(found)[: len(found)])

    strict_hits = sum(1 for pair in exact_pairs if pair in found)
    strict_recall = strict_hits / k if k else 0.0

    # Tie-aware: any found pair whose true weight reaches w_k counts as a hit.
    top_candidates = approx_top.head(k) if len(approx_top) else approx_top
    tie_hits = 0
    for pair in zip(top_candidates["src"].tolist(), top_candidates["dst"].tolist()) if len(top_candidates) else []:
        if pair in grouped.index and float(grouped.loc[pair]) >= w_k > 0:
            tie_hits += 1
    tie_aware_recall = min(1.0, tie_hits / k) if k else 0.0

    gamma = budget_gamma(width, candidate_capacity)
    core = [pair for pair in exact_pairs if total > 0 and (float(grouped.loc[pair]) - w_next) / total > gamma]
    if core:
        identifiable_recall: float | None = sum(1 for pair in core if pair in found) / len(core)
    else:
        identifiable_recall = None
    return IdentifiabilityReport(
        strict_recall=strict_recall,
        tie_aware_recall=tie_aware_recall,
        identifiable_core_size=len(core),
        identifiable_recall=identifiable_recall,
        identifiable=bool(core),
    )
