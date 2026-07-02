"""Conformal identifiability certificates for sketch-based top-k sensing.

Treats disjoint stream windows as exchangeable units. The per-window score is
the top-k ranking-gap share; a distribution-free lower bound on a future
window's score follows from the conformal order-statistic argument. Because
w_k >= Delta_k always, the combined sufficient condition
Delta_k/W > 2(eps + 1/(c+1)) implies the SpaceSaving retention condition
w_k/W > 1/(c+1), so one conformal bound certifies both retention and ranking.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalCertificate:
    n_calibration_windows: int
    alpha: float
    order_statistic_rank: int
    conformal_gap_lower_bound: float
    sketch_width: int
    sketch_depth: int
    candidate_capacity: int
    required_threshold: float
    certified: bool
    margin: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def window_statistics(
    edges: pd.DataFrame,
    window_edges: int,
    top_k: int = 20,
    value_column: str = "bytes",
) -> pd.DataFrame:
    """Per-window top-k statistics over disjoint consecutive windows."""

    n_windows = len(edges) // window_edges
    rows = []
    for index in range(n_windows):
        window = edges.iloc[index * window_edges : (index + 1) * window_edges]
        grouped = window.groupby(["src", "dst"], sort=False)[value_column].sum().sort_values(ascending=False)
        total = float(grouped.sum())
        if len(grouped) > top_k and total > 0:
            kth = float(grouped.iloc[top_k - 1])
            gap_share = max(0.0, (kth - float(grouped.iloc[top_k])) / total)
            kth_share = kth / total
        elif len(grouped) >= top_k and total > 0:
            kth = float(grouped.iloc[top_k - 1])
            gap_share = kth / total
            kth_share = kth / total
        else:
            gap_share = 0.0
            kth_share = 0.0
        rows.append(
            {
                "window": index,
                "window_edges": len(window),
                "unique_pairs": int(len(grouped)),
                "duplication_rate": 1.0 - len(grouped) / max(len(window), 1),
                "total_weight": total,
                "topk_gap_share": gap_share,
                "kth_weight_share": kth_share,
            }
        )
    return pd.DataFrame(rows)


def conformal_lower_bound(scores: np.ndarray | pd.Series, alpha: float) -> tuple[float, int]:
    """Distribution-free lower bound for one future exchangeable score.

    Returns (bound, rank). With m calibration scores and r = floor(alpha*(m+1)),
    the r-th smallest score L satisfies P(future < L) <= r/(m+1) <= alpha.
    When r < 1 the sample is too small for level alpha and the bound is -inf.
    """

    values = np.sort(np.asarray(scores, dtype=float))
    m = len(values)
    rank = math.floor(alpha * (m + 1))
    if rank < 1:
        return float("-inf"), 0
    return float(values[rank - 1]), rank


def budget_threshold(width: int, depth: int, candidate_capacity: int) -> float:
    """Sufficient-condition threshold 2(eps + 1/(c+1)) for the gap share."""

    del depth  # depth controls per-query failure probability, not the threshold
    return 2.0 * (math.e / width + 1.0 / (candidate_capacity + 1))


def conformal_certificate(
    calibration_scores: np.ndarray | pd.Series,
    alpha: float,
    width: int,
    depth: int,
    candidate_capacity: int,
) -> ConformalCertificate:
    """Certify a sketch budget for future windows at miscoverage level alpha."""

    bound, rank = conformal_lower_bound(calibration_scores, alpha)
    threshold = budget_threshold(width, depth, candidate_capacity)
    return ConformalCertificate(
        n_calibration_windows=int(len(calibration_scores)),
        alpha=alpha,
        order_statistic_rank=rank,
        conformal_gap_lower_bound=bound,
        sketch_width=width,
        sketch_depth=depth,
        candidate_capacity=candidate_capacity,
        required_threshold=threshold,
        certified=bool(bound > threshold),
        margin=float(bound - threshold) if math.isfinite(bound) else float("-inf"),
    )


def coverage_experiment(
    scores: np.ndarray | pd.Series,
    alphas: tuple[float, ...] = (0.05, 0.1, 0.2),
    n_calibration: int | None = None,
) -> pd.DataFrame:
    """Empirical coverage of the conformal bound on held-out windows.

    Splits sequentially: the first n_calibration windows calibrate, the rest
    test. Coverage is the fraction of test windows whose score is at least the
    calibrated lower bound; validity requires coverage >= 1 - alpha.
    """

    values = np.asarray(scores, dtype=float)
    if n_calibration is None:
        n_calibration = len(values) // 2
    calibration, test = values[:n_calibration], values[n_calibration:]
    rows = []
    for alpha in alphas:
        bound, rank = conformal_lower_bound(calibration, alpha)
        coverage = float(np.mean(test >= bound)) if len(test) else float("nan")
        rows.append(
            {
                "alpha": alpha,
                "n_calibration": len(calibration),
                "n_test": len(test),
                "order_statistic_rank": rank,
                "gap_lower_bound": bound,
                "empirical_coverage": coverage,
                "valid": bool(coverage >= 1.0 - alpha) if len(test) else None,
            }
        )
    return pd.DataFrame(rows)


def adaptive_alpha_trajectory(
    scores: np.ndarray | pd.Series,
    target_alpha: float = 0.1,
    gamma: float = 0.05,
    warmup: int = 20,
) -> pd.DataFrame:
    """Adaptive conformal inference over sequential windows (Gibbs-Candes).

    At each step the bound is calibrated on all previous windows at the current
    alpha_t; alpha_t moves by gamma * (target - error_t), keeping long-run
    miscoverage near target even when exchangeability fails (drift).
    """

    values = np.asarray(scores, dtype=float)
    alpha_t = target_alpha
    rows = []
    for t in range(warmup, len(values)):
        bound, rank = conformal_lower_bound(values[:t], min(max(alpha_t, 1e-6), 0.999))
        error = float(values[t] < bound)
        rows.append(
            {
                "window": t,
                "alpha_t": alpha_t,
                "gap_lower_bound": bound,
                "score": float(values[t]),
                "miscovered": error,
            }
        )
        alpha_t = alpha_t + gamma * (target_alpha - error)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.attrs["realized_miscoverage"] = float(frame["miscovered"].mean())
    return frame
