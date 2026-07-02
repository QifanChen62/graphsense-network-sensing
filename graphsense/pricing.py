"""Conformal budget pricing for sketch-based top-k identifiability.

Given a conformal lower bound L on the future-window gap share, find the
memory-minimal Count-Min width m and candidate capacity c that certify top-k
identifiability: L > 2(e/m + 1/(c+1)). Memory is modeled as
M(m, c) = depth * m * counter_bytes + c * candidate_bytes.

For fixed m the minimal capacity is closed-form,
    c_min(m) = ceil(1 / (L/2 - e/m)) - 1,   valid when m > 2e/L,
so pricing reduces to a one-dimensional search over m.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

COUNTER_BYTES = 8
CANDIDATE_BYTES = 32  # two 8-byte identifiers, an 8-byte count, amortized heap entry


@dataclass(frozen=True)
class BudgetPrice:
    gap_lower_bound: float
    alpha: float
    depth: int
    feasible: bool
    width: int
    capacity: int
    sketch_bytes: int
    candidate_bytes_total: int
    total_bytes: int
    margin: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def capacity_for_width(gap_lower_bound: float, width: int) -> int | None:
    """Minimal certified capacity at a given width, or None if infeasible."""

    slack = gap_lower_bound / 2.0 - math.e / width
    if slack <= 0:
        return None
    return max(1, math.ceil(1.0 / slack) - 1)


def certification_margin(gap_lower_bound: float, width: int, capacity: int) -> float:
    return gap_lower_bound - 2.0 * (math.e / width + 1.0 / (capacity + 1))


def price_budget(
    gap_lower_bound: float,
    alpha: float = 0.1,
    depth: int = 5,
    counter_bytes: int = COUNTER_BYTES,
    candidate_bytes: int = CANDIDATE_BYTES,
) -> BudgetPrice:
    """Algorithm 2: memory-minimal certified budget for one stream.

    Searches widths from just above the feasibility threshold 2e/L upward on a
    fine geometric grid with local integer refinement; each width gives the
    closed-form minimal capacity, so the search is one-dimensional.
    """

    if gap_lower_bound <= 0 or not math.isfinite(gap_lower_bound):
        return BudgetPrice(gap_lower_bound, alpha, depth, False, 0, 0, 0, 0, 0, float("-inf"))

    width_floor = int(math.ceil(2.0 * math.e / gap_lower_bound)) + 1
    best: tuple[int, int, int] | None = None  # (total, width, capacity)
    width = width_floor
    while width <= width_floor * 4096:
        capacity = capacity_for_width(gap_lower_bound, width)
        if capacity is not None:
            total = depth * width * counter_bytes + capacity * candidate_bytes
            if best is None or total < best[0]:
                best = (total, width, capacity)
        width = max(width + 1, int(width * 1.02))

    assert best is not None
    # Local refinement around the best width.
    _, best_width, _ = best
    for width in range(max(width_floor, int(best_width * 0.98)), int(best_width * 1.02) + 2):
        capacity = capacity_for_width(gap_lower_bound, width)
        if capacity is None:
            continue
        total = depth * width * counter_bytes + capacity * candidate_bytes
        if total < best[0]:
            best = (total, width, capacity)

    total, width, capacity = best
    return BudgetPrice(
        gap_lower_bound=gap_lower_bound,
        alpha=alpha,
        depth=depth,
        feasible=True,
        width=width,
        capacity=capacity,
        sketch_bytes=depth * width * counter_bytes,
        candidate_bytes_total=capacity * candidate_bytes,
        total_bytes=total,
        margin=certification_margin(gap_lower_bound, width, capacity),
    )


def best_index_at_memory(
    gap_lower_bound: float,
    total_bytes: int,
    depth: int = 5,
    counter_bytes: int = COUNTER_BYTES,
    candidate_bytes: int = CANDIDATE_BYTES,
) -> float:
    """Best achievable identifiability index I_k under a total-memory cap.

    Splits the budget between width and capacity to minimize the combined
    error 2(e/m + 1/(c+1)); returns L divided by that minimum (I_k >= 1 means
    the memory certifies).
    """

    if gap_lower_bound <= 0:
        return 0.0
    best_error = float("inf")
    for split in [i / 40.0 for i in range(1, 40)]:
        width = int(split * total_bytes / (depth * counter_bytes))
        capacity = int((1.0 - split) * total_bytes / candidate_bytes)
        if width < 8 or capacity < 1:
            continue
        error = 2.0 * (math.e / width + 1.0 / (capacity + 1))
        best_error = min(best_error, error)
    if not math.isfinite(best_error):
        return 0.0
    return gap_lower_bound / best_error
