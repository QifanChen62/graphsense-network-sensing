"""Certified router for GraphChallenge network-sensing workloads.

Given prefix measurements, a conformal gap lower bound, a memory budget, and
an execution mode, decide: the exact construction path (profiled on the
prefix, not threshold-guessed), whether sketch-based top-k is certifiable and
at what price, the sparse backend, and a safe fallback with the reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .pricing import BudgetPrice, price_budget

GB = 1_000_000_000


@dataclass(frozen=True)
class RouteDecision:
    exact_choice: str
    exact_source: str
    sketch_decision: str
    certified_price_bytes: int
    within_budget: bool
    backend: str
    online_route: str
    batch_route: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def route(
    prefix_sparse_seconds: float,
    prefix_pandas_seconds: float,
    gap_lower_bound: float,
    memory_budget_bytes: int,
    heuristic_choice: str = "sparse_direct",
    native_address_space: bool = False,
    depth: int = 5,
    profile_margin: float = 1.2,
) -> RouteDecision:
    """Decide exact path, sketch certification, backend, and fallbacks.

    The exact path combines two fallible signals -- shape-feature heuristics
    and prefix profiling -- and only commits when they agree; on disagreement
    or an indecisive profile it abstains to the safe sparse default. Neither
    signal alone is reliable (the heuristic misses on CTU-13, prefix profiling
    misses on hotspot at full scale), but their observed disagreement is
    exactly where abstention pays a small bounded penalty instead of a large
    wrong-choice penalty.
    """

    ratio = prefix_pandas_seconds / max(prefix_sparse_seconds, 1e-12)
    if ratio > profile_margin:
        profile_choice = "sparse_direct"
    elif ratio < 1.0 / profile_margin:
        profile_choice = "pandas_groupby"
    else:
        profile_choice = "indecisive"

    if profile_choice == heuristic_choice:
        exact_choice, exact_source = profile_choice, "heuristic and profile agree"
    elif profile_choice == "indecisive":
        exact_choice, exact_source = "sparse_direct", "profile indecisive; safe default"
    else:
        exact_choice, exact_source = "sparse_direct", "signals disagree; abstain to safe default"

    price: BudgetPrice = price_budget(gap_lower_bound, depth=depth)
    if not price.feasible:
        sketch_decision = "unidentifiable"
        reason = "top-k ranking gap is zero across calibration windows (all-tie); strict top-k identity carries no information"
    elif price.total_bytes > GB:
        sketch_decision = "not_worth_it"
        reason = f"certified price {price.total_bytes/1e9:.1f} GB exceeds sketching's useful range; exact construction is cheaper"
    elif price.total_bytes <= memory_budget_bytes:
        sketch_decision = "certifiable_within_budget"
        reason = f"conformal price {price.total_bytes/1e6:.1f} MB fits the {memory_budget_bytes/1e6:.0f} MB budget"
    else:
        sketch_decision = "certifiable_over_budget"
        reason = f"conformal price {price.total_bytes/1e6:.1f} MB exceeds the {memory_budget_bytes/1e6:.0f} MB budget; raise budget or fall back to exact"

    backend = "graphblas_hypersparse" if native_address_space else "scipy_csr"
    if exact_choice == "pandas_groupby":
        backend = "pandas_then_" + backend

    online_route = (
        f"certified sketch (width {price.width}, capacity {price.capacity})"
        if sketch_decision == "certifiable_within_budget"
        else f"exact {exact_choice} per window"
    )
    batch_route = f"exact {exact_choice} ({backend})"

    return RouteDecision(
        exact_choice=exact_choice,
        exact_source=exact_source,
        sketch_decision=sketch_decision,
        certified_price_bytes=int(price.total_bytes),
        within_budget=bool(price.feasible and price.total_bytes <= memory_budget_bytes),
        backend=backend,
        online_route=online_route,
        batch_route=batch_route,
        reason=reason,
    )
