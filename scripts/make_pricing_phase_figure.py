#!/usr/bin/env python3
"""Two-panel figure: conformal budget frontier and identifiability phase diagram.

Panel (a): best achievable identifiability index I_k as a function of total
sketch memory, per source; the horizontal line I_k = 1 marks certification,
and each curve crosses it at that stream's conformal memory price.

Panel (b): measured Recall@20 against log10 I_k for two decoder families
(SpaceSaving tracker and dyadic Count-Min descent) across all sources and
budgets. Points left of I_k = 1 with high recall reflect the screen's
conservatism (sufficient, not necessary); the soundness claim is one-sided:
no certified budget (right of the line) shows low recall.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import regime_shape
from graphsense.pricing import best_index_at_memory
from graphsense.synthetic import make_synthetic_edges

I_K_FLOOR = 1e-4  # plotting floor for I_k = 0 (all-tie official product)


def gap_share(edges: pd.DataFrame, top_k: int = 20, value_column: str = "bytes") -> float:
    grouped = edges.groupby(["src", "dst"], sort=False)[value_column].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    if len(grouped) <= top_k or total <= 0:
        return 0.0
    return max(0.0, float(grouped.iloc[top_k - 1] - grouped.iloc[top_k]) / total)


def index_for(gap: float, width: int, capacity: int) -> float:
    return gap / (2.0 * (math.e / width + 1.0 / (capacity + 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", default="results/conformal_certificates.csv")
    parser.add_argument("--candidate-sensitivity", default="results/candidate_sensitivity.csv")
    parser.add_argument("--ctu-sensitivity", default="results/ctu13_candidate_sensitivity.csv")
    parser.add_argument("--official-sensitivity", default="results/real_data_candidate_sensitivity.csv")
    parser.add_argument("--dyadic", default="results/dyadic_comparison.csv")
    parser.add_argument("--priced", default="results/priced_budget_recall.csv")
    parser.add_argument("--ctu-edges", default="data/real/ctu13_edges.csv")
    parser.add_argument("--phase-output", default="results/phase_diagram_points.csv")
    parser.add_argument("--figure", default="figures/pricing_phase.png")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--n-edges", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = ["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"]
    gaps: dict[str, float] = {}
    for regime in regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        edges = make_synthetic_edges(
            n_edges=args.n_edges,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )
        gaps[regime] = gap_share(edges)
    ctu_full = pd.read_csv(args.ctu_edges)
    gaps["ctu13_full"] = gap_share(ctu_full)
    gaps["ctu13_100k"] = gap_share(ctu_full.head(args.n_edges))
    gaps["official"] = 0.0

    points = []
    tracker = pd.read_csv(args.candidate_sensitivity)
    for row in tracker.itertuples():
        points.append(
            {
                "source": row.regime,
                "decoder": "spacesaving",
                "index": index_for(gaps[row.regime], row.sketch_width, row.candidate_capacity),
                "recall": row.topk_recall,
            }
        )
    ctu = pd.read_csv(args.ctu_sensitivity)
    for row in ctu.itertuples():
        points.append(
            {
                "source": "ctu13",
                "decoder": "spacesaving",
                "index": index_for(gaps["ctu13_full"], row.sketch_width, row.candidate_capacity),
                "recall": row.topk_recall,
            }
        )
    official = pd.read_csv(args.official_sensitivity)
    for row in official.itertuples():
        points.append({"source": "official", "decoder": "spacesaving", "index": 0.0, "recall": row.topk_recall})
    dyadic = pd.read_csv(args.dyadic)
    dyadic_gap_keys = {
        "ctu13_edges": "ctu13_100k",
        "official_prefix_edges": "official",
        "hotspot_zipf": "hotspot_zipf",
        "community_bursty": "community_bursty",
        "scanner_fanout": "scanner_fanout",
        "uniform_sparse": "uniform_sparse",
    }
    for row in dyadic.itertuples():
        gap = gaps[dyadic_gap_keys[row.source]]
        points.append(
            {
                "source": row.source,
                "decoder": "dyadic",
                "index": index_for(gap, 8192, row.dyadic_beam),
                "recall": row.dyadic_recall,
            }
        )
    priced_path = Path(args.priced)
    if priced_path.exists():
        priced = pd.read_csv(priced_path)
        priced_gap_keys = {"hotspot_zipf": "hotspot_zipf", "ctu13_edges": "ctu13_full"}
        for row in priced.itertuples():
            gap = gaps[priced_gap_keys[row.source]]
            points.append(
                {
                    "source": row.source,
                    "decoder": "spacesaving",
                    "index": index_for(gap, row.width, row.capacity),
                    "recall": row.topk_recall,
                }
            )
    phase = pd.DataFrame(points)
    Path(args.phase_output).parent.mkdir(parents=True, exist_ok=True)
    phase.to_csv(args.phase_output, index=False)

    certificates = pd.read_csv(args.certificates)
    bounds = certificates.groupby("source")["conformal_gap_lower_bound"].first()

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.0))
    memory_grid = np.logspace(5, 10.2, 120)
    for name, bound in bounds.items():
        label = name.replace("_edges", "").replace("official_prefix", "official (all-tie)")
        values = [best_index_at_memory(max(bound, 0.0), int(m)) for m in memory_grid]
        left.plot(memory_grid / 1e6, np.maximum(values, I_K_FLOOR / 10), label=label)
    left.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("total sketch memory (MB)")
    left.set_ylabel("best achievable $I_k$ at $\\alpha=0.1$")
    left.set_title("(a) conformal budget frontier")
    left.legend(fontsize=6, loc="upper left")

    markers = {"spacesaving": "o", "dyadic": "^"}
    for decoder, marker in markers.items():
        subset = phase[phase["decoder"] == decoder]
        x = np.log10(np.maximum(subset["index"].to_numpy(), I_K_FLOOR))
        right.scatter(x, subset["recall"], marker=marker, s=22, alpha=0.75, label=decoder)
    right.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    right.set_xlabel("$\\log_{10} I_k$ (all-tie floored at $10^{-4}$)")
    right.set_ylabel("Recall@20")
    right.set_title("(b) identifiability phase diagram")
    right.legend(fontsize=7, loc="upper left")

    figure.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=200)
    print(f"wrote {args.figure} and {args.phase_output} ({len(phase)} points)")
    certified = phase[phase["index"] >= 1.0]
    print(f"certified points: {len(certified)}; min recall among certified: {certified['recall'].min() if len(certified) else 'n/a'}")


if __name__ == "__main__":
    main()
