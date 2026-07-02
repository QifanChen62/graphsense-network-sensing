#!/usr/bin/env python3
"""Conformal identifiability certificates and coverage experiments.

Splits each input edge stream into disjoint windows, calibrates a conformal
lower bound on the top-k ranking-gap share, certifies sketch budgets at level
1 - alpha, and reports held-out empirical coverage plus an adaptive-conformal
trajectory for drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import regime_shape
from graphsense.conformal import (
    adaptive_alpha_trajectory,
    budget_threshold,
    conformal_certificate,
    coverage_experiment,
    window_statistics,
)
from graphsense.io import read_edges
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=[], help="real edge CSVs (src,dst,bytes)")
    parser.add_argument("--regimes", nargs="*", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--n-edges", type=int, default=5_000_000)
    parser.add_argument("--window-edges", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--budgets", nargs="+", default=["8192:5:512", "16384:5:5000"], help="width:depth:capacity")
    parser.add_argument("--target-alpha", type=float, default=0.1)
    parser.add_argument("--aci-gamma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--certificate-output", default="results/conformal_certificates.csv")
    parser.add_argument("--coverage-output", default="results/conformal_coverage.csv")
    parser.add_argument("--aci-output", default="results/conformal_aci.csv")
    parser.add_argument("--value-column", default="bytes")
    return parser.parse_args()


def _sources(args: argparse.Namespace):
    for input_path in args.inputs:
        edges = read_edges(input_path, value_column=args.value_column)
        yield Path(input_path).stem, edges
    for regime in args.regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        edges = make_synthetic_edges(
            n_edges=args.n_edges,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )
        yield regime, edges


def main() -> None:
    args = parse_args()
    certificate_rows = []
    coverage_rows = []
    aci_rows = []
    for name, edges in _sources(args):
        stats = window_statistics(edges, window_edges=args.window_edges, top_k=args.top_k, value_column=args.value_column)
        scores = stats["topk_gap_share"].to_numpy()
        n_calib = len(scores) // 2

        for budget in args.budgets:
            width, depth, capacity = (int(part) for part in budget.split(":"))
            certificate = conformal_certificate(scores[:n_calib], args.target_alpha, width, depth, capacity)
            certificate_rows.append({"source": name, "n_windows_total": len(scores), **certificate.to_dict()})

        coverage = coverage_experiment(scores, alphas=tuple(args.alphas), n_calibration=n_calib)
        coverage.insert(0, "source", name)
        coverage_rows.append(coverage)

        trajectory = adaptive_alpha_trajectory(scores, target_alpha=args.target_alpha, gamma=args.aci_gamma)
        if len(trajectory):
            aci_rows.append(
                {
                    "source": name,
                    "n_windows": len(scores),
                    "target_alpha": args.target_alpha,
                    "gamma": args.aci_gamma,
                    "realized_miscoverage": trajectory.attrs["realized_miscoverage"],
                    "final_alpha_t": float(trajectory["alpha_t"].iloc[-1]),
                }
            )

    certificates = pd.DataFrame(certificate_rows)
    coverages = pd.concat(coverage_rows, ignore_index=True)
    aci = pd.DataFrame(aci_rows)
    for path, frame in (
        (args.certificate_output, certificates),
        (args.coverage_output, coverages),
        (args.aci_output, aci),
    ):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"wrote {output} ({len(frame)} rows)")

    print("\n=== certificates (alpha={}) ===".format(args.target_alpha))
    print(
        certificates[
            ["source", "sketch_width", "candidate_capacity", "conformal_gap_lower_bound", "required_threshold", "certified", "margin"]
        ].to_string(index=False)
    )
    print("\n=== coverage ===")
    print(coverages[["source", "alpha", "gap_lower_bound", "empirical_coverage", "valid"]].to_string(index=False))
    if len(aci):
        print("\n=== adaptive conformal ===")
        print(aci.to_string(index=False))


if __name__ == "__main__":
    main()
