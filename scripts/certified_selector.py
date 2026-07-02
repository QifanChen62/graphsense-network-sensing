#!/usr/bin/env python3
"""Safety-certified shape-aware selector from an early edge-stream prefix."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.benchmark import regime_shape
from graphsense.io import read_edges
from graphsense.selector import early_stream_features, recommend_method
from graphsense.synthetic import make_synthetic_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results_v1/certified_selector_summary.csv")
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--regimes", nargs="+", default=["hotspot_zipf", "community_bursty", "scanner_fanout", "uniform_sparse"])
    parser.add_argument("--n-edges", type=int, default=5_000_000)
    parser.add_argument("--prefix-edges", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--value-column", default="bytes")
    parser.add_argument("--current-width", type=int, default=8192)
    parser.add_argument("--current-depth", type=int, default=5)
    parser.add_argument("--current-candidate-capacity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--materialize-full-stream",
        action="store_true",
        default=True,
        help="Generate full synthetic streams before taking the prefix so prefix statistics match the N-edge experiment.",
    )
    parser.add_argument(
        "--no-materialize-full-stream",
        dest="materialize_full_stream",
        action="store_false",
        help="Generate only the requested prefix for quicker smoke tests.",
    )
    return parser.parse_args()


def _certificate(prefix: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    value_column = args.value_column
    if value_column not in prefix.columns and "packets" in prefix.columns:
        value_column = "packets"
    features = early_stream_features(prefix, prefix_edges=args.prefix_edges, value_column=value_column)
    recommendation = recommend_method(features)
    grouped = prefix.groupby(["src", "dst"], sort=False)[value_column].sum().sort_values(ascending=False)
    total_weight = float(grouped.sum())
    top_k = grouped.head(args.top_k)
    p_k = float(top_k.sum() / total_weight) if total_weight > 0 else 0.0

    if len(grouped) > args.top_k and total_weight > 0:
        kth_weight = float(grouped.iloc[args.top_k - 1])
        next_weight = float(grouped.iloc[args.top_k])
        topk_gap_share = max(0.0, (kth_weight - next_weight) / total_weight)
    elif len(grouped) >= args.top_k and total_weight > 0:
        kth_weight = float(grouped.iloc[args.top_k - 1])
        topk_gap_share = kth_weight / total_weight
    else:
        kth_weight = 0.0
        topk_gap_share = 0.0

    # SpaceSaving retention bound: every key with true weight above W/(c+1) is
    # tracked, so c >= W/w_k - 1 guarantees all true top-k pairs are candidates.
    if kth_weight > 0:
        c_min = max(0, int(math.ceil(total_weight / kth_weight) - 1))
    else:
        c_min = math.inf

    if topk_gap_share > 0:
        epsilon_required = topk_gap_share / 2.0
        width_min = int(math.ceil(math.e / epsilon_required))
    else:
        epsilon_required = 0.0
        width_min = math.inf

    current_epsilon = math.e / args.current_width
    candidate_noise_share = 1.0 / (args.current_candidate_capacity + 1)
    combined_margin = topk_gap_share - 2.0 * (current_epsilon + candidate_noise_share)
    current_sketch_bytes = args.current_width * args.current_depth * 8

    if p_k <= 0:
        sketch_decision = "not_certified"
        reason = "top-k prefix share is zero"
    elif topk_gap_share <= 0:
        sketch_decision = "not_certified"
        reason = "top-k ranking gap is zero in the prefix"
    elif args.current_candidate_capacity < c_min:
        sketch_decision = "not_certified"
        reason = f"current candidate capacity {args.current_candidate_capacity} is below c_min {c_min}"
    elif args.current_width < width_min:
        sketch_decision = "not_certified"
        reason = f"current sketch width {args.current_width} is below width_min {width_min}"
    elif combined_margin <= 0:
        sketch_decision = "not_certified"
        reason = "combined Count-Min and candidate-error sufficient condition is not met"
    else:
        sketch_decision = "certified_safe"
        reason = "current width and candidate capacity satisfy the conservative sufficient condition"

    return {
        **features.to_dict(),
        **recommendation.to_dict(),
        "rho_unique_pair_ratio": features.unique_pairs / max(features.observed_edges, 1),
        "top_k": args.top_k,
        "topk_weight_share": p_k,
        "kth_weight_share": (kth_weight / total_weight) if total_weight > 0 else 0.0,
        "topk_gap_share": topk_gap_share,
        "required_candidate_capacity_c_min": c_min,
        "required_cms_epsilon_lt": epsilon_required,
        "required_cms_width_min": width_min,
        "current_sketch_width": args.current_width,
        "current_sketch_depth": args.current_depth,
        "current_candidate_capacity": args.current_candidate_capacity,
        "current_sketch_bytes": current_sketch_bytes,
        "combined_safety_margin": combined_margin,
        "sketch_safety_decision": sketch_decision,
        "sketch_safety_reason": reason,
    }


def _synthetic_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for regime in args.regimes:
        n_sources, n_destinations = regime_shape(regime, args.n_edges)
        n_generated = args.n_edges if args.materialize_full_stream else args.prefix_edges
        edges = make_synthetic_edges(
            n_edges=n_generated,
            n_sources=n_sources,
            n_destinations=n_destinations,
            seed=args.seed + args.n_edges + len(regime),
            regime=regime,
            label_mode="int",
        )
        prefix = edges.head(args.prefix_edges)
        rows.append(
            {
                "source": "synthetic",
                "regime": regime,
                "target_n_edges": args.n_edges,
                "prefix_edges": args.prefix_edges,
                **_certificate(prefix, args),
            }
        )
    return rows


def _input_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for input_path in args.inputs:
        path = Path(input_path)
        edges = read_edges(path, value_column=args.value_column)
        prefix = edges.head(args.prefix_edges)
        rows.append(
            {
                "source": str(path),
                "regime": path.stem,
                "target_n_edges": len(edges),
                "prefix_edges": min(args.prefix_edges, len(prefix)),
                **_certificate(prefix, args),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = _input_rows(args) if args.inputs else _synthetic_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")
    columns = [
        "regime",
        "rho_unique_pair_ratio",
        "duplication_rate",
        "topk_weight_share",
        "topk_gap_share",
        "required_candidate_capacity_c_min",
        "required_cms_width_min",
        "recommended_exact_method",
        "sketch_safety_decision",
        "sketch_safety_reason",
    ]
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
