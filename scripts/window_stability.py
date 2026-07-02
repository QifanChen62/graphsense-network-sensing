#!/usr/bin/env python3
"""Check selector and safety-screen stability across disjoint official-capture segments.

Splits a long official-prefix edge table into fixed-size windows, computes the
early-stream selector features and screen quantities per window, and reports
whether the recommendation and screen decision are stable across segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.selector import early_stream_features, recommend_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/real/official_prefix40m_edges.csv")
    parser.add_argument("--output", default="results/real_data_window_stability.csv")
    parser.add_argument("--window-edges", type=int, default=5_000_000)
    parser.add_argument("--prefix-edges", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--value-column", default="bytes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found; run scripts/fetch_official_prefix.py with a larger --bytes first")
    edges = pd.read_csv(input_path)
    n_windows = len(edges) // args.window_edges
    rows = []
    for index in range(n_windows):
        window = edges.iloc[index * args.window_edges : (index + 1) * args.window_edges]
        features = early_stream_features(window, prefix_edges=args.prefix_edges, value_column=args.value_column)
        recommendation = recommend_method(features)
        grouped = window.groupby(["src", "dst"], sort=False)[args.value_column].sum().sort_values(ascending=False)
        total_weight = float(grouped.sum())
        if len(grouped) > args.top_k and total_weight > 0:
            gap_share = max(0.0, float(grouped.iloc[args.top_k - 1] - grouped.iloc[args.top_k]) / total_weight)
        else:
            gap_share = 0.0
        rows.append(
            {
                "window": index,
                "start_edge": index * args.window_edges,
                "window_edges": len(window),
                "unique_pairs": int(window.drop_duplicates(["src", "dst"]).shape[0]),
                "duplication_rate": features.duplication_rate,
                "nnz_growth_rate": features.nnz_growth_rate,
                "edge_gini": features.edge_gini,
                "top_edge_share": features.top_edge_share,
                "recommended_exact_method": recommendation.recommended_exact_method,
                "topk_gap_share": gap_share,
                "screen_decision": "not_certified" if gap_share <= 0 else "gap_positive",
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} windows)")
    print(frame[["window", "duplication_rate", "nnz_growth_rate", "recommended_exact_method", "topk_gap_share", "screen_decision"]].to_string(index=False))
    stable = frame["recommended_exact_method"].nunique() == 1 and frame["screen_decision"].nunique() == 1
    print(f"stable across windows: {stable}")


if __name__ == "__main__":
    main()
