#!/usr/bin/env python3
"""Generate a compact figure for early-stream method-selector features."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

add_project_root()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/method_selector_summary.csv")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = [regime.replace("_", "\n") for regime in frame["regime"]]
    x = range(len(frame))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar([i - 0.18 for i in x], frame["duplication_rate"], width=0.36, label="Duplication rate")
    ax.bar([i + 0.18 for i in x], frame["top_edge_share"], width=0.36, label="Top-edge share")
    for i, row in frame.reset_index(drop=True).iterrows():
        marker = "OK" if bool(row["recommendation_matches"]) else "miss"
        ax.text(i, 1.02, marker, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Early-stream feature value")
    ax.set_title("Early traffic-shape features explain method recommendations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    output = outdir / "method_selector_features.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
