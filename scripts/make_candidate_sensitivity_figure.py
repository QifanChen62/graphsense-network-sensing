#!/usr/bin/env python3
"""Generate candidate-capacity sensitivity figure."""

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
    parser.add_argument("--input", default="results/candidate_sensitivity.csv")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for regime, part in frame.groupby("regime"):
        part = part.sort_values("candidate_capacity")
        ax.plot(part["candidate_capacity"], part["topk_recall"], marker="o", label=regime)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Candidate capacity")
    ax.set_ylabel("Top-20 heavy-hitter recall")
    ax.set_title("Candidate discovery sensitivity at fixed sketch width")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    output = outdir / "candidate_sensitivity.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
