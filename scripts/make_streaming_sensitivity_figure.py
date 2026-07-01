#!/usr/bin/env python3
"""Generate the streaming/sketch sensitivity figure."""

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
    parser.add_argument("--input", default="results/streaming_sensitivity.csv")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    for regime, part in frame.groupby("regime"):
        part = part.sort_values("sketch_width")
        axes[0].plot(part["sketch_width"], part["topk_recall"], marker="o", label=regime)
        axes[1].plot(part["sketch_width"], part["median_relative_error"], marker="s", label=regime)

    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("Sketch width")
    axes[1].set_xlabel("Sketch width")
    axes[0].set_ylabel("Top-20 recall")
    axes[1].set_ylabel("Median relative error")
    axes[0].set_title("Recall")
    axes[1].set_title("Weight error")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Streaming/sketch sensitivity at fixed candidate capacity")
    fig.tight_layout()
    output = outdir / "streaming_sensitivity.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
