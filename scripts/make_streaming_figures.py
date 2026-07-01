#!/usr/bin/env python3
"""Generate figures for exact sparse vs streaming/sketch benchmarks."""

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
    parser.add_argument("--input", default="results/streaming_benchmark.csv")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for (regime, method), part in frame.groupby(["regime", "method"]):
        ax.plot(part["n_edges"], part["seconds"], marker="o", label=f"{regime} / {method}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input edge records")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Exact sparse vs streaming/sketch runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    runtime_path = outdir / "streaming_runtime.png"
    fig.savefig(runtime_path, dpi=200)
    plt.close(fig)

    streaming = frame[frame["method"] == "streaming_sketch"].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for regime, part in streaming.groupby("regime"):
        ax.plot(part["n_edges"], part["topk_recall"], marker="s", label=regime)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Input edge records")
    ax.set_ylabel("Top-20 heavy-hitter recall")
    ax.set_title("Streaming/sketch heavy-hitter recall")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    recall_path = outdir / "streaming_topk_recall.png"
    fig.savefig(recall_path, dpi=200)
    plt.close(fig)

    print(f"wrote {runtime_path}")
    print(f"wrote {recall_path}")


if __name__ == "__main__":
    main()
