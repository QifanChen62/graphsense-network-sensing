"""Paper-ready benchmark plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _short_label(key: object) -> str:
    if not isinstance(key, tuple):
        return str(key)
    regime, method = key
    regime_name = str(regime).replace("_bursty", "").replace("_zipf", "").replace("_fanout", "").replace("_sparse", "")
    method_name = str(method).replace("sparse_direct", "sparse").replace("pandas_groupby", "pandas").replace("python_counter", "counter")
    return f"{regime_name}/{method_name}"


def make_benchmark_plots(input_csv: str | Path, outdir: str | Path) -> list[Path]:
    """Create runtime and memory plots from benchmark_summary.csv."""

    input_csv = Path(input_csv)
    frame = pd.read_csv(input_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stem = input_csv.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]

    group_cols = ["regime", "method", "n_edges"] if "regime" in frame.columns else ["method", "n_edges"]
    summary = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            seconds_mean=("seconds", "mean"),
            seconds_std=("seconds", "std"),
            construction_seconds_mean=("construction_seconds", "mean"),
            analytics_seconds_mean=("analytics_seconds", "mean"),
            edges_per_second_mean=("edges_per_second", "mean"),
            rss_mean=("max_rss_mb", "mean"),
            rss_bytes_per_edge_mean=("rss_bytes_per_edge", "mean"),
            nnz_per_edge_mean=("nnz_per_edge", "mean"),
        )
        .sort_values(group_cols)
    )
    grouped_path = outdir.parent / "results" / f"{stem}_grouped.csv"
    summary.to_csv(grouped_path, index=False)
    if stem == "benchmark":
        summary.to_csv(outdir.parent / "results" / "benchmark_grouped.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    label_cols = ["regime", "method"] if "regime" in summary.columns else ["method"]
    for key, part in summary.groupby(label_cols):
        label = _short_label(key)
        ax.plot(part["n_edges"], part["seconds_mean"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input edge records")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Traffic-matrix construction and summary runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    runtime_path = outdir / f"{stem}_runtime.png"
    fig.savefig(runtime_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(runtime_path)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for key, part in summary.groupby(label_cols):
        label = _short_label(key)
        ax.plot(part["n_edges"], part["rss_mean"], marker="s", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Input edge records")
    ax.set_ylabel("Peak RSS so far (MB)")
    ax.set_title("Memory footprint during commodity-CPU benchmark")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    memory_path = outdir / f"{stem}_memory.png"
    fig.savefig(memory_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(memory_path)

    return outputs
