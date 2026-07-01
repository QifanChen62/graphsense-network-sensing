#!/usr/bin/env python3
"""Generate network-sensing time-bin anomaly samples and summary curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_root

add_project_root()

from graphsense.synthetic import make_controlled_edges
from graphsense.timebin import add_timebin_anomaly_score, time_bin_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/timebin_anomaly_summary.csv")
    parser.add_argument("--scenario-output", default="results/timebin_anomaly_scenarios.csv")
    parser.add_argument("--figure", default="figures/timebin_anomaly.png")
    parser.add_argument("--n-edges", type=int, default=120000)
    parser.add_argument("--n-sources", type=int, default=4096)
    parser.add_argument("--n-destinations", type=int, default=4096)
    parser.add_argument("--n-time-bins", type=int, default=24)
    parser.add_argument("--anomaly-bin", type=int, default=17)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["volume_burst", "scanner_fanout", "destination_attack", "distributed_low_rate_scan"],
    )
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def _base_edges(args: argparse.Namespace, seed: int) -> pd.DataFrame:
    base = make_controlled_edges(
        n_edges=args.n_edges,
        n_sources=args.n_sources,
        n_destinations=args.n_destinations,
        target_nnz=min(args.n_edges, 80_000),
        zipf_skew=0.15,
        n_time_bins=args.n_time_bins,
        seed=seed,
        label_mode="int",
    )
    return base


def make_anomaly_edges(args: argparse.Namespace, scenario: str, scenario_index: int) -> pd.DataFrame:
    base = _base_edges(args, args.seed + 100 * scenario_index)
    rng = np.random.default_rng(args.seed + 1 + 100 * scenario_index)
    n_anomaly = max(1, args.n_edges // 12)

    if scenario == "volume_burst":
        sources = rng.integers(0, args.n_sources, size=n_anomaly)
        destinations = rng.integers(0, args.n_destinations, size=n_anomaly)
        packets = rng.gamma(shape=9.0, scale=12.0, size=n_anomaly)
        byte_floor, byte_ceiling = 512, 1500
    elif scenario == "scanner_fanout":
        scanner_count = max(4, args.n_sources // 300)
        sources = rng.integers(0, scanner_count, size=n_anomaly)
        destinations = rng.integers(0, args.n_destinations, size=n_anomaly)
        packets = rng.gamma(shape=2.0, scale=3.0, size=n_anomaly)
        byte_floor, byte_ceiling = 128, 800
    elif scenario == "destination_attack":
        destination_count = max(2, args.n_destinations // 512)
        sources = rng.integers(0, args.n_sources, size=n_anomaly)
        destinations = rng.integers(0, destination_count, size=n_anomaly)
        packets = rng.gamma(shape=3.0, scale=5.0, size=n_anomaly)
        byte_floor, byte_ceiling = 256, 1200
    elif scenario == "distributed_low_rate_scan":
        n_anomaly = max(1, args.n_edges // 10)
        sources = rng.integers(0, args.n_sources, size=n_anomaly)
        destinations = rng.integers(0, args.n_destinations, size=n_anomaly)
        packets = rng.gamma(shape=1.0, scale=1.0, size=n_anomaly)
        byte_floor, byte_ceiling = 64, 160
    else:
        raise ValueError(f"unknown anomaly scenario {scenario!r}")

    anomaly = pd.DataFrame(
        {
            "time_bin": args.anomaly_bin,
            "src": sources.astype(np.int64),
            "dst": destinations.astype(np.int64),
            "packets": packets.round(3),
            "bytes": np.maximum(1, np.round(packets * rng.integers(byte_floor, byte_ceiling, size=n_anomaly))).astype(np.int64),
            "regime": f"timebin_{scenario}",
            "target_nnz": np.nan,
        }
    )
    return pd.concat([base, anomaly], ignore_index=True)


def make_figure(summary: pd.DataFrame, output: Path, anomaly_bin: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(summary["scenario"].drop_duplicates())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(7.2, 7.8), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        part = summary[summary["scenario"] == scenario].sort_values("time_bin")
        ax.plot(part["time_bin"], part["anomaly_score"], marker="o", color="tab:red", label="Anomaly score")
        ax.axvline(anomaly_bin, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("Score", fontsize=8)
        ax.text(
            0.01,
            0.86,
            scenario.replace("_", " "),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Time-bin network-sensing anomalies across traffic patterns")
    axes[-1].set_xlabel("Time bin")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_frames = []
    scenario_rows = []
    for scenario_index, scenario in enumerate(args.scenarios):
        edges = make_anomaly_edges(args, scenario, scenario_index)
        summary = add_timebin_anomaly_score(time_bin_summary(edges))
        summary.insert(0, "scenario", scenario)
        summary["injected_anomaly"] = summary["time_bin"] == args.anomaly_bin
        summary_frames.append(summary)

        ranked = summary.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        injected = ranked[ranked["time_bin"] == args.anomaly_bin].iloc[0]
        scenario_rows.append(
            {
                "scenario": scenario,
                "anomaly_bin": args.anomaly_bin,
                "anomaly_rank": int(ranked.index[ranked["time_bin"] == args.anomaly_bin][0] + 1),
                "anomaly_score": float(injected["anomaly_score"]),
                "total_weight": float(injected["total_weight"]),
                "nnz": int(injected["nnz"]),
                "top_edge_share": float(injected["top_edge_share"]),
                "top_source_share": float(injected["top_source_share"]),
                "top_destination_share": float(injected["top_destination_share"]),
                "unique_sources": int(injected["unique_sources"]),
                "unique_destinations": int(injected["unique_destinations"]),
                "typical_total_weight_median": float(summary.loc[~summary["injected_anomaly"], "total_weight"].median()),
                "typical_nnz_median": float(summary.loc[~summary["injected_anomaly"], "nnz"].median()),
            }
        )

    summary = pd.concat(summary_frames, ignore_index=True)
    scenario_summary = pd.DataFrame(scenario_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    scenario_output = Path(args.scenario_output)
    scenario_output.parent.mkdir(parents=True, exist_ok=True)
    scenario_summary.to_csv(scenario_output, index=False)
    make_figure(summary, Path(args.figure), args.anomaly_bin)
    print(f"wrote {output}")
    print(f"wrote {scenario_output}")
    print(f"wrote {args.figure}")
    print(scenario_summary.to_string(index=False))


if __name__ == "__main__":
    main()
