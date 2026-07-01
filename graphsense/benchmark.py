"""Benchmark helpers for sparse traffic-matrix construction and analytics."""

from __future__ import annotations

import platform
import resource
import time
from dataclasses import asdict, dataclass
from typing import Callable

import pandas as pd

from .analytics import matrix_summary
from .baselines import counter_baseline, pandas_groupby_baseline
from .io import TrafficMatrix, edges_to_sparse
from .synthetic import make_synthetic_edges


Builder = Callable[[pd.DataFrame, str], TrafficMatrix]


@dataclass(frozen=True)
class BenchmarkRow:
    regime: str
    method: str
    n_edges: int
    repeat: int
    seconds: float
    construction_seconds: float
    analytics_seconds: float
    edges_per_second: float
    max_rss_mb: float
    rss_bytes_per_edge: float
    nnz: int
    nnz_per_edge: float
    n_sources: int
    n_destinations: int
    total_weight: float
    density: float
    row_entropy_bits: float
    column_entropy_bits: float
    edge_entropy_bits: float
    gini_edge_weight: float
    output_matches_sparse: bool
    python: str
    platform: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def sparse_direct(edges: pd.DataFrame, value_column: str = "bytes") -> TrafficMatrix:
    return edges_to_sparse(edges, value_column=value_column)


METHODS: dict[str, Builder] = {
    "sparse_direct": sparse_direct,
    "pandas_groupby": pandas_groupby_baseline,
    "python_counter": counter_baseline,
}


def _equivalent(left: TrafficMatrix, right: TrafficMatrix) -> bool:
    if left.matrix.shape != right.matrix.shape:
        return False
    difference = (left.matrix - right.matrix).tocoo()
    if difference.nnz == 0:
        return True
    return bool(abs(difference.data).max() < 1e-9)


def regime_shape(regime: str, n_edges: int) -> tuple[int, int]:
    """Choose matrix dimensions that produce different nnz/density regimes."""

    if regime == "hotspot_zipf":
        return max(512, int(n_edges**0.50)), max(512, int(n_edges**0.50))
    if regime == "community_bursty":
        return max(2048, int(n_edges**0.58)), max(2048, int(n_edges**0.58))
    if regime == "scanner_fanout":
        return max(4096, int(n_edges**0.62)), max(4096, int(n_edges**0.68))
    if regime == "uniform_sparse":
        return max(8192, int(n_edges**0.72)), max(8192, int(n_edges**0.72))
    raise ValueError(f"unknown regime {regime!r}")


def run_benchmark(
    sizes: list[int],
    repeats: int = 3,
    methods: tuple[str, ...] = ("sparse_direct", "pandas_groupby", "python_counter"),
    regimes: tuple[str, ...] = ("hotspot_zipf",),
    seed: int = 7,
) -> pd.DataFrame:
    """Run construction plus summary benchmarks over synthetic edge sizes."""

    rows: list[BenchmarkRow] = []
    for regime in regimes:
        for n_edges in sizes:
            n_sources, n_destinations = regime_shape(regime, n_edges)
            edges = make_synthetic_edges(
                n_edges=n_edges,
                n_sources=n_sources,
                n_destinations=n_destinations,
                seed=seed + n_edges + len(regime),
                regime=regime,
                label_mode="int",
            )
            reference = sparse_direct(edges)
            reference_summary = matrix_summary(reference)

            for repeat in range(repeats):
                for method in methods:
                    builder = METHODS[method]
                    construction_start = time.perf_counter()
                    traffic = builder(edges, "bytes")
                    construction_elapsed = time.perf_counter() - construction_start
                    analytics_start = time.perf_counter()
                    summary = matrix_summary(traffic)
                    analytics_elapsed = time.perf_counter() - analytics_start
                    elapsed = construction_elapsed + analytics_elapsed
                    rss_mb = _rss_mb()
                    rows.append(
                        BenchmarkRow(
                            regime=regime,
                            method=method,
                            n_edges=n_edges,
                            repeat=repeat,
                            seconds=elapsed,
                            construction_seconds=construction_elapsed,
                            analytics_seconds=analytics_elapsed,
                            edges_per_second=float(n_edges / elapsed) if elapsed > 0 else float("inf"),
                            max_rss_mb=rss_mb,
                            rss_bytes_per_edge=float(rss_mb * 1024 * 1024 / n_edges),
                            nnz=summary.nnz,
                            nnz_per_edge=float(summary.nnz / n_edges),
                            n_sources=summary.n_sources,
                            n_destinations=summary.n_destinations,
                            total_weight=summary.total_weight,
                            density=summary.density,
                            row_entropy_bits=summary.row_entropy_bits,
                            column_entropy_bits=summary.column_entropy_bits,
                            edge_entropy_bits=summary.edge_entropy_bits,
                            gini_edge_weight=summary.gini_edge_weight,
                            output_matches_sparse=_equivalent(traffic, reference)
                            and abs(summary.total_weight - reference_summary.total_weight) < 1e-9,
                            python=platform.python_version(),
                            platform=platform.platform(),
                        )
                    )
    return pd.DataFrame([row.to_dict() for row in rows])
