"""Synthetic sparse traffic-matrix generators for reproducible benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = ("uniform_sparse", "hotspot_zipf", "community_bursty", "scanner_fanout")


def _labels(values: np.ndarray, prefix: str, label_mode: str) -> np.ndarray:
    if label_mode == "int":
        return values.astype(np.int64)
    if label_mode == "str":
        return np.asarray([f"{prefix}{value:08d}" for value in values], dtype=object)
    raise ValueError("label_mode must be 'int' or 'str'")


def _zipf_prob(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    rng.shuffle(weights)
    return weights / weights.sum()


def make_synthetic_edges(
    n_edges: int,
    n_sources: int = 1000,
    n_destinations: int = 1000,
    alpha: float = 1.35,
    seed: int = 7,
    regime: str = "hotspot_zipf",
    label_mode: str = "str",
) -> pd.DataFrame:
    """Generate anonymized src-dst traffic records under a named regime.

    Regimes stress different sparse-matrix shapes:
    uniform_sparse creates high-nnz low-density matrices, hotspot_zipf creates
    repeated heavy hitters, community_bursty creates block structure, and
    scanner_fanout creates high source fanout toward many destinations.
    """

    rng = np.random.default_rng(seed)
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose one of {REGIMES}")

    if regime == "uniform_sparse":
        src = rng.integers(0, n_sources, size=n_edges, dtype=np.int64)
        dst = rng.integers(0, n_destinations, size=n_edges, dtype=np.int64)
        packets = rng.gamma(shape=2.0, scale=3.0, size=n_edges)
    elif regime == "hotspot_zipf":
        src = rng.choice(n_sources, size=n_edges, p=_zipf_prob(rng, n_sources, alpha))
        dst = rng.choice(n_destinations, size=n_edges, p=_zipf_prob(rng, n_destinations, alpha))
        packets = rng.lognormal(mean=1.2, sigma=0.9, size=n_edges)
    elif regime == "community_bursty":
        n_blocks = max(4, min(32, int(np.sqrt(min(n_sources, n_destinations)))))
        block = rng.integers(0, n_blocks, size=n_edges)
        src_width = max(1, n_sources // n_blocks)
        dst_width = max(1, n_destinations // n_blocks)
        src = (block * src_width + rng.integers(0, src_width, size=n_edges)) % n_sources
        local_dst = rng.random(n_edges) < 0.88
        dst_block = np.where(local_dst, block, rng.integers(0, n_blocks, size=n_edges))
        dst = (dst_block * dst_width + rng.integers(0, dst_width, size=n_edges)) % n_destinations
        burst = rng.random(n_edges) < 0.08
        packets = rng.gamma(shape=np.where(burst, 8.0, 1.5), scale=np.where(burst, 8.0, 2.5))
    else:
        scanner_count = max(8, min(n_sources // 20, 512))
        scanner = rng.random(n_edges) < 0.18
        src = np.where(
            scanner,
            rng.integers(0, scanner_count, size=n_edges),
            rng.integers(scanner_count, n_sources, size=n_edges),
        ).astype(np.int64)
        dst = rng.integers(0, n_destinations, size=n_edges, dtype=np.int64)
        packets = rng.gamma(shape=np.where(scanner, 1.1, 2.0), scale=np.where(scanner, 1.0, 5.0))

    bytes_ = np.maximum(1, np.round(packets * rng.integers(64, 1450, size=n_edges))).astype(np.int64)
    time_bin = rng.integers(0, 16, size=n_edges)

    return pd.DataFrame(
        {
            "time_bin": time_bin,
            "src": _labels(src, "s", label_mode),
            "dst": _labels(dst, "d", label_mode),
            "packets": packets.round(3),
            "bytes": bytes_,
            "regime": regime,
        }
    )


def make_controlled_edges(
    n_edges: int,
    n_sources: int,
    n_destinations: int,
    target_nnz: int | None = None,
    density: float | None = None,
    duplication_rate: float | None = None,
    zipf_skew: float = 0.0,
    community_blocks: int = 1,
    scanner_fraction: float = 0.0,
    burst_fraction: float = 0.0,
    n_time_bins: int = 16,
    seed: int = 7,
    label_mode: str = "str",
) -> pd.DataFrame:
    """Generate edges with explicit control over nnz/density and shape knobs."""

    if target_nnz is None and density is not None:
        target_nnz = int(round(density * n_sources * n_destinations))
    if target_nnz is None and duplication_rate is not None:
        target_nnz = int(round(n_edges * (1.0 - duplication_rate)))
    if target_nnz is None:
        target_nnz = min(n_edges, n_sources * n_destinations)
    target_nnz = max(1, min(int(target_nnz), n_edges, n_sources * n_destinations))

    rng = np.random.default_rng(seed)
    pair_ids = rng.choice(n_sources * n_destinations, size=target_nnz, replace=False)
    pool_src = pair_ids // n_destinations
    pool_dst = pair_ids % n_destinations

    if community_blocks > 1:
        block = rng.integers(0, community_blocks, size=target_nnz)
        src_width = max(1, n_sources // community_blocks)
        dst_width = max(1, n_destinations // community_blocks)
        pool_src = (block * src_width + rng.integers(0, src_width, size=target_nnz)) % n_sources
        pool_dst = (block * dst_width + rng.integers(0, dst_width, size=target_nnz)) % n_destinations

    if scanner_fraction > 0:
        scanner_count = max(1, min(n_sources, int(n_sources * scanner_fraction)))
        scanner_mask = rng.random(target_nnz) < min(scanner_fraction * 2, 0.9)
        pool_src = np.where(scanner_mask, rng.integers(0, scanner_count, size=target_nnz), pool_src)

    if zipf_skew > 0:
        ranks = np.arange(1, target_nnz + 1, dtype=np.float64)
        weights = ranks ** (-zipf_skew)
        weights = weights / weights.sum()
        chosen = rng.choice(target_nnz, size=n_edges, replace=True, p=weights)
    else:
        chosen = rng.integers(0, target_nnz, size=n_edges)

    src = pool_src[chosen]
    dst = pool_dst[chosen]
    burst = rng.random(n_edges) < burst_fraction
    packets = rng.gamma(shape=np.where(burst, 10.0, 2.0), scale=np.where(burst, 8.0, 3.0))
    bytes_ = np.maximum(1, np.round(packets * rng.integers(64, 1450, size=n_edges))).astype(np.int64)
    time_bin = np.where(
        burst,
        rng.integers(0, max(1, min(3, n_time_bins)), size=n_edges),
        rng.integers(0, n_time_bins, size=n_edges),
    )

    return pd.DataFrame(
        {
            "time_bin": time_bin,
            "src": _labels(src, "s", label_mode),
            "dst": _labels(dst, "d", label_mode),
            "packets": packets.round(3),
            "bytes": bytes_,
            "regime": "controlled",
            "target_nnz": target_nnz,
        }
    )


def write_synthetic_edges(path: str | Path, **kwargs) -> Path:
    """Write synthetic edges to CSV and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_synthetic_edges(**kwargs)
    frame.to_csv(path, index=False)
    return path
