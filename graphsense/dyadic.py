"""Dyadic (hierarchical) Count-Min heavy-hitter enumeration.

A second decoder family for the sparse-recovery view of sketching: instead of
tracking candidates per record (SpaceSaving), a tree of Count-Min sketches over
key prefixes is queried top-down at answer time, in the spirit of hierarchical
heavy hitters and group testing. Its discovery budget is the beam of prefixes
kept per level rather than a per-record candidate capacity. On flat (all-tie
or diffuse) traffic every prefix node looks alike, so descent has no signal to
follow -- the same identifiability limit that binds candidate tracking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MERSENNE = (1 << 61) - 1


class DyadicCountMin:
    """Count-Min sketches over every prefix length of an integer key space."""

    def __init__(self, key_bits: int, width: int = 8192, depth: int = 4, top_bits: int = 12, seed: int = 101) -> None:
        self.key_bits = int(key_bits)
        self.width = int(width)
        self.depth = int(depth)
        self.top_bits = min(int(top_bits), self.key_bits)
        rng = np.random.default_rng(seed)
        self.levels = list(range(self.top_bits, self.key_bits + 1))
        self.tables = {level: np.zeros((depth, width), dtype=np.float64) for level in self.levels}
        self._a = {level: rng.integers(1, _MERSENNE, size=depth, dtype=np.uint64) for level in self.levels}
        self._b = {level: rng.integers(0, _MERSENNE, size=depth, dtype=np.uint64) for level in self.levels}

    def _indices(self, level: int, prefixes: np.ndarray) -> np.ndarray:
        idx = np.empty((self.depth, len(prefixes)), dtype=np.int64)
        x = prefixes.astype(np.uint64)
        for row in range(self.depth):
            hashed = (self._a[level][row] * x + self._b[level][row]) % np.uint64(_MERSENNE)
            idx[row] = (hashed % np.uint64(self.width)).astype(np.int64)
        return idx

    def update_batch(self, keys: np.ndarray, weights: np.ndarray) -> None:
        keys = keys.astype(np.uint64)
        weights = weights.astype(np.float64)
        for level in self.levels:
            prefixes = keys >> np.uint64(self.key_bits - level)
            idx = self._indices(level, prefixes)
            for row in range(self.depth):
                np.add.at(self.tables[level][row], idx[row], weights)

    def estimate(self, level: int, prefixes: np.ndarray) -> np.ndarray:
        if len(prefixes) == 0:
            return np.zeros(0)
        idx = self._indices(level, prefixes)
        estimates = np.full(len(prefixes), np.inf)
        for row in range(self.depth):
            estimates = np.minimum(estimates, self.tables[level][row][idx[row]])
        return estimates

    def heavy_hitters(self, k: int, beam: int = 4096) -> pd.DataFrame:
        """Beam-limited top-down descent; returns leaf keys with estimates."""

        level = self.levels[0]
        prefixes = np.arange(1 << level, dtype=np.uint64)
        estimates = self.estimate(level, prefixes)
        order = np.argsort(estimates)[::-1][:beam]
        prefixes = prefixes[order]
        for level in self.levels[1:]:
            children = np.concatenate([prefixes << np.uint64(1), (prefixes << np.uint64(1)) | np.uint64(1)])
            estimates = self.estimate(level, children)
            order = np.argsort(estimates)[::-1][:beam]
            prefixes = children[order]
        estimates = self.estimate(self.key_bits, prefixes)
        order = np.argsort(estimates)[::-1][:k]
        return pd.DataFrame({"key": prefixes[order].astype(np.uint64), "estimated_weight": estimates[order]})

    @property
    def bytes(self) -> int:
        return int(sum(table.nbytes for table in self.tables.values()))


def edges_to_keys(edges: pd.DataFrame, value_column: str = "bytes") -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pack (src, dst) integer labels into single keys; returns (keys, weights, key_bits, dst_bits)."""

    src = edges["src"].to_numpy(dtype=np.uint64)
    dst = edges["dst"].to_numpy(dtype=np.uint64)
    dst_bits = max(int(dst.max()).bit_length(), 1)
    src_bits = max(int(src.max()).bit_length(), 1)
    keys = (src << np.uint64(dst_bits)) | dst
    weights = edges[value_column].to_numpy(dtype=np.float64)
    return keys, weights, src_bits + dst_bits, dst_bits


def dyadic_topk_recall(edges: pd.DataFrame, exact_top: pd.DataFrame, k: int = 20, width: int = 8192, depth: int = 4, beam: int = 4096, value_column: str = "bytes") -> dict[str, float]:
    """Recall of beam-descent dyadic enumeration against exact top-k pairs."""

    keys, weights, key_bits, dst_bits = edges_to_keys(edges, value_column)
    sketch = DyadicCountMin(key_bits=key_bits, width=width, depth=depth)
    sketch.update_batch(keys, weights)
    found = sketch.heavy_hitters(k=k, beam=beam)
    found_pairs = {(int(key) >> dst_bits, int(key) & ((1 << dst_bits) - 1)) for key in found["key"]}
    exact_pairs = {(int(row.src), int(row.dst)) for row in exact_top.itertuples()}
    recall = len(found_pairs & exact_pairs) / max(len(exact_pairs), 1)
    return {"topk_recall": recall, "sketch_bytes": sketch.bytes, "levels": len(sketch.levels)}
