"""Streaming/sketch analytics for edge-record network sensing."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def _stable_hash(value: object, seed: int) -> int:
    payload = f"{seed}:{value!r}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _entropy_from_counts(counts: dict[object, float]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probabilities = np.asarray([value / total for value in counts.values() if value > 0], dtype=np.float64)
    return float(-(probabilities * np.log2(probabilities)).sum())


class CountMinSketch:
    """Small Count-Min Sketch for positive weighted edge updates."""

    def __init__(self, width: int = 8192, depth: int = 5, seed: int = 17) -> None:
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        self.table = np.zeros((self.depth, self.width), dtype=np.float64)

    def _indices(self, key: object) -> list[int]:
        return [_stable_hash(key, self.seed + row * 104_729) % self.width for row in range(self.depth)]

    def update(self, key: object, weight: float) -> None:
        for row, col in enumerate(self._indices(key)):
            self.table[row, col] += weight

    def estimate(self, key: object) -> float:
        return float(min(self.table[row, col] for row, col in enumerate(self._indices(key))))

    @property
    def bytes(self) -> int:
        return int(self.table.nbytes)


class CandidateTracker:
    """Space-saving style candidate tracker for heavy edge keys.

    Eviction picks the minimum-count candidate via a lazy-deletion heap, so an
    all-unique stream costs O(log capacity) per record instead of a full scan.
    Ties on the minimum count are broken by heap insertion order.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = int(capacity)
        self.counts: dict[object, float] = {}
        self._heap: list[tuple[float, int, object]] = []
        self._order = 0

    def _push(self, key: object, count: float) -> None:
        self._order += 1
        heapq.heappush(self._heap, (count, self._order, key))

    def update(self, key: object, weight: float) -> None:
        if key in self.counts:
            self.counts[key] += weight
            self._push(key, self.counts[key])
            return
        if len(self.counts) < self.capacity:
            self.counts[key] = weight
            self._push(key, weight)
            return
        while True:
            count, _, victim = self._heap[0]
            if victim in self.counts and self.counts[victim] == count:
                break
            heapq.heappop(self._heap)  # stale entry
        heapq.heappop(self._heap)
        new_count = self.counts.pop(victim) + weight
        self.counts[key] = new_count
        self._push(key, new_count)

    def candidates(self) -> list[object]:
        return list(self.counts)


@dataclass(frozen=True)
class StreamingSummary:
    n_edges: int
    total_weight: float
    unique_sources: int
    unique_destinations: int
    row_entropy_bits: float
    column_entropy_bits: float
    sketch_bytes: int
    candidate_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def stream_summaries(
    edges: pd.DataFrame,
    value_column: str = "bytes",
    width: int = 8192,
    depth: int = 5,
    candidate_capacity: int = 512,
) -> tuple[StreamingSummary, pd.DataFrame]:
    """Compute streaming summaries and approximate heavy edge weights."""

    sketch = CountMinSketch(width=width, depth=depth)
    candidates = CandidateTracker(capacity=candidate_capacity)
    row_strength: dict[object, float] = {}
    col_strength: dict[object, float] = {}
    total_weight = 0.0

    for row in edges[["src", "dst", value_column]].itertuples(index=False, name=None):
        src, dst, weight = row
        weight = float(weight)
        key = (src, dst)
        total_weight += weight
        row_strength[src] = row_strength.get(src, 0.0) + weight
        col_strength[dst] = col_strength.get(dst, 0.0) + weight
        sketch.update(key, weight)
        candidates.update(key, weight)

    candidate_rows = []
    for key in candidates.candidates():
        src, dst = key
        candidate_rows.append({"src": src, "dst": dst, "estimated_weight": sketch.estimate(key)})
    top = pd.DataFrame(candidate_rows)
    if not top.empty:
        top = top.sort_values("estimated_weight", ascending=False).head(candidate_capacity).reset_index(drop=True)
        top.insert(0, "rank", np.arange(1, len(top) + 1))

    summary = StreamingSummary(
        n_edges=int(len(edges)),
        total_weight=float(total_weight),
        unique_sources=len(row_strength),
        unique_destinations=len(col_strength),
        row_entropy_bits=_entropy_from_counts(row_strength),
        column_entropy_bits=_entropy_from_counts(col_strength),
        sketch_bytes=sketch.bytes,
        candidate_count=len(candidates.counts),
    )
    return summary, top


def streaming_accuracy(exact_top: pd.DataFrame, approx_top: pd.DataFrame, k: int = 20) -> dict[str, float]:
    """Compare exact top-k edges with approximate streaming heavy hitters."""

    if exact_top.empty:
        return {"topk_recall": 1.0, "median_relative_error": 0.0, "max_relative_error": 0.0}
    exact = exact_top.head(k).copy()
    approx = approx_top.copy()
    exact_keys = set(zip(exact["src"], exact["dst"]))
    approx_keys = set(zip(approx["src"], approx["dst"]))
    recall = len(exact_keys & approx_keys) / max(len(exact_keys), 1)

    merged = exact.merge(approx, on=["src", "dst"], how="left")
    merged["estimated_weight"] = merged["estimated_weight"].fillna(0.0)
    denominator = merged["weight"].replace(0, np.nan)
    rel_error = ((merged["estimated_weight"] - merged["weight"]).abs() / denominator).fillna(0.0)
    return {
        "topk_recall": float(recall),
        "median_relative_error": float(rel_error.median()),
        "max_relative_error": float(rel_error.max()),
    }
