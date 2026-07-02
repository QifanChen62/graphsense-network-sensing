#!/usr/bin/env python3
"""Cross-validate the pcap-parsed edge table against official GraphBLAS matrices.

The official `20240507-142247.grb.tar` (8.6 GB) contains 128 inner tars, one per
2^23-packet segment, each holding 64 serialized GraphBLAS matrices (one per
2^17-packet window, UINT32 packet counts, 2^32 x 2^32 anonymized-address index
space). Inner tars are stored in build order, not data order; the member with
the earliest name timestamp (`tar/20240507-152247.8388608.tar`, byte offset
7,641,955,328, size 67,624,960) is the first data segment. Fetch it with:

    curl -r 7641955328-7709580287 \
      https://graphchallenge.s3.amazonaws.com/network2024/20240507-142247.grb.tar \
      -o inner_first.tar && mkdir grb_first && tar -xf inner_first.tar -C grb_first

The official matrices index addresses as little-endian uint32, while the pcap
parser records network byte order, so this script byteswaps before comparing.

Requires python-graphblas with suitesparse (needs numpy<2 wheels on Python 3.9;
use a side venv if the main environment runs numpy>=2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grb-dir", required=True, help="directory of extracted N.grb files for one segment")
    parser.add_argument("--edges", default="data/real/official_prefix40m_edges.csv")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--window-packets", type=int, default=131_072)
    parser.add_argument("--output", default="results/grb_cross_validation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import graphblas as gb

    needed = args.windows * args.window_packets
    ours = pd.read_csv(args.edges, nrows=needed)
    if len(ours) < needed:
        raise SystemExit(f"edge table has only {len(ours)} rows; need {needed}")
    src = ours["src"].to_numpy(dtype=np.uint32).byteswap().astype(np.uint64)
    dst = ours["dst"].to_numpy(dtype=np.uint32).byteswap().astype(np.uint64)
    our_keys = (src << np.uint64(32)) | dst

    rows = []
    for window in range(args.windows):
        blob = (Path(args.grb_dir) / f"{window}.grb").read_bytes()
        matrix = gb.Matrix.ss.deserialize(blob)
        grb_rows, grb_cols, grb_vals = matrix.to_coo()
        grb_keys = np.sort((grb_rows.astype(np.uint64) << np.uint64(32)) | grb_cols.astype(np.uint64))
        window_keys = np.sort(our_keys[window * args.window_packets : (window + 1) * args.window_packets])
        exact = len(grb_keys) == len(window_keys) and bool(np.array_equal(grb_keys, window_keys))
        rows.append(
            {
                "window": window,
                "grb_nvals": int(matrix.nvals),
                "grb_total": int(grb_vals.sum()),
                "grb_shape_log2": int(np.log2(matrix.shape[0])),
                "pairs_exact_match": exact,
            }
        )
        if not exact:
            print(f"window {window}: MISMATCH")

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    matched = int(frame["pairs_exact_match"].sum())
    print(f"wrote {output}")
    print(f"windows matched exactly: {matched}/{len(frame)}; total official nvals {int(frame['grb_nvals'].sum())}")


if __name__ == "__main__":
    main()
