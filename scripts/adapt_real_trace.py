#!/usr/bin/env python3
"""Convert a real flow capture (CTU-13 binetflow or generic CSV) to src,dst,bytes.

Addresses are relabeled to opaque integer identifiers via factorization, so the
output edge table carries no raw addresses. Flow order is preserved so that
window-based analyses see the capture's temporal structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/real/ctu13_edges.csv")
    parser.add_argument("--src-col", default="SrcAddr")
    parser.add_argument("--dst-col", default="DstAddr")
    parser.add_argument("--bytes-col", default="TotBytes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, usecols=[args.src_col, args.dst_col, args.bytes_col], low_memory=False)
    src_codes, _ = pd.factorize(frame[args.src_col])
    dst_codes, _ = pd.factorize(frame[args.dst_col])
    out = pd.DataFrame(
        {
            "src": src_codes,
            "dst": dst_codes,
            "bytes": pd.to_numeric(frame[args.bytes_col], errors="coerce").fillna(0).astype("int64"),
        }
    )
    out = out[out["bytes"] > 0].reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(
        f"wrote {output}: rows={len(out)} unique_src={out['src'].nunique()} "
        f"unique_dst={out['dst'].nunique()} "
        f"unique_pairs={len(out.drop_duplicates(['src', 'dst']))} "
        f"total_bytes={int(out['bytes'].sum())}"
    )


if __name__ == "__main__":
    main()
