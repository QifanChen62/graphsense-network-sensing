#!/usr/bin/env python3
"""Fetch and parse a byte-range prefix of the official anonymized capture.

Range-downloads the first N compressed bytes of the official GraphChallenge
pcap.zst, stream-decompresses the truncated frame, parses packet records, and
writes an anonymized src,dst,bytes edge table plus a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from _bootstrap import add_project_root

add_project_root()

from graphsense.pcap import iter_decompressed_chunks, pcap_stream_to_edges

DEFAULT_URL = "https://graphchallenge.s3.amazonaws.com/network2024/20240507-142247.pcap.zst"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--bytes", type=int, default=268_435_456, help="compressed prefix size to Range-request")
    parser.add_argument("--max-packets", type=int, default=5_000_000)
    parser.add_argument("--output", default="data/real/official_prefix_edges.csv")
    parser.add_argument("--cache", default="data/real/official_prefix.pcap.zst.part")
    parser.add_argument("--manifest", default="data/real/official_prefix_manifest.json")
    parser.add_argument("--force", action="store_true", help="re-download even when the cache is large enough")
    return parser.parse_args()


def download_prefix(url: str, target_bytes: int, cache_path: Path, force: bool) -> dict[str, object]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = cache_path.stat().st_size if cache_path.exists() else 0
    if force and existing:
        cache_path.unlink()
        existing = 0
    if existing >= target_bytes:
        print(f"cache {cache_path} already has {existing} bytes (>= {target_bytes}); skipping download")
        return {"http_status": None, "range_bytes_requested": target_bytes, "range_bytes_received": existing}

    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-{target_bytes - 1}"})
    received = existing
    with urllib.request.urlopen(request) as response:
        status = response.status
        if status not in (200, 206):
            raise RuntimeError(f"unexpected HTTP status {status} for {url}")
        if status == 200 and existing:
            raise RuntimeError("server ignored the Range resume request; rerun with --force")
        mode = "ab" if existing else "wb"
        with open(cache_path, mode) as handle:
            while received < target_bytes:
                chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, target_bytes - received))
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                print(f"downloaded {received}/{target_bytes} bytes", flush=True)
    return {"http_status": status, "range_bytes_requested": target_bytes, "range_bytes_received": received}


def main() -> None:
    args = parse_args()
    started = time.time()
    cache_path = Path(args.cache)
    download_info = download_prefix(args.url, args.bytes, cache_path, args.force)

    print("decompressing and parsing packet records...")
    edges, parse_info = pcap_stream_to_edges(iter_decompressed_chunks(cache_path), max_packets=args.max_packets)
    if edges.empty:
        raise RuntimeError("no packets parsed from the prefix; inspect the cache file")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output, index=False)
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    unique_pairs = len(edges.drop_duplicates(["src", "dst"]))
    import zstandard

    manifest = {
        "url": args.url,
        **download_info,
        **parse_info,
        "unique_src": int(edges["src"].nunique()),
        "unique_dst": int(edges["dst"].nunique()),
        "unique_pairs": int(unique_pairs),
        "duplication_rate": float(1.0 - unique_pairs / len(edges)),
        "total_bytes_field": int(edges["bytes"].sum()),
        "min_orig_len": int(edges["bytes"].min()),
        "max_orig_len": int(edges["bytes"].max()),
        "output": str(output),
        "output_sha256": sha256,
        "fetch_utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "zstandard_version": zstandard.__version__,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {output} ({len(edges)} edge records)")
    print(f"wrote {manifest_path}")
    for key in ("linktype", "record_size_detected", "parser_path", "packets_parsed", "unique_pairs", "duplication_rate"):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
