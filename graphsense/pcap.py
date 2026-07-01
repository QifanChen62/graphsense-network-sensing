"""Streaming pcap prefix parsing for the official anonymized network-sensing capture.

The official GraphChallenge capture is distributed as a zstd-compressed pcap. This
module parses a byte-range prefix of that stream into src/dst/bytes edge records
without requiring the full multi-gigabyte download: a truncated zstd frame is
decompressed as far as it goes, and a truncated final packet record is dropped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

GLOBAL_HEADER_BYTES = 24
RECORD_HEADER_BYTES = 16

# First four file bytes -> (struct byte-order prefix, nanosecond timestamps).
_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", False),
    b"\xa1\xb2\xc3\xd4": (">", False),
    b"\x4d\x3c\xb2\xa1": ("<", True),
    b"\xa1\xb2\x3c\x4d": (">", True),
}

# Link type -> byte offset of the IPv4 header within the packet payload.
_LINKTYPE_IP_OFFSETS = {
    0: 4,    # NULL/loopback
    1: 14,   # Ethernet
    101: 0,  # RAW IP
    113: 16, # Linux cooked capture
}


@dataclass(frozen=True)
class PcapMeta:
    """Byte order, timestamp unit, link type, and IPv4 offset for one capture."""

    byte_order: str
    nanosecond: bool
    linktype: int
    ip_offset: int
    magic_hex: str


def parse_pcap_global_header(buf: bytes) -> PcapMeta:
    """Parse the 24-byte pcap global header and resolve the IPv4 payload offset."""

    if len(buf) < GLOBAL_HEADER_BYTES:
        raise ValueError(f"need {GLOBAL_HEADER_BYTES} bytes for the pcap global header, got {len(buf)}")
    magic = bytes(buf[:4])
    if magic not in _PCAP_MAGICS:
        raise ValueError(f"unknown pcap magic {magic.hex()}; not a pcap stream")
    byte_order, nanosecond = _PCAP_MAGICS[magic]
    linktype = struct.unpack_from(byte_order + "I", buf, 20)[0]
    ip_offset = _LINKTYPE_IP_OFFSETS.get(linktype, 14)
    return PcapMeta(
        byte_order=byte_order,
        nanosecond=nanosecond,
        linktype=linktype,
        ip_offset=ip_offset,
        magic_hex=magic.hex(),
    )


def detect_uniform_record_size(buf: bytes, meta: PcapMeta, probe: int = 64) -> Optional[int]:
    """Return 16 + incl_len when the first `probe` records share one incl_len, else None."""

    header = struct.Struct(meta.byte_order + "IIII")
    offset = GLOBAL_HEADER_BYTES
    sizes = []
    while len(sizes) < probe and offset + RECORD_HEADER_BYTES <= len(buf):
        _, _, incl_len, _ = header.unpack_from(buf, offset)
        body_end = offset + RECORD_HEADER_BYTES + incl_len
        if body_end > len(buf):
            break
        sizes.append(incl_len)
        offset = body_end
    if len(sizes) < 2:
        return None
    if len(set(sizes)) != 1:
        return None
    return RECORD_HEADER_BYTES + sizes[0]


def resolve_ip_offset(buf: bytes, meta: PcapMeta, probe: int = 64, min_fraction: float = 0.9) -> tuple[int, float]:
    """Pick the IPv4 offset whose version nibble is 4 for >= min_fraction of probe records.

    Tries the linktype-derived offset first, then the other known offsets. Raises
    ValueError with a first-record hexdump when nothing looks like IPv4.
    """

    header = struct.Struct(meta.byte_order + "IIII")
    candidates = [meta.ip_offset] + [off for off in _LINKTYPE_IP_OFFSETS.values() if off != meta.ip_offset]
    best_offset, best_fraction = meta.ip_offset, 0.0
    for candidate in candidates:
        offset = GLOBAL_HEADER_BYTES
        seen = hits = 0
        while seen < probe and offset + RECORD_HEADER_BYTES <= len(buf):
            _, _, incl_len, _ = header.unpack_from(buf, offset)
            body_start = offset + RECORD_HEADER_BYTES
            body_end = body_start + incl_len
            if body_end > len(buf):
                break
            version_index = body_start + candidate
            if version_index < body_end and (buf[version_index] >> 4) == 4:
                hits += 1
            seen += 1
            offset = body_end
        fraction = hits / seen if seen else 0.0
        if fraction > best_fraction:
            best_offset, best_fraction = candidate, fraction
        if fraction >= min_fraction:
            return candidate, fraction
    first_record = bytes(buf[GLOBAL_HEADER_BYTES : GLOBAL_HEADER_BYTES + 96])
    raise ValueError(
        "no candidate IPv4 offset matched "
        f"(best offset {best_offset} at fraction {best_fraction:.2f}); "
        f"first record bytes: {first_record.hex()}"
    )


def parse_packets_numpy(data: bytes, meta: PcapMeta, record_size: int, max_packets: int, ip_offset: int) -> pd.DataFrame:
    """Vectorized parse of fixed-size records; assumes incl_len is uniform in data."""

    usable = (len(data) // record_size) * record_size
    if usable == 0:
        return pd.DataFrame({"src": [], "dst": [], "bytes": []})
    arr = np.frombuffer(data, dtype=np.uint8, count=usable).reshape(-1, record_size)
    if len(arr) > max_packets:
        arr = arr[:max_packets]
    length_dtype = "<u4" if meta.byte_order == "<" else ">u4"
    incl_len = arr[:, 8:12].copy().view(length_dtype).ravel()
    expected = record_size - RECORD_HEADER_BYTES
    if not np.all(incl_len == expected):
        raise ValueError("incl_len drifted inside a numpy batch; caller must fall back to struct parsing")
    src_start = RECORD_HEADER_BYTES + ip_offset + 12
    src = arr[:, src_start : src_start + 4].copy().view(">u4").ravel()
    dst = arr[:, src_start + 4 : src_start + 8].copy().view(">u4").ravel()
    orig_len = arr[:, 12:16].copy().view(length_dtype).ravel()
    return pd.DataFrame(
        {
            "src": src.astype(np.int64),
            "dst": dst.astype(np.int64),
            "bytes": orig_len.astype(np.int64),
        }
    )


def parse_packets_struct(chunks: Iterator[bytes], meta: PcapMeta, max_packets: int, ip_offset: int) -> pd.DataFrame:
    """Sequential parse tolerant of variable record sizes and truncated tails.

    `chunks` must start at the first record (global header already consumed).
    """

    header = struct.Struct(meta.byte_order + "IIII")
    buf = bytearray()
    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[int] = []
    src_field = struct.Struct(">II")
    done = False
    for chunk in chunks:
        buf.extend(chunk)
        offset = 0
        while offset + RECORD_HEADER_BYTES <= len(buf):
            _, _, incl_len, orig_len = header.unpack_from(buf, offset)
            body_end = offset + RECORD_HEADER_BYTES + incl_len
            if body_end > len(buf):
                break
            version_index = offset + RECORD_HEADER_BYTES + ip_offset
            if version_index + 20 <= body_end:
                src, dst = src_field.unpack_from(buf, version_index + 12)
                srcs.append(src)
                dsts.append(dst)
                weights.append(orig_len)
            offset = body_end
            if len(srcs) >= max_packets:
                done = True
                break
        del buf[:offset]
        if done:
            break
    return pd.DataFrame(
        {
            "src": pd.array(srcs, dtype="int64"),
            "dst": pd.array(dsts, dtype="int64"),
            "bytes": pd.array(weights, dtype="int64"),
        }
    )


def iter_decompressed_chunks(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    """Stream-decompress a possibly truncated zstd file, yielding what decodes cleanly."""

    import zstandard

    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with open(path, "rb") as handle:
        reader = dctx.stream_reader(handle, read_across_frames=True)
        while True:
            try:
                chunk = reader.read(chunk_size)
            except (zstandard.ZstdError, EOFError):
                # The prefix ends mid-frame; keep everything decoded so far.
                break
            if not chunk:
                break
            yield chunk


def pcap_stream_to_edges(chunks: Iterator[bytes], max_packets: int, probe: int = 64) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse a decompressed pcap byte stream into src/dst/bytes edge records.

    Buffers enough of the stream to read the global header and probe records,
    then uses the vectorized fixed-record path when incl_len is uniform and the
    struct path otherwise. Returns the edge table plus parser provenance info.
    """

    iterator = iter(chunks)
    buf = bytearray()
    probe_bytes = GLOBAL_HEADER_BYTES + 64 * 1024
    for chunk in iterator:
        buf.extend(chunk)
        if len(buf) >= probe_bytes:
            break
    meta = parse_pcap_global_header(bytes(buf[:GLOBAL_HEADER_BYTES]))
    ip_offset, ipv4_fraction = resolve_ip_offset(bytes(buf), meta, probe=probe)
    record_size = detect_uniform_record_size(bytes(buf), meta, probe=probe)
    body = bytes(buf[GLOBAL_HEADER_BYTES:])

    def remaining() -> Iterator[bytes]:
        if body:
            yield body
        for chunk in iterator:
            yield chunk

    parser_path = "numpy" if record_size is not None else "struct"
    if record_size is not None:
        frames = []
        carry = bytearray()
        parsed = 0
        stream = remaining()
        for chunk in stream:
            carry.extend(chunk)
            usable = (len(carry) // record_size) * record_size
            if usable == 0:
                continue
            take = min(usable, (max_packets - parsed) * record_size)
            try:
                frame = parse_packets_numpy(bytes(carry[:take]), meta, record_size, max_packets - parsed, ip_offset)
            except ValueError:
                # incl_len drifted after the probe window: finish with the
                # struct parser from the unconsumed remainder of the stream.
                parser_path = "numpy_then_struct"
                def rest() -> Iterator[bytes]:
                    yield bytes(carry)
                    for later in stream:
                        yield later
                frames.append(parse_packets_struct(rest(), meta, max_packets - parsed, ip_offset))
                break
            frames.append(frame)
            parsed += len(frame)
            del carry[:take]
            if parsed >= max_packets:
                break
        edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"src": [], "dst": [], "bytes": []})
    else:
        edges = parse_packets_struct(remaining(), meta, max_packets, ip_offset)

    info: dict[str, object] = {
        "pcap_magic_hex": meta.magic_hex,
        "byte_order": meta.byte_order,
        "nanosecond": meta.nanosecond,
        "linktype": meta.linktype,
        "ip_offset": ip_offset,
        "ipv4_fraction_probe": ipv4_fraction,
        "record_size_detected": record_size,
        "parser_path": parser_path,
        "packets_parsed": int(len(edges)),
    }
    return edges, info
