#!/usr/bin/env python3
"""Optional wrapper for official GraphChallenge reference commands.

Set GRAPHCHALLENGE_REF_DIR to a checked-out reference repository and pass a
command after --. This keeps the local reproducibility path independent from
large external data and reference build requirements.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", default=os.environ.get("GRAPHCHALLENGE_REF_DIR"))
    parser.add_argument("command", nargs=argparse.REMAINDER, help="reference command after --")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ref_dir:
        raise SystemExit("Set --ref-dir or GRAPHCHALLENGE_REF_DIR to the official reference checkout.")
    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        raise SystemExit(f"reference directory not found: {ref_dir}")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Pass the reference command after --, for example: scripts/run_reference.py -- make")
    print(f"running in {ref_dir}: {' '.join(command)}")
    raise SystemExit(subprocess.call(command, cwd=ref_dir))


if __name__ == "__main__":
    main()
