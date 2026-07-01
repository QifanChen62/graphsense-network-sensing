#!/usr/bin/env python3
"""Check local dependencies without downloading data."""

from __future__ import annotations

import importlib
import os
import platform


PACKAGES = ["numpy", "scipy", "pandas", "matplotlib"]


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphsense-mpl")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    for package in PACKAGES:
        module = importlib.import_module(package)
        print(f"{package}={getattr(module, '__version__', 'unknown')}")
    print("setup_check=ok")


if __name__ == "__main__":
    main()
