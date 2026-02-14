#!/usr/bin/env python3
"""
Repo-local entrypoint so we don't require installing the package.

Usage:
  python3 sow.py --help
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()
    from sow.cli import main as cli_main  # noqa: PLC0415

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())

