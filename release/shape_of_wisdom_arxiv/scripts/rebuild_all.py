#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from shape_of_wisdom_release.build_assets import rebuild_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the arXiv release from included derived data.")
    parser.add_argument("--skip-pdf", action="store_true", help="Regenerate data assets but skip tectonic.")
    parser.add_argument("--offline", action="store_true", help="Compile with tectonic --only-cached.")
    args = parser.parse_args()
    rebuild_all(build_pdf=not args.skip_pdf, offline=args.offline)


if __name__ == "__main__":
    main()
