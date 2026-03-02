#!/usr/bin/env python3
"""Quick CPU-only regeneration of paper figures from stored parquet data.

This is a thin wrapper around sow.v2.figures.paper_figures that avoids
running the full 15-stage pipeline.  Use it when iterating on figure
aesthetics or after editing style.py / paper_figures.py.

Usage:
    python scripts/v2/regenerate_paper_figures.py                  # defaults
    python scripts/v2/regenerate_paper_figures.py \
        --parquet-dir results/parquet \
        --output-dir paper/final_paper/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `sow.*` resolves
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.figures.paper_figures import generate_all_figures, render_preview_pngs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate all paper figures (CPU-only, no inference).",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=REPO_ROOT / "results" / "parquet",
        help="Directory containing pre-computed parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper" / "final_paper" / "figures",
        help="Directory for output PDF figures.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Optional path to prompt JSONL (for fig6 prompt text).",
    )
    parser.add_argument(
        "--spans",
        type=Path,
        default=None,
        help="Optional path to spans.jsonl (for fig6 span structure).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run post-generation checks and render PNG previews.",
    )
    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"ERROR: parquet directory not found: {args.parquet_dir}", file=sys.stderr)
        return 1

    paths = generate_all_figures(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        prompts_path=args.prompts,
        spans_path=args.spans,
    )

    if args.verify:
        print("\nVerification:")
        previews = render_preview_pngs(figure_paths=paths)
        preview_dir = (REPO_ROOT / "tmp" / "fig_previews")
        for p in paths:
            status = "PASS"
            reason = ""
            if not p.exists():
                status = "FAIL"
                reason = "missing file"
            elif p.stat().st_size <= 0:
                status = "FAIL"
                reason = "empty file"
            else:
                try:
                    from pypdf import PdfReader  # type: ignore
                    pages = len(PdfReader(str(p)).pages)
                    if pages != 1:
                        status = "FAIL"
                        reason = f"expected 1 page, got {pages}"
                except Exception as exc:
                    reason = f"pdf check skipped ({exc})"
            suffix = f" - {reason}" if reason else ""
            print(f"  [{status}] {p.name}{suffix}")
        if previews:
            print(f"  [PASS] Rendered {len(previews)} PNG previews to {preview_dir}")
        else:
            print("  [WARN] PNG previews not rendered (missing pdftoppm/qlmanage?)")

    print(f"\n{len(paths)} figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
