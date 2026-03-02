#!/usr/bin/env python3
"""Figure quality control for Paper 1 / Part I.

Usage::

    python scripts/part1/figure_qc.py paper/part1/figures/
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

EXPECTED_FIGURES = [
    "fig1_examples.pdf",
    "fig2_delta_distribution.pdf",
    "fig3_decision_space_trajectories.pdf",
    "fig4_flow_field.pdf",
    "fig5_commitment_and_flips.pdf",
    "fig6_robustness.pdf",
]

MIN_SIZE_BYTES = 15_000


def check_figures(fig_dir: Path) -> list[str]:
    """Run all QC checks. Returns list of failure messages."""
    failures: list[str] = []

    for name in EXPECTED_FIGURES:
        path = fig_dir / name
        if not path.exists():
            failures.append(f"MISSING: {name}")
            continue

        size = path.stat().st_size
        if size < MIN_SIZE_BYTES:
            failures.append(f"TOO SMALL: {name} is {size} bytes (min {MIN_SIZE_BYTES})")

        # Read PDF header
        content = path.read_bytes()
        if not content.startswith(b"%PDF"):
            failures.append(f"NOT PDF: {name} does not start with %PDF header")
            continue

        # Check for page count (look for /Type /Page entries)
        text = content.decode("latin-1", errors="replace")
        import re

        pages = re.findall(r"/Type\s*/Page[^s]", text)
        if len(pages) == 0:
            failures.append(f"NO PAGES: {name} has 0 detected pages")
        elif len(pages) > 1:
            failures.append(f"MULTI-PAGE: {name} has {len(pages)} pages (expected 1)")

        # Check for MediaBox (bounding box)
        mediaboxes = re.findall(
            r"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", text
        )
        if not mediaboxes:
            failures.append(f"NO MEDIABOX: {name} has no bounding box")
        else:
            for mb in mediaboxes:
                x1, y1, x2, y2 = [float(v) for v in mb]
                if x2 - x1 < 10 or y2 - y1 < 10:
                    failures.append(
                        f"TINY BBOX: {name} has bounding box "
                        f"({x1},{y1},{x2},{y2}) — likely blank"
                    )

        # Check for vector vs raster content
        n_images = text.count("/Subtype /Image")
        n_stream = text.count("stream")
        if n_images > 0 and n_stream > 0:
            image_ratio = n_images / n_stream
            if image_ratio > 0.5:
                failures.append(
                    f"RASTER WARNING: {name} has {n_images} image objects "
                    f"out of {n_stream} streams — may be largely raster"
                )

        # Check fonts embedded (look for /Type /Font)
        n_fonts = text.count("/Type /Font")
        if n_fonts == 0:
            # Not necessarily an error — some minimal PDFs may not embed fonts
            pass  # Informational only

    return failures


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: figure_qc.py <figures_dir>")
        return 1

    fig_dir = Path(sys.argv[1])
    if not fig_dir.is_dir():
        print(f"ERROR: {fig_dir} is not a directory")
        return 1

    failures = check_figures(fig_dir)

    if failures:
        print(f"FIGURE QC FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    else:
        print(f"FIGURE QC PASSED — {len(EXPECTED_FIGURES)} figures OK ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
