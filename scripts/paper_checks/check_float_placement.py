#!/usr/bin/env python3
"""Check that no figures appear after the references section.

Parses the .aux file to extract page numbers of figure labels and the
references start label, then verifies all figures come before references.

Usage:
    python3 scripts/paper_checks/check_float_placement.py paper/arxiv_submission/main.aux
"""
from __future__ import annotations
import re, sys
from pathlib import Path


def parse_aux_pages(aux_path: Path) -> dict[str, int]:
    """Extract label→page mappings from .aux newlabel entries."""
    pages: dict[str, int] = {}
    text = aux_path.read_text()
    # Pattern: \newlabel{LABEL}{{NUMBER}{PAGE}{...}{...}{...}}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", text):
        label, page = m.group(1), int(m.group(2))
        pages[label] = page
    return pages


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_float_placement.py <main.aux>")
        return 1

    aux_path = Path(sys.argv[1])
    if not aux_path.exists():
        print(f"ERROR: {aux_path} not found. Compile the paper first.")
        return 1

    pages = parse_aux_pages(aux_path)

    ref_page = pages.get("sec:references_start")
    if ref_page is None:
        print("WARNING: \\label{sec:references_start} not found in .aux")
        print("  Add \\label{sec:references_start} before \\bibliography in main.tex")
        return 0  # Warning, not failure

    fig_labels = {k: v for k, v in pages.items() if k.startswith("fig:")}
    failures = []
    for label, page in sorted(fig_labels.items()):
        if page >= ref_page:
            failures.append(f"  FAIL: {label} on page {page} (references start on page {ref_page})")

    if failures:
        print(f"FLOAT PLACEMENT CHECK FAILED — {len(failures)} figure(s) after references:")
        for f in failures:
            print(f)
        return 1
    else:
        print(f"FLOAT PLACEMENT CHECK PASSED — all {len(fig_labels)} figures before references (page {ref_page})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
