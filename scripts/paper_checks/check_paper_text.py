#!/usr/bin/env python3
"""Paper text quality checks. Fails the build if any check is violated.

Usage:
    python3 scripts/paper_checks/check_paper_text.py paper/arxiv_submission/main.tex
"""
from __future__ import annotations
import re, sys
from pathlib import Path


ABSTRACT_FORBIDDEN = [
    r"\b12345\b",
    r"\bseed\b",
    r"parquet",
    r"results/",
    r"scripts/",
    r"paper/part1",
    r"BUILD_INFO",
    r"SHA-256",
    r"sha256",
    r"cached artifacts",
    r"offline",
    r"HF_HUB",
]

BODY_FORBIDDEN = [
    (r"embedding\s+layer", "Use 'first logged layer (layer 0)' instead"),
]


def extract_abstract(tex: str) -> str:
    """Extract text between \\begin{abstract} and \\end{abstract}."""
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    return m.group(1) if m else ""


def check_abstract(abstract: str) -> list[str]:
    failures = []
    for pattern in ABSTRACT_FORBIDDEN:
        if re.search(pattern, abstract, re.IGNORECASE):
            failures.append(f"ABSTRACT contains forbidden token: '{pattern}'")
    return failures


def check_body(tex: str) -> list[str]:
    failures = []
    for pattern, msg in BODY_FORBIDDEN:
        if re.search(pattern, tex, re.IGNORECASE):
            failures.append(f"BODY: '{pattern}' found — {msg}")

    # Check "decision boundary" is qualified
    for m in re.finditer(r"decision\s+boundary", tex, re.IGNORECASE):
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(tex), m.end() + 80)
        context = tex[ctx_start:ctx_end].lower()
        if "zero-margin" not in context and "argmax" not in context and "softmax" not in context:
            failures.append(f"BODY: unqualified 'decision boundary' — specify zero-margin or argmax")

    return failures


def check_figure_refs(tex: str) -> list[str]:
    """Verify every \\label{{fig:X}} has at least one \\ref or \\cref mention."""
    failures = []
    labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", tex))
    for label in sorted(labels):
        short = label.replace("fig:", "")
        if label not in tex.replace(f"\\label{{{label}}}", ""):
            # Check for cref/ref
            if not re.search(rf"\\(?:c?ref|Cref)\{{[^}}]*{re.escape(label)}[^}}]*\}}", tex):
                failures.append(f"UNREFERENCED FIGURE: {label} has a label but is never \\cref'd or \\ref'd")
    return failures


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_paper_text.py <main.tex>")
        return 1

    tex = Path(sys.argv[1]).read_text()
    abstract = extract_abstract(tex)

    all_failures = []
    all_failures.extend(check_abstract(abstract))
    all_failures.extend(check_body(tex))
    all_failures.extend(check_figure_refs(tex))

    if all_failures:
        print(f"PAPER TEXT CHECK FAILED — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  ✗ {f}")
        return 1
    else:
        print("PAPER TEXT CHECK PASSED ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
