#!/usr/bin/env python3
"""Quality gates for Paper 1. All must pass or build fails.

Usage:
    python3 scripts/part1/quality_gates.py --paper-dir paper/part1
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path


def check_forbidden_phrases(tex_text: str, defs: dict) -> list[str]:
    """Check for forbidden ambiguous phrases."""
    failures = []
    for phrase in defs.get("forbidden_ambiguous_phrases", []):
        if phrase.startswith("about ") or phrase.startswith("approximately") or phrase.startswith("roughly"):
            # Regex patterns
            pattern = phrase.replace("\\d+", r"\d+")
            if re.search(pattern, tex_text):
                failures.append(f"FORBIDDEN PHRASE: found pattern '{phrase}'")
        else:
            # Literal
            if phrase.lower() in tex_text.lower():
                failures.append(f"FORBIDDEN PHRASE: '{phrase}' found in paper text")
    return failures


def check_figures_exist(paper_dir: Path) -> list[str]:
    """Check all expected figure PDFs exist and are non-trivial."""
    expected = [
        "figA_why_delta_soft.pdf",
        "fig2_examples.pdf",
        "fig3_distributions.pdf",
        "fig4_decision_space.pdf",
        "fig5_flow_field.pdf",
        "fig6_commitment.pdf",
        "fig7_robustness.pdf",
    ]
    failures = []
    for name in expected:
        p = paper_dir / "figures" / name
        if not p.exists():
            failures.append(f"MISSING FIGURE: {name}")
        elif p.stat().st_size < 5000:
            failures.append(f"TINY FIGURE: {name} is {p.stat().st_size} bytes")
        elif not p.read_bytes().startswith(b"%PDF"):
            failures.append(f"NOT PDF: {name}")
    return failures


def check_build_info(paper_dir: Path) -> list[str]:
    """Check BUILD_INFO.json exists and has required fields."""
    bi_path = paper_dir / "BUILD_INFO.json"
    if not bi_path.exists():
        return ["MISSING: BUILD_INFO.json"]
    bi = json.loads(bi_path.read_text())
    failures = []
    for key in ["parquet_dir", "git_commit", "build_timestamp_utc", "file_hashes"]:
        if key not in bi:
            failures.append(f"BUILD_INFO missing key: {key}")
    if "figure_hashes" not in bi:
        failures.append("BUILD_INFO missing figure_hashes")
    return failures


def check_auto_numbers(paper_dir: Path) -> list[str]:
    """Check auto_numbers.tex exists and is non-empty."""
    an = paper_dir / "auto_numbers.tex"
    if not an.exists():
        return ["MISSING: auto_numbers.tex"]
    text = an.read_text()
    macros = re.findall(r"\\newcommand", text)
    if len(macros) < 30:
        return [f"auto_numbers.tex has only {len(macros)} macros (expected ≥30)"]
    return []


def check_terminology(tex_text: str) -> list[str]:
    """Check terminology compliance."""
    failures = []
    # "embedding layer" without qualification
    if re.search(r"embedding\s+layer", tex_text, re.IGNORECASE):
        failures.append("TERMINOLOGY: 'embedding layer' found — should be 'first logged layer (layer 0)'")

    # "decision boundary" without qualification
    for match in re.finditer(r"decision\s+boundary", tex_text, re.IGNORECASE):
        # Check if qualified (nearby text mentions zero-margin or p(correct))
        start = max(0, match.start() - 100)
        end = min(len(tex_text), match.end() + 100)
        context = tex_text[start:end]
        if "zero-margin" not in context.lower() and "delta" not in context.lower():
            failures.append(f"TERMINOLOGY: unqualified 'decision boundary' near: ...{tex_text[match.start()-30:match.end()+30]}...")

    # "phase transition"
    if "phase transition" in tex_text.lower():
        failures.append("TERMINOLOGY: 'phase transition' found — regimes vary smoothly, not discontinuously")

    return failures


def check_no_freehand_pct(tex_text: str) -> list[str]:
    """Check for freehand percentage ranges in prose (not in macros/tables)."""
    failures = []
    # Look for patterns like "50--65%" or "60-80%" or "about 60%"
    patterns = [
        r"\d+--\d+\\?%",
        r"\d+-\d+\\?%",
        r"about\s+\d+\\?%",
        r"roughly\s+\d+\\?%",
        r"approximately\s+\d+\\?%",
    ]
    for pat in patterns:
        matches = re.findall(pat, tex_text)
        for m in matches:
            failures.append(f"FREEHAND PCT: '{m}' — use a computed macro instead")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", type=Path, default=Path("paper/part1"))
    args = ap.parse_args()
    pdir = args.paper_dir.resolve()

    # Load definitions
    defs_path = pdir / "definitions_and_numbers.json"
    defs = json.loads(defs_path.read_text()) if defs_path.exists() else {}

    # Load paper text
    tex_candidates = ["paper.tex", "main.tex", "part1.tex"]
    tex_text = ""
    for name in tex_candidates:
        p = pdir / name
        if p.exists():
            tex_text = p.read_text()
            break

    all_failures = []

    # 1) Forbidden phrases
    all_failures.extend(check_forbidden_phrases(tex_text, defs))

    # 2) Figure existence
    all_failures.extend(check_figures_exist(pdir))

    # 3) BUILD_INFO
    all_failures.extend(check_build_info(pdir))

    # 4) auto_numbers.tex
    all_failures.extend(check_auto_numbers(pdir))

    # 5) Terminology
    all_failures.extend(check_terminology(tex_text))

    # 6) Freehand percentages
    all_failures.extend(check_no_freehand_pct(tex_text))

    # Report
    results = {
        "total_checks": 6,
        "failures": all_failures,
        "pass": len(all_failures) == 0,
    }
    out_path = pdir / "QUALITY_GATES.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    if all_failures:
        print(f"QUALITY GATES FAILED — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  ✗ {f}")
        return 1
    else:
        print("QUALITY GATES PASSED ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
