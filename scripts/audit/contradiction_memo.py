#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from _audit_common import REPO_ROOT, default_paths, write_json


def _extract_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p in low for p in patterns):
            out.append({"path": str(path.relative_to(REPO_ROOT)), "line": i, "text": line.strip()})
    return out


def _line_contains(path: Path, needle: str) -> int | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if needle in line:
            return i
    return None


def _find_substitution_panel_ref(v3_path: Path) -> dict[str, Any]:
    lines = v3_path.read_text(encoding="utf-8").splitlines()
    text_ref_line = None
    caption_sub_line = None
    for i, line in enumerate(lines, start=1):
        if "Substituting attention scalars" in line and "fig:causal" in line:
            text_ref_line = i
        if "(c)}~Substitution summary" in line:
            caption_sub_line = i
    mismatch = False
    if text_ref_line is not None:
        txt = lines[text_ref_line - 1]
        mismatch = bool(re.search(r"\\cref\{fig:causal\}b", txt))
    return {
        "text_reference_line": text_ref_line,
        "caption_substitution_line": caption_sub_line,
        "panel_mismatch_detected": mismatch,
    }


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Generate contradiction-resolution memo from code + paper sources.")
    ap.add_argument("--out-json", type=Path, default=paths.audit / "contradiction_memo.json")
    args = ap.parse_args()

    v1 = REPO_ROOT / "paper" / "final_paper" / "paper_publish.tex"
    v2 = REPO_ROOT / "paper" / "final_paper" / "paper_publish_v2.tex"
    v3 = REPO_ROOT / "paper" / "final_paper" / "paper_publish_v3.tex"
    patching_py = REPO_ROOT / "src" / "sow" / "v2" / "causal" / "patching.py"
    stage09 = REPO_ROOT / "scripts" / "v2" / "09_causal_tests.py"

    patterns = [
        "substitut",
        "attention is more",
        "mlp substitution",
        "inject",
        "stable-wrong",
        "stable_wrong",
    ]
    paper_hits = []
    for p in [v3, v2, v1]:
        if p.exists():
            paper_hits.extend(_extract_hits(p, patterns))

    patch_bug_line = _line_contains(patching_py, "success_key = next((k for k in successes.keys() if k[0] == str(mid)), None)")
    failing_set_line = _line_contains(stage09, 'failing = merged[merged["trajectory_type"].isin(["unstable_wrong", "stable_wrong"])]')
    panel_ref = _find_substitution_panel_ref(v3)

    payload = {
        "contradiction_candidates": paper_hits,
        "code_evidence": {
            "patching_single_source_selection": {
                "path": str(patching_py.relative_to(REPO_ROOT)),
                "line": patch_bug_line,
                "summary": (
                    "Legacy substitution picks first stable-correct source per model; "
                    "results are row-order sensitive and do not represent robust all-pairs transfer."
                ),
            },
            "failing_set_scope": {
                "path": str(stage09.relative_to(REPO_ROOT)),
                "line": failing_set_line,
                "summary": "Legacy causal test includes stable_wrong + unstable_wrong in failing set.",
            },
            "v3_panel_reference": panel_ref,
        },
        "resolved_interpretation": {
            "what_happened": (
                "The mismatch is not evidence that two independent mechanisms disagree; "
                "it mainly comes from experimental-definition drift and a legacy pairing bug."
            ),
            "conceptual_mismatch": [
                "Scalar substitution under a linearized bookkeeping model is not the same as full activation patching.",
                "Legacy pairing used one fixed source prompt per model, making attention outcomes unstable to row order.",
                "Some prose compares stable-wrong-only claims to code that mixes unstable-wrong targets.",
            ],
            "corrected_claim_for_v3": (
                "Under cached scalar-substitution accounting, MLP substitution yields larger positive "
                "margin shifts than attention in the audited pair sets, but this conclusion is conditional "
                "on the pairing protocol, layer range, and readout definition."
            ),
        },
    }
    write_json(args.out_json, payload)
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

