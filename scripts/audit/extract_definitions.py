#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _audit_common import REPO_ROOT, default_paths, write_json


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_line(path: Path, needle: str) -> int:
    lines = _read_lines(path)
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return idx
    raise RuntimeError(f"needle not found in {path}: {needle}")


def _snippet(path: Path, start: int, end: int) -> str:
    lines = _read_lines(path)
    s = max(1, int(start))
    e = min(len(lines), int(end))
    return "\n".join(lines[s - 1 : e])


def _entry(*, name: str, file_path: Path, line_start: int, line_end: int, variable_names: list[str], interpretation: str) -> dict[str, Any]:
    rel = str(file_path.relative_to(REPO_ROOT))
    return {
        "name": str(name),
        "variable_names": list(variable_names),
        "source_path": rel,
        "source_line_start": int(line_start),
        "source_line_end": int(line_end),
        "code": _snippet(file_path, line_start, line_end),
        "interpretation": str(interpretation),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract canonical metric/component definitions from code.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=default_paths().audit / "definitions_extracted.json",
    )
    args = ap.parse_args()

    metrics_py = REPO_ROOT / "src" / "sow" / "v2" / "metrics.py"
    traj_py = REPO_ROOT / "src" / "sow" / "v2" / "trajectory_types.py"
    buckets_py = REPO_ROOT / "src" / "sow" / "token_buckets" / "option_buckets.py"
    tracing_py = REPO_ROOT / "scripts" / "v2" / "07_run_tracing.py"
    patching_py = REPO_ROOT / "src" / "sow" / "v2" / "causal" / "patching.py"
    stage09_py = REPO_ROOT / "scripts" / "v2" / "09_causal_tests.py"

    definitions: list[dict[str, Any]] = []

    # closest competitor k*(l)
    k_line = _find_line(metrics_py, "def _competitor_from_logits")
    definitions.append(
        _entry(
            name="closest_competitor_k_star_l",
            file_path=metrics_py,
            line_start=k_line,
            line_end=k_line + 11,
            variable_names=["comp", "competitor", "correct_key", "candidate_logits"],
            interpretation=(
                "k*(l) is reselected at each layer as argmax over non-correct options "
                "from per-layer candidate logits; it can change with layer."
            ),
        )
    )

    # delta(l)
    delta_line = _find_line(metrics_py, "delta = float(logits.get(ck, 0.0)) - float(logits.get(comp, 0.0))")
    definitions.append(
        _entry(
            name="delta_l",
            file_path=metrics_py,
            line_start=delta_line - 2,
            line_end=delta_line + 3,
            variable_names=["delta", "ck", "comp"],
            interpretation=(
                "delta(l) is correct-choice logit minus current competitor logit, "
                "computed at every layer from candidate logits."
            ),
        )
    )

    # option token set variants
    variants_line = _find_line(buckets_py, 'VARIANT_TEMPLATES = ["{L}", " {L}", "\\n{L}", "({L})", "{L}.", "{L}:"]')
    definitions.append(
        _entry(
            name="option_token_set_variants",
            file_path=buckets_py,
            line_start=variants_line - 3,
            line_end=variants_line + 4,
            variable_names=["VARIANT_TEMPLATES", "LETTER_OPTIONS", "build_buckets_from_tokenizer"],
            interpretation=(
                "Bucketed option-token set includes bare, leading-space, newline, parenthesized, "
                "dot, and colon variants for A/B/C/D."
            ),
        )
    )

    # stability rule
    stable_line = _find_line(traj_py, "stable = (late_flip_count <= int(max_late_flip_count)) and (min_abs_delta_tail >= float(min_abs_delta_tail_floor))")
    definitions.append(
        _entry(
            name="trajectory_stability",
            file_path=traj_py,
            line_start=stable_line - 6,
            line_end=stable_line + 7,
            variable_names=[
                "tail_len",
                "max_late_flip_count",
                "min_abs_delta_tail_floor",
                "late_flip_count",
                "min_abs_delta_tail",
            ],
            interpretation=(
                "Stability is defined on tail sign-flip count and minimum absolute delta in tail window; "
                "it is computed on delta/sign(delta), not on drift."
            ),
        )
    )

    # s_attn / s_mlp
    s_attn_line = _find_line(tracing_py, "s_attn = float(delta_after_attn - delta_in)")
    definitions.append(
        _entry(
            name="s_attn_s_mlp",
            file_path=tracing_py,
            line_start=s_attn_line - 5,
            line_end=s_attn_line + 6,
            variable_names=[
                "h_in",
                "attn_vec",
                "mlp_vec",
                "delta_in",
                "delta_after_attn",
                "delta_after_mlp",
                "s_attn",
                "s_mlp",
            ],
            interpretation=(
                "s_attn is margin change from adding attention update to h_in; "
                "s_mlp is incremental margin change from adding MLP update after attention."
            ),
        )
    )

    # substitution recurrence
    patch_line = _find_line(patching_py, "patched[i] = fail_drift[i] - fail_comp[i] + succ_comp[i]")
    definitions.append(
        _entry(
            name="substitution_recurrence",
            file_path=patching_py,
            line_start=patch_line - 5,
            line_end=patch_line + 12,
            variable_names=["fail_drift", "fail_comp", "succ_comp", "patched", "trace"],
            interpretation=(
                "At substituted layers drift is fail_drift - fail_component + success_component; "
                "patched delta trajectory is reconstructed by cumulative sum from failing start delta."
            ),
        )
    )

    mismatches: list[dict[str, Any]] = [
        {
            "mismatch_id": "readout_bucket_vs_single_token",
            "where_a": "src/sow/v2/metrics.py",
            "where_b": "scripts/v2/07_run_tracing.py",
            "why_it_matters": (
                "Decision metrics use bucketed option logits from baseline extraction, "
                "while tracing/scalars use single-token A/B/C/D logits. "
                "This can change competitor identity, delta scale, and sign for some rows."
            ),
            "paper_standardization_recommendation": (
                "Treat tracing decomposition claims as single-token readout claims; "
                "keep bucket-readout claims separate and explicitly caveated."
            ),
        },
        {
            "mismatch_id": "failing_set_scope_drift",
            "where_a": "scripts/v2/09_causal_tests.py",
            "where_b": "paper/final_paper/paper_publish_v3.tex",
            "why_it_matters": (
                "Code uses failing = stable_wrong + unstable_wrong for substitution, "
                "while prose often reads as stable_wrong-only."
            ),
            "paper_standardization_recommendation": (
                "Either restrict code and report to stable_wrong-only or explicitly label the broader failing set."
            ),
        },
        {
            "mismatch_id": "substitution_source_pairing_bug",
            "where_a": "src/sow/v2/causal/patching.py",
            "where_b": "results/parquet/patching_results.parquet",
            "why_it_matters": (
                "Source trajectory is selected by first key per model, so all failing prompts for a model "
                "are patched from one stable-correct source; results become row-order sensitive."
            ),
            "paper_standardization_recommendation": (
                "Use pair-set definitions that are explicit and order-invariant (all-pairs or seeded one-to-one)."
            ),
        },
        {
            "mismatch_id": "causal_panel_reference_drift",
            "where_a": "paper/final_paper/paper_publish_v3.tex:446",
            "where_b": "paper/final_paper/paper_publish_v3.tex:463-467",
            "why_it_matters": (
                "Text references substitution as panel (b) while caption assigns substitution to panel (c)."
            ),
            "paper_standardization_recommendation": "Update in-text references to panel (c) for substitution.",
        },
        {
            "mismatch_id": "competitor_reselection_across_layers",
            "where_a": "src/sow/v2/metrics.py",
            "where_b": "scripts/audit/substitution_rederive.py (blocked modes)",
            "why_it_matters": (
                "Cached tracing scalars embed one competitor mode; alternate competitor definitions "
                "cannot be recomputed for substitution without additional cached logits."
            ),
            "paper_standardization_recommendation": (
                "State competitor mode explicitly and mark competitor-sensitivity as blocked pending extra inference."
            ),
        },
    ]

    payload = {
        "definitions": definitions,
        "definition_consistency_diff": mismatches,
    }
    write_json(args.out_json, payload)
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

