from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_controls import paired_coverage_summary
from io_utils import read_jsonl, write_json_atomic, write_text_atomic
from path_manager import ExperimentPaths
from qc_reports import build_qc_summary, render_qc_markdown
from run_manifest import load_run_manifest, resolve_manifest_path, update_run_manifest, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_panel_name


def _paired_readout_coverage(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        key = (row.get("example_id"), row.get("model_name"), row.get("permutation_id"))
        groups.setdefault(key, set()).add(row.get("readout_id"))
    total = len(groups)
    fully_paired = sum(
        1
        for readouts in groups.values()
        if {"semantic_exact", "templated_semantic", "letter_label"}.issubset(readouts)
    )
    return {
        "groups_total": total,
        "groups_fully_paired": fully_paired,
        "groups_missing_any_readout": total - fully_paired,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")

    explicit_panel = None
    if args.smoke or args.panel is not None:
        explicit_panel = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    manifest = load_run_manifest(paths, run_id=args.run_id, panel_name=explicit_panel)
    validate_run_manifest_config_hashes(paths, manifest)

    examples = read_jsonl(resolve_manifest_path(paths, manifest, "prepared_examples"))
    audits = read_jsonl(resolve_manifest_path(paths, manifest, "token_audits"))
    failures = read_jsonl(resolve_manifest_path(paths, manifest, "run_controls_failures"))
    score_manifest = read_jsonl(resolve_manifest_path(paths, manifest, "score_manifest"))
    paired = paired_coverage_summary(score_manifest, required_fields=["cache_key", "example_id", "model_name"])
    paired.update(_paired_readout_coverage(score_manifest))
    summary = build_qc_summary(
        canonical_examples=examples,
        token_audits=audits,
        failures=failures,
        paired_coverage=paired,
    )
    write_json_atomic(resolve_manifest_path(paths, manifest, "qc_summary_json"), summary)
    write_text_atomic(resolve_manifest_path(paths, manifest, "qc_summary_markdown"), render_qc_markdown(summary))
    update_run_manifest(
        paths,
        manifest,
        stage_name="run_qc",
        updates={
            "example_count": len(examples),
            "token_audit_count": len(audits),
            "failure_count": len(failures),
        },
    )


if __name__ == "__main__":
    main()
