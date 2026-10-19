from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from io_utils import read_jsonl, read_yaml, write_json_atomic
from path_manager import ExperimentPaths
from positivity_criteria import evaluate_model_dataset_positivity, summarize_family_counts, summarize_model_counts
from run_manifest import load_run_manifest, resolve_manifest_path, update_run_manifest, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_panel_name
from stats_bootstrap import grouped_bootstrap


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

    metrics_cfg = read_yaml(paths.configs_dir / "metrics.yaml")["metrics"]
    rows = read_jsonl(resolve_manifest_path(paths, manifest, "primary_metrics"))
    simplified = []
    for row in rows:
        if row.get("readout_id") != "semantic_exact" or row.get("permutation_id") != "identity":
            continue
        anchor = row["metrics"].get("final_winner_anchor", {})
        simplified.append(
            {
                "group_id": f"{row.get('dataset_name')}::{row.get('example_id')}",
                "delta_margin": anchor.get("delta_margin"),
            }
        )
    valid = [row for row in simplified if row["delta_margin"] is not None]
    if valid:
        stats = grouped_bootstrap(
            valid,
            group_field="group_id",
            value_fn=lambda sampled: sum(item["delta_margin"] for item in sampled) / len(sampled),
        )
    else:
        stats = {"note": "no valid rows"}
    write_json_atomic(resolve_manifest_path(paths, manifest, "bootstrap_summary"), stats)
    criteria_cfg = metrics_cfg["positivity_criteria"]
    evaluations = evaluate_model_dataset_positivity(rows, criteria_cfg=criteria_cfg)
    write_json_atomic(resolve_manifest_path(paths, manifest, "model_dataset_positivity"), evaluations)
    write_json_atomic(resolve_manifest_path(paths, manifest, "model_positivity_summary"), summarize_model_counts(evaluations))
    write_json_atomic(resolve_manifest_path(paths, manifest, "family_positivity_summary"), summarize_family_counts(evaluations))
    update_run_manifest(
        paths,
        manifest,
        stage_name="run_stats",
        updates={
            "metric_row_count": len(rows),
            "valid_bootstrap_rows": len(valid),
            "positivity_evaluation_count": len(evaluations),
        },
    )


if __name__ == "__main__":
    main()
