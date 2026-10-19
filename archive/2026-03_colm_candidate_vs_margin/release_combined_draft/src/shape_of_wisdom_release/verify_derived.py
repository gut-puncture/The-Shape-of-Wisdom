from __future__ import annotations

import json
from pathlib import Path

from .build_assets import DERIVED_PATH, MECHANISTIC_PATH, MODEL_ORDER, PRIMARY_DATASETS, RELEASE_ROOT, load_jsonl, write_json


REQUIRED_COLUMNS = {
    "cache_key",
    "dataset_name",
    "example_id",
    "final_winner",
    "model_family",
    "model_name",
    "n_layers",
    "norm_d2",
    "norm_d1",
    "norm_d50",
    "norm_delta_21",
    "norm_delta_250",
    "write_share_after_d2",
    "write_share_after_d1",
    "write_share_last20",
    "trajectory_regime",
    "leader_switch_count",
    "final_correct",
}


def verify_derived() -> dict[str, object]:
    rows = load_jsonl(DERIVED_PATH)
    mech_rows = load_jsonl(MECHANISTIC_PATH)
    failures: list[str] = []
    if not rows:
        failures.append("derived data is empty")
    columns = set(rows[0]) if rows else set()
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        failures.append(f"missing required columns: {missing}")
    if any("cache_relpath" in row for row in rows):
        failures.append("derived data contains local cache_relpath values")
    datasets = {str(row["dataset_name"]) for row in rows}
    expected_datasets = set(PRIMARY_DATASETS)
    if datasets != expected_datasets:
        failures.append(f"unexpected dataset set: {sorted(datasets)}")
    forbidden = ["mmlu_abstract_algebra", "MMLU Abstract Algebra", "Abstract Algebra"]
    for row in rows:
        rendered = json.dumps(row, sort_keys=True)
        if any(text in rendered for text in forbidden):
            failures.append("tiny dataset text leaked into trajectory rows")
            break
    models = {str(row["model_name"]) for row in rows}
    if models != set(MODEL_ORDER):
        failures.append(f"unexpected model set: {sorted(models)}")
    if any(str(row["model_name"]) not in MODEL_ORDER for row in rows):
        failures.append("unknown model appears in rows")
    if any(row.get("missing_option_scores") for row in rows):
        failures.append("missing option scores are present")
    if any(row.get("missing_options_any_layer") for row in rows):
        failures.append("missing options are present")
    analyses = {str(row.get("analysis")) for row in mech_rows}
    required_analyses = {
        "drift_decomposition",
        "component_depth_bin",
        "span_effect_by_model",
        "span_deletion_summary",
        "negative_control",
        "substitution_sensitivity",
    }
    missing_analyses = sorted(required_analyses - analyses)
    if missing_analyses:
        failures.append(f"missing mechanistic analyses: {missing_analyses}")
    summary = {
        "rows": len(rows),
        "mechanistic_rows": len(mech_rows),
        "datasets": sorted(datasets),
        "models": sorted(models),
        "mechanistic_analyses": sorted(analyses),
        "failures": failures,
    }
    write_json(RELEASE_ROOT / "qa" / "derived_verification.json", summary)
    if failures:
        raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    summary = verify_derived()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
