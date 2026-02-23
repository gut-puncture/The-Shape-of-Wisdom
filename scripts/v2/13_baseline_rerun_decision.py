#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _common import baseline_output_path, load_experiment_config, resolve_models, run_v2_root_for, write_json


REQUIRED_DECISION_METRICS_COLUMNS = {
    "model_id",
    "prompt_uid",
    "layer_index",
    "correct_key",
    "delta",
    "is_correct",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="V2: baseline rerun decision contract")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "experiment_v2.yaml"))
    ap.add_argument("--final-publication-run", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]

    out_root = run_v2_root_for(args.run_id)
    decision_path = out_root / "decision_metrics.parquet"
    decision_exists = decision_path.exists()
    decision_has_columns = False
    observed_model_ids: list[str] = []
    if decision_exists:
        try:
            dm = pd.read_parquet(decision_path)
        except Exception:
            dm = pd.DataFrame()
        cols = set(dm.columns.tolist())
        decision_has_columns = REQUIRED_DECISION_METRICS_COLUMNS.issubset(cols)
        observed_model_ids = sorted(dm["model_id"].astype(str).drop_duplicates().tolist()) if ("model_id" in dm.columns and not dm.empty) else []
    expected_models_in_metrics = bool(expected_model_ids) and all(mid in set(observed_model_ids) for mid in expected_model_ids)

    baseline_outputs_present = {}
    for model_id in expected_model_ids:
        baseline_outputs_present[model_id] = baseline_output_path(args.run_id, model_id).exists()
    baseline_outputs_complete = bool(expected_model_ids) and all(bool(v) for v in baseline_outputs_present.values())

    remediation_contract_intact = bool(decision_exists and decision_has_columns and expected_models_in_metrics)

    if args.final_publication_run:
        requires_full_regen = True
        rationale = (
            "Final publication-grade run requires fresh 3-model baseline regeneration with a fresh run_id and full thermal-resume logging."
        )
        policy_mode = "final_publication_run"
    else:
        requires_full_regen = not bool(remediation_contract_intact and baseline_outputs_complete)
        if requires_full_regen:
            rationale = (
                "Remediation verification cannot safely reuse baseline because layerwise/logit reproducibility contract is incomplete."
            )
        else:
            rationale = (
                "Remediation verification may reuse baseline outputs because layerwise/logit contracts are intact and correctness is top1-logit authoritative."
            )
        policy_mode = "remediation_verification"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(args.run_id),
        "policy_mode": policy_mode,
        "requires_full_baseline_regeneration": bool(requires_full_regen),
        "rationale": str(rationale),
        "expected_model_ids": expected_model_ids,
        "observed_model_ids": observed_model_ids,
        "checks": {
            "decision_metrics_exists": bool(decision_exists),
            "decision_metrics_required_columns": bool(decision_has_columns),
            "expected_models_in_metrics": bool(expected_models_in_metrics),
            "baseline_outputs_complete": bool(baseline_outputs_complete),
            "baseline_outputs_present_by_model": {str(k): bool(v) for k, v in baseline_outputs_present.items()},
        },
    }
    out_path = out_root / "meta" / "stage13_baseline_rerun_decision.json"
    write_json(out_path, payload)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
