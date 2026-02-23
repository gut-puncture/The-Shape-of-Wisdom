#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import base_parser, load_experiment_config, resolve_models, run_v2_root_for, write_json, write_parquet
from sow.v2.trajectory_types import classify_trajectory_table, type_counts


def main() -> int:
    ap = base_parser("V2: classify prompts into four trajectory types")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]
    validators_cfg = cfg.get("validators") or {}
    stage03_cfg = validators_cfg.get("stage03_trajectory") or {}
    required_types = [str(x) for x in (stage03_cfg.get("required_types") or ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"])]
    min_count_per_type_per_model = int(stage03_cfg.get("min_count_per_type_per_model", 1))
    tail_len = int(stage03_cfg.get("tail_len", 8))
    max_late_flip_count = int(stage03_cfg.get("max_late_flip_count", 0))
    min_abs_delta_tail_floor = float(stage03_cfg.get("min_abs_delta_tail_floor", 0.5))

    out_root = run_v2_root_for(args.run_id)
    in_path = out_root / "decision_metrics.parquet"
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    metrics = pd.read_parquet(in_path)
    if args.model_name:
        metrics = metrics[metrics["model_id"].str.contains(args.model_name, na=False, case=False)]
    if args.max_prompts > 0 and not metrics.empty:
        keep = set(metrics["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        metrics = metrics[metrics["prompt_uid"].isin(keep)]

    prompt_types_new = classify_trajectory_table(
        metrics,
        tail_len=int(tail_len),
        max_late_flip_count=int(max_late_flip_count),
        min_abs_delta_tail_floor=float(min_abs_delta_tail_floor),
    )

    out_types = out_root / "prompt_types.parquet"
    if args.resume and out_types.exists():
        old = pd.read_parquet(out_types)
        prompt_types = pd.concat([old, prompt_types_new], ignore_index=True)
        prompt_types = prompt_types.drop_duplicates(subset=["model_id", "prompt_uid"], keep="last")
    else:
        prompt_types = prompt_types_new

    counts = type_counts(prompt_types)
    per_model_type_counts: dict[str, dict[str, int]] = {}
    for model_id, sub in prompt_types.groupby("model_id", sort=True):
        vc = sub["trajectory_type"].value_counts().to_dict()
        per_model_type_counts[str(model_id)] = {str(k): int(v) for k, v in vc.items()}
    observed_models = sorted(per_model_type_counts.keys())
    if expected_model_ids:
        expected_models_present = all(mid in per_model_type_counts for mid in expected_model_ids)
    else:
        expected_models_present = len(observed_models) > 0
    model_ids_for_gating = expected_model_ids if expected_model_ids else observed_models
    required_types_per_model = bool(model_ids_for_gating) and all(
        all(str(t) in (per_model_type_counts.get(mid) or {}) for t in required_types) for mid in model_ids_for_gating
    )
    min_count_ok = bool(model_ids_for_gating) and all(
        all(int((per_model_type_counts.get(mid) or {}).get(str(t), 0)) >= int(min_count_per_type_per_model) for t in required_types)
        for mid in model_ids_for_gating
    )
    gates = {
        "expected_models_present": bool(expected_models_present),
        "required_types_per_model": bool(required_types_per_model),
        "min_count_per_type_per_model": bool(min_count_ok),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    pass_flag = len(failing_gates) == 0
    write_parquet(out_types, prompt_types)
    write_json(out_root / "type_counts.json", counts)
    write_json(
        out_root / "03_classify_trajectories.report.json",
        {
            "pass": bool(pass_flag),
            "rows": int(prompt_types.shape[0]),
            "counts": counts,
            "per_model_type_counts": per_model_type_counts,
            "expected_model_ids": expected_model_ids,
            "observed_model_ids": observed_models,
            "trajectory_stability_contract": {
                "required_types": required_types,
                "min_count_per_type_per_model": int(min_count_per_type_per_model),
                "tail_len": int(tail_len),
                "max_late_flip_count": int(max_late_flip_count),
                "min_abs_delta_tail_floor": float(min_abs_delta_tail_floor),
            },
            "gates": gates,
            "failing_gates": failing_gates,
            "out_path": str(out_types),
        },
    )
    print(str(out_types))
    return 0 if bool(pass_flag) else 2


if __name__ == "__main__":
    raise SystemExit(main())
