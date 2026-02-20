#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load_experiment_config, resolve_models, run_v2_root_for, write_json, write_parquet
from sow.v2.causal.ablations import run_component_ablation
from sow.v2.causal.patching import run_activation_patching


def main() -> int:
    ap = base_parser("V2: run causal ablations and activation patching")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]
    causal_cfg = cfg.get("causal") or {}
    validators_cfg = cfg.get("validators") or {}
    stage09_cfg = validators_cfg.get("stage09") or {}
    target_layers = [int(x) for x in (causal_cfg.get("ablation_target_layers") or list(range(20, 28)))]
    min_ablation_rows = int(stage09_cfg.get("min_ablation_rows", 1))
    min_patching_rows = int(stage09_cfg.get("min_patching_rows", 1))

    out_root = run_v2_root_for(args.run_id)
    ablation_out = out_root / "ablation_results.parquet"
    patching_out = out_root / "patching_results.parquet"

    tracing_path = out_root / "tracing_scalars.parquet"
    types_path = out_root / "prompt_types.parquet"
    if not tracing_path.exists() or not types_path.exists():
        raise SystemExit("missing tracing_scalars.parquet or prompt_types.parquet")

    tracing = pd.read_parquet(tracing_path)
    ptypes = pd.read_parquet(types_path)

    if args.model_name:
        tracing = tracing[tracing["model_id"].str.contains(args.model_name, na=False, case=False)]
        ptypes = ptypes[ptypes["model_id"].str.contains(args.model_name, na=False, case=False)]

    merged = tracing.merge(ptypes[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")
    if args.max_prompts > 0 and not merged.empty:
        keep = set(merged["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        merged = merged[merged["prompt_uid"].isin(keep)]

    abl_attn = run_component_ablation(merged, component="attention", target_layers=target_layers)
    abl_mlp = run_component_ablation(merged, component="mlp", target_layers=target_layers)
    ablation = pd.concat([abl_attn, abl_mlp], ignore_index=True)

    failing = merged[merged["trajectory_type"].isin(["unstable_wrong", "stable_wrong"])]
    success = merged[merged["trajectory_type"].isin(["stable_correct"]) ]
    patch_attn = run_activation_patching(failing, success, component="attention", target_layers=target_layers)
    patch_mlp = run_activation_patching(failing, success, component="mlp", target_layers=target_layers)
    patching = pd.concat([patch_attn, patch_mlp], ignore_index=True)

    write_parquet(ablation_out, ablation)
    write_parquet(patching_out, patching)

    ablation_components = set(ablation["component"].astype(str).tolist()) if (not ablation.empty and "component" in ablation.columns) else set()
    patching_components = set(patching["component"].astype(str).tolist()) if (not patching.empty and "component" in patching.columns) else set()
    ablation_model_ids = set(ablation["model_id"].astype(str).tolist()) if (not ablation.empty and "model_id" in ablation.columns) else set()
    patching_model_ids = set(patching["model_id"].astype(str).tolist()) if (not patching.empty and "model_id" in patching.columns) else set()
    if expected_model_ids:
        ablation_expected_models_present = all(mid in ablation_model_ids for mid in expected_model_ids)
        patching_expected_models_present = all(mid in patching_model_ids for mid in expected_model_ids)
    else:
        ablation_expected_models_present = len(ablation_model_ids) > 0
        patching_expected_models_present = len(patching_model_ids) > 0
    component_coverage = {
        "ablation": {
            "attention": "attention" in ablation_components,
            "mlp": "mlp" in ablation_components,
        },
        "patching": {
            "attention": "attention" in patching_components,
            "mlp": "mlp" in patching_components,
        },
    }
    model_coverage = {
        "expected_models": expected_model_ids,
        "ablation_present": sorted(ablation_model_ids),
        "patching_present": sorted(patching_model_ids),
    }
    gates = {
        "min_ablation_rows": int(ablation.shape[0]) >= int(min_ablation_rows),
        "min_patching_rows": int(patching.shape[0]) >= int(min_patching_rows),
        "ablation_expected_models_present": bool(ablation_expected_models_present),
        "patching_expected_models_present": bool(patching_expected_models_present),
        "ablation_components_complete": all(component_coverage["ablation"].values()),
        "patching_components_complete": all(component_coverage["patching"].values()),
        "ablation_delta_shift_finite": bool(
            (not ablation.empty)
            and ("delta_shift" in ablation.columns)
            and np.isfinite(ablation["delta_shift"].to_numpy(dtype=float)).all()
        ),
        "patching_delta_shift_finite": bool(
            (not patching.empty)
            and ("delta_shift" in patching.columns)
            and np.isfinite(patching["delta_shift"].to_numpy(dtype=float)).all()
        ),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    pass_flag = len(failing_gates) == 0

    write_json(
        out_root / "09_causal_tests.report.json",
        {
            "pass": bool(pass_flag),
            "ablation_rows": int(ablation.shape[0]),
            "patching_rows": int(patching.shape[0]),
            "target_layers": target_layers,
            "component_coverage": component_coverage,
            "model_coverage": model_coverage,
            "gates": gates,
            "failing_gates": failing_gates,
        },
    )
    print(str(out_root / "ablation_results.parquet"))
    return 0 if bool(pass_flag) else 2


if __name__ == "__main__":
    raise SystemExit(main())
