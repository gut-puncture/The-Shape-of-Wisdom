#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import base_parser, load_experiment_config, run_v2_root_for, write_json, write_parquet
from sow.v2.causal.ablations import run_component_ablation
from sow.v2.causal.patching import run_activation_patching


def main() -> int:
    ap = base_parser("V2: run causal ablations and activation patching")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    causal_cfg = cfg.get("causal") or {}
    target_layers = [int(x) for x in (causal_cfg.get("ablation_target_layers") or list(range(20, 28)))]

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
        tracing = tracing[tracing["model_id"].str.contains(args.model_name, na=False)]
        ptypes = ptypes[ptypes["model_id"].str.contains(args.model_name, na=False)]

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

    write_json(
        out_root / "09_causal_tests.report.json",
        {
            "pass": True,
            "ablation_rows": int(ablation.shape[0]),
            "patching_rows": int(patching.shape[0]),
            "target_layers": target_layers,
        },
    )
    print(str(out_root / "ablation_results.parquet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
