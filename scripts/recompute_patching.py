#!/usr/bin/env python3
"""Recompute patching_results.parquet from stored tracing_scalars.parquet.

This is a pure numpy computation from stored artifacts — no model inference.
Reads tracing_scalars.parquet and prompt_types.parquet, runs the fixed
run_activation_patching() function, and writes patching_results.parquet.

Usage:
    python scripts/recompute_patching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from sow.v2.causal.patching import run_activation_patching

PARQUET_DIR = REPO_ROOT / "results" / "parquet"
TARGET_LAYERS = list(range(20, 28))


def main() -> int:
    tracing_path = PARQUET_DIR / "tracing_scalars.parquet"
    types_path = PARQUET_DIR / "prompt_types.parquet"

    if not tracing_path.exists():
        print(f"ERROR: {tracing_path} not found", file=sys.stderr)
        return 1
    if not types_path.exists():
        print(f"ERROR: {types_path} not found", file=sys.stderr)
        return 1

    print(f"Loading {tracing_path} ...")
    tracing = pd.read_parquet(tracing_path)
    ptypes = pd.read_parquet(types_path)

    merged = tracing.merge(
        ptypes[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"],
        how="left",
    )
    print(f"  Merged rows: {len(merged)}")

    failing = merged[merged["trajectory_type"].isin(["unstable_wrong", "stable_wrong"])]
    success = merged[merged["trajectory_type"].isin(["stable_correct"])]
    print(f"  Failing trajectories: {failing['prompt_uid'].nunique()} prompts")
    print(f"  Success trajectories: {success['prompt_uid'].nunique()} prompts")

    print("Running attention patching ...")
    patch_attn = run_activation_patching(
        failing, success, component="attention", target_layers=TARGET_LAYERS
    )
    print(f"  Attention patching rows: {len(patch_attn)}")

    print("Running MLP patching ...")
    patch_mlp = run_activation_patching(
        failing, success, component="mlp", target_layers=TARGET_LAYERS
    )
    print(f"  MLP patching rows: {len(patch_mlp)}")

    patching = pd.concat([patch_attn, patch_mlp], ignore_index=True)
    out_path = PARQUET_DIR / "patching_results.parquet"
    patching.to_parquet(out_path, index=False)
    print(f"\nWrote {len(patching)} rows to {out_path}")

    # Quick summary
    for comp in ["attention", "mlp"]:
        vals = patching[patching["component"] == comp]["delta_shift"]
        frac_pos = float((vals > 0).mean())
        mean_shift = float(vals.mean())
        print(f"  {comp}: mean_shift={mean_shift:.3f}, frac_positive={frac_pos:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
