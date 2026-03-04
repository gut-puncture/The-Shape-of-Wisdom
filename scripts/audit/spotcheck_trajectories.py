#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _audit_common import default_paths, ensure_dir, read_parquet_required, write_csv, write_json


TARGET_CATEGORIES = ["stable_correct", "stable_wrong", "unstable_wrong"]


def _sample_prompts(
    prompt_types: pd.DataFrame,
    tracing: pd.DataFrame,
    *,
    total: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    available = prompt_types.merge(
        tracing[["model_id", "prompt_uid"]].drop_duplicates(),
        on=["model_id", "prompt_uid"],
        how="inner",
    )
    parts: list[pd.DataFrame] = []
    targets = {
        "stable_correct": max(1, int(round(total * 0.35))),
        "stable_wrong": max(1, int(round(total * 0.35))),
        "unstable_wrong": max(1, int(total) - 2 * max(1, int(round(total * 0.35)))),
    }
    for cat in TARGET_CATEGORIES:
        sub = available[available["trajectory_type"].astype(str) == cat].copy()
        if sub.empty:
            continue
        k = min(int(targets[cat]), int(sub.shape[0]))
        idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        parts.append(sub.loc[idx])
    sampled = pd.concat(parts, ignore_index=True) if parts else available.head(0).copy()
    if sampled.shape[0] < total:
        remaining = available[~available.set_index(["model_id", "prompt_uid"]).index.isin(sampled.set_index(["model_id", "prompt_uid"]).index)]
        need = min(int(total - sampled.shape[0]), int(remaining.shape[0]))
        if need > 0:
            idx = rng.choice(remaining.index.to_numpy(), size=need, replace=False)
            sampled = pd.concat([sampled, remaining.loc[idx]], ignore_index=True)
    return sampled.sort_values(["trajectory_type", "model_id", "prompt_uid"]).reset_index(drop=True)


def _safe_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Spot-check ~20 trajectories across categories from cached artifacts.")
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--out-dir", type=Path, default=paths.audit)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    decision = read_parquet_required(args.parquet_dir / "decision_metrics.parquet")
    tracing = read_parquet_required(args.parquet_dir / "tracing_scalars.parquet")
    prompt_types = read_parquet_required(args.parquet_dir / "prompt_types.parquet")

    sampled = _sample_prompts(prompt_types, tracing, total=int(args.n_prompts), seed=int(args.seed))
    if sampled.empty:
        raise SystemExit("no sampled prompts available in tracing subset")

    key = sampled[["model_id", "prompt_uid", "trajectory_type", "is_correct"]].drop_duplicates()
    t_sub = tracing.merge(key, on=["model_id", "prompt_uid"], how="inner")
    d_sub = decision.merge(
        key[["model_id", "prompt_uid"]],
        on=["model_id", "prompt_uid"],
        how="inner",
    )[["model_id", "prompt_uid", "layer_index", "competitor", "delta", "drift"]].rename(
        columns={"competitor": "competitor_bucket", "delta": "delta_bucket", "drift": "drift_bucket"}
    )
    merged = t_sub.merge(
        d_sub,
        on=["model_id", "prompt_uid", "layer_index"],
        how="left",
    )
    merged = merged.sort_values(["trajectory_type", "model_id", "prompt_uid", "layer_index"]).reset_index(drop=True)

    merged["delta_next"] = (
        merged.groupby(["model_id", "prompt_uid"], sort=False)["delta"]
        .shift(-1)
        .astype(float)
    )
    merged["delta_step"] = merged["delta_next"] - merged["delta"]
    merged["drift_vs_delta_residual"] = merged["delta_step"] - merged["drift"]
    merged["decomp_residual"] = merged["drift"] - (merged["s_attn"] + merged["s_mlp"])
    merged["competitor_match_bucket_vs_tracing"] = (
        merged["competitor_key"].astype(str) == merged["competitor_bucket"].astype(str)
    )

    cols = [
        "model_id",
        "prompt_uid",
        "trajectory_type",
        "is_correct",
        "layer_index",
        "correct_key",
        "competitor_key",
        "competitor_bucket",
        "competitor_match_bucket_vs_tracing",
        "delta",
        "delta_bucket",
        "drift",
        "drift_bucket",
        "s_attn",
        "s_mlp",
        "delta_step",
        "drift_vs_delta_residual",
        "decomp_residual",
    ]
    out_csv = args.out_dir / "trajectory_spotcheck_layers.csv"
    write_csv(out_csv, merged[cols])

    drift_res = _safe_float_series(merged["drift_vs_delta_residual"])
    decomp_res = _safe_float_series(merged["decomp_residual"])
    summary = {
        "sampled_prompt_count": int(sampled.shape[0]),
        "sampled_by_category": (
            sampled["trajectory_type"].value_counts().sort_index().to_dict()
            if "trajectory_type" in sampled.columns
            else {}
        ),
        "sampled_examples": sampled.head(20).to_dict(orient="records"),
        "drift_identity_check": {
            "mean_abs_delta_step_minus_drift": float(np.nanmean(np.abs(drift_res))),
            "p95_abs_delta_step_minus_drift": float(np.nanpercentile(np.abs(drift_res), 95)),
            "max_abs_delta_step_minus_drift": float(np.nanmax(np.abs(drift_res))),
        },
        "decomposition_accounting_check": {
            "mean_abs_drift_minus_sattn_smlp": float(np.nanmean(np.abs(decomp_res))),
            "p95_abs_drift_minus_sattn_smlp": float(np.nanpercentile(np.abs(decomp_res), 95)),
            "max_abs_drift_minus_sattn_smlp": float(np.nanmax(np.abs(decomp_res))),
        },
        "competitor_match_rate_bucket_vs_tracing": float(
            merged["competitor_match_bucket_vs_tracing"].astype(float).mean()
        ),
        "out_csv": str(out_csv),
    }
    out_json = args.out_dir / "trajectory_spotcheck_summary.json"
    write_json(out_json, summary)
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

