#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from _audit_common import default_paths, read_parquet_required, require_paths, write_json


REQUIRED_PARQUET = [
    "decision_metrics.parquet",
    "prompt_types.parquet",
    "tracing_scalars.parquet",
    "ablation_results.parquet",
    "patching_results.parquet",
    "attention_mass_by_span.parquet",
    "attention_contrib_by_span.parquet",
    "span_effects.parquet",
    "span_labels.parquet",
    "span_deletion_causal.parquet",
    "negative_controls.parquet",
]


def _finite_report(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"nan_count": 0, "inf_count": 0, "columns": {}}
    if df.empty:
        return out
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        out["columns"][str(col)] = {"nan_count": nan_count, "inf_count": inf_count}
        out["nan_count"] += nan_count
        out["inf_count"] += inf_count
    return out


def _duplicate_count(df: pd.DataFrame, key_cols: list[str]) -> int:
    if df.empty:
        return 0
    if any(c not in df.columns for c in key_cols):
        return -1
    return int(df.duplicated(subset=key_cols, keep=False).sum())


def _layer_completeness(
    df: pd.DataFrame,
    *,
    key_cols: tuple[str, str],
    layer_col: str = "layer_index",
) -> dict[str, Any]:
    if df.empty:
        return {"groups": 0, "incomplete_groups": 0, "out_of_order_groups": 0, "examples": []}
    k1, k2 = key_cols
    model_layers = (
        df.groupby(k1, sort=False)[layer_col]
        .max()
        .astype(int)
        .add(1)
        .to_dict()
    )
    incomplete = 0
    out_of_order = 0
    examples: list[dict[str, Any]] = []
    groups = 0
    for (mid, puid), g in df.groupby([k1, k2], sort=False):
        groups += 1
        layers = pd.to_numeric(g[layer_col], errors="coerce").dropna().astype(int).tolist()
        expected_n = int(model_layers.get(str(mid), 0))
        expected = list(range(expected_n))
        sorted_layers = sorted(layers)
        incomplete_flag = sorted_layers != expected
        order_flag = layers != sorted_layers
        if incomplete_flag:
            incomplete += 1
        if order_flag:
            out_of_order += 1
        if (incomplete_flag or order_flag) and len(examples) < 20:
            examples.append(
                {
                    "model_id": str(mid),
                    "prompt_uid": str(puid),
                    "observed_min": int(min(layers)) if layers else None,
                    "observed_max": int(max(layers)) if layers else None,
                    "observed_count": int(len(layers)),
                    "expected_count": int(expected_n),
                    "order_ok": not order_flag,
                }
            )
    return {
        "groups": int(groups),
        "incomplete_groups": int(incomplete),
        "out_of_order_groups": int(out_of_order),
        "examples": examples,
    }


def _script_safety_scan(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    dangerous_patterns = [
        "AutoModelForCausalLM.from_pretrained",
        "AutoTokenizer.from_pretrained",
        ".generate(",
        "torch.inference_mode(",
        "run_baseline_for_model(",
        "assert_inference_allowed(",
    ]
    safe: list[dict[str, Any]] = []
    dangerous: list[dict[str, Any]] = []
    candidates = sorted((repo_root / "scripts").rglob("*.py"))
    for path in candidates:
        rel = str(path.relative_to(repo_root))
        txt = path.read_text(encoding="utf-8")
        hits = [p for p in dangerous_patterns if p in txt]
        entry = {"path": rel, "danger_signals": hits}
        if hits:
            dangerous.append(entry)
        else:
            safe.append(entry)
    return {"safe_scripts": safe, "dangerous_scripts": dangerous}


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Run cached-artifact integrity checks (no inference).")
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--reports-dir", type=Path, default=paths.reports)
    ap.add_argument("--config", type=Path, default=paths.repo / "configs" / "experiment_v2.yaml")
    ap.add_argument("--out-json", type=Path, default=paths.audit / "artifact_integrity.json")
    args = ap.parse_args()

    require_paths([args.parquet_dir / name for name in REQUIRED_PARQUET])
    require_paths([args.config])

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    trajectory_cfg = (((cfg.get("validators") or {}).get("stage03_trajectory") or {}))
    tail_len = int(trajectory_cfg.get("tail_len", 8))
    max_late_flip_count = int(trajectory_cfg.get("max_late_flip_count", 0))
    min_abs_delta_tail_floor = float(trajectory_cfg.get("min_abs_delta_tail_floor", 0.3))
    deterministic_seed = int(((cfg.get("execution") or {}).get("deterministic_seed", 12345)))

    # Load artifacts
    frames: dict[str, pd.DataFrame] = {}
    shape_summary: dict[str, Any] = {}
    finite_summary: dict[str, Any] = {}
    for name in REQUIRED_PARQUET:
        df = read_parquet_required(args.parquet_dir / name)
        frames[name] = df
        shape_summary[name] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
        finite_summary[name] = _finite_report(df)

    decision = frames["decision_metrics.parquet"]
    prompt_types = frames["prompt_types.parquet"]
    tracing = frames["tracing_scalars.parquet"]

    duplicates = {
        "decision_metrics": {
            "keys": ["model_id", "prompt_uid", "layer_index"],
            "duplicate_rows": _duplicate_count(decision, ["model_id", "prompt_uid", "layer_index"]),
        },
        "prompt_types": {
            "keys": ["model_id", "prompt_uid"],
            "duplicate_rows": _duplicate_count(prompt_types, ["model_id", "prompt_uid"]),
        },
        "tracing_scalars": {
            "keys": ["model_id", "prompt_uid", "layer_index"],
            "duplicate_rows": _duplicate_count(tracing, ["model_id", "prompt_uid", "layer_index"]),
        },
        "attention_mass_by_span": {
            "keys": ["model_id", "prompt_uid", "layer_index", "span_label"],
            "duplicate_rows": _duplicate_count(
                frames["attention_mass_by_span.parquet"],
                ["model_id", "prompt_uid", "layer_index", "span_label"],
            ),
        },
        "attention_contrib_by_span": {
            "keys": ["model_id", "prompt_uid", "layer_index", "span_label"],
            "duplicate_rows": _duplicate_count(
                frames["attention_contrib_by_span.parquet"],
                ["model_id", "prompt_uid", "layer_index", "span_label"],
            ),
        },
        "patching_results": {
            "keys": ["model_id", "prompt_uid_fail", "prompt_uid_success", "component", "target_layers"],
            "duplicate_rows": _duplicate_count(
                frames["patching_results.parquet"],
                ["model_id", "prompt_uid_fail", "prompt_uid_success", "component", "target_layers"],
            ),
        },
        "ablation_results": {
            "keys": ["model_id", "prompt_uid", "component", "target_layers"],
            "duplicate_rows": _duplicate_count(
                frames["ablation_results.parquet"],
                ["model_id", "prompt_uid", "component", "target_layers"],
            ),
        },
        "span_effects": {
            "keys": ["model_id", "prompt_uid", "span_id"],
            "duplicate_rows": _duplicate_count(frames["span_effects.parquet"], ["model_id", "prompt_uid", "span_id"]),
        },
        "span_labels": {
            "keys": ["model_id", "prompt_uid", "span_id"],
            "duplicate_rows": _duplicate_count(frames["span_labels.parquet"], ["model_id", "prompt_uid", "span_id"]),
        },
    }

    layer_checks = {
        "decision_metrics": _layer_completeness(decision, key_cols=("model_id", "prompt_uid")),
        "tracing_scalars": _layer_completeness(tracing, key_cols=("model_id", "prompt_uid")),
    }

    # Reclassification agreement from raw decision trajectories.
    from sow.v2.trajectory_types import classify_trajectory_table  # noqa: PLC0415

    rederived = classify_trajectory_table(
        decision,
        tail_len=tail_len,
        max_late_flip_count=max_late_flip_count,
        min_abs_delta_tail_floor=min_abs_delta_tail_floor,
    )
    merged = prompt_types.merge(
        rederived[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"],
        how="outer",
        suffixes=("_stored", "_rederived"),
        indicator=True,
    )
    mismatch = merged[
        (merged["_merge"] != "both")
        | (merged["trajectory_type_stored"].astype(str) != merged["trajectory_type_rederived"].astype(str))
    ]
    reclassification = {
        "stored_rows": int(prompt_types.shape[0]),
        "rederived_rows": int(rederived.shape[0]),
        "agreement_rows": int(merged.shape[0] - mismatch.shape[0]),
        "mismatch_rows": int(mismatch.shape[0]),
        "agreement_rate": float((merged.shape[0] - mismatch.shape[0]) / max(1, merged.shape[0])),
        "mismatch_examples": mismatch.head(20).to_dict(orient="records"),
        "settings": {
            "tail_len": tail_len,
            "max_late_flip_count": max_late_flip_count,
            "min_abs_delta_tail_floor": min_abs_delta_tail_floor,
        },
    }

    # Range sanity checks.
    final_layers = decision.sort_values("layer_index").groupby(["model_id", "prompt_uid"], as_index=False).tail(1)
    final_delta = pd.to_numeric(final_layers["delta"], errors="coerce").to_numpy(dtype=np.float64)
    final_correct = final_layers["is_correct"].astype(bool).to_numpy()
    final_delta_sign_acc = float(((final_delta > 0) == final_correct).mean()) if final_delta.size else 0.0

    range_checks = {
        "decision_final_delta": {
            "mean": float(np.nanmean(final_delta)) if final_delta.size else 0.0,
            "std": float(np.nanstd(final_delta)) if final_delta.size else 0.0,
            "p05": float(np.nanpercentile(final_delta, 5)) if final_delta.size else 0.0,
            "p50": float(np.nanpercentile(final_delta, 50)) if final_delta.size else 0.0,
            "p95": float(np.nanpercentile(final_delta, 95)) if final_delta.size else 0.0,
            "sign_matches_is_correct_rate": final_delta_sign_acc,
        },
        "s_attn": {
            "mean": float(np.nanmean(pd.to_numeric(tracing["s_attn"], errors="coerce"))),
            "std": float(np.nanstd(pd.to_numeric(tracing["s_attn"], errors="coerce"))),
            "p01": float(np.nanpercentile(pd.to_numeric(tracing["s_attn"], errors="coerce"), 1)),
            "p99": float(np.nanpercentile(pd.to_numeric(tracing["s_attn"], errors="coerce"), 99)),
        },
        "s_mlp": {
            "mean": float(np.nanmean(pd.to_numeric(tracing["s_mlp"], errors="coerce"))),
            "std": float(np.nanstd(pd.to_numeric(tracing["s_mlp"], errors="coerce"))),
            "p01": float(np.nanpercentile(pd.to_numeric(tracing["s_mlp"], errors="coerce"), 1)),
            "p99": float(np.nanpercentile(pd.to_numeric(tracing["s_mlp"], errors="coerce"), 99)),
        },
    }

    sampling_checks: dict[str, Any] = {
        "deterministic_seed_from_config": deterministic_seed,
        "tracing_subset_balance_check": {},
    }
    subset_report = args.reports_dir / "06_select_tracing_subset.report.json"
    if subset_report.exists():
        rep = yaml.safe_load(subset_report.read_text(encoding="utf-8")) or {}
        models = rep.get("models") or {}
        for mid, payload in models.items():
            traj_counts = (payload or {}).get("trajectory_counts") or {}
            sampling_checks["tracing_subset_balance_check"][str(mid)] = traj_counts
    else:
        sampling_checks["tracing_subset_balance_check"] = {"warning": f"missing {subset_report}"}

    script_safety = _script_safety_scan(default_paths().repo)
    dangerous_paths = {x["path"] for x in script_safety["dangerous_scripts"]}
    script_safety["named_script_status"] = [
        {
            "path": "scripts/v2/00a_generate_baseline_outputs.py",
            "status": "dangerous" if "scripts/v2/00a_generate_baseline_outputs.py" in dangerous_paths else "safe",
        },
        {
            "path": "scripts/v2/05_span_counterfactuals.py",
            "status": "dangerous" if "scripts/v2/05_span_counterfactuals.py" in dangerous_paths else "safe",
        },
        {
            "path": "scripts/v2/07_run_tracing.py",
            "status": "dangerous" if "scripts/v2/07_run_tracing.py" in dangerous_paths else "safe",
        },
        {
            "path": "scripts/v2/08_attention_and_mlp_decomposition.py",
            "status": "dangerous" if "scripts/v2/08_attention_and_mlp_decomposition.py" in dangerous_paths else "safe",
        },
        {
            "path": "scripts/v2/regenerate_paper_figures.py",
            "status": "dangerous" if "scripts/v2/regenerate_paper_figures.py" in dangerous_paths else "safe",
        },
    ]

    pass_checks = {
        "all_required_artifacts_present": True,
        "no_nan_or_inf": all(
            (int(v.get("nan_count", 0)) == 0 and int(v.get("inf_count", 0)) == 0)
            for v in finite_summary.values()
        ),
        "no_duplicate_keys": all(
            int(payload.get("duplicate_rows", 0)) == 0
            for payload in duplicates.values()
            if int(payload.get("duplicate_rows", 0)) >= 0
        ),
        "decision_layers_complete": int(layer_checks["decision_metrics"]["incomplete_groups"]) == 0,
        "tracing_layers_complete": int(layer_checks["tracing_scalars"]["incomplete_groups"]) == 0,
        "prompt_type_reclassification_matches": int(reclassification["mismatch_rows"]) == 0,
    }
    fail_reasons = [k for k, v in pass_checks.items() if not bool(v)]

    payload = {
        "pass": len(fail_reasons) == 0,
        "failing_checks": fail_reasons,
        "shape_summary": shape_summary,
        "finite_summary": finite_summary,
        "duplicate_checks": duplicates,
        "layer_checks": layer_checks,
        "reclassification": reclassification,
        "range_checks": range_checks,
        "sampling_checks": sampling_checks,
        "script_safety_table": script_safety,
        "gates": pass_checks,
    }
    write_json(args.out_json, payload)
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
