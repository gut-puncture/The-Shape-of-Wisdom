#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load_experiment_config, resolve_models, run_v2_root_for, write_json, write_parquet
from sow.v2.causal.span_deletion import compare_evidence_vs_distractor, run_negative_controls, summarize_span_deletion_effects
from sow.v2.stats import benjamini_hochberg, bootstrap_ci, permutation_test_mean_diff


def _bootstrap_gap_ci(ev: np.ndarray, ds: np.ndarray, *, seed: int) -> dict[str, float]:
    ev_arr = np.asarray(ev, dtype=np.float64)
    ds_arr = np.asarray(ds, dtype=np.float64)
    ev_arr = ev_arr[np.isfinite(ev_arr)]
    ds_arr = ds_arr[np.isfinite(ds_arr)]
    if ev_arr.size == 0 or ds_arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(int(seed))
    draws = np.empty((2000,), dtype=np.float64)
    for i in range(draws.size):
        ev_m = float(np.mean(rng.choice(ev_arr, size=ev_arr.size, replace=True)))
        ds_m = float(np.mean(rng.choice(ds_arr, size=ds_arr.size, replace=True)))
        draws[i] = ev_m - ds_m
    return {
        "mean": float(np.mean(draws)),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
    }


def _control_value(df: pd.DataFrame, key: str) -> float:
    if df.empty:
        return 0.0
    sub = df[df["control"].astype(str) == str(key)]
    if sub.empty:
        return 0.0
    return float(sub["mean_effect_delta"].iloc[0])


def _split_array(arr: np.ndarray, *, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return x.copy(), x.copy()
    frac = min(max(float(train_fraction), 0.05), 0.95)
    n_train = int(round(float(x.size) * frac))
    n_train = max(1, min(int(x.size) - 1, n_train))
    return x[:n_train].copy(), x[n_train:].copy()


def _split_prompt_subsets(sub: pd.DataFrame, *, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if sub.empty:
        return sub.copy(), sub.copy(), "prompt_uid"
    if "prompt_uid" not in sub.columns:
        frac = min(max(float(train_fraction), 0.05), 0.95)
        n = int(sub.shape[0])
        if n <= 1:
            return sub.copy(), sub.copy(), "row_fallback"
        n_train = int(round(float(n) * frac))
        n_train = max(1, min(n - 1, n_train))
        return sub.iloc[:n_train].copy(), sub.iloc[n_train:].copy(), "row_fallback"

    prompt_uids = sorted(sub["prompt_uid"].astype(str).drop_duplicates().tolist())
    if len(prompt_uids) <= 1:
        return sub.copy(), sub.copy(), "prompt_uid"
    frac = min(max(float(train_fraction), 0.05), 0.95)
    n_train_prompts = int(round(float(len(prompt_uids)) * frac))
    n_train_prompts = max(1, min(len(prompt_uids) - 1, n_train_prompts))
    train_uids = set(prompt_uids[:n_train_prompts])
    train = sub[sub["prompt_uid"].astype(str).isin(train_uids)].copy()
    test = sub[~sub["prompt_uid"].astype(str).isin(train_uids)].copy()
    return train, test, "prompt_uid"


def main() -> int:
    ap = base_parser("V2: run span deletion causal summaries and negative controls")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]
    validators_cfg = cfg.get("validators") or {}
    stage10_cfg = validators_cfg.get("stage10") or {}
    min_evidence_rows = int(stage10_cfg.get("min_evidence_rows", 5))
    min_distractor_rows = int(stage10_cfg.get("min_distractor_rows", 5))
    alpha = float(stage10_cfg.get("alpha", 0.05))
    min_gap_ci_lo = float(stage10_cfg.get("min_gap_ci_lo", 0.0))
    min_observed_minus_shuffled = float(stage10_cfg.get("min_observed_minus_shuffled", 0.05))
    min_observed_minus_sign_flipped = float(stage10_cfg.get("min_observed_minus_sign_flipped", 0.05))
    split_train_fraction = float(stage10_cfg.get("split_train_fraction", 0.7))
    min_train_rows = int(stage10_cfg.get("min_train_rows", 0))
    min_test_rows = int(stage10_cfg.get("min_test_rows", 0))
    require_split_direction_match = bool(stage10_cfg.get("require_split_direction_match", False))

    out_root = run_v2_root_for(args.run_id)
    span_del_out = out_root / "span_deletion_causal.parquet"
    neg_out = out_root / "negative_controls.parquet"

    labels_path = out_root / "span_labels.parquet"
    if not labels_path.exists():
        raise SystemExit(f"missing span labels: {labels_path}")

    labels = pd.read_parquet(labels_path)
    if args.model_name and "model_id" in labels.columns:
        labels = labels[labels["model_id"].astype(str).str.contains(args.model_name, na=False, case=False)]
    if args.max_prompts > 0 and not labels.empty:
        keep = set(labels["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        labels = labels[labels["prompt_uid"].isin(keep)]

    span_del = summarize_span_deletion_effects(labels)
    neg = run_negative_controls(labels, seed=123)
    stats = compare_evidence_vs_distractor(labels)

    stats_per_model: dict[str, dict[str, float | int | bool | dict[str, float]]] = {}
    model_ids: list[str] = []
    p_values: list[float] = []
    if not labels.empty and "model_id" in labels.columns:
        for i, (model_id, sub) in enumerate(labels.groupby("model_id", sort=True)):
            sub_ord = sub.sort_values(["prompt_uid", "span_id"]).copy() if ("prompt_uid" in sub.columns and "span_id" in sub.columns) else sub.copy()
            train_sub, test_sub, split_unit = _split_prompt_subsets(sub_ord, train_fraction=float(split_train_fraction))
            train_prompts = set(train_sub["prompt_uid"].astype(str).drop_duplicates().tolist()) if "prompt_uid" in train_sub.columns else set()
            test_prompts = set(test_sub["prompt_uid"].astype(str).drop_duplicates().tolist()) if "prompt_uid" in test_sub.columns else set()
            overlap_prompt_count = int(len(train_prompts & test_prompts))

            ev_sub = sub_ord.loc[sub_ord["span_label"].astype(str) == "evidence", ["prompt_uid", "span_id", "effect_delta"]].copy()
            ds_sub = sub_ord.loc[sub_ord["span_label"].astype(str) == "distractor", ["prompt_uid", "span_id", "effect_delta"]].copy()
            ev = ev_sub["effect_delta"].to_numpy(dtype=np.float64)
            ds = ds_sub["effect_delta"].to_numpy(dtype=np.float64)
            ev = ev[np.isfinite(ev)]
            ds = ds[np.isfinite(ds)]

            ev_train_series = train_sub.loc[train_sub["span_label"].astype(str) == "evidence", "effect_delta"] if not train_sub.empty else pd.Series(dtype=float)
            ds_train_series = train_sub.loc[train_sub["span_label"].astype(str) == "distractor", "effect_delta"] if not train_sub.empty else pd.Series(dtype=float)
            ev_test_series = test_sub.loc[test_sub["span_label"].astype(str) == "evidence", "effect_delta"] if not test_sub.empty else pd.Series(dtype=float)
            ds_test_series = test_sub.loc[test_sub["span_label"].astype(str) == "distractor", "effect_delta"] if not test_sub.empty else pd.Series(dtype=float)

            ev_train = ev_train_series.to_numpy(dtype=np.float64)
            ds_train = ds_train_series.to_numpy(dtype=np.float64)
            ev_test = ev_test_series.to_numpy(dtype=np.float64)
            ds_test = ds_test_series.to_numpy(dtype=np.float64)
            ev_train = ev_train[np.isfinite(ev_train)]
            ds_train = ds_train[np.isfinite(ds_train)]
            ev_test = ev_test[np.isfinite(ev_test)]
            ds_test = ds_test[np.isfinite(ds_test)]

            all_prompts = set(sub_ord["prompt_uid"].astype(str).drop_duplicates().tolist()) if "prompt_uid" in sub_ord.columns else set()
            # Fallback only when a single prompt makes prompt-level split impossible.
            if split_unit == "prompt_uid" and int(len(all_prompts)) <= 1:
                ev_train, ev_test = _split_array(ev, train_fraction=float(split_train_fraction))
                ds_train, ds_test = _split_array(ds, train_fraction=float(split_train_fraction))
                split_unit = "row_fallback"
                overlap_prompt_count = 0

            split_train_rows = int(min(ev_train.size, ds_train.size))
            split_test_rows = int(min(ev_test.size, ds_test.size))
            train_gap = float(np.mean(ev_train) - np.mean(ds_train)) if (ev_train.size > 0 and ds_train.size > 0) else 0.0
            test_gap = float(np.mean(ev_test) - np.mean(ds_test)) if (ev_test.size > 0 and ds_test.size > 0) else 0.0
            split_direction_consistent = bool(np.sign(train_gap) == np.sign(test_gap) and np.sign(train_gap) != 0.0)
            model_neg = run_negative_controls(sub, seed=123 + i)
            observed = _control_value(model_neg, "observed")
            shuffled = _control_value(model_neg, "shuffled")
            sign_flipped = _control_value(model_neg, "sign_flipped")
            p_val = float(permutation_test_mean_diff(ev.tolist(), ds.tolist(), n_permutations=5000, seed=42 + i))
            p_values.append(p_val)
            model_ids.append(str(model_id))
            gap_ci = _bootstrap_gap_ci(ev, ds, seed=99 + i)
            stats_per_model[str(model_id)] = {
                "evidence_n": int(ev.size),
                "distractor_n": int(ds.size),
                "evidence_ci": bootstrap_ci(ev.tolist(), seed=202 + i),
                "distractor_ci": bootstrap_ci(ds.tolist(), seed=303 + i),
                "gap_ci": gap_ci,
                "p_value": p_val,
                "observed": float(observed),
                "shuffled": float(shuffled),
                "sign_flipped": float(sign_flipped),
                "observed_minus_shuffled": float(observed - shuffled),
                "observed_minus_sign_flipped": float(observed - sign_flipped),
                "split": {
                    "split_unit": str(split_unit),
                    "train_rows": int(split_train_rows),
                    "test_rows": int(split_test_rows),
                    "train_gap": float(train_gap),
                    "test_gap": float(test_gap),
                    "direction_consistent": bool(split_direction_consistent),
                    "train_prompt_count": int(len(train_prompts)),
                    "test_prompt_count": int(len(test_prompts)),
                    "overlap_prompt_count": int(overlap_prompt_count),
                },
            }

    bh_flags = benjamini_hochberg(p_values, alpha=float(alpha))
    for idx, model_id in enumerate(model_ids):
        ms = stats_per_model.get(model_id, {})
        ms["bh_significant"] = bool(bh_flags[idx]) if idx < len(bh_flags) else False
        stats_per_model[model_id] = ms

    has_models = len(stats_per_model) > 0
    observed_model_ids = sorted(stats_per_model.keys())
    if expected_model_ids:
        expected_models_present = all(mid in stats_per_model for mid in expected_model_ids)
    else:
        expected_models_present = bool(has_models)
    gates = {
        "has_models": bool(has_models),
        "expected_models_present": bool(expected_models_present),
        "min_evidence_rows": bool(
            has_models and all(int(ms.get("evidence_n", 0)) >= int(min_evidence_rows) for ms in stats_per_model.values())
        ),
        "min_distractor_rows": bool(
            has_models and all(int(ms.get("distractor_n", 0)) >= int(min_distractor_rows) for ms in stats_per_model.values())
        ),
        "gap_ci_lo": bool(
            has_models and all(float((ms.get("gap_ci") or {}).get("lo", 0.0)) >= float(min_gap_ci_lo) for ms in stats_per_model.values())
        ),
        "bh_significant": bool(
            has_models and all(bool(ms.get("bh_significant", False)) for ms in stats_per_model.values())
        ),
        "observed_minus_shuffled": bool(
            has_models
            and all(
                float(ms.get("observed_minus_shuffled", 0.0)) >= float(min_observed_minus_shuffled) for ms in stats_per_model.values()
            )
        ),
        "observed_minus_sign_flipped": bool(
            has_models
            and all(
                float(ms.get("observed_minus_sign_flipped", 0.0)) >= float(min_observed_minus_sign_flipped)
                for ms in stats_per_model.values()
            )
        ),
        "split_train_rows_min": bool(
            has_models
            and all(int((ms.get("split") or {}).get("train_rows", 0)) >= int(min_train_rows) for ms in stats_per_model.values())
        ),
        "split_test_rows_min": bool(
            has_models
            and all(int((ms.get("split") or {}).get("test_rows", 0)) >= int(min_test_rows) for ms in stats_per_model.values())
        ),
        "split_direction_consistent": bool(
            (not require_split_direction_match)
            or (
                has_models
                and all(bool((ms.get("split") or {}).get("direction_consistent", False)) for ms in stats_per_model.values())
            )
        ),
        "split_prompt_disjoint": bool(
            has_models
            and all(int((ms.get("split") or {}).get("overlap_prompt_count", 0)) == 0 for ms in stats_per_model.values())
        ),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    pass_flag = len(failing_gates) == 0

    write_parquet(span_del_out, span_del)
    write_parquet(neg_out, neg)
    write_json(
        out_root / "10_causal_validation_tools.report.json",
        {
            "pass": bool(pass_flag),
            "span_deletion_rows": int(span_del.shape[0]),
            "negative_control_rows": int(neg.shape[0]),
            "evidence_vs_distractor": stats,
            "stats_per_model": stats_per_model,
            "multiple_comparison": {
                "alpha": float(alpha),
                "expected_model_ids": expected_model_ids,
                "observed_model_ids": observed_model_ids,
                "model_ids": model_ids,
                "p_values": p_values,
                "bh_significant": bh_flags,
            },
            "split_contract": {
                "split_train_fraction": float(split_train_fraction),
                "min_train_rows": int(min_train_rows),
                "min_test_rows": int(min_test_rows),
                "require_split_direction_match": bool(require_split_direction_match),
            },
            "gates": gates,
            "failing_gates": failing_gates,
        },
    )
    print(str(out_root / "span_deletion_causal.parquet"))
    return 0 if bool(pass_flag) else 2


if __name__ == "__main__":
    raise SystemExit(main())
