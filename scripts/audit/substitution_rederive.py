#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from _audit_common import default_paths, ensure_dir, read_parquet_required, write_csv, write_json


@dataclass
class PromptTrace:
    model_id: str
    prompt_uid: str
    trajectory_type: str
    example_id: str
    coarse_domain: str
    prompt_len: float
    final_entropy: float
    layers: np.ndarray
    delta: np.ndarray
    drift: np.ndarray
    s_attn: np.ndarray
    s_mlp: np.ndarray


def _load_prompt_length_by_uid(spans_jsonl: Path) -> dict[str, float]:
    if not spans_jsonl.exists():
        return {}
    out: dict[str, float] = {}
    with spans_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            uid = str(rec.get("prompt_uid") or "")
            if not uid:
                continue
            end_char = rec.get("end_char")
            try:
                val = float(end_char)
            except Exception:
                continue
            out[uid] = max(float(out.get(uid, 0.0)), float(val))
    return out


def _build_prompt_traces(
    tracing: pd.DataFrame,
    prompt_types: pd.DataFrame,
    decision: pd.DataFrame,
    *,
    spans_jsonl: Path,
) -> tuple[dict[tuple[str, str], PromptTrace], dict[str, list[str]]]:
    ptype = prompt_types[["model_id", "prompt_uid", "trajectory_type"]].drop_duplicates()
    meta = (
        decision.sort_values("layer_index")
        .groupby(["model_id", "prompt_uid"], as_index=False)
        .agg(
            example_id=("example_id", "first"),
            coarse_domain=("coarse_domain", "first"),
            final_entropy=("entropy", "last"),
        )
    )
    prompt_len_by_uid = _load_prompt_length_by_uid(spans_jsonl)

    merged = tracing.merge(ptype, on=["model_id", "prompt_uid"], how="left").merge(
        meta,
        on=["model_id", "prompt_uid"],
        how="left",
    )
    merged = merged.reset_index(drop=True)
    merged["_row_idx"] = np.arange(merged.shape[0], dtype=np.int64)

    prompt_order_by_model: dict[str, list[str]] = {}
    first_occurrence = merged.drop_duplicates(subset=["model_id", "prompt_uid"], keep="first")
    for _, row in first_occurrence.iterrows():
        mid = str(row["model_id"])
        uid = str(row["prompt_uid"])
        if mid not in prompt_order_by_model:
            prompt_order_by_model[mid] = []
        prompt_order_by_model[mid].append(uid)

    traces: dict[tuple[str, str], PromptTrace] = {}
    for (mid, uid), g in merged.groupby(["model_id", "prompt_uid"], sort=False):
        g = g.sort_values("layer_index", kind="stable")
        model_id = str(mid)
        prompt_uid = str(uid)
        traces[(model_id, prompt_uid)] = PromptTrace(
            model_id=model_id,
            prompt_uid=prompt_uid,
            trajectory_type=str(g["trajectory_type"].iloc[0]),
            example_id=str(g["example_id"].iloc[0]),
            coarse_domain=str(g["coarse_domain"].iloc[0]),
            prompt_len=float(prompt_len_by_uid.get(prompt_uid, np.nan)),
            final_entropy=float(pd.to_numeric(g["final_entropy"], errors="coerce").iloc[0]),
            layers=pd.to_numeric(g["layer_index"], errors="coerce").to_numpy(dtype=np.int64),
            delta=pd.to_numeric(g["delta"], errors="coerce").to_numpy(dtype=np.float64),
            drift=pd.to_numeric(g["drift"], errors="coerce").to_numpy(dtype=np.float64),
            s_attn=pd.to_numeric(g["s_attn"], errors="coerce").to_numpy(dtype=np.float64),
            s_mlp=pd.to_numeric(g["s_mlp"], errors="coerce").to_numpy(dtype=np.float64),
        )
    return traces, prompt_order_by_model


def _pair_id(
    *,
    model_id: str,
    source_uid: str,
    target_uid: str,
    pairing_mode: str,
    normalization_mode: str,
    layer_range_mode: str,
    failing_set_mode: str,
) -> str:
    raw = "|".join(
        [
            str(model_id),
            str(source_uid),
            str(target_uid),
            str(pairing_mode),
            str(normalization_mode),
            str(layer_range_mode),
            str(failing_set_mode),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _layer_set_for_mode(*, mode: str, n_layers: int, paper_default: list[int]) -> list[int]:
    if int(n_layers) <= 0:
        return []
    if mode == "paper-default":
        xs = [int(l) for l in paper_default if 0 <= int(l) < int(n_layers)]
        return xs if xs else list(range(n_layers))
    out: list[int] = []
    for li in range(int(n_layers)):
        depth = float(li) / float(max(1, n_layers - 1))
        if mode == "early" and depth < (1.0 / 3.0):
            out.append(li)
        elif mode == "mid" and (1.0 / 3.0) <= depth < (2.0 / 3.0):
            out.append(li)
        elif mode == "late" and depth >= (2.0 / 3.0):
            out.append(li)
    return out if out else list(range(n_layers))


def _build_norm_tables(
    tracing: pd.DataFrame,
) -> tuple[dict[tuple[str, int], tuple[float, float]], dict[tuple[str, int], tuple[float, float]]]:
    attn: dict[tuple[str, int], tuple[float, float]] = {}
    mlp: dict[tuple[str, int], tuple[float, float]] = {}
    g = tracing.groupby(["model_id", "layer_index"], sort=False)
    for (mid, li), sub in g:
        a = pd.to_numeric(sub["s_attn"], errors="coerce").to_numpy(dtype=np.float64)
        m = pd.to_numeric(sub["s_mlp"], errors="coerce").to_numpy(dtype=np.float64)
        a_mean, a_std = float(np.nanmean(a)), float(np.nanstd(a))
        m_mean, m_std = float(np.nanmean(m)), float(np.nanstd(m))
        attn[(str(mid), int(li))] = (a_mean, a_std)
        mlp[(str(mid), int(li))] = (m_mean, m_std)
    return attn, mlp


def _normalize_component(
    values: np.ndarray,
    *,
    model_id: str,
    layers: np.ndarray,
    mode: str,
    stats: dict[tuple[str, int], tuple[float, float]],
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if mode == "raw":
        return arr
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.size):
        key = (str(model_id), int(layers[i]))
        mean, std = stats.get(key, (0.0, 0.0))
        if std <= 1e-12:
            out[i] = 0.0
            continue
        if mode == "zscore_per_model_layer":
            out[i] = (arr[i] - mean) / std
        elif mode == "std_per_model_layer":
            out[i] = arr[i] / std
        else:
            out[i] = arr[i]
    return out


def _choose_source_for_target(
    *,
    mode: str,
    model_id: str,
    target: PromptTrace,
    success_pool: list[PromptTrace],
    rng: np.random.Generator,
) -> PromptTrace | None:
    if not success_pool:
        return None
    if mode == "one_to_one_seeded":
        return success_pool[int(rng.integers(0, len(success_pool)))]
    if mode == "domain_matched_seeded":
        same = [p for p in success_pool if p.coarse_domain == target.coarse_domain]
        pool = same if same else success_pool
        return pool[int(rng.integers(0, len(pool)))]
    if mode == "domain_length_matched_seeded":
        # Prefer domain matches, then nearest prompt length (fallback to final entropy).
        same = [p for p in success_pool if p.coarse_domain == target.coarse_domain]
        pool = same if same else success_pool
        if not pool:
            return None
        target_len = target.prompt_len
        target_ent = target.final_entropy

        def _dist(p: PromptTrace) -> tuple[float, float]:
            len_dist = (
                abs(float(p.prompt_len) - float(target_len))
                if np.isfinite(p.prompt_len) and np.isfinite(target_len)
                else np.inf
            )
            ent_dist = (
                abs(float(p.final_entropy) - float(target_ent))
                if np.isfinite(p.final_entropy) and np.isfinite(target_ent)
                else np.inf
            )
            return (len_dist, ent_dist)

        dists = np.asarray([_dist(p) for p in pool], dtype=np.float64)
        best = np.nanmin(dists[:, 0])
        if np.isfinite(best):
            idxs = np.where(np.isclose(dists[:, 0], best, atol=1e-9))[0]
        else:
            best_e = np.nanmin(dists[:, 1])
            if np.isfinite(best_e):
                idxs = np.where(np.isclose(dists[:, 1], best_e, atol=1e-9))[0]
            else:
                idxs = np.arange(len(pool))
        return pool[int(rng.choice(idxs))]
    return success_pool[int(rng.integers(0, len(success_pool)))]


def _build_pairs(
    *,
    pairing_mode: str,
    model_id: str,
    success_pool: list[PromptTrace],
    fail_pool: list[PromptTrace],
    prompt_order: list[str],
    seed: int,
) -> list[tuple[PromptTrace, PromptTrace]]:
    if not success_pool or not fail_pool:
        return []
    rng = np.random.default_rng(int(seed))
    if pairing_mode == "legacy_first_per_model":
        order_rank = {uid: i for i, uid in enumerate(prompt_order)}
        success_pool_sorted = sorted(success_pool, key=lambda p: order_rank.get(p.prompt_uid, 10**9))
        src = success_pool_sorted[0]
        return [(src, t) for t in fail_pool]
    if pairing_mode == "all_pairs_within_model":
        return [(s, t) for t in fail_pool for s in success_pool]
    out: list[tuple[PromptTrace, PromptTrace]] = []
    for t in fail_pool:
        src = _choose_source_for_target(
            mode=pairing_mode,
            model_id=model_id,
            target=t,
            success_pool=success_pool,
            rng=rng,
        )
        if src is not None:
            out.append((src, t))
    return out


def _compute_pair_row(
    *,
    source: PromptTrace,
    target: PromptTrace,
    component: str,
    component_target: np.ndarray,
    component_source: np.ndarray,
    layers_substituted: list[int],
    tail_len: int,
    pairing_mode: str,
    normalization_mode: str,
    layer_range_mode: str,
    failing_set_mode: str,
    seed: int,
) -> dict[str, Any] | None:
    fail_delta = target.delta
    fail_drift = target.drift
    fail_li = target.layers
    n = min(fail_delta.size, fail_drift.size, fail_li.size, component_target.size, component_source.size)
    if n <= 0:
        return None
    target_set = set(int(x) for x in layers_substituted)
    patched = fail_drift[:n].copy()
    for i in range(n):
        if int(fail_li[i]) in target_set:
            patched[i] = fail_drift[i] - component_target[i] + component_source[i]

    trace = np.zeros((n + 1,), dtype=np.float64)
    trace[0] = float(fail_delta[0])
    for i in range(n):
        trace[i + 1] = trace[i] + patched[i]

    base_final = float(fail_delta[n - 1])
    patched_final = float(trace[-1])
    shift = float(patched_final - base_final)
    tail_entry = max(0, int(n - int(tail_len)))
    tail_entry_shift = float(trace[tail_entry] - fail_delta[tail_entry])
    mean_shift = float(np.nanmean(trace[:n] - fail_delta[:n]))

    pid = _pair_id(
        model_id=target.model_id,
        source_uid=source.prompt_uid,
        target_uid=target.prompt_uid,
        pairing_mode=pairing_mode,
        normalization_mode=normalization_mode,
        layer_range_mode=layer_range_mode,
        failing_set_mode=failing_set_mode,
    )
    return {
        "pair_id": pid,
        "model_id": target.model_id,
        "source_prompt_uid": source.prompt_uid,
        "target_prompt_uid": target.prompt_uid,
        "source_example_id": source.example_id,
        "target_example_id": target.example_id,
        "category_source": source.trajectory_type,
        "category_target": target.trajectory_type,
        "layers_substituted": ",".join(str(int(x)) for x in layers_substituted),
        "component": component,
        "delta_final_base": base_final,
        "delta_final_patched": patched_final,
        "delta_shift": shift,
        "delta_tail_entry_shift": tail_entry_shift,
        "delta_mean_shift": mean_shift,
        "pairing_mode": pairing_mode,
        "normalization_mode": normalization_mode,
        "layer_range_mode": layer_range_mode,
        "failing_set_mode": failing_set_mode,
        "seed": int(seed),
    }


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Independent substitution re-derivation from cached tracing artifacts.")
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--config", type=Path, default=paths.repo / "configs" / "experiment_v2.yaml")
    ap.add_argument("--spans-jsonl", type=Path, default=paths.results / "spans.jsonl")
    ap.add_argument("--out-csv", type=Path, default=paths.audit / "substitution_pairs_vnext.csv")
    ap.add_argument("--out-json", type=Path, default=paths.audit / "substitution_rederive_diagnostics.json")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ensure_dir(args.out_csv.parent)
    tracing = read_parquet_required(args.parquet_dir / "tracing_scalars.parquet")
    prompt_types = read_parquet_required(args.parquet_dir / "prompt_types.parquet")
    decision = read_parquet_required(args.parquet_dir / "decision_metrics.parquet")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    causal_cfg = cfg.get("causal") or {}
    validators_cfg = cfg.get("validators") or {}
    stage03_cfg = validators_cfg.get("stage03_trajectory") or {}
    tail_len = int(stage03_cfg.get("tail_len", 8))
    paper_layers = [int(x) for x in (causal_cfg.get("patching_target_layers") or list(range(20, 28)))]

    traces, prompt_order_by_model = _build_prompt_traces(
        tracing=tracing,
        prompt_types=prompt_types,
        decision=decision,
        spans_jsonl=args.spans_jsonl,
    )
    if not traces:
        raise SystemExit("no trace trajectories available")

    attn_norm_stats, mlp_norm_stats = _build_norm_tables(tracing)
    pairing_modes = [
        "legacy_first_per_model",
        "all_pairs_within_model",
        "one_to_one_seeded",
        "domain_matched_seeded",
        "domain_length_matched_seeded",
    ]
    normalization_modes = ["raw", "zscore_per_model_layer", "std_per_model_layer"]
    layer_range_modes = ["early", "mid", "late", "paper-default"]
    failing_set_modes = ["stable_wrong_only", "stable_wrong_plus_unstable_wrong"]
    baseline_setting = {
        "pairing_mode": "all_pairs_within_model",
        "normalization_mode": "raw",
        "layer_range_mode": "paper-default",
        "failing_set_mode": "stable_wrong_plus_unstable_wrong",
    }
    settings: list[dict[str, str]] = []
    # Baseline.
    settings.append(dict(baseline_setting))
    # Pairing sensitivity.
    for pairing_mode in pairing_modes:
        settings.append({**baseline_setting, "pairing_mode": pairing_mode})
    # Layer-range sensitivity.
    for layer_range_mode in layer_range_modes:
        settings.append({**baseline_setting, "layer_range_mode": layer_range_mode})
    # Normalization sensitivity.
    for normalization_mode in normalization_modes:
        settings.append({**baseline_setting, "normalization_mode": normalization_mode})
    # Failing-set sensitivity.
    for failing_set_mode in failing_set_modes:
        settings.append({**baseline_setting, "failing_set_mode": failing_set_mode})
    # Deduplicate while preserving order.
    _seen = set()
    unique_settings: list[dict[str, str]] = []
    for s in settings:
        key = (
            s["pairing_mode"],
            s["normalization_mode"],
            s["layer_range_mode"],
            s["failing_set_mode"],
        )
        if key in _seen:
            continue
        _seen.add(key)
        unique_settings.append(s)

    model_ids = sorted({k[0] for k in traces.keys()})
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "pairing_modes": pairing_modes,
        "normalization_modes": normalization_modes,
        "layer_range_modes": layer_range_modes,
        "failing_set_modes": failing_set_modes,
        "evaluated_settings": unique_settings,
        "red_flags": {},
    }

    for setting in unique_settings:
        failing_set_mode = str(setting["failing_set_mode"])
        pairing_mode = str(setting["pairing_mode"])
        normalization_mode = str(setting["normalization_mode"])
        layer_range_mode = str(setting["layer_range_mode"])
        for mid in model_ids:
            prompt_uids = [uid for uid in prompt_order_by_model.get(mid, []) if (mid, uid) in traces]
            pool = [traces[(mid, uid)] for uid in prompt_uids]
            success_pool = [p for p in pool if p.trajectory_type == "stable_correct"]
            if failing_set_mode == "stable_wrong_only":
                fail_pool = [p for p in pool if p.trajectory_type == "stable_wrong"]
            else:
                fail_pool = [p for p in pool if p.trajectory_type in {"stable_wrong", "unstable_wrong"}]
            if not success_pool or not fail_pool:
                continue
            pairs = _build_pairs(
                pairing_mode=pairing_mode,
                model_id=mid,
                success_pool=success_pool,
                fail_pool=fail_pool,
                prompt_order=prompt_order_by_model.get(mid, []),
                seed=int(args.seed) + int(sum(ord(ch) for ch in mid)),
            )
            if not pairs:
                continue
            # Same pair-set for both components.
            comp_rows: dict[str, list[dict[str, Any]]] = {"attention": [], "mlp": []}
            for source, target in pairs:
                n_layers = int(min(target.layers.size, source.layers.size))
                layers_substituted = _layer_set_for_mode(
                    mode=layer_range_mode,
                    n_layers=n_layers,
                    paper_default=paper_layers,
                )
                attn_target = _normalize_component(
                    target.s_attn,
                    model_id=mid,
                    layers=target.layers,
                    mode=normalization_mode,
                    stats=attn_norm_stats,
                )
                attn_source = _normalize_component(
                    source.s_attn,
                    model_id=mid,
                    layers=source.layers,
                    mode=normalization_mode,
                    stats=attn_norm_stats,
                )
                mlp_target = _normalize_component(
                    target.s_mlp,
                    model_id=mid,
                    layers=target.layers,
                    mode=normalization_mode,
                    stats=mlp_norm_stats,
                )
                mlp_source = _normalize_component(
                    source.s_mlp,
                    model_id=mid,
                    layers=source.layers,
                    mode=normalization_mode,
                    stats=mlp_norm_stats,
                )
                row_a = _compute_pair_row(
                    source=source,
                    target=target,
                    component="attention",
                    component_target=attn_target,
                    component_source=attn_source,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode=pairing_mode,
                    normalization_mode=normalization_mode,
                    layer_range_mode=layer_range_mode,
                    failing_set_mode=failing_set_mode,
                    seed=int(args.seed),
                )
                row_m = _compute_pair_row(
                    source=source,
                    target=target,
                    component="mlp",
                    component_target=mlp_target,
                    component_source=mlp_source,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode=pairing_mode,
                    normalization_mode=normalization_mode,
                    layer_range_mode=layer_range_mode,
                    failing_set_mode=failing_set_mode,
                    seed=int(args.seed),
                )
                if row_a:
                    comp_rows["attention"].append(row_a)
                if row_m:
                    comp_rows["mlp"].append(row_m)
            a_keys = {r["pair_id"] for r in comp_rows["attention"]}
            m_keys = {r["pair_id"] for r in comp_rows["mlp"]}
            if a_keys != m_keys:
                raise RuntimeError(
                    "pair-set mismatch across components for "
                    f"{pairing_mode}/{normalization_mode}/{layer_range_mode}/{failing_set_mode} model={mid}"
                )
            rows.extend(comp_rows["attention"])
            rows.extend(comp_rows["mlp"])

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        raise SystemExit("substitution re-derivation produced no rows")

    # Red-flag assertions.
    if not (out["category_source"].astype(str) == "stable_correct").all():
        raise RuntimeError("source category is not uniformly stable_correct")
    valid_targets = {"stable_wrong", "unstable_wrong"}
    if not out["category_target"].astype(str).isin(valid_targets).all():
        raise RuntimeError("target category has unexpected values")
    delta_shift_check = np.abs(
        pd.to_numeric(out["delta_shift"], errors="coerce")
        - (
            pd.to_numeric(out["delta_final_patched"], errors="coerce")
            - pd.to_numeric(out["delta_final_base"], errors="coerce")
        )
    )
    if float(np.nanmax(delta_shift_check)) > 1e-9:
        raise RuntimeError("delta_shift orientation check failed")

    # Sign convention sanity using final-layer decision metrics on traced prompts.
    traced_keys = out[["model_id", "target_prompt_uid"]].drop_duplicates().rename(columns={"target_prompt_uid": "prompt_uid"})
    final_dm = (
        decision.merge(traced_keys, on=["model_id", "prompt_uid"], how="inner")
        .sort_values("layer_index")
        .groupby(["model_id", "prompt_uid"], as_index=False)
        .tail(1)
    )
    sign_match_rate = float(((final_dm["delta"] > 0) == final_dm["is_correct"].astype(bool)).mean()) if not final_dm.empty else 0.0
    diagnostics["red_flags"]["final_delta_sign_match_rate_on_targets"] = sign_match_rate
    diagnostics["red_flags"]["final_delta_sign_match_threshold"] = 0.95
    diagnostics["red_flags"]["final_delta_sign_match_pass"] = bool(sign_match_rate >= 0.95)

    # Legacy row-order sensitivity diagnostic.
    subset = out[
        (out["pairing_mode"] == "legacy_first_per_model")
        & (out["normalization_mode"] == "raw")
        & (out["layer_range_mode"] == "paper-default")
        & (out["failing_set_mode"] == "stable_wrong_plus_unstable_wrong")
    ].copy()
    legacy_base = (
        subset.groupby("component", as_index=False)["delta_shift"].mean().set_index("component")["delta_shift"].to_dict()
        if not subset.empty
        else {}
    )
    diagnostics["legacy_base_mean_shift"] = legacy_base

    # all_pairs row-order invariance diagnostic (set-based).
    all_pairs_subset = out[
        (out["pairing_mode"] == "all_pairs_within_model")
        & (out["normalization_mode"] == "raw")
        & (out["layer_range_mode"] == "paper-default")
        & (out["failing_set_mode"] == "stable_wrong_plus_unstable_wrong")
    ].copy()
    diagnostics["all_pairs_counts"] = (
        all_pairs_subset.groupby("component")["pair_id"].nunique().to_dict() if not all_pairs_subset.empty else {}
    )
    diagnostics["all_pairs_base_mean_shift"] = (
        all_pairs_subset.groupby("component")["delta_shift"].mean().to_dict() if not all_pairs_subset.empty else {}
    )

    # Row-order perturbation diagnostic for legacy pairing.
    perturb_rows: list[dict[str, Any]] = []
    perturb_seeds = [11, 29, 47]
    for ps in perturb_seeds:
        tmp_rows: list[dict[str, Any]] = []
        for mid in model_ids:
            prompt_uids = [uid for uid in prompt_order_by_model.get(mid, []) if (mid, uid) in traces]
            pool = [traces[(mid, uid)] for uid in prompt_uids]
            success_pool = [p for p in pool if p.trajectory_type == "stable_correct"]
            fail_pool = [p for p in pool if p.trajectory_type in {"stable_wrong", "unstable_wrong"}]
            if not success_pool or not fail_pool:
                continue
            rng = np.random.default_rng(int(ps) + int(sum(ord(ch) for ch in mid)))
            shuffled_success_uids = [p.prompt_uid for p in success_pool]
            rng.shuffle(shuffled_success_uids)
            remaining = [uid for uid in prompt_uids if uid not in set(shuffled_success_uids)]
            perturbed_order = list(shuffled_success_uids) + list(remaining)
            pairs = _build_pairs(
                pairing_mode="legacy_first_per_model",
                model_id=mid,
                success_pool=success_pool,
                fail_pool=fail_pool,
                prompt_order=perturbed_order,
                seed=int(args.seed),
            )
            for source, target in pairs:
                n_layers = int(min(target.layers.size, source.layers.size))
                layers_substituted = _layer_set_for_mode(
                    mode="paper-default",
                    n_layers=n_layers,
                    paper_default=paper_layers,
                )
                row_a = _compute_pair_row(
                    source=source,
                    target=target,
                    component="attention",
                    component_target=target.s_attn,
                    component_source=source.s_attn,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode="legacy_first_per_model",
                    normalization_mode="raw",
                    layer_range_mode="paper-default",
                    failing_set_mode="stable_wrong_plus_unstable_wrong",
                    seed=int(ps),
                )
                row_m = _compute_pair_row(
                    source=source,
                    target=target,
                    component="mlp",
                    component_target=target.s_mlp,
                    component_source=source.s_mlp,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode="legacy_first_per_model",
                    normalization_mode="raw",
                    layer_range_mode="paper-default",
                    failing_set_mode="stable_wrong_plus_unstable_wrong",
                    seed=int(ps),
                )
                if row_a:
                    tmp_rows.append(row_a)
                if row_m:
                    tmp_rows.append(row_m)
        if tmp_rows:
            tmp_df = pd.DataFrame.from_records(tmp_rows)
            perturb_rows.append(
                {
                    "seed": int(ps),
                    "mean_shift_by_component": tmp_df.groupby("component")["delta_shift"].mean().to_dict(),
                    "frac_positive_by_component": tmp_df.groupby("component").apply(lambda g: float((g["delta_shift"] > 0).mean())).to_dict(),
                }
            )
    diagnostics["legacy_row_order_perturbation"] = perturb_rows
    if perturb_rows and legacy_base:
        diffs = []
        for rec in perturb_rows:
            means = rec.get("mean_shift_by_component") or {}
            for comp, base_val in legacy_base.items():
                diffs.append(abs(float(means.get(comp, 0.0)) - float(base_val)))
        diagnostics["legacy_row_order_max_abs_mean_diff"] = float(max(diffs)) if diffs else 0.0
        diagnostics["legacy_row_order_sensitive"] = bool((float(max(diffs)) if diffs else 0.0) > 1e-9)
    else:
        diagnostics["legacy_row_order_max_abs_mean_diff"] = 0.0
        diagnostics["legacy_row_order_sensitive"] = False

    # All-pairs should be order-invariant (set-wise and summary-wise).
    all_pairs_perturb_rows: list[dict[str, Any]] = []
    for ps in [71]:
        tmp_rows: list[dict[str, Any]] = []
        for mid in model_ids:
            prompt_uids = [uid for uid in prompt_order_by_model.get(mid, []) if (mid, uid) in traces]
            pool = [traces[(mid, uid)] for uid in prompt_uids]
            success_pool = [p for p in pool if p.trajectory_type == "stable_correct"]
            fail_pool = [p for p in pool if p.trajectory_type in {"stable_wrong", "unstable_wrong"}]
            if not success_pool or not fail_pool:
                continue
            rng = np.random.default_rng(int(ps) + int(sum(ord(ch) for ch in mid)))
            shuffled_success = list(success_pool)
            shuffled_fail = list(fail_pool)
            rng.shuffle(shuffled_success)
            rng.shuffle(shuffled_fail)
            pairs = _build_pairs(
                pairing_mode="all_pairs_within_model",
                model_id=mid,
                success_pool=shuffled_success,
                fail_pool=shuffled_fail,
                prompt_order=prompt_order_by_model.get(mid, []),
                seed=int(args.seed),
            )
            for source, target in pairs:
                n_layers = int(min(target.layers.size, source.layers.size))
                layers_substituted = _layer_set_for_mode(
                    mode="paper-default",
                    n_layers=n_layers,
                    paper_default=paper_layers,
                )
                row_a = _compute_pair_row(
                    source=source,
                    target=target,
                    component="attention",
                    component_target=target.s_attn,
                    component_source=source.s_attn,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode="all_pairs_within_model",
                    normalization_mode="raw",
                    layer_range_mode="paper-default",
                    failing_set_mode="stable_wrong_plus_unstable_wrong",
                    seed=int(ps),
                )
                row_m = _compute_pair_row(
                    source=source,
                    target=target,
                    component="mlp",
                    component_target=target.s_mlp,
                    component_source=source.s_mlp,
                    layers_substituted=layers_substituted,
                    tail_len=tail_len,
                    pairing_mode="all_pairs_within_model",
                    normalization_mode="raw",
                    layer_range_mode="paper-default",
                    failing_set_mode="stable_wrong_plus_unstable_wrong",
                    seed=int(ps),
                )
                if row_a:
                    tmp_rows.append(row_a)
                if row_m:
                    tmp_rows.append(row_m)
        if tmp_rows:
            tdf = pd.DataFrame.from_records(tmp_rows)
            all_pairs_perturb_rows.append(
                {
                    "seed": int(ps),
                    "mean_shift_by_component": tdf.groupby("component")["delta_shift"].mean().to_dict(),
                    "pair_count_by_component": tdf.groupby("component")["pair_id"].nunique().to_dict(),
                }
            )
    diagnostics["all_pairs_order_perturbation"] = all_pairs_perturb_rows
    if all_pairs_perturb_rows and diagnostics["all_pairs_base_mean_shift"]:
        diffs = []
        base_means = diagnostics["all_pairs_base_mean_shift"]
        for rec in all_pairs_perturb_rows:
            means = rec.get("mean_shift_by_component") or {}
            for comp, base_val in base_means.items():
                diffs.append(abs(float(means.get(comp, 0.0)) - float(base_val)))
        diagnostics["all_pairs_order_max_abs_mean_diff"] = float(max(diffs)) if diffs else 0.0
        diagnostics["all_pairs_order_invariant"] = bool((float(max(diffs)) if diffs else 0.0) <= 1e-9)
    else:
        diagnostics["all_pairs_order_max_abs_mean_diff"] = 0.0
        diagnostics["all_pairs_order_invariant"] = True

    write_csv(args.out_csv, out)
    write_json(args.out_json, diagnostics)
    print(str(args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
