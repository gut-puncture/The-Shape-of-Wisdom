#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from _common import (
    base_parser,
    baseline_manifest_path,
    load_experiment_config,
    resolve_models,
    run_v2_root_for,
    write_json,
    write_text_atomic,
)
from sow.io_jsonl import iter_jsonl


def _domain_difficulty_table(sub: pd.DataFrame, *, manifest_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame(columns=["coarse_domain", "n_prompts", "correct_rate", "difficulty"])
    tmp = sub.copy()
    tmp["coarse_domain"] = [str((manifest_map.get(str(u)) or {}).get("coarse_domain") or "unknown") for u in tmp["prompt_uid"].tolist()]
    if "is_correct" not in tmp.columns:
        tmp["is_correct"] = False
    out = (
        tmp.groupby("coarse_domain", as_index=False)
        .agg(n_prompts=("prompt_uid", "nunique"), correct_rate=("is_correct", "mean"))
        .sort_values(["correct_rate", "n_prompts", "coarse_domain"], ascending=[True, False, True])
    )
    out["difficulty"] = 1.0 - out["correct_rate"].astype(float)
    return out


def _select_balanced_subset(
    sub: pd.DataFrame,
    *,
    per_model_target: int,
    min_difficult_share: float,
    required_types: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if sub.empty:
        return sub, {
            "required_types": list(required_types),
            "per_type_target": {},
            "available_by_type": {},
            "selected_by_type": {},
            "trajectory_quota_feasible": False,
            "infeasible_types": list(required_types),
            "selected_total": 0,
            "target_total": int(per_model_target),
            "target_met": False,
        }
    rtypes = [str(t) for t in required_types] if required_types else sorted(sub["trajectory_type"].astype(str).drop_duplicates().tolist())
    n_types = max(1, int(len(rtypes)))
    base = int(per_model_target // n_types)
    rem = int(per_model_target % n_types)
    per_type_target = {t: int(base + (1 if i < rem else 0)) for i, t in enumerate(rtypes)}
    selected_parts = []
    available_by_type: dict[str, int] = {}
    selected_by_type: dict[str, int] = {}
    infeasible_types: list[str] = []
    for t in rtypes:
        g = sub[sub["trajectory_type"].astype(str) == str(t)].sort_values(["is_difficult", "prompt_uid"], ascending=[False, True]).copy()
        tgt = int(per_type_target.get(str(t), 0))
        available_by_type[str(t)] = int(g.shape[0])
        if int(g.shape[0]) < int(tgt):
            infeasible_types.append(str(t))
        hard_target = int(math.ceil(float(tgt) * float(min_difficult_share)))
        hard = g[g["is_difficult"]].head(hard_target)
        rem_df = g[~g["prompt_uid"].isin(set(hard["prompt_uid"].tolist()))].head(max(0, tgt - int(hard.shape[0])))
        picked = pd.concat([hard, rem_df], ignore_index=True)
        if int(picked.shape[0]) < int(tgt):
            extra = g[~g["prompt_uid"].isin(set(picked["prompt_uid"].tolist()))].head(int(tgt - int(picked.shape[0])))
            picked = pd.concat([picked, extra], ignore_index=True)
        picked = picked.head(int(tgt))
        selected_by_type[str(t)] = int(picked.shape[0])
        selected_parts.append(picked)
    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["prompt_uid"]) if selected_parts else pd.DataFrame(columns=sub.columns)
    selected = selected.head(int(per_model_target)).copy()
    quota = {
        "required_types": rtypes,
        "per_type_target": {str(k): int(v) for k, v in per_type_target.items()},
        "available_by_type": {str(k): int(v) for k, v in available_by_type.items()},
        "selected_by_type": {str(k): int(v) for k, v in selected_by_type.items()},
        "trajectory_quota_feasible": bool(len(infeasible_types) == 0),
        "infeasible_types": sorted(infeasible_types),
        "selected_total": int(selected.shape[0]),
        "target_total": int(per_model_target),
        "target_met": int(selected.shape[0]) == int(per_model_target),
    }
    return selected, quota


def main() -> int:
    ap = base_parser("V2: select tracing subset balanced by trajectory type")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    per_model_target_cfg = int((cfg.get("sampling") or {}).get("tracing_prompts_per_model", 600))
    per_model_target = int(per_model_target_cfg)
    if int(args.max_prompts) > 0:
        per_model_target = min(int(per_model_target), int(args.max_prompts))
    per_model_target = max(1, int(per_model_target))
    validators_cfg = cfg.get("validators") or {}
    stage06_cfg = validators_cfg.get("stage06_tracing_subset") or {}
    difficult_top_k = max(1, int(stage06_cfg.get("difficult_domain_top_k", 1)))
    min_difficult_share = float(stage06_cfg.get("min_difficult_share", 0.30))
    min_domains_covered = int(stage06_cfg.get("min_domains_covered", 1))
    min_prompts_per_domain = int(stage06_cfg.get("min_prompts_per_domain", 10))
    max_domain_share = float(stage06_cfg.get("max_domain_share", 1.0))
    stage03_cfg = validators_cfg.get("stage03_trajectory") or {}
    required_types = [str(x) for x in (stage03_cfg.get("required_types") or ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"])]

    out_root = run_v2_root_for(args.run_id)
    types_path = out_root / "prompt_types.parquet"
    if not types_path.exists():
        raise SystemExit(f"missing input: {types_path}")
    types = pd.read_parquet(types_path)

    manifest_map = {}
    for row in iter_jsonl(baseline_manifest_path(args.run_id)):
        manifest_map[str(row.get("prompt_uid") or "")] = row

    report = {"pass": True, "models": {}}
    model_passes = []
    for model in models:
        model_id = str(model["model_id"])
        sub = types[types["model_id"] == model_id].copy()
        if sub.empty:
            report["models"][model_id] = {"selected": 0, "pass": False, "reason": "no_prompts"}
            model_passes.append(False)
            continue

        if "is_correct" not in sub.columns:
            sub["is_correct"] = False

        dtab = _domain_difficulty_table(sub, manifest_map=manifest_map)
        difficult_domains = dtab.head(difficult_top_k)["coarse_domain"].astype(str).tolist()
        prompt_to_domain = {str(uid): str((manifest_map.get(str(uid)) or {}).get("coarse_domain") or "unknown") for uid in sub["prompt_uid"].tolist()}
        sub["coarse_domain"] = [prompt_to_domain.get(str(uid), "unknown") for uid in sub["prompt_uid"].tolist()]
        sub["is_difficult"] = sub["coarse_domain"].astype(str).isin(set(difficult_domains))
        sub = sub.sort_values(["trajectory_type", "is_difficult", "prompt_uid"], ascending=[True, False, True])

        selected, quota = _select_balanced_subset(
            sub,
            per_model_target=int(per_model_target),
            min_difficult_share=float(min_difficult_share),
            required_types=required_types,
        )

        records = []
        for _, row in selected.iterrows():
            puid = str(row["prompt_uid"])
            m = manifest_map.get(puid, {})
            records.append(
                {
                    "prompt_uid": puid,
                    "example_id": str(m.get("example_id") or row.get("prompt_uid")),
                    "trajectory_type": str(row["trajectory_type"]),
                    "prompt_text": str(m.get("prompt_text") or ""),
                    "correct_key": str(m.get("correct_key") or "A"),
                    "coarse_domain": str(m.get("coarse_domain") or "unknown"),
                    "is_difficult_domain": bool(str(m.get("coarse_domain") or "unknown") in set(difficult_domains)),
                }
            )

        out_path = out_root / f"tracing_subset_{model_id.replace('/', '__')}.json"
        write_text_atomic(out_path, json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

        selected_df = pd.DataFrame.from_records(records)
        available_difficult_count = int(sub["is_difficult"].sum())
        actual_difficult_count = int(selected_df["is_difficult_domain"].sum()) if (not selected_df.empty and "is_difficult_domain" in selected_df.columns) else 0
        target_difficult_count = int(math.ceil(float(min_difficult_share) * float(max(1, selected_df.shape[0]))))
        difficult_feasible = available_difficult_count >= target_difficult_count
        difficult_ok = difficult_feasible and (actual_difficult_count >= target_difficult_count)
        selected_type_counts = selected_df["trajectory_type"].astype(str).value_counts().to_dict() if (not selected_df.empty and "trajectory_type" in selected_df.columns) else {}
        per_type_target = quota.get("per_type_target") or {}
        trajectory_ok = all(int(selected_type_counts.get(str(t), 0)) == int(per_type_target.get(str(t), 0)) for t in required_types)
        selected_domain_counts = (
            selected_df["coarse_domain"].astype(str).value_counts().to_dict()
            if (not selected_df.empty and "coarse_domain" in selected_df.columns)
            else {}
        )
        selected_domain_count = len(selected_domain_counts)
        selected_total = int(len(records))
        max_selected_share = float(max(selected_domain_counts.values()) / max(1, selected_total)) if selected_domain_counts else 0.0
        domain_coverage_ok = int(selected_domain_count) >= int(min_domains_covered)
        if int(min_prompts_per_domain) > 0:
            domain_floor_ok = bool(
                selected_domain_count > 0
                and all(int(v) >= int(min_prompts_per_domain) for v in selected_domain_counts.values())
            )
        else:
            domain_floor_ok = bool(selected_domain_count > 0)
        domain_max_share_ok = bool(selected_domain_count > 0) and float(max_selected_share) <= float(max_domain_share)
        gates = {
            "nonempty_selection": len(records) > 0,
            "selection_target_met": int(selected_total) == int(per_model_target),
            "trajectory_quota_feasible": bool(quota.get("trajectory_quota_feasible", False)),
            "trajectory_balance": bool(trajectory_ok),
            "difficult_domain_feasible": bool(difficult_feasible),
            "difficult_domain_balance": bool(difficult_ok),
            "domain_coverage": bool(domain_coverage_ok),
            "domain_prompt_floor": bool(domain_floor_ok),
            "domain_max_share": bool(domain_max_share_ok),
        }
        failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
        model_pass = len(failing_gates) == 0
        model_passes.append(model_pass)

        report["models"][model_id] = {
            "selected": int(len(records)),
            "target_selected": int(per_model_target),
            "out_path": str(out_path),
            "trajectory_counts": {str(k): int(v) for k, v in selected_type_counts.items()},
            "trajectory_balance": quota,
            "difficult_domains": difficult_domains,
            "domain_difficulty": dtab.to_dict(orient="records"),
            "difficult_domain_balance": {
                "min_difficult_share": float(min_difficult_share),
                "available_difficult_count": int(available_difficult_count),
                "target_difficult_count": int(target_difficult_count),
                "actual_difficult_count": int(actual_difficult_count),
                "actual_difficult_share": float(actual_difficult_count / max(1, len(records))),
            },
            "domain_replication": {
                "min_domains_covered": int(min_domains_covered),
                "selected_domain_count": int(selected_domain_count),
                "min_prompts_per_domain": int(min_prompts_per_domain),
                "selected_domain_counts": {str(k): int(v) for k, v in selected_domain_counts.items()},
                "max_domain_share": float(max_domain_share),
                "actual_max_domain_share": float(max_selected_share),
            },
            "gates": gates,
            "failing_gates": failing_gates,
            "pass": bool(model_pass),
        }

    report["pass"] = bool(len(model_passes) > 0 and all(model_passes))
    write_json(out_root / "06_select_tracing_subset.report.json", report)
    print(str(out_root / "06_select_tracing_subset.report.json"))
    return 0 if bool(report["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
