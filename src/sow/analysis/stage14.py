from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.token_buckets.option_buckets import model_fs_id


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_uids(path: Path) -> List[str]:
    uids = []
    for r in iter_jsonl(path):
        uids.append(str(r["prompt_uid"]))
    return uids


def _load_inference_sentinel(run_dir: Path, *, condition: str, model_id: str) -> Dict[str, Any]:
    p = run_dir / "sentinels" / f"inference_{condition}.{model_fs_id(model_id)}.done"
    if not p.exists():
        raise FileNotFoundError(f"missing inference sentinel: {p}")
    obj = _load_json(p)
    out_path = Path(str(obj.get("output_path") or ""))
    if not out_path.exists():
        raise FileNotFoundError(f"missing output referenced by sentinel: {out_path}")
    if sha256_file(out_path) != str(obj.get("output_sha256") or ""):
        raise ValueError(f"output sha256 mismatch vs sentinel: {out_path}")
    obj["output_path"] = str(out_path)
    return obj


@dataclass
class _Agg:
    count: int
    one_token_ok: int
    resolved: int
    correct_resolved: int
    flip_sum: int


def _empty_agg() -> _Agg:
    return _Agg(count=0, one_token_ok=0, resolved=0, correct_resolved=0, flip_sum=0)


def _agg_add(agg: _Agg, row: Dict[str, Any]) -> None:
    agg.count += 1
    if bool(row.get("first_token_is_option_letter")):
        agg.one_token_ok += 1
    if str(row.get("parser_status") or "") == "resolved":
        agg.resolved += 1
        if row.get("is_correct") is True:
            agg.correct_resolved += 1
    agg.flip_sum += int(row.get("flip_count") or 0)


def _finalize_agg(agg: _Agg) -> Dict[str, Any]:
    if agg.count <= 0:
        return {"n": 0}
    acc_res = (agg.correct_resolved / agg.resolved) if agg.resolved else None
    return {
        "n": int(agg.count),
        "one_token_compliance_rate": float(agg.one_token_ok / agg.count),
        "parser_resolved_rate": float(agg.resolved / agg.count),
        "accuracy_on_resolved": float(acc_res) if acc_res is not None else None,
        "mean_flip_count": float(agg.flip_sum / agg.count),
    }


def run_stage14_analysis(
    *,
    run_id: str,
    run_dir: Path,
    cfg: Dict[str, Any],
    baseline_manifest: Path,
    robustness_manifest: Path,
    baseline_wrapper_id: str,
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    Stage 14: deterministic analysis producing JSON/CSV artifacts.

    This is intentionally conservative: it focuses on behavioral + trajectory metrics
    and avoids heavy vector-geometry processing (which would require parsing and
    aggregating 128D vectors per layer for every row).
    """
    models = list(cfg["models"])
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_uids = _load_manifest_uids(baseline_manifest)
    rob_uids = _load_manifest_uids(robustness_manifest)
    base_uid_set = set(base_uids)
    rob_uid_set = set(rob_uids)

    # Outputs.
    per_prompt_csv = _next_available_path(analysis_dir / "per_prompt_metrics.csv")
    layerwise_csv = _next_available_path(analysis_dir / "layerwise_aggregates.csv")
    robust_delta_csv = _next_available_path(analysis_dir / "robustness_deltas.csv")
    commit_csv = _next_available_path(analysis_dir / "commitment_hist.csv")

    # Prepare writers.
    pp_cols = [
        "run_id",
        "model_id",
        "model_revision",
        "condition",
        "prompt_uid",
        "example_id",
        "wrapper_id",
        "coarse_domain",
        "parser_status",
        "parsed_choice",
        "is_correct",
        "first_token_is_option_letter",
        "flip_count",
        *[f"commitment_layer@{t}" for t in thresholds],
    ]
    lw_cols = ["run_id", "model_id", "model_revision", "condition", "wrapper_id", "layer_index", "mean_entropy", "mean_margin"]
    rd_cols = [
        "run_id",
        "model_id",
        "model_revision",
        "wrapper_id",
        "accuracy_on_resolved_baseline",
        "accuracy_on_resolved_wrapper",
        "delta_accuracy_on_resolved",
        "parser_resolved_rate_baseline",
        "parser_resolved_rate_wrapper",
        "delta_parser_resolved_rate",
        "one_token_compliance_rate_baseline",
        "one_token_compliance_rate_wrapper",
        "delta_one_token_compliance_rate",
        "n_wrapper",
    ]
    ch_cols = ["run_id", "model_id", "model_revision", "condition", "wrapper_id", "threshold", "commitment_layer", "count"]

    per_model_summary: Dict[str, Any] = {}
    validation: Dict[str, Any] = {"models": []}

    with per_prompt_csv.open("w", newline="", encoding="utf-8") as f_pp, layerwise_csv.open("w", newline="", encoding="utf-8") as f_lw, robust_delta_csv.open("w", newline="", encoding="utf-8") as f_rd, commit_csv.open("w", newline="", encoding="utf-8") as f_ch:
        w_pp = csv.DictWriter(f_pp, fieldnames=pp_cols)
        w_lw = csv.DictWriter(f_lw, fieldnames=lw_cols)
        w_rd = csv.DictWriter(f_rd, fieldnames=rd_cols)
        w_ch = csv.DictWriter(f_ch, fieldnames=ch_cols)
        w_pp.writeheader()
        w_lw.writeheader()
        w_rd.writeheader()
        w_ch.writeheader()

        for m in models:
            model_id = str(m["model_id"])
            revision = str(m["revision"])

            # Load inference outputs from sentinels.
            s_base = _load_inference_sentinel(run_dir, condition="baseline", model_id=model_id)
            s_rob = _load_inference_sentinel(run_dir, condition="robustness", model_id=model_id)
            p_base = Path(s_base["output_path"])
            p_rob = Path(s_rob["output_path"])

            # Aggregations by wrapper for each condition.
            agg_base_by_wrap: Dict[str, _Agg] = defaultdict(_empty_agg)
            agg_rob_by_wrap: Dict[str, _Agg] = defaultdict(_empty_agg)
            agg_base_overall = _empty_agg()
            agg_rob_overall = _empty_agg()

            # Layerwise sums by wrapper (entropy + margin).
            lw_sum_ent_base: Dict[str, np.ndarray] = {}
            lw_sum_mar_base: Dict[str, np.ndarray] = {}
            lw_cnt_base: Dict[str, int] = defaultdict(int)

            lw_sum_ent_rob: Dict[str, np.ndarray] = {}
            lw_sum_mar_rob: Dict[str, np.ndarray] = {}
            lw_cnt_rob: Dict[str, int] = defaultdict(int)

            # Commitment hist by wrapper + threshold.
            commit_hist_base: Dict[Tuple[str, float], Dict[Optional[int], int]] = defaultdict(lambda: defaultdict(int))
            commit_hist_rob: Dict[Tuple[str, float], Dict[Optional[int], int]] = defaultdict(lambda: defaultdict(int))

            # Coverage validation: prompt_uids set matches CCC manifests.
            seen_base: set[str] = set()
            seen_rob: set[str] = set()

            def handle_row(*, condition: str, row: Dict[str, Any]) -> None:
                wrapper = str(row.get("wrapper_id") or "")
                lw = row.get("layerwise") or []
                if not isinstance(lw, list) or not lw:
                    raise ValueError("missing layerwise")
                n_layers = len(lw)
                ent = np.asarray([float(x["candidate_entropy"]) for x in lw], dtype=np.float64)
                mar = np.asarray([float(x["top2_margin_prob"]) for x in lw], dtype=np.float64)

                if condition == "baseline":
                    if wrapper not in lw_sum_ent_base:
                        lw_sum_ent_base[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                        lw_sum_mar_base[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                    lw_sum_ent_base[wrapper] += ent
                    lw_sum_mar_base[wrapper] += mar
                    lw_cnt_base[wrapper] += 1
                    _agg_add(agg_base_by_wrap[wrapper], row)
                    _agg_add(agg_base_overall, row)
                else:
                    if wrapper not in lw_sum_ent_rob:
                        lw_sum_ent_rob[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                        lw_sum_mar_rob[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                    lw_sum_ent_rob[wrapper] += ent
                    lw_sum_mar_rob[wrapper] += mar
                    lw_cnt_rob[wrapper] += 1
                    _agg_add(agg_rob_by_wrap[wrapper], row)
                    _agg_add(agg_rob_overall, row)

                # Commitment hist.
                cdict = row.get("commitment_layer_by_margin_threshold") or {}
                for t in thresholds:
                    key = str(float(t))
                    v = cdict.get(key)
                    layer = int(v) if isinstance(v, int) else None
                    if condition == "baseline":
                        commit_hist_base[(wrapper, float(t))][layer] += 1
                    else:
                        commit_hist_rob[(wrapper, float(t))][layer] += 1

                # Per-prompt metrics CSV.
                pp = {
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_revision": revision,
                    "condition": condition,
                    "prompt_uid": str(row.get("prompt_uid") or ""),
                    "example_id": str(row.get("example_id") or ""),
                    "wrapper_id": wrapper,
                    "coarse_domain": str(row.get("coarse_domain") or "unknown"),
                    "parser_status": str(row.get("parser_status") or ""),
                    "parsed_choice": row.get("parsed_choice"),
                    "is_correct": row.get("is_correct"),
                    "first_token_is_option_letter": bool(row.get("first_token_is_option_letter")),
                    "flip_count": int(row.get("flip_count") or 0),
                }
                for t in thresholds:
                    pp[f"commitment_layer@{t}"] = (row.get("commitment_layer_by_margin_threshold") or {}).get(str(float(t)))
                w_pp.writerow(pp)

            # Process baseline.
            for row in iter_jsonl(p_base):
                puid = str(row.get("prompt_uid") or "")
                seen_base.add(puid)
                handle_row(condition="baseline", row=row)
            # Process robustness.
            for row in iter_jsonl(p_rob):
                puid = str(row.get("prompt_uid") or "")
                seen_rob.add(puid)
                handle_row(condition="robustness", row=row)

            missing_base = sorted(base_uid_set - seen_base)
            extra_base = sorted(seen_base - base_uid_set)
            missing_rob = sorted(rob_uid_set - seen_rob)
            extra_rob = sorted(seen_rob - rob_uid_set)

            validation["models"].append(
                {
                    "model_id": model_id,
                    "model_revision": revision,
                    "baseline": {"missing_uids": missing_base[:20], "extra_uids": extra_base[:20], "n": len(seen_base)},
                    "robustness": {"missing_uids": missing_rob[:20], "extra_uids": extra_rob[:20], "n": len(seen_rob)},
                }
            )

            # Write layerwise aggregates.
            def write_layerwise(condition: str, sums_ent: Dict[str, np.ndarray], sums_mar: Dict[str, np.ndarray], counts: Dict[str, int]) -> None:
                for wrapper, ent_sum in sorted(sums_ent.items(), key=lambda kv: kv[0]):
                    mar_sum = sums_mar[wrapper]
                    cnt = int(counts[wrapper])
                    for li in range(int(ent_sum.shape[0])):
                        w_lw.writerow(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "model_revision": revision,
                                "condition": condition,
                                "wrapper_id": wrapper,
                                "layer_index": int(li),
                                "mean_entropy": float(ent_sum[li] / cnt) if cnt else None,
                                "mean_margin": float(mar_sum[li] / cnt) if cnt else None,
                            }
                        )

            write_layerwise("baseline", lw_sum_ent_base, lw_sum_mar_base, lw_cnt_base)
            write_layerwise("robustness", lw_sum_ent_rob, lw_sum_mar_rob, lw_cnt_rob)

            # Write commitment hists.
            def write_commit(condition: str, hist: Dict[Tuple[str, float], Dict[Optional[int], int]]) -> None:
                for (wrapper, t), mp in sorted(hist.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                    for layer, cnt in sorted(mp.items(), key=lambda kv: (-1 if kv[0] is None else int(kv[0]))):
                        w_ch.writerow(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "model_revision": revision,
                                "condition": condition,
                                "wrapper_id": wrapper,
                                "threshold": float(t),
                                "commitment_layer": layer,
                                "count": int(cnt),
                            }
                        )

            write_commit("baseline", commit_hist_base)
            write_commit("robustness", commit_hist_rob)

            # Wrapper-wise robustness deltas relative to baseline (resolved accuracy, resolved rate, one-token rate).
            base_overall = _finalize_agg(agg_base_by_wrap.get(str(baseline_wrapper_id)) or _empty_agg())
            base_acc = base_overall.get("accuracy_on_resolved")
            base_res = base_overall.get("parser_resolved_rate")
            base_one = base_overall.get("one_token_compliance_rate")

            for wrapper, agg in sorted(agg_rob_by_wrap.items(), key=lambda kv: kv[0]):
                wr = _finalize_agg(agg)
                w_rd.writerow(
                    {
                        "run_id": run_id,
                        "model_id": model_id,
                        "model_revision": revision,
                        "wrapper_id": wrapper,
                        "accuracy_on_resolved_baseline": base_acc,
                        "accuracy_on_resolved_wrapper": wr.get("accuracy_on_resolved"),
                        "delta_accuracy_on_resolved": (wr.get("accuracy_on_resolved") - base_acc) if (wr.get("accuracy_on_resolved") is not None and base_acc is not None) else None,
                        "parser_resolved_rate_baseline": base_res,
                        "parser_resolved_rate_wrapper": wr.get("parser_resolved_rate"),
                        "delta_parser_resolved_rate": (wr.get("parser_resolved_rate") - base_res) if (wr.get("parser_resolved_rate") is not None and base_res is not None) else None,
                        "one_token_compliance_rate_baseline": base_one,
                        "one_token_compliance_rate_wrapper": wr.get("one_token_compliance_rate"),
                        "delta_one_token_compliance_rate": (wr.get("one_token_compliance_rate") - base_one) if (wr.get("one_token_compliance_rate") is not None and base_one is not None) else None,
                        "n_wrapper": int(wr.get("n") or 0),
                    }
                )

            per_model_summary[model_id] = {"baseline_overall": _finalize_agg(agg_base_overall), "robustness_overall": _finalize_agg(agg_rob_overall)}

    # Final report JSON.
    report = {
        "pass": True,
        "run_id": run_id,
        "thresholds": [float(x) for x in thresholds],
        "inputs": {
            "baseline_manifest_path": str(baseline_manifest),
            "baseline_manifest_sha256": sha256_file(baseline_manifest),
            "robustness_manifest_path": str(robustness_manifest),
            "robustness_manifest_sha256": sha256_file(robustness_manifest),
        },
        "artifacts": {
            "per_prompt_metrics_csv": str(per_prompt_csv),
            "per_prompt_metrics_sha256": sha256_file(per_prompt_csv),
            "layerwise_aggregates_csv": str(layerwise_csv),
            "layerwise_aggregates_sha256": sha256_file(layerwise_csv),
            "robustness_deltas_csv": str(robust_delta_csv),
            "robustness_deltas_sha256": sha256_file(robust_delta_csv),
            "commitment_hist_csv": str(commit_csv),
            "commitment_hist_sha256": sha256_file(commit_csv),
        },
        "per_model_summary": per_model_summary,
        "validation": validation,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report
