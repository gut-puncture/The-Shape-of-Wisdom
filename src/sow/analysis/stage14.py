from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sow.hashing import sha256_file
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


def _load_inference_sentinel(run_dir: Path, *, condition: str, model_id: str, model_revision: str, run_id: str) -> Dict[str, Any]:
    p = run_dir / "sentinels" / f"inference_{condition}.{model_fs_id(model_id)}.done"
    if not p.exists():
        raise FileNotFoundError(f"missing inference sentinel: {p}")
    obj = _load_json(p)
    # Fail-fast: analysis must be anchored to stage-owned, self-describing sentinels.
    if obj.get("stage") != f"inference_{condition}":
        raise ValueError(f"inference sentinel stage mismatch: expected inference_{condition}")
    if obj.get("run_id") != str(run_id):
        raise ValueError("inference sentinel run_id mismatch")
    if obj.get("model_id") != str(model_id):
        raise ValueError("inference sentinel model_id mismatch")
    if obj.get("model_revision") != str(model_revision):
        raise ValueError("inference sentinel model_revision mismatch")
    out_path = Path(str(obj.get("output_path") or ""))
    if not out_path.exists():
        raise FileNotFoundError(f"missing output referenced by sentinel: {out_path}")
    if sha256_file(out_path) != str(obj.get("output_sha256") or ""):
        raise ValueError(f"output sha256 mismatch vs sentinel: {out_path}")
    obj["output_path"] = str(out_path)
    return obj


def _resolve_topology_layer_index(*, requested_layer: int, n_layers: int) -> int:
    if int(n_layers) <= 0:
        raise ValueError("n_layers must be positive")
    if int(requested_layer) == -1:
        return int(n_layers) - 1
    if int(requested_layer) < 0 or int(requested_layer) >= int(n_layers):
        raise ValueError(f"invalid topology layer index {requested_layer}; n_layers={n_layers}")
    return int(requested_layer)


def _convergence_index_from_entropy(entropy: float) -> float:
    # candidate entropy is on 4 options -> max entropy ln(4)
    return float(np.clip(1.0 - (float(entropy) / float(np.log(4.0))), 0.0, 1.0))


def _project_centroids_to_2d(x: np.ndarray) -> np.ndarray:
    # x: (n_domains, dim)
    if x.ndim != 2:
        raise ValueError("centroid matrix must be 2D")
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(2, int(vt.shape[0]))
    coords = centered @ vt[:k, :].T
    if k == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=np.float64)], axis=1)
    return coords.astype(np.float64, copy=False)


def _plot_convergence_commitment(
    *,
    out_path: Path,
    model_id: str,
    mean_entropy: np.ndarray,
    mean_margin: np.ndarray,
    commitment_hist: Dict[Optional[int], int],
    commit_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(int(mean_entropy.shape[0]))
    conv = np.asarray([_convergence_index_from_entropy(float(v)) for v in mean_entropy], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    ax0.plot(x, conv, label="convergence_index", linewidth=2.0)
    ax0.plot(x, mean_margin, label="mean_margin", linewidth=2.0)
    ax0.set_title("Convergence and Margin by Layer")
    ax0.set_xlabel("layer_index")
    ax0.set_ylabel("value")
    ax0.set_ylim(bottom=0.0)
    ax0.grid(True, alpha=0.2)
    ax0.legend(loc="best")

    ax1 = axes[1]
    # None means "never committed by threshold"
    layers = sorted(commitment_hist.keys(), key=lambda v: (-1 if v is None else int(v)))
    labels = ["None" if v is None else str(int(v)) for v in layers]
    vals = [int(commitment_hist.get(v, 0)) for v in layers]
    xh = np.arange(len(labels))
    ax1.bar(xh, vals)
    ax1.set_xticks(xh)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title(f"Commitment Histogram @ margin={float(commit_threshold):.2f}")
    ax1.set_xlabel("commitment_layer")
    ax1.set_ylabel("count")
    ax1.grid(True, axis="y", alpha=0.2)

    fig.suptitle(model_id)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_domain_topology(
    *,
    out_path: Path,
    model_id: str,
    layer_index: int,
    domains: List[str],
    counts: List[int],
    coords: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if len(domains) == 0:
        ax.text(0.5, 0.5, "No domains", ha="center", va="center")
    else:
        sizes = 40.0 + 12.0 * np.sqrt(np.asarray(counts, dtype=np.float64))
        colors = np.arange(len(domains), dtype=np.float64)
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=colors, cmap="tab20", alpha=0.85, edgecolors="black", linewidths=0.3)
        for i, d in enumerate(domains):
            ax.annotate(str(d), (float(coords[i, 0]), float(coords[i, 1])), fontsize=8)
    ax.set_title(f"Domain Topology Centroids (layer={layer_index})")
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.grid(True, alpha=0.2)
    fig.suptitle(model_id)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
    robustness_manifest: Optional[Path],
    baseline_wrapper_id: str,
    thresholds: List[float],
    include_robustness: bool = True,
    topology_layer: int = -1,
) -> Dict[str, Any]:
    """
    Stage 14 deterministic analysis producing CSV/JSON artifacts and summary plots.

    Modes:
    - include_robustness=True: baseline + robustness deltas (legacy behavior)
    - include_robustness=False: baseline-only mechanistic analysis (faster)
    """
    if include_robustness and robustness_manifest is None:
        raise ValueError("robustness_manifest is required when include_robustness=True")

    models = list(cfg["models"])
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = analysis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_uids = _load_manifest_uids(baseline_manifest)
    base_uid_set = set(base_uids)
    rob_uid_set: set[str] = set()
    if include_robustness and robustness_manifest is not None:
        rob_uid_set = set(_load_manifest_uids(robustness_manifest))

    # Outputs.
    per_prompt_csv = _next_available_path(analysis_dir / "per_prompt_metrics.csv")
    layerwise_csv = _next_available_path(analysis_dir / "layerwise_aggregates.csv")
    robust_delta_csv = _next_available_path(analysis_dir / "robustness_deltas.csv")
    commit_csv = _next_available_path(analysis_dir / "commitment_hist.csv")
    convergence_csv = _next_available_path(analysis_dir / "convergence_by_layer.csv")
    topology_centroids_csv = _next_available_path(analysis_dir / "domain_topology_centroids.csv")
    topology_dists_csv = _next_available_path(analysis_dir / "domain_topology_pairwise_distances.csv")

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
    cv_cols = [
        "run_id",
        "model_id",
        "model_revision",
        "condition",
        "wrapper_id",
        "layer_index",
        "mean_entropy",
        "convergence_index",
        "mean_margin",
        "n_rows",
    ]
    tc_cols = [
        "run_id",
        "model_id",
        "model_revision",
        "condition",
        "layer_index",
        "coarse_domain",
        "n_rows",
        "centroid_norm_l2",
        "pc1",
        "pc2",
    ]
    td_cols = [
        "run_id",
        "model_id",
        "model_revision",
        "condition",
        "layer_index",
        "coarse_domain_a",
        "coarse_domain_b",
        "centroid_distance_l2",
    ]

    per_model_summary: Dict[str, Any] = {}
    validation: Dict[str, Any] = {
        "models": [],
        "include_robustness": bool(include_robustness),
        "topology_layer_requested": int(topology_layer),
    }
    errors: List[str] = []
    figure_files: List[Path] = []

    with (
        per_prompt_csv.open("w", newline="", encoding="utf-8") as f_pp,
        layerwise_csv.open("w", newline="", encoding="utf-8") as f_lw,
        robust_delta_csv.open("w", newline="", encoding="utf-8") as f_rd,
        commit_csv.open("w", newline="", encoding="utf-8") as f_ch,
        convergence_csv.open("w", newline="", encoding="utf-8") as f_cv,
        topology_centroids_csv.open("w", newline="", encoding="utf-8") as f_tc,
        topology_dists_csv.open("w", newline="", encoding="utf-8") as f_td,
    ):
        w_pp = csv.DictWriter(f_pp, fieldnames=pp_cols)
        w_lw = csv.DictWriter(f_lw, fieldnames=lw_cols)
        w_rd = csv.DictWriter(f_rd, fieldnames=rd_cols)
        w_ch = csv.DictWriter(f_ch, fieldnames=ch_cols)
        w_cv = csv.DictWriter(f_cv, fieldnames=cv_cols)
        w_tc = csv.DictWriter(f_tc, fieldnames=tc_cols)
        w_td = csv.DictWriter(f_td, fieldnames=td_cols)

        w_pp.writeheader()
        w_lw.writeheader()
        w_rd.writeheader()
        w_ch.writeheader()
        w_cv.writeheader()
        w_tc.writeheader()
        w_td.writeheader()

        for m in models:
            model_id = str(m["model_id"])
            revision = str(m["revision"])

            # Load inference outputs from sentinels.
            s_base = _load_inference_sentinel(run_dir, condition="baseline", model_id=model_id, model_revision=revision, run_id=run_id)
            p_base = Path(s_base["output_path"])
            p_rob: Optional[Path] = None
            if include_robustness:
                s_rob = _load_inference_sentinel(run_dir, condition="robustness", model_id=model_id, model_revision=revision, run_id=run_id)
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

            # Baseline-only topology accumulators by domain.
            topo_sum_by_domain: Dict[str, np.ndarray] = {}
            topo_count_by_domain: Dict[str, int] = defaultdict(int)
            topo_dim: List[Optional[int]] = [None]
            topo_layer_used: List[Optional[int]] = [None]

            # Coverage validation: prompt_uids set matches manifests.
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
                    if wrapper != str(baseline_wrapper_id):
                        raise ValueError(f"unexpected wrapper_id in baseline outputs: {wrapper}")
                    if wrapper not in lw_sum_ent_base:
                        lw_sum_ent_base[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                        lw_sum_mar_base[wrapper] = np.zeros((n_layers,), dtype=np.float64)
                    lw_sum_ent_base[wrapper] += ent
                    lw_sum_mar_base[wrapper] += mar
                    lw_cnt_base[wrapper] += 1
                    _agg_add(agg_base_by_wrap[wrapper], row)
                    _agg_add(agg_base_overall, row)

                    li = _resolve_topology_layer_index(requested_layer=int(topology_layer), n_layers=n_layers)
                    topo_layer_used[0] = int(li)
                    proj = np.asarray(lw[li].get("projected_hidden_128") or [], dtype=np.float64)
                    if proj.ndim != 1 or int(proj.shape[0]) <= 0:
                        raise ValueError("missing/invalid projected_hidden_128 for topology")
                    if topo_dim[0] is None:
                        topo_dim[0] = int(proj.shape[0])
                    if int(proj.shape[0]) != int(topo_dim[0]):
                        raise ValueError("projected_hidden_128 dimension mismatch across rows")
                    dom = str(row.get("coarse_domain") or "unknown")
                    if dom not in topo_sum_by_domain:
                        topo_sum_by_domain[dom] = np.zeros((int(topo_dim[0]),), dtype=np.float64)
                    topo_sum_by_domain[dom] += proj
                    topo_count_by_domain[dom] += 1
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

            # Process robustness (optional).
            if include_robustness and p_rob is not None:
                for row in iter_jsonl(p_rob):
                    puid = str(row.get("prompt_uid") or "")
                    seen_rob.add(puid)
                    handle_row(condition="robustness", row=row)

            missing_base = sorted(base_uid_set - seen_base)
            extra_base = sorted(seen_base - base_uid_set)
            missing_rob = sorted(rob_uid_set - seen_rob)
            extra_rob = sorted(seen_rob - rob_uid_set)

            if missing_base or extra_base:
                errors.append(
                    f"{model_id}@{revision} baseline coverage mismatch: "
                    f"missing={len(missing_base)} extra={len(extra_base)}"
                )
            if include_robustness and (missing_rob or extra_rob):
                errors.append(
                    f"{model_id}@{revision} robustness coverage mismatch: "
                    f"missing={len(missing_rob)} extra={len(extra_rob)}"
                )

            validation_entry: Dict[str, Any] = {
                "model_id": model_id,
                "model_revision": revision,
                "baseline": {"missing_uids": missing_base[:20], "extra_uids": extra_base[:20], "n": len(seen_base)},
            }
            if include_robustness:
                validation_entry["robustness"] = {"missing_uids": missing_rob[:20], "extra_uids": extra_rob[:20], "n": len(seen_rob)}
            else:
                validation_entry["robustness"] = {"skipped": True}
            validation["models"].append(validation_entry)

            # Write layerwise aggregates and convergence summaries.
            def write_layerwise(condition: str, sums_ent: Dict[str, np.ndarray], sums_mar: Dict[str, np.ndarray], counts: Dict[str, int]) -> None:
                for wrapper, ent_sum in sorted(sums_ent.items(), key=lambda kv: kv[0]):
                    mar_sum = sums_mar[wrapper]
                    cnt = int(counts[wrapper])
                    for li in range(int(ent_sum.shape[0])):
                        mean_ent = float(ent_sum[li] / cnt) if cnt else None
                        mean_mar = float(mar_sum[li] / cnt) if cnt else None
                        w_lw.writerow(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "model_revision": revision,
                                "condition": condition,
                                "wrapper_id": wrapper,
                                "layer_index": int(li),
                                "mean_entropy": mean_ent,
                                "mean_margin": mean_mar,
                            }
                        )
                        w_cv.writerow(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "model_revision": revision,
                                "condition": condition,
                                "wrapper_id": wrapper,
                                "layer_index": int(li),
                                "mean_entropy": mean_ent,
                                "convergence_index": _convergence_index_from_entropy(mean_ent) if mean_ent is not None else None,
                                "mean_margin": mean_mar,
                                "n_rows": int(cnt),
                            }
                        )

            write_layerwise("baseline", lw_sum_ent_base, lw_sum_mar_base, lw_cnt_base)
            if include_robustness:
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
            if include_robustness:
                write_commit("robustness", commit_hist_rob)

            # Wrapper-wise robustness deltas relative to baseline.
            if include_robustness:
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

            # Domain topology summaries from baseline outputs only.
            sorted_domains = sorted(topo_sum_by_domain.keys())
            if not sorted_domains:
                errors.append(f"{model_id}@{revision} has no baseline topology rows")
            else:
                means = []
                counts = []
                for d in sorted_domains:
                    n = int(topo_count_by_domain[d])
                    if n <= 0:
                        continue
                    means.append(topo_sum_by_domain[d] / float(n))
                    counts.append(n)
                if means:
                    mat = np.stack(means, axis=0).astype(np.float64, copy=False)
                    coords = _project_centroids_to_2d(mat)
                    li_used = int(topo_layer_used[0] if topo_layer_used[0] is not None else -1)
                    for i, d in enumerate(sorted_domains):
                        w_tc.writerow(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "model_revision": revision,
                                "condition": "baseline",
                                "layer_index": li_used,
                                "coarse_domain": d,
                                "n_rows": int(counts[i]),
                                "centroid_norm_l2": float(np.linalg.norm(mat[i])),
                                "pc1": float(coords[i, 0]),
                                "pc2": float(coords[i, 1]),
                            }
                        )
                    for i in range(len(sorted_domains)):
                        for j in range(i + 1, len(sorted_domains)):
                            dist = float(np.linalg.norm(mat[i] - mat[j]))
                            w_td.writerow(
                                {
                                    "run_id": run_id,
                                    "model_id": model_id,
                                    "model_revision": revision,
                                    "condition": "baseline",
                                    "layer_index": li_used,
                                    "coarse_domain_a": sorted_domains[i],
                                    "coarse_domain_b": sorted_domains[j],
                                    "centroid_distance_l2": dist,
                                }
                            )

                    # Plot convergence + commitment + topology for this model.
                    base_wrapper = str(baseline_wrapper_id)
                    base_cnt = int(lw_cnt_base.get(base_wrapper, 0))
                    if base_cnt <= 0:
                        errors.append(f"{model_id}@{revision} missing baseline wrapper aggregate for plotting")
                    else:
                        mean_ent = lw_sum_ent_base[base_wrapper] / float(base_cnt)
                        mean_mar = lw_sum_mar_base[base_wrapper] / float(base_cnt)
                        t_plot = 0.1 if any(abs(float(t) - 0.1) < 1e-9 for t in thresholds) else float(thresholds[0])
                        c_hist = dict(commit_hist_base.get((base_wrapper, float(t_plot))) or {})

                        fig_cc = _next_available_path(fig_dir / f"{model_fs_id(model_id)}_convergence_commitment.png")
                        fig_top = _next_available_path(fig_dir / f"{model_fs_id(model_id)}_domain_topology.png")
                        try:
                            _plot_convergence_commitment(
                                out_path=fig_cc,
                                model_id=model_id,
                                mean_entropy=np.asarray(mean_ent, dtype=np.float64),
                                mean_margin=np.asarray(mean_mar, dtype=np.float64),
                                commitment_hist=c_hist,
                                commit_threshold=float(t_plot),
                            )
                            _plot_domain_topology(
                                out_path=fig_top,
                                model_id=model_id,
                                layer_index=li_used,
                                domains=sorted_domains,
                                counts=[int(x) for x in counts],
                                coords=np.asarray(coords, dtype=np.float64),
                            )
                            figure_files.extend([fig_cc, fig_top])
                        except Exception as exc:
                            errors.append(f"{model_id}@{revision} plotting failure: {type(exc).__name__}: {exc}")
                else:
                    errors.append(f"{model_id}@{revision} could not compute topology means")

            if include_robustness:
                per_model_summary[model_id] = {
                    "baseline_overall": _finalize_agg(agg_base_overall),
                    "robustness_overall": _finalize_agg(agg_rob_overall),
                }
            else:
                per_model_summary[model_id] = {
                    "baseline_overall": _finalize_agg(agg_base_overall),
                    "robustness_overall": {"skipped": True},
                }

    ok = not errors

    inputs_obj: Dict[str, Any] = {
        "baseline_manifest_path": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "include_robustness": bool(include_robustness),
    }
    if robustness_manifest is not None:
        inputs_obj["robustness_manifest_path"] = str(robustness_manifest)
        inputs_obj["robustness_manifest_sha256"] = sha256_file(robustness_manifest)

    report = {
        "pass": bool(ok),
        "run_id": run_id,
        "thresholds": [float(x) for x in thresholds],
        "topology_layer_requested": int(topology_layer),
        "inputs": inputs_obj,
        "artifacts": {
            "per_prompt_metrics_csv": str(per_prompt_csv),
            "per_prompt_metrics_sha256": sha256_file(per_prompt_csv),
            "layerwise_aggregates_csv": str(layerwise_csv),
            "layerwise_aggregates_sha256": sha256_file(layerwise_csv),
            "robustness_deltas_csv": str(robust_delta_csv),
            "robustness_deltas_sha256": sha256_file(robust_delta_csv),
            "commitment_hist_csv": str(commit_csv),
            "commitment_hist_sha256": sha256_file(commit_csv),
            "convergence_by_layer_csv": str(convergence_csv),
            "convergence_by_layer_sha256": sha256_file(convergence_csv),
            "domain_topology_centroids_csv": str(topology_centroids_csv),
            "domain_topology_centroids_sha256": sha256_file(topology_centroids_csv),
            "domain_topology_pairwise_distances_csv": str(topology_dists_csv),
            "domain_topology_pairwise_distances_sha256": sha256_file(topology_dists_csv),
            "figure_files": [
                {"path": str(p), "sha256": sha256_file(p)}
                for p in sorted(figure_files, key=lambda x: str(x))
            ],
        },
        "per_model_summary": per_model_summary,
        "validation": validation,
        "errors": errors[:50],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report
