#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import base_parser, run_v2_root_for, write_json, write_text_atomic
from sow.v2.assets import write_phase_diagram, write_sha_manifest, write_trajectory_plots

REPO_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_FILES = [
    "00_run_experiment.report.json",
    "00a_generate_baseline_outputs.report.json",
    "layerwise.parquet",
    "decision_metrics.parquet",
    "prompt_types.parquet",
    "type_counts.json",
    "basin_gap.parquet",
    "spans.jsonl",
    "span_effects.parquet",
    "span_labels.parquet",
    "span_paraphrase_stability.parquet",
    "tracing_scalars.parquet",
    "attention_mass_by_span.parquet",
    "attention_contrib_by_span.parquet",
    "ablation_results.parquet",
    "patching_results.parquet",
    "span_deletion_causal.parquet",
    "negative_controls.parquet",
    "01_extract_baseline.report.json",
    "02_compute_decision_metrics.report.json",
    "03_classify_trajectories.report.json",
    "04_region_analysis.report.json",
    "05_span_counterfactuals.report.json",
    "06_select_tracing_subset.report.json",
    "07_run_tracing.report.json",
    "08_attention_and_mlp_decomposition.report.json",
    "09_causal_tests.report.json",
    "10_causal_validation_tools.report.json",
    "11_generate_paper_assets.report.json",
    "meta/readiness_audit.json",
    "meta/run_start_metadata_snapshot.json",
]

REQUIRED_METADATA_FILES = [
    REPO_ROOT / "configs" / "experiment_v2.yaml",
    REPO_ROOT / "docs" / "PAPER_OBJECTIVE_V3.md",
    REPO_ROOT / "docs" / "IMPLEMENTATION_PLAN_V3.md",
    REPO_ROOT / "docs" / "MODEL_NUANCES_V2.md",
    REPO_ROOT / "docs" / "PREREGISTERED_HYPOTHESES_V3.md",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _evaluate_metadata_immutability(snapshot_path: Path) -> dict[str, object]:
    if not snapshot_path.exists():
        return {
            "pass": False,
            "reason": "missing_snapshot",
            "snapshot_path": str(snapshot_path),
            "changed_files": [],
            "missing_files": [str(snapshot_path)],
        }

    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "pass": False,
            "reason": f"invalid_snapshot_json: {exc}",
            "snapshot_path": str(snapshot_path),
            "changed_files": [],
            "missing_files": [],
        }

    records = payload.get("files") or []
    changed_files: list[str] = []
    missing_files: list[str] = []
    for rec in records:
        p = Path(str((rec or {}).get("path") or ""))
        expected = str((rec or {}).get("sha256") or "")
        if not p.exists():
            missing_files.append(str(p))
            continue
        got = _sha256(p)
        if (not expected) or (got != expected):
            changed_files.append(str(p))

    pass_flag = (len(changed_files) == 0) and (len(missing_files) == 0)
    return {
        "pass": bool(pass_flag),
        "reason": None if pass_flag else "metadata_changed_or_missing",
        "snapshot_path": str(snapshot_path),
        "changed_files": sorted(changed_files),
        "missing_files": sorted(missing_files),
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_routing_vs_contribution_figure(*, out_root: Path, out_path: Path) -> bool:
    mass_path = out_root / "attention_mass_by_span.parquet"
    contrib_path = out_root / "attention_contrib_by_span.parquet"
    if (not mass_path.exists()) or (not contrib_path.exists()):
        return False
    mass = pd.read_parquet(mass_path)
    contrib = pd.read_parquet(contrib_path)
    if mass.empty or contrib.empty:
        return False

    mg = mass.groupby("span_label", as_index=False)["attention_mass"].mean()
    cg = contrib.groupby("span_label", as_index=False)["attention_contribution"].mean()
    merged = mg.merge(cg, on="span_label", how="outer").fillna(0.0)
    if merged.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = list(range(len(merged)))
    ax1.bar(x, merged["attention_mass"].tolist(), alpha=0.55, label="routing mass")
    ax1.set_xlabel("span")
    ax1.set_ylabel("attention mass")
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["span_label"].tolist(), rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, merged["attention_contribution"].tolist(), color="#d62728", marker="o", label="decision contribution")
    ax2.set_ylabel("attention contribution")
    ax1.set_title("Attention Routing vs Decision Contribution")
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_causal_panel_figure(*, out_root: Path, out_path: Path) -> bool:
    ablation_path = out_root / "ablation_results.parquet"
    patching_path = out_root / "patching_results.parquet"
    if (not ablation_path.exists()) or (not patching_path.exists()):
        return False
    ablation = pd.read_parquet(ablation_path)
    patching = pd.read_parquet(patching_path)
    if ablation.empty or patching.empty:
        return False

    ab = ablation.groupby("component", as_index=False)["delta_shift"].mean()
    pa = patching.groupby("component", as_index=False)["delta_shift"].mean()
    if ab.empty or pa.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].bar(ab["component"].tolist(), ab["delta_shift"].tolist(), color="#1f77b4", alpha=0.8)
    axes[0].set_title("Ablation Mean Delta Shift")
    axes[0].set_ylabel("delta shift")
    axes[0].grid(alpha=0.2)
    axes[1].bar(pa["component"].tolist(), pa["delta_shift"].tolist(), color="#2ca02c", alpha=0.8)
    axes[1].set_title("Patching Mean Delta Shift")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Causal Validation Panel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    ap = base_parser("V2: generate final paper assets and final artifact folder")
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    metrics_path = out_root / "decision_metrics.parquet"
    types_path = out_root / "prompt_types.parquet"
    if not metrics_path.exists() or not types_path.exists():
        raise SystemExit("missing decision_metrics.parquet or prompt_types.parquet")

    metrics = pd.read_parquet(metrics_path)
    if args.model_name:
        metrics = metrics[metrics["model_id"].str.contains(args.model_name, na=False)]
    if args.max_prompts > 0 and not metrics.empty:
        keep = set(metrics["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        metrics = metrics[metrics["prompt_uid"].isin(keep)]

    fig_dir = out_root / "figures"
    write_phase_diagram(metrics, out_path=fig_dir / "fig_phase_diagram_modelX.png")
    traj_files = write_trajectory_plots(metrics, out_dir=fig_dir)
    routing_fig = fig_dir / "fig_routing_vs_contribution_modelX.png"
    causal_fig = fig_dir / "fig_causal_panel_modelX.png"
    region_fig = out_root / "fig_region_entry_exit.png"
    _write_routing_vs_contribution_figure(out_root=out_root, out_path=routing_fig)
    _write_causal_panel_figure(out_root=out_root, out_path=causal_fig)

    final_root = REPO_ROOT / "artifacts" / "final_result_v2" / args.run_id
    final_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_root / "meta" / "run_start_metadata_snapshot.json"
    metadata_immutability = _evaluate_metadata_immutability(snapshot_path)

    copied = []
    missing = []
    deferred_required = {"11_generate_paper_assets.report.json"}
    for name in REQUIRED_FILES:
        if name in deferred_required:
            continue
        src = out_root / name
        dst = final_root / name
        if _copy_if_exists(src, dst):
            copied.append(str(dst))
        else:
            missing.append(name)

    for src in REQUIRED_METADATA_FILES:
        dst = final_root / "metadata" / src.name
        if _copy_if_exists(src, dst):
            copied.append(str(dst))
        else:
            missing.append(f"metadata/{src.name}")

    for fig in [
        fig_dir / "fig_phase_diagram_modelX.png",
        *traj_files,
        region_fig,
        routing_fig,
        causal_fig,
        *sorted(fig_dir.glob("fig_motion_decomposition_*.png")),
    ]:
        if fig.exists():
            dst = final_root / "figures" / fig.name
            _copy_if_exists(fig, dst)
            copied.append(str(dst))

    methods_md = final_root / "methods.md"
    results_md = final_root / "results_summary.md"
    write_text_atomic(
        methods_md,
        "# Methods\n\n"
        "This run follows PAPER_OBJECTIVE_V3 and IMPLEMENTATION_PLAN_V3 with deterministic metrics, trajectory typing, "
        "span operationalization, tracing decomposition, and causal validation controls.\n",
    )
    write_text_atomic(
        results_md,
        "# Results Summary\n\n"
        f"- Decision-metric rows: {int(metrics.shape[0])}\n"
        f"- Prompt count: {int(metrics['prompt_uid'].nunique()) if not metrics.empty else 0}\n"
        f"- Generated figures: {len(list((final_root / 'figures').glob('*.png')))}\n",
    )

    gates = {
        "required_files_present": len(missing) == 0,
        "metadata_immutability": bool(metadata_immutability.get("pass")),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])

    final_report = {
        "pass": len(failing_gates) == 0,
        "run_id": args.run_id,
        "artifact_root": str(final_root),
        "copied_files": copied,
        "missing_required_files": missing,
        "metadata_immutability": metadata_immutability,
        "gates": gates,
        "failing_gates": failing_gates,
    }

    stage11_report_path = out_root / "11_generate_paper_assets.report.json"
    write_json(stage11_report_path, final_report)
    stage11_bundle_path = final_root / "11_generate_paper_assets.report.json"
    copied.append(str(stage11_bundle_path))
    final_report["copied_files"] = copied
    write_json(stage11_bundle_path, final_report)
    write_json(final_root / "final_report.json", final_report)
    write_sha_manifest(root_dir=final_root, out_path=final_root / "sha256_manifest.json")
    print(str(final_root))
    return 0 if len(failing_gates) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
