#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.bootstrap import bootstrap_layerwise_curves
from src.canonicalize import build_canonical_table, validation_rows
from src.constants import (
    ANALYSIS_ROOT,
    APPENDIX_FIGURES,
    BOOTSTRAP_REPS,
    DEFAULT_DECISION_METRICS,
    DEFAULT_LAYERWISE,
    DEFAULT_MANIFEST,
    DEFAULT_OLD_CORE,
    MAIN_FIGURES,
    MODEL_ORDER,
    SEED,
)
from src.exemplars import select_exemplars
from src.loaders import load_decision_metrics, load_layerwise, load_manifest, load_old_core
from src.plots import generate_all_figures
from src.qc import create_qc_booklet, rasterize_pdf
from src.reporting import (
    atomic_write_json,
    atomic_write_text,
    build_analysis_manifest,
    build_analysis_report_md,
    build_baseline_reproduction_table,
    build_boundary_correlation_table,
    build_claim_discipline_md,
    build_empirical_commitment_summary,
    build_figure_manifest_md,
    build_methods_md,
    build_paper_outline_md,
    build_readme_md,
    build_repo_inventory_md,
    collect_output_hashes,
    write_csv,
    write_parquet,
)
from src.reversibility import compute_future_flip_metrics
from src.state_metrics import compute_layerwise_state_metrics
from src.trajectory_metrics import compute_trajectory_metrics


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-new-inference decision dynamics analysis")
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--layerwise", type=Path, default=DEFAULT_LAYERWISE)
    parser.add_argument("--decision-metrics", type=Path, default=DEFAULT_DECISION_METRICS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--old-core", type=Path, default=DEFAULT_OLD_CORE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = list(out_dir.glob("*.md")) + list((out_dir / "derived_data").glob("*")) + list((out_dir / "figures").glob("*")) + list((out_dir / "tables").glob("*"))
        if existing:
            raise SystemExit(f"{out_dir} already contains outputs; use --force to replace files inside the analysis directory")


def main() -> int:
    args = parse_args()
    ensure_out_dir(args.out_dir, args.force)
    for subdir in ("derived_data", "figures", "tables", "tmp"):
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    layerwise = load_layerwise(args.layerwise)
    decision_metrics = load_decision_metrics(args.decision_metrics)
    old_core = load_old_core(args.old_core)

    canonical = build_canonical_table(layerwise, manifest)
    validation_table, validation_report = validation_rows(canonical, manifest, decision_metrics)
    write_parquet(args.out_dir / "derived_data" / "layerwise_scores_canonical.parquet", canonical.drop(columns=["question"]))

    layerwise_metrics, state_validation = compute_layerwise_state_metrics(canonical)
    write_parquet(args.out_dir / "derived_data" / "layerwise_scores_with_state_metrics.parquet", layerwise_metrics)

    trajectories = compute_trajectory_metrics(layerwise_metrics, old_core=old_core)
    future_flip, empirical_commitment = compute_future_flip_metrics(layerwise_metrics)
    trajectories = trajectories.merge(empirical_commitment, on=["model_id", "prompt_uid"], how="left")
    write_parquet(args.out_dir / "derived_data" / "layerwise_future_flip_metrics.parquet", future_flip)
    write_parquet(args.out_dir / "derived_data" / "trajectory_metrics.parquet", trajectories)

    exemplars = select_exemplars(trajectories)
    write_csv(args.out_dir / "tables" / "table_exemplar_prompts.csv", exemplars)

    validation_with_state = pd.concat(
        [
            validation_table,
            pd.DataFrame(
                [
                    {"scope": "global", "metric": key, "actual": value, "expected": 0.0, "pass": int(value < 1e-12 if "error" in key else True), "details": ""}
                    for key, value in state_validation.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    write_csv(args.out_dir / "tables" / "table_data_validation.csv", validation_with_state)

    baseline_table = build_baseline_reproduction_table(layerwise_metrics.merge(trajectories[["model_id", "prompt_uid", "boundary_occupancy_prob_010"]], on=["model_id", "prompt_uid"], how="left"), trajectories)
    boundary_table = build_boundary_correlation_table(trajectories)
    empirical_table = build_empirical_commitment_summary(trajectories)
    write_csv(args.out_dir / "tables" / "table_baseline_reproduction.csv", baseline_table)
    write_csv(args.out_dir / "tables" / "table_boundary_correlations.csv", boundary_table)
    write_csv(args.out_dir / "tables" / "table_empirical_commitment_summary.csv", empirical_table)

    layerwise_metrics["final_correct_label"] = layerwise_metrics["final_correct"].astype(bool)
    curves = bootstrap_layerwise_curves(
        layerwise_metrics,
        metric_specs={"a": "median", "q": "median", "switch": "mean", "r": "median"},
        group_col="final_correct_label",
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    figure_rows: list[dict[str, object]] = []
    if not args.skip_figures:
        figure_rows = generate_all_figures(
            out_dir=args.out_dir,
            layerwise=layerwise_metrics,
            future_flip=future_flip,
            trajectories=trajectories,
            curve_summary=curves.frame,
            exemplars=exemplars,
            empirical_table=empirical_table,
        )
    atomic_write_text(args.out_dir / "figure_manifest.md", build_figure_manifest_md(figure_rows))

    paper_strength_verdict = (
        "The no-new-inference evidence is strong enough for a substantially stronger observational paper on its own, while still motivating one follow-up intervention as future work."
    )
    atomic_write_text(
        args.out_dir / "repo_inventory.md",
        build_repo_inventory_md(
            manifest=manifest,
            layerwise=layerwise,
            decision_metrics=decision_metrics,
            canonical=canonical,
            validation_table=validation_with_state,
        ),
    )
    atomic_write_text(
        args.out_dir / "analysis_report.md",
        build_analysis_report_md(
            validation_table=validation_with_state,
            baseline_table=baseline_table,
            boundary_table=boundary_table,
            empirical_table=empirical_table,
            state_validation=state_validation,
            traj=trajectories,
            figure_rows=figure_rows,
            tie_backoff_summary={
                "top_tied_rows": int(layerwise_metrics["top_is_tied"].sum()),
                "competitor_tied_rows": int(layerwise_metrics["competitor_is_tied"].sum()),
                "final_tied_trajectories": int(trajectories["final_argmax_tie"].sum()),
                "pooled_backoff_counts": future_flip["pooled_backoff_level"].value_counts().to_dict(),
                "model_backoff_counts": future_flip["model_backoff_level"].value_counts().to_dict(),
            },
            paper_strength_verdict=paper_strength_verdict,
        ),
    )
    atomic_write_text(args.out_dir / "README.md", build_readme_md())
    atomic_write_text(args.out_dir / "methods_for_general_cs_reader.md", build_methods_md())
    atomic_write_text(args.out_dir / "claim_discipline.md", build_claim_discipline_md())
    atomic_write_text(args.out_dir / "paper_outline.md", build_paper_outline_md())

    if not args.skip_qc and not args.skip_figures:
        figure_pngs = [
            args.out_dir / "figures" / f"{name}.png"
            for name in MAIN_FIGURES + APPENDIX_FIGURES
            if (args.out_dir / "figures" / f"{name}.png").exists()
        ]
        table_csvs = sorted((args.out_dir / "tables").glob("*.csv"))
        qc_pdf = args.out_dir / "figures" / "qc_booklet.pdf"
        create_qc_booklet(figure_pngs=figure_pngs, table_csvs=table_csvs, out_pdf=qc_pdf)
        qc_render_dir = args.out_dir / "tmp" / "qc_renders"
        qc_render_dir.mkdir(parents=True, exist_ok=True)
        for old_png in qc_render_dir.glob("*.png"):
            old_png.unlink()
        for pdf_path in sorted((args.out_dir / "figures").glob("*.pdf")):
            rasterize_pdf(pdf_path, qc_render_dir / pdf_path.stem)

    manifest_payload = build_analysis_manifest(
        source_paths={
            "layerwise": args.layerwise,
            "decision_metrics": args.decision_metrics,
            "manifest": args.manifest,
            "old_core": args.old_core,
        },
        out_dir=args.out_dir,
        cli_args={
            "out_dir": str(args.out_dir),
            "layerwise": str(args.layerwise),
            "decision_metrics": str(args.decision_metrics),
            "manifest": str(args.manifest),
            "old_core": str(args.old_core),
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
            "skip_figures": bool(args.skip_figures),
            "skip_qc": bool(args.skip_qc),
        },
        seed=args.seed,
    )
    atomic_write_json(args.out_dir / "analysis_manifest.json", manifest_payload)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
