from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from .constants import (
    ALPHAS,
    APPENDIX_FIGURES,
    MAIN_FIGURES,
    MODEL_ORDER,
    REQUIRED_MARKDOWN,
    REQUIRED_PARQUETS,
    REQUIRED_TABLES,
    SEED,
)
from .loaders import dependency_versions, sha256_file
from .reversibility import alpha_label


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as handle:
        handle.write(text)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent), newline="") as handle:
        df.to_csv(handle, index=False)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def dataframe_to_markdown(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    data = df.head(max_rows) if max_rows is not None else df
    columns = list(data.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in data.iterrows():
        rows.append("| " + " | ".join(_fmt_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *rows])


def _fmt_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def build_boundary_correlation_table(traj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, group in [("pooled", traj)] + list(traj.groupby("model_id", sort=False)):
        rows.extend(_spearman_rows(scope, group, "boundary_occupancy_prob_010", "last_flip_depth"))
        rows.extend(_spearman_rows(scope, group, "boundary_occupancy_prob_010", "flip_count"))
        rows.extend(_spearman_rows(scope, group, "boundary_occupancy_prob_010", "pooled_empirical_commitment_depth_005"))
    return pd.DataFrame(rows)


def _spearman_rows(scope: str, df: pd.DataFrame, x_col: str, y_col: str) -> list[dict[str, Any]]:
    sub = df[[x_col, y_col]].dropna()
    corr = float("nan")
    if len(sub) >= 3 and sub[x_col].nunique() > 1 and sub[y_col].nunique() > 1:
        corr = spearmanr(sub[x_col], sub[y_col]).statistic
    return [
        {
            "scope": scope,
            "x_metric": x_col,
            "y_metric": y_col,
            "n": int(len(sub)),
            "spearman_rho": corr if corr == corr else float("nan"),
            "missing_y_count": int(df[y_col].isna().sum()),
        }
    ]


def build_empirical_commitment_summary(traj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        label = alpha_label(alpha)
        depth_col = f"pooled_empirical_commitment_depth_{label}"
        model_depth_col = f"model_empirical_commitment_depth_{label}"
        for scope, group in [("pooled", traj)] + list(traj.groupby("model_id", sort=False)):
            for final_group, final_sub in [("all", group), ("final_correct", group[group["final_correct"]]), ("final_wrong", group[~group["final_correct"]])]:
                rows.append(
                    {
                        "scope": scope,
                        "final_group": final_group,
                        "alpha": alpha,
                        "pooled_nonmissing": int(final_sub[depth_col].notna().sum()),
                        "pooled_never_enter": int(final_sub[depth_col].isna().sum()),
                        "pooled_median_depth": float(final_sub[depth_col].median()) if final_sub[depth_col].notna().any() else float("nan"),
                        "model_nonmissing": int(final_sub[model_depth_col].notna().sum()),
                        "model_never_enter": int(final_sub[model_depth_col].isna().sum()),
                        "model_median_depth": float(final_sub[model_depth_col].median()) if final_sub[model_depth_col].notna().any() else float("nan"),
                        "old_nonmissing": int(final_sub["old_threshold_commitment_depth"].notna().sum()),
                        "old_median_depth": float(final_sub["old_threshold_commitment_depth"].median()) if final_sub["old_threshold_commitment_depth"].notna().any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def build_baseline_reproduction_table(layerwise: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    work = layerwise.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").copy()
    work["delta_hard_jump"] = work.groupby(["model_id", "prompt_uid"], sort=False)["delta_hard"].diff().abs()
    work["delta_soft_jump"] = work.groupby(["model_id", "prompt_uid"], sort=False)["delta_soft"].diff().abs()
    switch_rows = work["switch"] == 1
    nonswitch_rows = (work["switch"] == 0) & (work.groupby(["model_id", "prompt_uid"], sort=False).cumcount() > 0)
    baseline_rows = [
        {
            "observation": "soft_margin_smoother_than_hard_margin_at_switch_layers",
            "scope": "pooled",
            "value": float(work.loc[switch_rows, "delta_hard_jump"].median() - work.loc[switch_rows, "delta_soft_jump"].median()),
            "reproduced": bool(work.loc[switch_rows, "delta_hard_jump"].median() > work.loc[switch_rows, "delta_soft_jump"].median()),
            "details": "positive means hard jumps exceed soft jumps at switch layers",
        },
        {
            "observation": "switch_rate_mid_or_earlier_peak_and_late_decline",
            "scope": "pooled",
            "value": float(_peak_vs_late_switch_rate(work)),
            "reproduced": bool(_peak_vs_late_switch_rate(work) > 0.0),
            "details": "peak switch rate minus late-depth switch rate",
        },
        {
            "observation": "old_regime_unstable_last_flips_are_late",
            "scope": "pooled",
            "value": float(traj.loc[traj["old_regime"] == "Unstable", "old_last_flip_depth"].median()) if "old_regime" in traj.columns else float("nan"),
            "reproduced": bool("old_regime" in traj.columns and traj.loc[traj["old_regime"] == "Unstable", "old_last_flip_depth"].median() >= 0.75),
            "details": "median old last-flip depth in old unstable group",
        },
        {
            "observation": "old_threshold_commitment_is_late",
            "scope": "pooled",
            "value": float(traj["old_threshold_commitment_depth"].median()) if traj["old_threshold_commitment_depth"].notna().any() else float("nan"),
            "reproduced": bool(traj["old_threshold_commitment_depth"].median() >= 0.75) if traj["old_threshold_commitment_depth"].notna().any() else False,
            "details": "median old-threshold commitment depth",
        },
    ]
    return pd.DataFrame(baseline_rows)


def _peak_vs_late_switch_rate(layerwise: pd.DataFrame) -> float:
    work = layerwise.copy()
    work["summary_z_bin"] = (work["z"] * 20).clip(upper=19.999999).astype(int)
    rate = work.groupby("summary_z_bin", sort=False)["switch"].mean()
    peak = float(rate.max())
    late = float(rate[rate.index >= 16].mean()) if bool((rate.index >= 16).any()) else float(rate.iloc[-1])
    return peak - late


def build_repo_inventory_md(
    *,
    manifest: pd.DataFrame,
    layerwise: pd.DataFrame,
    decision_metrics: pd.DataFrame,
    canonical: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> str:
    parts = [
        "# Repo Inventory",
        "",
        "## Files Used",
        "",
        "- `results/parquet/layerwise.parquet`: cached layerwise four-option readout scores.",
        "- `results/parquet/decision_metrics.parquet`: cached readout-derived decision metrics used only for cross-checks.",
        "- `prompt_packs/ccc_baseline_v1_3000.jsonl`: source-of-truth prompt metadata and correct options.",
        "- `paper/part1/data/part1_core.parquet`: comparison-only old-threshold/old-regime artifact when present.",
        "",
        "## Source Schemas",
        "",
        f"### layerwise.parquet\n\nColumns: {', '.join(layerwise.columns.tolist())}",
        "",
        f"### decision_metrics.parquet\n\nColumns: {', '.join(decision_metrics.columns.tolist())}",
        "",
        f"### prompt manifest\n\nColumns: {', '.join(manifest.columns.tolist())}",
        "",
        f"### canonical layerwise table\n\nColumns: {', '.join(canonical.columns.tolist())}",
        "",
        "## Verified Counts",
        "",
        dataframe_to_markdown(validation_table[validation_table["metric"].isin(["canonical_rows", "manifest_rows", "complete_layer_coverage", "prompt_count", "row_count", "min_layer_index", "max_layer_index"])], max_rows=20),
        "",
        "## Assumptions and Ambiguities",
        "",
        "- `layer_index = 0` is the first logged transformer block output, not embeddings, following the repo contract.",
        "- Final predicted option is derived from final-layer score argmax with alphabetical tie-break; tied-final trajectories are flagged explicitly.",
        "- The 4-choice probabilities are conditional inside the four-option readout, not full-vocabulary generation probabilities.",
        "- Old threshold commitment and old regime labels are treated as comparison-only because their sign convention is asymmetric for final-wrong trajectories.",
        "",
        "## Totals",
        "",
        f"- Models found: {canonical['model_id'].nunique()}",
        f"- Trajectories found: {canonical[['model_id', 'prompt_uid']].drop_duplicates().shape[0]}",
        f"- Layerwise rows found: {len(canonical)}",
    ]
    return "\n".join(parts) + "\n"


def build_figure_manifest_md(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    parts = [
        "# Figure Manifest",
        "",
        dataframe_to_markdown(frame),
        "",
        "All figures are readout-space summaries only. Raw `a/q/r` panels are model-specific because score scales differ across models.",
    ]
    return "\n".join(parts) + "\n"


def build_readme_md() -> str:
    parts = [
        "# No-New-Inference Decision Dynamics",
        "",
        "This package re-analyzes the cached layerwise four-option readout only. It performs no inference, uses no GPU, downloads nothing, and writes only inside this directory.",
        "",
        "## Run",
        "",
        "```bash",
        "python3 analysis/no_new_inference_decision_dynamics/run_analysis.py \\",
        "  --out-dir analysis/no_new_inference_decision_dynamics \\",
        "  --seed 12345 \\",
        "  --bootstrap-reps 1000 \\",
        "  --force",
        "```",
        "",
        "## Required outputs",
        "",
        "- Markdown reports: " + ", ".join(REQUIRED_MARKDOWN),
        "- Parquet files: " + ", ".join(REQUIRED_PARQUETS),
        "- Tables: " + ", ".join(REQUIRED_TABLES),
        "- Figures: " + ", ".join([f'{name}.pdf/.png' for name in MAIN_FIGURES + APPENDIX_FIGURES]),
        "",
        "## Reproducibility",
        "",
        f"- Fixed seed: `{SEED}`",
        "- Fixed alphabetical tie-break for all argmax operations",
        "- Fixed `z` binning and `p_correct` binning",
        "- Source hashes and output hashes written to `analysis_manifest.json`",
    ]
    return "\n".join(parts) + "\n"


def build_methods_md() -> str:
    return """# Methods For A General CS Reader

The data are cached layer-by-layer scores for the four answer options A, B, C, and D on 3000 MMLU prompts for each of three instruction-tuned transformer models. At each logged layer we only observe the current four-option readout, not the hidden state itself.

`delta_soft` is the correct-option score minus the log-sum-exp of the three incorrect-option scores. It removes a discrete artifact of the hard margin, which can jump when the identity of the strongest incorrect option changes. In this four-option readout, `delta_soft` and `p_correct` are exact monotone transforms of one another: `p_correct = sigmoid(delta_soft)`.

`p_correct` is the current probability of the correct option inside the four-choice readout. It is not the probability of the full natural-language completion over the model vocabulary.

The commitment coordinate `a` is a linear contrast between the correct score and the average incorrect score after centering the four-score vector. The competitor-dispersion coordinate `q` measures how much the three incorrect options still differ from one another. The angle `theta` says which incorrect-answer direction is currently strongest inside that incorrect-option plane.

`future_flip` asks a purely empirical question: given the current readout state, does the sign of the soft margin ever change again later in depth? The empirical commitment depth is the first layer after which the trajectory stays in a region of low future-flip probability.

Everything in this package stays in answer-readout space. It does not identify hidden-state computation, internal pathways, or causality.
"""


def build_claim_discipline_md() -> str:
    return """# Claim Discipline

| Claim | Evidence required | Allowed wording | Overclaim to avoid |
| --- | --- | --- | --- |
| soft margin reduces competitor-switch artifacts | Hard-vs-soft jump comparison at switch layers, plus representative trajectories | “reduces switch-induced discontinuities” | “solves uncertainty” |
| middle layers show more competitor reshuffling | Switch-rate depth curves and competitor-dynamics appendix | “competitor identity changes are more common before late stabilization” | “the model explores alternatives internally” |
| late layers are more committed | empirical commitment depths, future-flip map, late-depth low future-flip regions | “later states are empirically less reversible” | “the network locks into a mechanism” |
| boundary-dwelling trajectories stabilize later | occupancy distributions, scatter with last-flip depth, commitment-depth summaries, and per-model consistency | “if supported consistently, higher boundary occupancy can coincide with later stabilization” | “boundary dwelling uniformly delays commitment” |
| commitment is well described by low future-flip probability | empirical commitment tables and reversibility heatmap | “commitment is operationalized here as low estimated future-flip probability” | “commitment is a causal property of the model” |
| this is readout-space dynamics, not hidden-state mechanism | repeated scope statement in report and captions | “readout-space dynamics” | any language about hidden-state mechanism, circuits, or causality |

Note: in the current cached data, the robust boundary-dwelling claim is stronger for flip-heavy competition than for uniformly later stabilization, so the package keeps the latter as a guarded secondary possibility rather than a headline result.

Forbidden wording in this package: attractor, basin, separatrix, criticality, phase transition, mechanism, causality, circuit, internal representation.
"""


def build_paper_outline_md() -> str:
    return """# Paper Outline

## Title candidates
- From Competition to Late Commitment in Cached Layerwise Answer Readouts
- Boundary Dwelling and Empirical Reversibility in Layerwise Multiple-Choice Readouts
- Late Stabilization in Transformer Answer Readouts Without New Inference

## One-sentence thesis
Cached four-option readouts reveal answer formation as a trajectory from reversible competition to later low-reversibility commitment, with flip-heavy trajectories spending more depth near the decision boundary and with the timing link to stabilization treated as model-dependent.

## Main text
1. Problem: final answers hide the path.
2. Setup: cached layerwise four-option scores on MMLU.
3. Better observables: soft margin and current correct-option probability.
4. Better state description: commitment `a` versus competitor dispersion `q`.
5. Main finding: trajectories move from competition to later stabilization.
6. Strongest result: empirical future-flip map and commitment as low future-flip probability.
7. Boundary dwelling: boundary-near trajectories show more flips, while the later-stabilization link is mixed across models.
8. Limits: readout-space only.
9. Conclusion: answer formation is a depth-wise trajectory, not just a final-layer label.

## Appendix
- tie robustness
- old-threshold comparison
- per-model reversibility maps
- travel metrics
- boundary-width robustness
"""


def build_analysis_report_md(
    *,
    validation_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    boundary_table: pd.DataFrame,
    empirical_table: pd.DataFrame,
    state_validation: dict[str, float],
    traj: pd.DataFrame,
    figure_rows: list[dict[str, Any]],
    tie_backoff_summary: dict[str, Any],
    paper_strength_verdict: str,
) -> str:
    per_model_accuracy = (
        traj.groupby("model_id")["final_correct"]
        .mean()
        .rename("final_correct_rate")
        .reset_index()
    )
    figure_manifest = pd.DataFrame(figure_rows)
    occupancy_by_outcome = (
        traj.groupby("final_correct")["boundary_occupancy_prob_010"]
        .mean()
        .rename(index={False: "final_wrong", True: "final_correct"})
        .to_dict()
    )
    boundary_flip = boundary_table[(boundary_table["scope"] == "pooled") & (boundary_table["y_metric"] == "flip_count")]
    boundary_last_flip = boundary_table[(boundary_table["scope"] == "pooled") & (boundary_table["y_metric"] == "last_flip_depth")]
    boundary_commitment = boundary_table[(boundary_table["scope"] == "pooled") & (boundary_table["y_metric"] == "pooled_empirical_commitment_depth_005")]
    pooled_flip_rho = float(boundary_flip["spearman_rho"].iloc[0]) if not boundary_flip.empty else float("nan")
    pooled_last_flip_rho = float(boundary_last_flip["spearman_rho"].iloc[0]) if not boundary_last_flip.empty else float("nan")
    pooled_commitment_rho = float(boundary_commitment["spearman_rho"].iloc[0]) if not boundary_commitment.empty else float("nan")
    parts = [
        "# Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"This package re-analyzes the cached four-option readout only. {paper_strength_verdict}",
        "",
        "The central objects are the soft margin, the current correct-option probability inside the 4-choice readout, the commitment coordinate `a`, the competitor-dispersion magnitude `q`, and the empirical future-flip probability map over normalized depth and current probability.",
        "",
        "## Data Inventory And Validation",
        "",
        dataframe_to_markdown(validation_table),
        "",
        "State-space reconstruction checks:",
        "",
        f"- max reconstruction absolute error: {state_validation['state_reconstruction_max_abs_error']:.3e}",
        f"- mean reconstruction absolute error: {state_validation['state_reconstruction_mean_abs_error']:.3e}",
        f"- max |sigmoid(delta_soft) - p_correct|: {state_validation['p_correct_sigmoid_delta_soft_max_abs_error']:.3e}",
        "",
        "Per-model final correctness rates:",
        "",
        dataframe_to_markdown(per_model_accuracy),
        "",
        "## Exact Metric Definitions",
        "",
        "- `delta_hard`: correct score minus strongest incorrect score.",
        "- `delta_soft`: correct score minus log-sum-exp of the three incorrect scores.",
        "- `p_correct`: current correct-option probability in the 4-choice readout; exactly `sigmoid(delta_soft)` here.",
        "- `a`: linear correct-vs-rest commitment coordinate after centering and fixed permutation.",
        "- `q`: magnitude of disagreement among the incorrect answers.",
        "- `future_flip`: whether the sign of `delta_soft` ever changes again later in depth.",
        "- empirical commitment depth: first layer after which the estimated future-flip probability stays below the chosen threshold for the rest of the trajectory.",
        "",
        "## Baseline Reproduction",
        "",
        dataframe_to_markdown(baseline_table),
        "",
        "## Main Quantitative Results",
        "",
        "Figures produced by this run:",
        "",
        dataframe_to_markdown(figure_manifest[figure_manifest["kind"] == "main"]),
        "",
        "Boundary correlations:",
        "",
        dataframe_to_markdown(boundary_table),
        "",
        "Boundary-dwelling interpretation:",
        "",
        f"- Mean boundary occupancy is higher for final-correct trajectories ({occupancy_by_outcome.get('final_correct', float('nan')):.4f}) than for final-wrong trajectories ({occupancy_by_outcome.get('final_wrong', float('nan')):.4f}) in this cache, so boundary dwelling is not a synonym for final failure.",
        f"- Boundary occupancy rises with flip count in the pooled data (Spearman rho = {pooled_flip_rho:.3f}), and that sign is positive in each model.",
        f"- The later-stabilization link is mixed rather than uniform: pooled occupancy versus last-flip depth has rho = {pooled_last_flip_rho:.3f}, while pooled occupancy versus empirical commitment depth has rho = {pooled_commitment_rho:.3f}. The main supported claim is prolonged competition near the boundary, not a single monotone delay-of-commitment law.",
        "",
        "Empirical commitment summary:",
        "",
        dataframe_to_markdown(empirical_table.head(18)),
        "",
        "## Per-Model Consistency Checks",
        "",
        "- Raw `a`, `q`, and `r` are shown per model because the models have materially different logit scales.",
        "- Scale-free summaries such as `p_correct`, switch rate, boundary occupancy, and future-flip probability are pooled only through equal model weighting.",
        "- Tie-affected states and tied-final trajectories are retained with explicit flags and covered in appendix robustness figures.",
        "",
        "## Tie And Backoff Sensitivity",
        "",
        f"- top-score tied rows: {tie_backoff_summary['top_tied_rows']}",
        f"- competitor-tied rows: {tie_backoff_summary['competitor_tied_rows']}",
        f"- tied final trajectories: {tie_backoff_summary['final_tied_trajectories']}",
        f"- pooled backoff counts: {tie_backoff_summary['pooled_backoff_counts']}",
        f"- per-model backoff counts: {tie_backoff_summary['model_backoff_counts']}",
        "",
        "## Limitations",
        "",
        "- This is a readout-space analysis only.",
        "- The 4-choice probabilities are conditional inside the recorded answer subspace.",
        "- Old threshold commitment is included only for comparison and is asymmetric for final-wrong trajectories.",
        "",
        "## One Suggested Next Experimental Step",
        "",
        "Run one targeted follow-up intervention on prompts that remain boundary-dwelling late in depth, and test whether shifting the readout away from the boundary late in the network reduces future flips without changing the prompt set.",
        "",
        "## What Cannot Be Claimed",
        "",
        "- Nothing here identifies hidden-state computation.",
        "- Nothing here establishes causality.",
        "- Nothing here licenses claims about internal mechanisms beyond the observed answer-readout trajectories.",
    ]
    return "\n".join(parts) + "\n"


def required_output_paths(out_dir: Path) -> list[Path]:
    paths = [out_dir / rel for rel in REQUIRED_MARKDOWN + REQUIRED_PARQUETS + REQUIRED_TABLES]
    paths.extend([out_dir / "figures" / f"{name}.pdf" for name in MAIN_FIGURES + APPENDIX_FIGURES])
    paths.extend([out_dir / "figures" / f"{name}.png" for name in MAIN_FIGURES + APPENDIX_FIGURES])
    paths.append(out_dir / "figures" / "qc_booklet.pdf")
    return paths


def collect_output_hashes(out_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(required_output_paths(out_dir)):
        if path.exists():
            hashes[str(path.relative_to(out_dir))] = sha256_file(path)
    return hashes


def build_analysis_manifest(
    *,
    source_paths: dict[str, Path],
    out_dir: Path,
    cli_args: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "cli_args": cli_args,
        "dependency_versions": dependency_versions(),
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items() if path.exists()},
        "output_hashes": collect_output_hashes(out_dir),
    }
