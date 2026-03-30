#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs" / "runs" / "paper_merged_9x1000_20260330T122258Z"
PAPER_DIR = ROOT / "paper" / "colm2026"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"
GEN_DIR = PAPER_DIR / "generated"
PROVENANCE_PATH = PAPER_DIR / "provenance.json"

PRIMARY_READOUT = "semantic_exact"
PRIMARY_PERMUTATION = "identity"
PRIMARY_ANCHOR = "final_winner_anchor"
SEED = 20260330

DATASET_ORDER = [
    "ai2_arc_challenge",
    "commonsenseqa_validation",
    "mmlu_abstract_algebra",
]
DATASET_DISPLAY = {
    "ai2_arc_challenge": "ARC-Challenge",
    "commonsenseqa_validation": "CommonsenseQA",
    "mmlu_abstract_algebra": "MMLU Abstract Algebra",
}
FAMILY_ORDER = [
    "qwen2.5_instruct",
    "gemma2_it",
    "llama_instruct",
]
FAMILY_DISPLAY = {
    "qwen2.5_instruct": "Qwen 2.5 Instruct",
    "gemma2_it": "Gemma 2 IT",
    "llama_instruct": "Llama Instruct",
}
MODEL_ORDER = [
    "qwen2p5_1p5b_instruct",
    "qwen2p5_3b_instruct",
    "qwen2p5_7b_instruct",
    "gemma2_2b_it",
    "gemma2_9b_it",
    "gemma2_27b_it",
    "llama3p2_1b_instruct",
    "llama3p2_3b_instruct",
    "llama3p1_8b_instruct",
]
MODEL_DISPLAY = {
    "qwen2p5_1p5b_instruct": "Qwen 2.5 1.5B",
    "qwen2p5_3b_instruct": "Qwen 2.5 3B",
    "qwen2p5_7b_instruct": "Qwen 2.5 7B",
    "gemma2_2b_it": "Gemma 2 2B",
    "gemma2_9b_it": "Gemma 2 9B",
    "gemma2_27b_it": "Gemma 2 27B",
    "llama3p2_1b_instruct": "Llama 3.2 1B",
    "llama3p2_3b_instruct": "Llama 3.2 3B",
    "llama3p1_8b_instruct": "Llama 3.1 8B",
}
READOUT_DISPLAY = {
    "semantic_exact": "Semantic exact",
    "templated_semantic": "Templated semantic",
    "letter_label": "Letter label",
}
PERMUTATION_DISPLAY = {
    "identity": "Identity",
    "reverse": "Reverse",
}

COLORS = {
    "ink": "#1C2733",
    "plausible": "#4C7F94",
    "committed": "#1E2B39",
    "sand": "#D7C6A8",
    "grid": "#E6E1D8",
    "muted_text": "#5F615B",
    "qwen2.5_instruct": "#B76343",
    "gemma2_it": "#8B8A4D",
    "llama_instruct": "#375C88",
    "identity": "#1E2B39",
    "reverse": "#6C8EA4",
}


@dataclass
class PaperContext:
    metrics_df: pd.DataFrame
    score_df: pd.DataFrame
    canonical_examples: dict[str, dict[str, Any]]
    primary_df: pd.DataFrame
    positivity_df: pd.DataFrame
    family_summary_df: pd.DataFrame
    bootstrap_summary: dict[str, Any]
    qc_summary: dict[str, Any]
    selection_manifest: dict[str, Any]
    run_manifest: dict[str, Any]
    model_registry: list[dict[str, Any]]
    readout_cfg: dict[str, Any]
    prompt_cfg: dict[str, Any]
    controls_cfg: dict[str, Any]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Book Antiqua", "URW Palladio L", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": "#3D403B",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def fmt_pct(value: float, *, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}\\%"


def fmt_float(value: float, *, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pretty_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(name, name)


def pretty_family(name: str) -> str:
    return FAMILY_DISPLAY.get(name, name)


def pretty_model(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def read_configs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    readout_cfg = yaml.safe_load((ROOT / "configs" / "readouts.yaml").read_text())
    prompt_cfg = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text())
    controls_cfg = yaml.safe_load((ROOT / "configs" / "controls.yaml").read_text())
    return models_cfg, readout_cfg, prompt_cfg, controls_cfg


def load_context() -> PaperContext:
    models_cfg, readout_cfg, prompt_cfg, controls_cfg = read_configs()
    run_manifest = load_json(RUN_DIR / "run_manifest.json")
    source_models = set(run_manifest["source_models"])
    model_registry = [row for row in models_cfg if row["name"] in source_models]
    model_registry.sort(key=lambda row: MODEL_ORDER.index(row["name"]))

    canonical_examples = {
        row["example_id"]: row for row in load_jsonl(RUN_DIR / "prepared_data" / "canonical_examples.jsonl")
    }
    metrics_df = normalize_metrics(load_jsonl(RUN_DIR / "metrics" / "primary_metrics.jsonl"))
    score_df = pd.DataFrame(load_jsonl(RUN_DIR / "metrics" / "score_manifest.jsonl"))
    positivity_df = pd.DataFrame(load_json(RUN_DIR / "stats" / "model_dataset_positivity.json"))
    family_summary_df = pd.DataFrame(load_json(RUN_DIR / "stats" / "family_positivity_summary.json"))
    bootstrap_summary = load_json(RUN_DIR / "stats" / "bootstrap_summary.json")
    qc_summary = load_json(RUN_DIR / "qc" / "qc_summary.json")
    selection_manifest = load_json(RUN_DIR / "manifests" / "prepared_selection_manifest.json")
    primary_df = positivity_df[
        (positivity_df["criterion_label"] == "primary_rule")
        & (positivity_df["anchor"] == PRIMARY_ANCHOR)
        & (positivity_df["readout_id"] == PRIMARY_READOUT)
        & (positivity_df["permutation_id"] == PRIMARY_PERMUTATION)
    ].copy()
    primary_df["dataset_name"] = pd.Categorical(primary_df["dataset_name"], DATASET_ORDER, ordered=True)
    primary_df["model_name"] = pd.Categorical(primary_df["model_name"], MODEL_ORDER, ordered=True)
    primary_df = primary_df.sort_values(["dataset_name", "model_name"]).reset_index(drop=True)
    return PaperContext(
        metrics_df=metrics_df,
        score_df=score_df,
        canonical_examples=canonical_examples,
        primary_df=primary_df,
        positivity_df=positivity_df,
        family_summary_df=family_summary_df,
        bootstrap_summary=bootstrap_summary,
        qc_summary=qc_summary,
        selection_manifest=selection_manifest,
        run_manifest=run_manifest,
        model_registry=model_registry,
        readout_cfg=readout_cfg,
        prompt_cfg=prompt_cfg,
        controls_cfg=controls_cfg,
    )


def normalize_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"][PRIMARY_ANCHOR]
        profile = metrics["winner_runner_contrast_by_layer"]
        n_layers = len(profile)
        denom = max(1, n_layers - 1)
        out.append(
            {
                "cache_key": row["cache_key"],
                "dataset_name": row["dataset_name"],
                "example_id": row["example_id"],
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "model_size": row["model_size"],
                "readout_id": row["readout_id"],
                "permutation_id": row["permutation_id"],
                "anchor_option": metrics["anchor_option"],
                "d_id": metrics["d_id"],
                "d_top1": metrics["d_top1"],
                "d_margin": metrics["d_margin"],
                "delta_margin": metrics["delta_margin"],
                "delta_top1": metrics["delta_top1"],
                "final_contrast": metrics["final_contrast"],
                "write_center_of_mass": metrics["write_center_of_mass"],
                "late_write_mass_last_20pct": metrics["late_write_mass_last_20pct"],
                "winner_runner_contrast_by_layer": profile,
                "positive_write_increments": metrics["positive_write_increments"],
                "n_layers": n_layers,
                "norm_d_id": None if metrics["d_id"] is None else metrics["d_id"] / denom,
                "norm_d_top1": None if metrics["d_top1"] is None else metrics["d_top1"] / denom,
                "norm_d_margin": None if metrics["d_margin"] is None else metrics["d_margin"] / denom,
                "norm_delta_margin": None
                if metrics["d_id"] is None or metrics["d_margin"] is None
                else (metrics["d_margin"] - metrics["d_id"]) / denom,
            }
        )
    df = pd.DataFrame(out)
    df["dataset_name"] = pd.Categorical(df["dataset_name"], DATASET_ORDER, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df["model_family"] = pd.Categorical(df["model_family"], FAMILY_ORDER, ordered=True)
    return df.sort_values(["dataset_name", "model_name", "example_id", "readout_id", "permutation_id"]).reset_index(drop=True)


def semantic_identity_valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["readout_id"] == PRIMARY_READOUT)
        & (df["permutation_id"] == PRIMARY_PERMUTATION)
        & (df["d_margin"].notna())
    ].copy()


def compute_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(values <= threshold).mean() for threshold in grid])


def short_question(text: str, *, width: int = 44, max_lines: int = 2) -> str:
    compact = " ".join(text.split())
    lines = textwrap.wrap(compact, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    trimmed[-1] = trimmed[-1].rstrip(". ") + "..."
    return "\n".join(trimmed)


def select_exemplars(context: PaperContext) -> list[pd.Series]:
    df = semantic_identity_valid(context.metrics_df)
    df = df[df["final_contrast"].fillna(-1.0) > 0.20].copy()
    df["selection_score"] = (
        df["norm_delta_margin"].fillna(-1.0)
        + 0.08 * df["final_contrast"].fillna(0.0)
        - 0.03 * df["norm_d_id"].fillna(1.0)
    )
    preferences = [
        ("qwen2.5_instruct", "ai2_arc_challenge"),
        ("gemma2_it", "commonsenseqa_validation"),
        ("llama_instruct", "commonsenseqa_validation"),
        ("qwen2.5_instruct", "mmlu_abstract_algebra"),
    ]
    chosen: list[pd.Series] = []
    used: set[tuple[str, str, str]] = set()
    for family, dataset in preferences:
        subset = df[(df["model_family"] == family) & (df["dataset_name"] == dataset)].sort_values(
            ["selection_score", "final_contrast"], ascending=False
        )
        for _, row in subset.iterrows():
            key = (str(row["model_name"]), str(row["dataset_name"]), str(row["example_id"]))
            if key in used:
                continue
            chosen.append(row)
            used.add(key)
            break
    if len(chosen) < 4:
        extras = df.sort_values(["selection_score", "final_contrast"], ascending=False)
        for _, row in extras.iterrows():
            key = (str(row["model_name"]), str(row["dataset_name"]), str(row["example_id"]))
            if key in used:
                continue
            chosen.append(row)
            used.add(key)
            if len(chosen) == 4:
                break
    return chosen[:4]


def build_flagship_figure(context: PaperContext) -> dict[str, Any]:
    df = semantic_identity_valid(context.metrics_df)
    grid = np.linspace(0.0, 1.0, 401)
    pooled_plausible = compute_cdf(df["norm_d_id"].to_numpy(), grid)
    pooled_committed = compute_cdf(df["norm_d_margin"].to_numpy(), grid)
    checkpoints = [0.2, 0.5, 0.8]
    checkpoint_stats = []
    for checkpoint in checkpoints:
        plausible = float((df["norm_d_id"] <= checkpoint).mean())
        committed = float((df["norm_d_margin"] <= checkpoint).mean())
        checkpoint_stats.append(
            {
                "x": checkpoint,
                "plausible": plausible,
                "committed": committed,
            }
        )

    fig = plt.figure(figsize=(7.05, 6.7))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[2.35, 0.9, 1.45], hspace=0.32, wspace=0.2)
    ax_pooled = fig.add_subplot(gs[0, :])
    ax_strip = fig.add_subplot(gs[1, :], sharex=ax_pooled)
    facet_axes = [fig.add_subplot(gs[2, idx], sharex=ax_pooled, sharey=ax_pooled) for idx in range(3)]

    ax_pooled.fill_between(grid, pooled_committed, pooled_plausible, color=COLORS["sand"], alpha=0.45, zorder=1)
    ax_pooled.plot(grid, pooled_plausible, color=COLORS["plausible"], lw=2.9, zorder=3)
    ax_pooled.plot(grid, pooled_committed, color=COLORS["committed"], lw=2.9, zorder=4)
    ax_pooled.grid(True, axis="y")
    ax_pooled.set_xlim(0.0, 1.0)
    ax_pooled.set_ylim(0.0, 1.02)
    ax_pooled.set_ylabel("Share of evaluable cases")
    ax_pooled.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_pooled.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_pooled.set_title("Plausibility comes early. Commitment comes late.")

    label_x = 0.73
    ax_pooled.text(
        label_x,
        np.interp(label_x, grid, pooled_plausible) + 0.03,
        "Eventual answer\nalready plausible",
        color=COLORS["plausible"],
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax_pooled.text(
        label_x,
        max(0.05, np.interp(label_x, grid, pooled_committed) - 0.08),
        "Eventual answer\nalready committed",
        color=COLORS["committed"],
        fontsize=8.5,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    callout_positions = {
        0.2: (0.06, 0.64),
        0.5: (0.33, 0.56),
        0.8: (0.58, 0.44),
    }
    for stats in checkpoint_stats:
        x = stats["x"]
        plausible = stats["plausible"]
        committed = stats["committed"]
        mid = committed + 0.5 * (plausible - committed)
        tx, ty = callout_positions[x]
        ax_pooled.annotate(
            f"{int(x * 100)}% depth\n{plausible * 100:.0f}% plausible\n{committed * 100:.0f}% committed",
            xy=(x, mid),
            xytext=(tx, ty),
            fontsize=7.8,
            color=COLORS["ink"],
            ha="left",
            va="top",
            arrowprops={"arrowstyle": "-", "color": COLORS["muted_text"], "lw": 0.8},
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
                "linewidth": 0.7,
                "alpha": 0.94,
            },
        )

    rng = np.random.default_rng(SEED)
    sample_n = min(320, len(df))
    strip_sample = df.sample(sample_n, random_state=SEED).copy()
    strip_sample["midpoint"] = 0.5 * (strip_sample["norm_d_id"] + strip_sample["norm_d_margin"])
    strip_sample = strip_sample.sort_values(["midpoint", "norm_delta_margin"]).reset_index(drop=True)
    y_positions = np.linspace(0.05, 0.95, len(strip_sample))
    for y, (_, row) in zip(y_positions, strip_sample.iterrows()):
        ax_strip.plot(
            [row["norm_d_id"], row["norm_d_margin"]],
            [y, y],
            color=COLORS[str(row["model_family"])],
            alpha=0.15,
            lw=0.8,
            solid_capstyle="round",
        )
    ax_strip.set_ylim(0.0, 1.0)
    ax_strip.set_yticks([])
    ax_strip.grid(False)
    ax_strip.text(
        0.0,
        1.03,
        "Sampled model-prompt pairs: interval from first plausibility to stable commitment on the same depth axis",
        transform=ax_strip.transAxes,
        fontsize=8,
        color=COLORS["muted_text"],
        ha="left",
        va="bottom",
    )

    for axis, dataset in zip(facet_axes, DATASET_ORDER):
        subset = df[df["dataset_name"] == dataset]
        plausible = compute_cdf(subset["norm_d_id"].to_numpy(), grid)
        committed = compute_cdf(subset["norm_d_margin"].to_numpy(), grid)
        axis.fill_between(grid, committed, plausible, color=COLORS["sand"], alpha=0.36, zorder=1)
        axis.plot(grid, plausible, color=COLORS["plausible"], lw=2.0, zorder=3)
        axis.plot(grid, committed, color=COLORS["committed"], lw=2.0, zorder=4)
        axis.grid(True, axis="y")
        axis.set_title(f"{pretty_dataset(dataset)} (n={len(subset):,})", fontsize=8.8)
        axis.set_ylim(0.0, 1.02)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_xlabel("Depth")
    facet_axes[0].set_ylabel("Share")

    figure_path = FIG_DIR / "figure1_flagship_plausibility_commitment.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {
        "figure_path": figure_path.name,
        "pooled_at_20_plausible": checkpoint_stats[0]["plausible"],
        "pooled_at_20_committed": checkpoint_stats[0]["committed"],
        "pooled_at_50_plausible": checkpoint_stats[1]["plausible"],
        "pooled_at_50_committed": checkpoint_stats[1]["committed"],
        "pooled_at_80_plausible": checkpoint_stats[2]["plausible"],
        "pooled_at_80_committed": checkpoint_stats[2]["committed"],
        "pooled_valid_cases": int(len(df)),
    }


def build_forest_figure(context: PaperContext) -> dict[str, Any]:
    df = context.primary_df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 5.2), sharey=True)
    ci_hi_values = []
    for item in df["bootstrap"]:
        if isinstance(item, dict):
            ci_hi_values.append(float(item.get("ci_hi", 0.0)))
    x_max = max(max(ci_hi_values, default=0.0), 35.0)
    x_max = 5 * math.ceil((x_max + 3.0) / 5.0)
    y_positions = np.arange(len(MODEL_ORDER))[::-1]
    model_to_y = {model: y for model, y in zip(MODEL_ORDER, y_positions)}

    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset_name"] == dataset].copy()
        axis.axvline(0.0, color=COLORS["grid"], lw=1.0, zorder=0)
        for family_idx in range(1, len(FAMILY_ORDER)):
            boundary_y = y_positions[len(FAMILY_ORDER[:family_idx]) * 3 - 1] - 0.5
            axis.axhline(boundary_y, color=COLORS["grid"], lw=0.8, zorder=0)
        for _, row in subset.iterrows():
            y = model_to_y[str(row["model_name"])]
            family = str(row["model_family"])
            color = COLORS[family]
            if row["status"] == "evaluated":
                ci = row["bootstrap"]
                axis.plot(
                    [ci["ci_lo"], ci["ci_hi"]],
                    [y, y],
                    color=color,
                    lw=2.0,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.scatter(
                    [row["observed_estimate"]],
                    [y],
                    color=color,
                    edgecolors="white",
                    linewidths=0.8,
                    s=38,
                    zorder=3,
                )
            else:
                axis.scatter([x_max - 2.0], [y], marker="x", color=COLORS["muted_text"], s=34, zorder=3)
                axis.text(
                    x_max - 1.2,
                    y,
                    "insuf.",
                    fontsize=7.2,
                    color=COLORS["muted_text"],
                    ha="left",
                    va="center",
                )
        axis.set_title(pretty_dataset(dataset), fontsize=9.5)
        axis.set_xlim(-1.0, x_max)
        axis.set_ylim(-0.8, len(MODEL_ORDER) - 0.2)
        axis.xaxis.set_major_locator(MultipleLocator(5))
        axis.grid(True, axis="x")
        axis.set_xlabel("Median commitment lag (layers)")
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([pretty_model(model) for model in MODEL_ORDER])
    axes[0].set_ylabel("Model")
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)

    family_centers = {
        "qwen2.5_instruct": np.mean([model_to_y[MODEL_ORDER[idx]] for idx in range(0, 3)]),
        "gemma2_it": np.mean([model_to_y[MODEL_ORDER[idx]] for idx in range(3, 6)]),
        "llama_instruct": np.mean([model_to_y[MODEL_ORDER[idx]] for idx in range(6, 9)]),
    }
    for family, y in family_centers.items():
        axes[0].text(
            -0.56,
            y,
            pretty_family(family),
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.1,
            color=COLORS[family],
            fontweight="bold",
        )

    figure_path = FIG_DIR / "figure2_primary_forest_plot.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {"figure_path": figure_path.name}


def build_controls_figure(context: PaperContext) -> dict[str, Any]:
    metrics_df = context.metrics_df.copy()
    metrics_df["valid"] = metrics_df["delta_margin"].notna()
    valid_rates = (
        metrics_df.groupby(["readout_id", "permutation_id"], observed=True)["valid"]
        .mean()
        .reset_index()
        .sort_values(["readout_id", "permutation_id"])
    )

    pivot = context.score_df.pivot_table(
        index=["model_name", "dataset_name", "example_id", "permutation_id"],
        columns="readout_id",
        values="final_winner",
        aggfunc="first",
    ).reset_index()
    agreement_rows = []
    for permutation_id in ["identity", "reverse"]:
        subset = pivot[pivot["permutation_id"] == permutation_id].copy()
        semantic = subset["semantic_exact"]
        for control in ["templated_semantic", "letter_label"]:
            agreement_rows.append(
                {
                    "permutation_id": permutation_id,
                    "comparison": control,
                    "agreement": float((subset[control] == semantic).mean()),
                }
            )
    agreement_df = pd.DataFrame(agreement_rows)
    all_same_rate = float(
        ((pivot["semantic_exact"] == pivot["templated_semantic"]) & (pivot["semantic_exact"] == pivot["letter_label"])).mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), gridspec_kw={"wspace": 0.28})

    x_positions = np.arange(3)
    offsets = {"identity": -0.03, "reverse": 0.03}
    for permutation_id in ["identity", "reverse"]:
        subset = valid_rates[valid_rates["permutation_id"] == permutation_id].set_index("readout_id")
        y = [subset.loc[readout, "valid"] for readout in ["semantic_exact", "templated_semantic", "letter_label"]]
        axes[0].plot(
            x_positions + offsets[permutation_id],
            y,
            marker="o",
            ms=5,
            lw=2.0,
            color=COLORS[permutation_id],
            linestyle="-" if permutation_id == "identity" else "--",
        )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([READOUT_DISPLAY[key] for key in ["semantic_exact", "templated_semantic", "letter_label"]])
    axes[0].set_ylim(0.45, 1.0)
    axes[0].grid(True, axis="y")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_ylabel("Valid-rate for $\\Delta_{\\mathrm{margin}}$")
    axes[0].set_title("A. Control readouts change coverage", fontsize=9.5)
    axes[0].text(
        0.01,
        0.02,
        "Solid = identity, dashed = reverse",
        transform=axes[0].transAxes,
        fontsize=7.6,
        color=COLORS["muted_text"],
        ha="left",
        va="bottom",
    )

    x_positions = np.arange(2)
    for permutation_id in ["identity", "reverse"]:
        subset = agreement_df[agreement_df["permutation_id"] == permutation_id].set_index("comparison")
        y = [subset.loc[key, "agreement"] for key in ["templated_semantic", "letter_label"]]
        axes[1].plot(
            x_positions + offsets[permutation_id],
            y,
            marker="o",
            ms=5,
            lw=2.0,
            color=COLORS[permutation_id],
            linestyle="-" if permutation_id == "identity" else "--",
        )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(["Templated vs.\nsemantic", "Label vs.\nsemantic"])
    axes[1].set_ylim(0.35, 1.0)
    axes[1].grid(True, axis="y")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].set_ylabel("Final-winner agreement")
    axes[1].set_title("B. Controls are not interchangeable", fontsize=9.5)
    axes[1].text(
        0.0,
        0.03,
        f"All three readouts pick the same final winner for {all_same_rate * 100:.0f}% of fully paired cases.",
        transform=axes[1].transAxes,
        fontsize=7.6,
        color=COLORS["muted_text"],
        ha="left",
        va="bottom",
    )

    figure_path = FIG_DIR / "figure3_control_readout_comparison.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {
        "figure_path": figure_path.name,
        "valid_rates": {
            row["readout_id"] + "__" + row["permutation_id"]: float(row["valid"])
            for row in valid_rates.to_dict("records")
        },
        "agreement_rates": {
            row["comparison"] + "__" + row["permutation_id"]: float(row["agreement"])
            for row in agreement_df.to_dict("records")
        },
        "all_same_rate": all_same_rate,
    }


def build_exemplars_figure(context: PaperContext) -> dict[str, Any]:
    exemplars = select_exemplars(context)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.3), gridspec_kw={"hspace": 0.48, "wspace": 0.26})
    fig.subplots_adjust(top=0.84)
    axes_flat = axes.flatten()

    legend_handles = [
        Line2D([0], [0], color="#444444", lw=1.8, label="winner-runner contrast"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["plausible"], markersize=6, label="first plausible"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="#6D6D6D", markersize=6, label="stable top-1"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=COLORS["committed"], markersize=6, label="margin-stable"),
    ]

    chosen_records: list[dict[str, Any]] = []
    for axis, row in zip(axes_flat, exemplars):
        profile = np.array(row["winner_runner_contrast_by_layer"], dtype=float)
        x = np.linspace(0.0, 1.0, len(profile))
        family = str(row["model_family"])
        axis.axhline(0.0, color=COLORS["grid"], lw=0.9, zorder=0)
        axis.plot(x, profile, color=COLORS[family], lw=2.0, zorder=2)
        axis.fill_between(x, 0.0, profile, where=profile >= 0.0, color=COLORS[family], alpha=0.10)
        markers = [
            ("d_id", "o", COLORS["plausible"]),
            ("d_top1", "^", "#6D6D6D"),
            ("d_margin", "s", COLORS["committed"]),
        ]
        for field, marker, color in markers:
            depth = row[field]
            if depth is None:
                continue
            idx = int(depth)
            axis.scatter(
                [x[idx]],
                [profile[idx]],
                marker=marker,
                s=32,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        dataset_name = str(row["dataset_name"])
        example = context.canonical_examples[str(row["example_id"])]
        axis.set_title(
            f"{pretty_model(str(row['model_name']))} on {pretty_dataset(dataset_name)}",
            fontsize=8.9,
        )
        axis.text(
            0.02,
            0.98,
            short_question(example["question_canonical"]),
            transform=axis.transAxes,
            fontsize=7.2,
            color=COLORS["muted_text"],
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-1.05, 1.05)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_xlabel("Normalized depth")
        axis.set_ylabel("Winner-runner contrast")
        chosen_records.append(
            {
                "model_name": str(row["model_name"]),
                "dataset_name": dataset_name,
                "example_id": str(row["example_id"]),
                "question": example["question_canonical"],
            }
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handlelength=1.9,
        columnspacing=1.1,
    )
    figure_path = FIG_DIR / "figure4_prompt_level_exemplars.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {"figure_path": figure_path.name, "exemplars": chosen_records}


def build_selection_figure(context: PaperContext) -> dict[str, Any]:
    selections = pd.DataFrame(context.selection_manifest["selections"])
    selections["dataset"] = pd.Categorical(selections["dataset"], DATASET_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.4), gridspec_kw={"wspace": 0.28})
    option_markers = {3: "s", 4: "o", 5: "^"}
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = selections[selections["dataset"] == dataset].copy()
        subset = subset.sort_values("question_length").reset_index(drop=True)
        x = np.arange(len(subset))
        colors = []
        for reason in subset["selection_reason"]:
            if str(reason).startswith("majority_option_count_quantile"):
                colors.append(COLORS["sand"])
            else:
                colors.append("#D9D9D9")
        for option_count in sorted(subset["option_count"].unique()):
            opt_subset = subset[subset["option_count"] == option_count]
            idx = opt_subset.index.to_numpy()
            axis.scatter(
                x[idx],
                opt_subset["question_length"],
                s=24,
                marker=option_markers.get(int(option_count), "o"),
                color=[colors[i] for i in idx],
                edgecolors=COLORS["ink"],
                linewidths=0.25,
                alpha=0.92,
            )
        axis.set_title(pretty_dataset(dataset), fontsize=9.0)
        axis.set_xlabel("Selected example rank")
        axis.grid(True, axis="y")
        axis.set_ylabel("Question length")
    figure_path = FIG_DIR / "appendix_selection_profile.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {"figure_path": figure_path.name}


def build_qc_figure(context: PaperContext) -> dict[str, Any]:
    qc = context.qc_summary
    token_counts = qc["token_flag_counts"]
    example_counts = qc["example_flag_counts"]
    labels = [
        "Tokenization collision",
        "Long tokenization",
        "Duplicate option text",
        "Noncanonical labels",
    ]
    values = [
        token_counts.get("tokenization_collision", 0),
        token_counts.get("suspiciously_long_option_tokenization", 0),
        example_counts.get("duplicate_option_text", 0),
        example_counts.get("noncanonical_option_labels", 0),
    ]
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=[COLORS["sand"], COLORS["plausible"], "#D1CECA", "#B8B6B0"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(True, axis="x")
    ax.set_xlabel("Count")
    ax.set_title("QC-visible anomalies are logged, not hidden", fontsize=9.5)
    figure_path = FIG_DIR / "appendix_qc_anomalies.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    return {"figure_path": figure_path.name}


def write_tables(context: PaperContext, figure_stats: dict[str, Any]) -> None:
    write_text(TABLE_DIR / "table1_design_summary.tex", build_design_table(context))
    write_text(TABLE_DIR / "table2_primary_summary.tex", build_primary_summary_table(context))
    write_text(TABLE_DIR / "appendix_model_registry.tex", build_model_registry_table(context))
    write_text(TABLE_DIR / "appendix_full_positivity.tex", build_full_positivity_table(context))
    write_text(TABLE_DIR / "appendix_qc_summary.tex", build_qc_table(context))
    write_text(TABLE_DIR / "appendix_readout_defs.tex", build_readout_table(context))


def build_design_table(context: PaperContext) -> str:
    canonical_examples = pd.DataFrame(context.canonical_examples.values())
    dataset_counts = canonical_examples.groupby("dataset_name").size().to_dict()
    rows = [
        (
            "Models",
            "Nine instruction-tuned checkpoints from Qwen 2.5 Instruct (1.5B, 3B, 7B), "
            "Gemma 2 IT (2B, 9B, 27B), and Llama Instruct (3.2 1B, 3.2 3B, 3.1 8B).",
        ),
        (
            "Dataset slice",
            f"{pretty_dataset('ai2_arc_challenge')} ({dataset_counts.get('ai2_arc_challenge', 0)}), "
            f"{pretty_dataset('commonsenseqa_validation')} ({dataset_counts.get('commonsenseqa_validation', 0)}), "
            f"and {pretty_dataset('mmlu_abstract_algebra')} ({dataset_counts.get('mmlu_abstract_algebra', 0)}).",
        ),
        (
            "Prompting",
            "Model-specific chat or raw prompt templates. Every prompt ends with the literal suffix "
            "\\texttt{\"Final answer:\"}.",
        ),
        (
            "Readouts",
            "Primary: the exact option text after \\texttt{\"Final answer:\"}, for example "
            "\\texttt{\"photosynthesis\"}. Controls: the literal continuation "
            "\\texttt{\"The answer is \\{option\\_text\\}\"} and the bare answer letter, such as "
            "\\texttt{\"A\"}.",
        ),
        (
            "Permutations",
            "Identity ordering and reverse ordering, kept paired on the same example subset.",
        ),
        (
            "Primary rule",
            "For each model-dataset cell, report the median commitment lag "
            "$\\Delta_{\\mathrm{margin}} = d_{\\mathrm{margin}} - d_{\\mathrm{id}}$ with a grouped-bootstrap 95\\% confidence interval; "
            "the cell is positive iff the lower confidence bound is greater than zero.",
        ),
    ]
    body = "\n".join(
        f"{latex_escape(label)} & {value} \\\\" for label, value in rows
    )
    return (
        "\\begin{tabularx}{\\linewidth}{@{}lX@{}}\n"
        "\\toprule\n"
        "\\textbf{Aspect} & \\textbf{Configuration}\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )


def build_primary_summary_table(context: PaperContext) -> str:
    primary = context.primary_df.copy()
    summary_rows = []
    for dataset in DATASET_ORDER:
        subset = primary[primary["dataset_name"] == dataset]
        summary_rows.append(
            (
                pretty_dataset(dataset),
                int((subset["positive"] == True).sum()),
                int((subset["status"] == "evaluated").sum()),
                int((subset["status"] == "insufficient_n").sum()),
            )
        )
    summary_rows.append(
        (
            "All datasets",
            int((primary["positive"] == True).sum()),
            int((primary["status"] == "evaluated").sum()),
            int((primary["status"] == "insufficient_n").sum()),
        )
    )
    body = "\n".join(
        f"{latex_escape(name)} & {positive} & {evaluated} & {insufficient} \\\\"
        for name, positive, evaluated, insufficient in summary_rows
    )
    return (
        "\\begin{tabular}{@{}lccc@{}}\n"
        "\\toprule\n"
        "Dataset & Positive & Evaluable & Insufficient \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def build_model_registry_table(context: PaperContext) -> str:
    rows = []
    for row in context.model_registry:
        repo_name = str(row["model_id"]).split("/")[-1]
        revision_short = str(row["model_revision"])[:12]
        rows.append(
            (
                pretty_family(str(row["family"])),
                pretty_model(str(row["name"])),
                latex_escape(str(row["size"])),
                latex_escape(repo_name),
                f"\\texttt{{{latex_escape(revision_short)}}}",
            )
        )
    body = "\n".join(" & ".join(parts) + r" \\" for parts in rows)
    return (
        "\\begin{tabularx}{\\linewidth}{@{}lllXX@{}}\n"
        "\\toprule\n"
        "Family & Model & Size & Repository & Revision prefix \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )


def build_full_positivity_table(context: PaperContext) -> str:
    rows = []
    for _, row in context.primary_df.iterrows():
        if row["status"] == "evaluated":
            estimate = f"{int(row['observed_estimate'])}"
            ci = f"[{int(row['bootstrap']['ci_lo'])}, {int(row['bootstrap']['ci_hi'])}]"
            decision = "positive" if row["positive"] else "negative"
        else:
            estimate = "--"
            ci = "--"
            decision = "insufficient n"
        rows.append(
            (
                pretty_dataset(str(row["dataset_name"])),
                pretty_model(str(row["model_name"])),
                f"{int(row['n_valid'])}/{int(row['n_total'])}",
                estimate,
                ci,
                decision,
            )
        )
    body = "\n".join(" & ".join(latex_escape(part) for part in parts) + r" \\" for parts in rows)
    return (
        "\\begin{tabularx}{\\textwidth}{@{}llcccc@{}}\n"
        "\\toprule\n"
        "Dataset & Model & Valid / total & Median lag & 95\\% CI & Primary decision \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )


def build_qc_table(context: PaperContext) -> str:
    qc = context.qc_summary
    rows = [
        ("Prepared examples", str(qc["total_examples"])),
        ("Token audits", str(qc["total_token_audits"])),
        ("Tokenization collisions", str(qc["token_flag_counts"].get("tokenization_collision", 0))),
        ("Suspiciously long option tokenizations", str(qc["token_flag_counts"].get("suspiciously_long_option_tokenization", 0))),
        ("Duplicate option text", str(qc["example_flag_counts"].get("duplicate_option_text", 0))),
        ("Noncanonical option labels", str(qc["example_flag_counts"].get("noncanonical_option_labels", 0))),
        ("Unpaired rows", str(qc["paired_coverage"].get("unpaired_rows", 0))),
        ("Failure rows", str(qc["total_failures"])),
    ]
    body = "\n".join(f"{latex_escape(k)} & {latex_escape(v)} \\\\" for k, v in rows)
    return (
        "\\begin{tabular}{@{}lc@{}}\n"
        "\\toprule\n"
        "QC item & Count \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def build_readout_table(context: PaperContext) -> str:
    readout_rows = [
        ("semantic_exact", "Exact option text after \\texttt{\"Final answer:\"}, e.g. \\texttt{\"photosynthesis\"}", "Mean log-probability"),
        ("templated_semantic", "Literal continuation \\texttt{\"The answer is \\{option\\_text\\}\"}", "Mean log-probability"),
        ("letter_label", "Bare answer letter after \\texttt{\"Final answer:\"}, e.g. \\texttt{\"A\"}", "Mean log-probability"),
    ]
    body = "\n".join(f"{latex_escape(name)} & {surface} & {latex_escape(norm)} \\\\" for name, surface, norm in readout_rows)
    return (
        "\\begin{tabularx}{\\linewidth}{@{}lXl@{}}\n"
        "\\toprule\n"
        "Readout & Continuation surface & Normalization \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_paper_numbers(context: PaperContext, figure_stats: dict[str, Any]) -> None:
    primary = context.primary_df.copy()
    secondary = context.positivity_df[
        (context.positivity_df["criterion_label"] == "secondary_robustness")
        & (context.positivity_df["anchor"] == PRIMARY_ANCHOR)
        & (context.positivity_df["readout_id"] == PRIMARY_READOUT)
        & (context.positivity_df["permutation_id"] == PRIMARY_PERMUTATION)
    ].copy()
    gold = context.positivity_df[
        (context.positivity_df["criterion_label"] == "gold_answer_sensitivity")
        & (context.positivity_df["anchor"] == "gold_answer_anchor")
        & (context.positivity_df["readout_id"] == PRIMARY_READOUT)
        & (context.positivity_df["permutation_id"] == PRIMARY_PERMUTATION)
    ].copy()
    arc_csqa = primary[primary["dataset_name"].isin(["ai2_arc_challenge", "commonsenseqa_validation"])]
    gold_arc_csqa = gold[gold["dataset_name"].isin(["ai2_arc_challenge", "commonsenseqa_validation"])]
    mmlu = primary[primary["dataset_name"] == "mmlu_abstract_algebra"]
    gold_mmlu = gold[gold["dataset_name"] == "mmlu_abstract_algebra"]
    semantic_df = semantic_identity_valid(context.metrics_df)
    selection_counts = (
        pd.DataFrame(context.canonical_examples.values())
        .groupby("dataset_name")
        .size()
        .to_dict()
    )
    macros = [
        macro("MergedRunId", latex_escape(context.run_manifest["merged_run_id"])),
        macro("PreparedExampleCount", str(context.run_manifest["counts"]["prepared_examples"])),
        macro("ScoreRowCount", f"{context.run_manifest['counts']['score_rows']:,}"),
        macro("MetricRowCount", f"{context.run_manifest['counts']['metric_rows']:,}"),
        macro("TokenAuditCount", f"{context.run_manifest['counts']['token_audits']:,}"),
        macro("SemanticIdentityValidCases", f"{len(semantic_df):,}"),
        macro("PooledLagMedian", fmt_float(context.bootstrap_summary["estimate_median"], digits=1)),
        macro("PooledLagCILow", fmt_float(context.bootstrap_summary["ci_lo"], digits=1)),
        macro("PooledLagCIHigh", fmt_float(context.bootstrap_summary["ci_hi"], digits=1)),
        macro("FlagshipTwentyPlausiblePct", fmt_pct(figure_stats["flagship"]["pooled_at_20_plausible"])),
        macro("FlagshipTwentyCommittedPct", fmt_pct(figure_stats["flagship"]["pooled_at_20_committed"])),
        macro("FlagshipFiftyPlausiblePct", fmt_pct(figure_stats["flagship"]["pooled_at_50_plausible"])),
        macro("FlagshipFiftyCommittedPct", fmt_pct(figure_stats["flagship"]["pooled_at_50_committed"])),
        macro("FlagshipEightyPlausiblePct", fmt_pct(figure_stats["flagship"]["pooled_at_80_plausible"])),
        macro("FlagshipEightyCommittedPct", fmt_pct(figure_stats["flagship"]["pooled_at_80_committed"])),
        macro("ArcPromptCount", str(selection_counts.get("ai2_arc_challenge", 0))),
        macro("CsqaPromptCount", str(selection_counts.get("commonsenseqa_validation", 0))),
        macro("MmluPromptCount", str(selection_counts.get("mmlu_abstract_algebra", 0))),
        macro("OverallPrimaryPositiveCells", str(int((primary["positive"] == True).sum()))),
        macro("OverallPrimaryEvaluableCells", str(int((primary["status"] == "evaluated").sum()))),
        macro("OverallPrimaryInsufficientCells", str(int((primary["status"] == "insufficient_n").sum()))),
        macro("OverallSecondaryPositiveCells", str(int((secondary["positive"] == True).sum()))),
        macro("OverallSecondaryEvaluableCells", str(int((secondary["status"] == "evaluated").sum()))),
        macro("OverallSecondaryInsufficientCells", str(int((secondary["status"] == "insufficient_n").sum()))),
        macro("ArcCsqaPositiveCells", str(int((arc_csqa["positive"] == True).sum()))),
        macro("ArcCsqaEvaluableCells", str(int((arc_csqa["status"] == "evaluated").sum()))),
        macro("MmluPositiveCells", str(int((mmlu["positive"] == True).sum()))),
        macro("MmluEvaluableCells", str(int((mmlu["status"] == "evaluated").sum()))),
        macro("MmluInsufficientCells", str(int((mmlu["status"] == "insufficient_n").sum()))),
        macro("GoldArcCsqaPositiveCells", str(int((gold_arc_csqa["positive"] == True).sum()))),
        macro("GoldArcCsqaEvaluableCells", str(int((gold_arc_csqa["status"] == "evaluated").sum()))),
        macro("GoldMmluInsufficientCells", str(int((gold_mmlu["status"] == "insufficient_n").sum()))),
        macro("SemanticIdentityValidRate", fmt_pct(figure_stats["controls"]["valid_rates"]["semantic_exact__identity"], digits=0)),
        macro("TemplatedIdentityValidRate", fmt_pct(figure_stats["controls"]["valid_rates"]["templated_semantic__identity"], digits=0)),
        macro("LetterIdentityValidRate", fmt_pct(figure_stats["controls"]["valid_rates"]["letter_label__identity"], digits=0)),
        macro("TemplatedAgreementIdentity", fmt_pct(figure_stats["controls"]["agreement_rates"]["templated_semantic__identity"], digits=0)),
        macro("LetterAgreementIdentity", fmt_pct(figure_stats["controls"]["agreement_rates"]["letter_label__identity"], digits=0)),
        macro("AllSameReadoutRate", fmt_pct(figure_stats["controls"]["all_same_rate"], digits=0)),
        macro("DuplicateOptionCount", str(context.qc_summary["example_flag_counts"].get("duplicate_option_text", 0))),
        macro("TokenCollisionCount", str(context.qc_summary["token_flag_counts"].get("tokenization_collision", 0))),
    ]
    content = "% Auto-generated by scripts/paper/build_colm2026_assets.py\n" + "\n".join(macros) + "\n"
    write_text(GEN_DIR / "paper_numbers.tex", content)


def build_provenance(context: PaperContext, figure_stats: dict[str, Any]) -> None:
    input_paths = [
        RUN_DIR / "run_manifest.json",
        RUN_DIR / "metrics" / "primary_metrics.jsonl",
        RUN_DIR / "metrics" / "score_manifest.jsonl",
        RUN_DIR / "stats" / "bootstrap_summary.json",
        RUN_DIR / "stats" / "model_dataset_positivity.json",
        RUN_DIR / "stats" / "family_positivity_summary.json",
        RUN_DIR / "qc" / "qc_summary.json",
        RUN_DIR / "prepared_data" / "canonical_examples.jsonl",
        RUN_DIR / "manifests" / "prepared_selection_manifest.json",
        ROOT / "configs" / "metrics.yaml",
        ROOT / "configs" / "models.yaml",
        ROOT / "configs" / "readouts.yaml",
        ROOT / "configs" / "prompts.yaml",
        ROOT / "configs" / "controls.yaml",
    ]
    generated_paths = sorted(
        [path for path in FIG_DIR.glob("*.pdf")]
        + [path for path in TABLE_DIR.glob("*.tex")]
        + [GEN_DIR / "paper_numbers.tex"]
    )
    provenance = {
        "merged_run_id": context.run_manifest["merged_run_id"],
        "source_runs": context.run_manifest["source_runs"],
        "source_models": context.run_manifest["source_models"],
        "config_diff_policy": context.run_manifest.get("config_diff_policy"),
        "selection_manifest_sha256": context.selection_manifest["selection_manifest_sha256"],
        "input_hashes": {str(path.relative_to(ROOT)): sha256_path(path) for path in input_paths},
        "generated_hashes": {str(path.relative_to(ROOT)): sha256_path(path) for path in generated_paths if path.exists()},
        "summary": {
            "prepared_examples": context.run_manifest["counts"]["prepared_examples"],
            "score_rows": context.run_manifest["counts"]["score_rows"],
            "metric_rows": context.run_manifest["counts"]["metric_rows"],
            "token_audits": context.run_manifest["counts"]["token_audits"],
            "primary_positive_cells": int((context.primary_df["positive"] == True).sum()),
            "primary_evaluable_cells": int((context.primary_df["status"] == "evaluated").sum()),
            "flagship_20pct_plausible": figure_stats["flagship"]["pooled_at_20_plausible"],
            "flagship_20pct_committed": figure_stats["flagship"]["pooled_at_20_committed"],
        },
    }
    write_text(PROVENANCE_PATH, json.dumps(provenance, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    context = load_context()
    figure_stats = {
        "flagship": build_flagship_figure(context),
        "forest": build_forest_figure(context),
        "controls": build_controls_figure(context),
        "exemplars": build_exemplars_figure(context),
        "selection": build_selection_figure(context),
        "qc": build_qc_figure(context),
    }
    write_tables(context, figure_stats)
    write_paper_numbers(context, figure_stats)
    build_provenance(context, figure_stats)


if __name__ == "__main__":
    main()
