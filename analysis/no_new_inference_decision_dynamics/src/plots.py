from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from .constants import (
    APPENDIX_FIGURES,
    COLOR_BOUNDARY,
    COLOR_FINAL_CORRECT,
    COLOR_FINAL_WRONG,
    COLOR_MODEL,
    COLOR_NEUTRAL,
    DEPTH_CMAP,
    MAIN_FIGURES,
    MODEL_ORDER,
    MODEL_SHORT,
)
from .qc import normalize_pdf


def configure_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.10,
            "grid.linewidth": 0.3,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> tuple[Path, Path]:
    pdf_path = out_dir / "figures" / f"{stem}.pdf"
    png_path = out_dir / "figures" / f"{stem}.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    normalize_pdf(pdf_path, title=stem)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def generate_all_figures(
    *,
    out_dir: Path,
    layerwise: pd.DataFrame,
    future_flip: pd.DataFrame,
    trajectories: pd.DataFrame,
    curve_summary: pd.DataFrame,
    exemplars: pd.DataFrame,
    empirical_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    configure_style()
    rows: list[dict[str, Any]] = []
    rows.extend(_fig01(out_dir, layerwise, trajectories, exemplars))
    rows.extend(_fig02(out_dir, curve_summary))
    rows.extend(_fig03(out_dir, future_flip))
    rows.extend(_fig04(out_dir, trajectories, empirical_table))
    rows.extend(_fig05(out_dir, layerwise, exemplars))
    rows.extend(_appendix_figures(out_dir, layerwise, future_flip, trajectories, curve_summary))
    return rows


def _manifest_row(stem: str, claim: str, inputs: str, panels: str, kind: str) -> dict[str, Any]:
    return {"kind": kind, "filename": f"{stem}.pdf", "claim": claim, "inputs": inputs, "panels": panels, "qc_status": "generated"}


def _fig01(out_dir: Path, layerwise: pd.DataFrame, trajectories: pd.DataFrame, exemplars: pd.DataFrame) -> list[dict[str, Any]]:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(["correct", "wrong 1", "wrong 2", "wrong 3"], [1.8, 0.4, 0.2, -0.1], color=[COLOR_FINAL_CORRECT, "#b5c0ca", "#c8d0d8", "#d6dde3"])
    ax_a.axhline(0.0, color="#222222", lw=0.7)
    ax_a.set_title("A. One layer, decomposed")
    ax_a.set_ylabel("Score")
    ax_a.text(0.02, 0.96, "soft margin = correct - logsumexp(wrong)\n\na = correct-vs-rest\nq = disagreement among wrong answers", transform=ax_a.transAxes, va="top", fontsize=9)

    exemplar_ids = exemplars.drop_duplicates("model_id").sort_values("model_id").head(3)
    path_grid = gs[0, 1].subgridspec(3, 1, hspace=0.35)
    density_grid = gs[0, 2].subgridspec(3, 1, hspace=0.35)
    for row_idx, (_, exemplar) in enumerate(exemplar_ids.iterrows()):
        sub = layerwise[(layerwise["model_id"] == exemplar["model_id"]) & (layerwise["prompt_uid"] == exemplar["prompt_uid"])].sort_values("layer_index")
        ax = fig.add_subplot(path_grid[row_idx, 0])
        _plot_depth_colored_path(ax, sub["a"], sub["q"], sub["z"], title=f"B. {MODEL_SHORT[exemplar['model_id']]}")
        if row_idx == 0:
            ax.set_ylabel("competitor dispersion q")
        ax.set_xlabel("commitment a")
    for row_idx, model_id in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(density_grid[row_idx, 0])
        model_sub = layerwise[layerwise["model_id"] == model_id]
        _density_overlay(ax, model_sub)
        ax.set_title(f"C. {MODEL_SHORT[model_id]}")
        ax.set_xlabel("commitment a")
        if row_idx == 0:
            ax.set_ylabel("competitor dispersion q")
    save_figure(fig, "fig01_state_decomposition", out_dir)
    return [_manifest_row("fig01_state_decomposition", "state decomposition and answer-readout coordinates", "layerwise_scores_with_state_metrics.parquet", "schematic + depth-colored paths + early/mid/late densities", "main")]


def _plot_depth_colored_path(ax: plt.Axes, a: pd.Series, q: pd.Series, z: pd.Series, *, title: str) -> None:
    cmap = plt.get_cmap(DEPTH_CMAP)
    norm = Normalize(vmin=0.0, vmax=1.0)
    for idx in range(1, len(a)):
        ax.plot([a.iloc[idx - 1], a.iloc[idx]], [q.iloc[idx - 1], q.iloc[idx]], color=cmap(norm(float(z.iloc[idx]))), lw=2.0, alpha=0.95)
    ax.scatter(a.iloc[0], q.iloc[0], color="#111111", s=26, marker="o", label="start")
    ax.scatter(a.iloc[-1], q.iloc[-1], color=COLOR_FINAL_CORRECT, s=28, marker="s", label="end")
    ax.set_title(title)


def _density_overlay(ax: plt.Axes, model_sub: pd.DataFrame) -> None:
    windows = {
        "early": ((model_sub["z"] <= 0.2), "#88929d"),
        "middle": (((model_sub["z"] >= 0.4) & (model_sub["z"] <= 0.6)), COLOR_BOUNDARY),
        "late": ((model_sub["z"] >= 0.8), COLOR_FINAL_CORRECT),
    }
    for label, (mask, color) in windows.items():
        sub = model_sub[mask]
        if sub.empty:
            continue
        hist, xedges, yedges = np.histogram2d(sub["a"], sub["q"], bins=30)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters, indexing="xy")
        if np.any(hist > 0):
            ax.contour(X, Y, hist.T, levels=3, colors=[color], linewidths=1.2)
    ax.set_xlim(model_sub["a"].quantile(0.01), model_sub["a"].quantile(0.99))
    ax.set_ylim(max(0.0, model_sub["q"].quantile(0.01)), model_sub["q"].quantile(0.99))


def _fig02(out_dir: Path, curves: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = {True: ("Final correct", COLOR_FINAL_CORRECT), False: ("Final wrong", COLOR_FINAL_WRONG)}
    metrics = [("a", "commitment a"), ("q", "competitor dispersion q"), ("switch", "switch rate"), ("r", "decisiveness r")]
    fig, axes = plt.subplots(len(metrics), len(MODEL_ORDER), figsize=(14, 10), sharex=True)
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, model_id in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = curves[(curves["metric"] == metric) & (curves["model_id"] == model_id)]
            for group_value, (label, color) in label_map.items():
                line = sub[sub["group"] == group_value].sort_values("summary_z_bin")
                if line.empty:
                    continue
                x = (line["summary_z_bin"] + 0.5) / 20.0
                ax.plot(x, line["center"], color=color, lw=2.2, label=label)
                ax.fill_between(x, line["ci_low"], line["ci_high"], color=color, alpha=0.18)
            if row_idx == 0:
                ax.set_title(MODEL_SHORT[model_id])
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("normalized depth z")
    axes[0, 0].legend(frameon=False, loc="upper left")
    save_figure(fig, "fig02_depthwise_dynamics", out_dir)
    return [_manifest_row("fig02_depthwise_dynamics", "depth-wise competition and commitment summaries", "bootstrap curve summaries", "model-specific small multiples for a, q, switch rate, and r", "main")]


def _fig03(out_dir: Path, future_flip: pd.DataFrame) -> list[dict[str, Any]]:
    grid = future_flip.groupby(["z_bin", "p_bin"], sort=False)["pooled_future_flip_prob"].median().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(grid.T, origin="lower", aspect="auto", extent=[0, 1, 0, 1], cmap="cividis", vmin=0.0, vmax=1.0)
    ax.axhline(0.5, color="white", lw=1.2, linestyle="--")
    ax.set_xlabel("normalized depth z")
    ax.set_ylabel("current p_correct")
    ax.set_title("Empirical future-flip probability")
    plt.colorbar(image, ax=ax, label="Pr(any future sign flip)")
    save_figure(fig, "fig03_reversibility_map", out_dir)
    return [_manifest_row("fig03_reversibility_map", "empirical reversibility map over depth and current probability", "layerwise_future_flip_metrics.parquet", "pooled heatmap with boundary line", "main")]


def _fig04(out_dir: Path, trajectories: pd.DataFrame, empirical_table: pd.DataFrame) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_a, ax_b, ax_c = axes
    data = [trajectories.loc[trajectories["final_correct"], "boundary_occupancy_prob_010"], trajectories.loc[~trajectories["final_correct"], "boundary_occupancy_prob_010"]]
    parts = ax_a.violinplot(data, positions=[1, 2], showmedians=True, widths=0.8)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLOR_FINAL_CORRECT if idx == 0 else COLOR_FINAL_WRONG)
        body.set_alpha(0.35)
    ax_a.set_xticks([1, 2], ["final correct", "final wrong"])
    ax_a.set_title("A. Boundary occupancy")
    ax_a.set_ylabel("fraction of layers with |p-0.5| < 0.10")

    scatter = trajectories.dropna(subset=["last_flip_depth"]).copy()
    ax_b.scatter(scatter["boundary_occupancy_prob_010"], scatter["last_flip_depth"], s=14, alpha=0.25, color=COLOR_BOUNDARY)
    if not scatter.empty:
        bins = np.linspace(float(scatter["boundary_occupancy_prob_010"].min()), float(scatter["boundary_occupancy_prob_010"].max()), 11)
        scatter["bin"] = np.clip(np.digitize(scatter["boundary_occupancy_prob_010"], bins, right=True) - 1, 0, 9)
        trend = scatter.groupby("bin", sort=False).agg(x=("boundary_occupancy_prob_010", "median"), y=("last_flip_depth", "median"))
        ax_b.plot(trend["x"], trend["y"], color=COLOR_NEUTRAL, lw=2.0)
    ax_b.set_title("B. Boundary dwelling vs last flip")
    ax_b.set_xlabel("boundary occupancy (prob 0.10)")
    ax_b.set_ylabel("last flip depth")

    box_data = [
        trajectories.loc[trajectories["boundary_occupancy_quartile"] == quartile, "pooled_empirical_commitment_depth_005"].dropna().to_numpy(dtype=float)
        for quartile in range(1, 5)
    ]
    box_data = [values if len(values) else np.asarray([np.nan]) for values in box_data]
    ax_c.boxplot(box_data, labels=["Q1", "Q2", "Q3", "Q4"], patch_artist=True, boxprops={"facecolor": "#d6cfde", "alpha": 0.8})
    ax_c.set_ylim(0, 1)
    ax_c.set_title("C. Commitment depth by occupancy quartile")
    ax_c.set_xlabel("within-model occupancy quartile")
    ax_c.set_ylabel("pooled commitment depth (alpha=0.05)")
    save_figure(fig, "fig04_boundary_dwellers", out_dir)
    return [_manifest_row("fig04_boundary_dwellers", "boundary dwelling tracks flip-heavy trajectories, with a mixed link to later stabilization", "trajectory_metrics.parquet + table_empirical_commitment_summary.csv", "distribution + scatter + commitment-depth summary", "main")]


def _fig05(out_dir: Path, layerwise: pd.DataFrame, exemplars: pd.DataFrame) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(12, max(8, 2.8 * len(exemplars))))
    if len(exemplars) == 1:
        axes = np.asarray([axes])
    for row_idx, exemplar in enumerate(exemplars.itertuples(index=False)):
        sub = layerwise[(layerwise["model_id"] == exemplar.model_id) & (layerwise["prompt_uid"] == exemplar.prompt_uid)].sort_values("layer_index")
        ax_left, ax_right = axes[row_idx]
        color = COLOR_FINAL_CORRECT if exemplar.final_correct else COLOR_FINAL_WRONG
        if exemplar.group == "boundary_dwelling":
            color = COLOR_BOUNDARY
        ax_left.plot(sub["z"], sub["p_correct"], color=color, lw=2.2)
        ax_left.axhline(0.5, color="#222222", lw=0.7, linestyle="--")
        ax_left.set_ylim(0, 1)
        ax_left.set_ylabel("p_correct")
        ax_left.set_title(f"{exemplar.group} | {MODEL_SHORT[exemplar.model_id]}", fontsize=8)
        _plot_depth_colored_path(ax_right, sub["a"], sub["q"], sub["z"], title="a-q path")
        ax_right.set_ylabel("q")
        ax_right.set_xlabel("a")
        ax_left.set_xlabel("normalized depth z")
    save_figure(fig, "fig05_exemplar_trajectories", out_dir)
    return [_manifest_row("fig05_exemplar_trajectories", "representative stable, wrong, and boundary-dwelling trajectories", "layerwise_scores_with_state_metrics.parquet + table_exemplar_prompts.csv", "paired p_correct and a-q views for algorithmic exemplars", "main")]


def _appendix_figures(
    out_dir: Path,
    layerwise: pd.DataFrame,
    future_flip: pd.DataFrame,
    trajectories: pd.DataFrame,
    curves: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    rows.append(_appendix_baseline_margin(out_dir, layerwise))
    rows.append(_appendix_baseline_switch(out_dir, trajectories))
    rows.append(_appendix_model_reversibility(out_dir, future_flip))
    rows.append(_appendix_boundary_widths(out_dir, trajectories))
    rows.append(_appendix_travel(out_dir, trajectories))
    rows.append(_appendix_old_vs_empirical(out_dir, trajectories))
    rows.append(_appendix_competitor_dynamics(out_dir, layerwise, curves))
    return rows


def _appendix_baseline_margin(out_dir: Path, layerwise: pd.DataFrame) -> dict[str, Any]:
    sub = layerwise.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").copy()
    sub["delta_hard_jump"] = sub.groupby(["model_id", "prompt_uid"], sort=False)["delta_hard"].diff().abs()
    sub["delta_soft_jump"] = sub.groupby(["model_id", "prompt_uid"], sort=False)["delta_soft"].diff().abs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    prompt_scores = (
        sub.groupby(["model_id", "prompt_uid"], sort=False)
        .apply(lambda frame: float((frame.loc[frame["switch"] == 1, "delta_hard_jump"] - frame.loc[frame["switch"] == 1, "delta_soft_jump"]).sum()), include_groups=False)
        .reset_index(name="artifact_score")
        .sort_values(["artifact_score", "prompt_uid"], ascending=[False, True], kind="stable")
    )
    exemplar = prompt_scores.iloc[0]
    prompt = sub[(sub["model_id"] == exemplar["model_id"]) & (sub["prompt_uid"] == exemplar["prompt_uid"])]
    axes[0].plot(prompt["z"], prompt["delta_hard"], color=COLOR_FINAL_WRONG, label="hard", lw=1.8)
    axes[0].plot(prompt["z"], prompt["delta_soft"], color=COLOR_FINAL_CORRECT, label="soft", lw=1.8)
    axes[0].legend(frameon=False)
    axes[0].set_title("Representative margins")
    axes[1].hist(sub["delta_hard_jump"].dropna(), bins=40, alpha=0.45, color=COLOR_FINAL_WRONG, label="hard")
    axes[1].hist(sub["delta_soft_jump"].dropna(), bins=40, alpha=0.45, color=COLOR_FINAL_CORRECT, label="soft")
    axes[1].set_title("Jump distributions")
    axes[1].legend(frameon=False)
    save_figure(fig, "appendix_baseline_margin_smoothing", out_dir)
    return _manifest_row("appendix_baseline_margin_smoothing", "hard-vs-soft margin appendix", "layerwise_scores_with_state_metrics.parquet", "representative prompt + jump histograms", "appendix")


def _appendix_baseline_switch(out_dir: Path, trajectories: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(trajectories["last_flip_depth"].dropna(), bins=30, color=COLOR_BOUNDARY, alpha=0.7)
    axes[0].set_title("Last flip depth")
    axes[1].hist(trajectories["old_threshold_commitment_depth"].dropna(), bins=30, color=COLOR_NEUTRAL, alpha=0.7)
    axes[1].set_title("Old-threshold commitment depth")
    save_figure(fig, "appendix_baseline_switch_commitment", out_dir)
    return _manifest_row("appendix_baseline_switch_commitment", "baseline switch and old commitment appendix", "trajectory_metrics.parquet", "last-flip and old-commitment histograms", "appendix")


def _appendix_model_reversibility(out_dir: Path, future_flip: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(14, 4), sharey=True)
    for ax, model_id in zip(axes, MODEL_ORDER):
        grid = future_flip[future_flip["model_id"] == model_id].groupby(["z_bin", "p_bin"], sort=False)["model_future_flip_prob"].median().unstack(fill_value=np.nan)
        ax.imshow(grid.T, origin="lower", aspect="auto", extent=[0, 1, 0, 1], cmap="cividis", vmin=0.0, vmax=1.0)
        ax.axhline(0.5, color="white", lw=1.0, linestyle="--")
        ax.set_title(MODEL_SHORT[model_id])
        ax.set_xlabel("z")
    axes[0].set_ylabel("p_correct")
    save_figure(fig, "appendix_per_model_reversibility_maps", out_dir)
    return _manifest_row("appendix_per_model_reversibility_maps", "per-model reversibility maps", "layerwise_future_flip_metrics.parquet", "one heatmap per model", "appendix")


def _appendix_boundary_widths(out_dir: Path, trajectories: pd.DataFrame) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 4))
    widths = ["boundary_occupancy_prob_005", "boundary_occupancy_prob_010", "boundary_occupancy_margin_025", "boundary_occupancy_margin_050"]
    data = [trajectories[column].dropna().to_numpy(dtype=float) for column in widths]
    ax.boxplot(data, labels=["p±0.05", "p±0.10", "m±0.25", "m±0.50"])
    ax.set_title("Boundary-width robustness")
    save_figure(fig, "appendix_boundary_width_robustness", out_dir)
    return _manifest_row("appendix_boundary_width_robustness", "boundary width robustness", "trajectory_metrics.parquet", "occupancy distributions for multiple thresholds", "appendix")


def _appendix_travel(out_dir: Path, trajectories: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(trajectories["path_length"], bins=30, color=COLOR_NEUTRAL, alpha=0.7)
    axes[0].set_title("Path length")
    axes[1].hist(trajectories["angular_travel"], bins=30, color=COLOR_BOUNDARY, alpha=0.7)
    axes[1].set_title("Angular travel")
    axes[2].hist(trajectories["radial_travel"], bins=30, color=COLOR_FINAL_CORRECT, alpha=0.7)
    axes[2].set_title("Radial travel")
    save_figure(fig, "appendix_travel_metric_distributions", out_dir)
    return _manifest_row("appendix_travel_metric_distributions", "trajectory travel distributions", "trajectory_metrics.parquet", "path, angular, and radial travel histograms", "appendix")


def _appendix_old_vs_empirical(out_dir: Path, trajectories: pd.DataFrame) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6, 5))
    sub = trajectories.dropna(subset=["old_threshold_commitment_depth", "pooled_empirical_commitment_depth_005"])
    ax.scatter(sub["old_threshold_commitment_depth"], sub["pooled_empirical_commitment_depth_005"], s=16, alpha=0.25, color=COLOR_MODEL[MODEL_ORDER[0]])
    ax.plot([0, 1], [0, 1], color="#222222", lw=0.8, linestyle="--")
    ax.set_xlabel("old-threshold commitment depth")
    ax.set_ylabel("empirical commitment depth (alpha=0.05)")
    ax.set_title("Old vs empirical commitment")
    save_figure(fig, "appendix_old_vs_empirical_commitment", out_dir)
    return _manifest_row("appendix_old_vs_empirical_commitment", "old-threshold versus empirical commitment comparison", "trajectory_metrics.parquet", "scatter of old and empirical commitment depths", "appendix")


def _appendix_competitor_dynamics(out_dir: Path, layerwise: pd.DataFrame, curves: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sub = layerwise.dropna(subset=["next_switch"]).copy()
    sub["q_rank_bin"] = pd.qcut(sub["q"], q=10, labels=False, duplicates="drop")
    q_switch = sub.groupby("q_rank_bin", sort=False)["next_switch"].mean()
    axes[0].plot((q_switch.index.to_numpy(dtype=float) + 0.5) / max(len(q_switch), 1), q_switch.to_numpy(dtype=float), color=COLOR_BOUNDARY, lw=2.0)
    axes[0].set_title("Next-switch probability vs q")
    axes[0].set_xlabel("q rank bin")
    axes[0].set_ylabel("Pr(next switch)")
    switch_curves = curves[(curves["metric"] == "switch")]
    for model_id in MODEL_ORDER:
        model_sub = switch_curves[(switch_curves["model_id"] == model_id) & (switch_curves["group"] == True)].sort_values("summary_z_bin")
        axes[1].plot((model_sub["summary_z_bin"] + 0.5) / 20.0, model_sub["center"], lw=2.0, color=COLOR_MODEL[model_id], label=MODEL_SHORT[model_id])
    axes[1].legend(frameon=False)
    axes[1].set_title("Final-correct switch rate by model")
    axes[1].set_xlabel("normalized depth z")
    axes[1].set_ylabel("switch rate")
    save_figure(fig, "appendix_competitor_dynamics", out_dir)
    return _manifest_row("appendix_competitor_dynamics", "competitor-dynamics appendix", "layerwise_scores_with_state_metrics.parquet + bootstrap curves", "next-switch vs q and per-model switch curves", "appendix")
