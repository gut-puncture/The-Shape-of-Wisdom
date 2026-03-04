#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _audit_common import default_paths, ensure_dir, read_parquet_required, require_paths, write_json


COLORS = {
    "stable_correct": "#1f77b4",
    "stable_wrong": "#d62728",
    "unstable_correct": "#2ca02c",
    "unstable_wrong": "#ff7f0e",
    "attention": "#1f77b4",
    "mlp": "#d62728",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 220,
        }
    )


def _short_model(mid: str) -> str:
    if "Qwen" in mid:
        return "Qwen2.5-7B"
    if "Llama" in mid:
        return "Llama-3.1-8B"
    if "Mistral" in mid:
        return "Mistral-7B"
    return mid


def _save(fig: plt.Figure, out_base: Path) -> list[str]:
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def _depth(df: pd.DataFrame) -> pd.Series:
    max_layer = df.groupby("model_id")["layer_index"].transform("max").astype(float)
    return pd.to_numeric(df["layer_index"], errors="coerce").astype(float) / max_layer.clip(lower=1.0)


def fig_trajectory_primitives(dm: pd.DataFrame, out_dir: Path) -> list[str]:
    merged = dm.copy()
    merged["depth"] = _depth(merged)
    merged["depth_bin"] = (merged["depth"] * 20).round() / 20.0
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharex=True)
    for ax, metric, label in [
        (axes[0], "delta", "Logit margin δ"),
        (axes[1], "drift", "Drift Δδ per layer"),
        (axes[2], "boundary", "Boundary |δ|"),
    ]:
        agg = (
            merged.groupby(["trajectory_type", "depth_bin"])[metric]
            .agg(
                median="median",
                q25=lambda x: float(np.nanpercentile(x, 25)),
                q75=lambda x: float(np.nanpercentile(x, 75)),
            )
            .reset_index()
        )
        for t in ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]:
            s = agg[agg["trajectory_type"] == t].sort_values("depth_bin")
            if s.empty:
                continue
            ax.plot(s["depth_bin"], s["median"], lw=2.0, color=COLORS[t], label=t.replace("_", "-"))
            ax.fill_between(s["depth_bin"], s["q25"], s["q75"], color=COLORS[t], alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.6)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Trajectory primitives with median + IQR ribbons", y=1.12)
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_1_trajectory_primitives")


def fig_stability_map(pt: pd.DataFrame, out_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    order = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    for ax, t in zip(axes.reshape(-1), order):
        sub = pt[pt["trajectory_type"] == t]
        ax.scatter(
            pd.to_numeric(sub["min_abs_delta_tail"], errors="coerce"),
            pd.to_numeric(sub["late_flip_count"], errors="coerce"),
            s=12,
            alpha=0.35,
            color=COLORS[t],
            edgecolors="none",
        )
        ax.axvline(0.3, color="#444444", lw=1.0, ls="--", alpha=0.8)
        ax.axhline(0.5, color="#444444", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(t.replace("_", "-"))
        ax.set_xlabel("Tail minimum |δ|")
        ax.set_ylabel("Late sign flips")
        ax.grid(alpha=0.22)
    fig.suptitle("Stability map (operational thresholds shown as dashed lines)")
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_2_stability_map")


def fig_decomposition(tracing: pd.DataFrame, out_dir: Path) -> list[str]:
    merged = tracing.copy()
    merged["depth"] = _depth(merged)
    merged["depth_bin"] = (merged["depth"] * 20).round() / 20.0
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True, sharey=True)
    order = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    for ax, t in zip(axes.reshape(-1), order):
        sub = merged[merged["trajectory_type"] == t]
        if sub.empty:
            continue
        for comp, col in [("s_attn", "attention"), ("s_mlp", "mlp")]:
            agg = (
                sub.groupby("depth_bin")[comp]
                .agg(
                    median="median",
                    q25=lambda x: float(np.nanpercentile(x, 25)),
                    q75=lambda x: float(np.nanpercentile(x, 75)),
                )
                .reset_index()
                .sort_values("depth_bin")
            )
            ax.plot(agg["depth_bin"], agg["median"], lw=2.0, color=COLORS[col], label=col)
            ax.fill_between(agg["depth_bin"], agg["q25"], agg["q75"], color=COLORS[col], alpha=0.18, linewidth=0)
        ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.6)
        ax.set_title(t.replace("_", "-"))
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Contribution to δ per layer")
        ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Component decomposition by trajectory type (median + IQR)")
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_3_decomposition_aggregates")


def fig_substitution_sensitivity(summary: pd.DataFrame, out_dir: Path) -> list[str]:
    groups = ["pairing_mode", "layer_range_mode", "normalization_mode", "failing_set_mode"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
    blocked = summary["blocked_reason"].fillna("").astype(str).str.strip()
    for ax, group in zip(axes.reshape(-1), groups):
        sub = summary[(summary["setting_group"] == group) & (blocked == "")]
        if sub.empty:
            ax.set_visible(False)
            continue
        names = sorted(sub["setting_name"].astype(str).unique().tolist())
        x = np.arange(len(names), dtype=float)
        width = 0.36
        for j, comp in enumerate(["attention", "mlp"]):
            s = sub[sub["component"] == comp].set_index("setting_name")
            means = [float(s.loc[n, "mean"]) if n in s.index else 0.0 for n in names]
            errs_lo = [
                (float(s.loc[n, "mean"]) - float(s.loc[n, "ci_lo"])) if n in s.index else 0.0
                for n in names
            ]
            errs_hi = [
                (float(s.loc[n, "ci_hi"]) - float(s.loc[n, "mean"])) if n in s.index else 0.0
                for n in names
            ]
            xpos = x + (-0.5 + j) * width
            ax.bar(
                xpos,
                means,
                width=width,
                color=COLORS[comp],
                alpha=0.78,
                label=comp,
                yerr=np.vstack([errs_lo, errs_hi]),
                capsize=2.2,
                error_kw={"elinewidth": 0.8, "alpha": 0.8},
            )
        ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Mean Δ logit margin")
        ax.set_title(group.replace("_", " "))
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Substitution sensitivity (mean Δδ with bootstrap 95% CI)")
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_4_substitution_sensitivity")


def fig_linearization_fidelity(
    by_layer: pd.DataFrame,
    by_type: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    axes[0].plot(by_layer["layer_index"], by_layer["mae_unit"], color=COLORS["attention"], lw=2.2, label="unit")
    axes[0].plot(by_layer["layer_index"], by_layer["mae_ols"], color=COLORS["mlp"], lw=2.2, label="ols")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Mean absolute error")
    axes[0].set_title("Reconstruction error vs layer")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    x = np.arange(by_type.shape[0], dtype=float)
    axes[1].bar(x - 0.16, by_type["mae_unit"], width=0.32, color=COLORS["attention"], alpha=0.8, label="unit")
    axes[1].bar(x + 0.16, by_type["mae_ols"], width=0.32, color=COLORS["mlp"], alpha=0.8, label="ols")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(by_type["trajectory_type"].astype(str).str.replace("_", "-"), rotation=15)
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Error by trajectory type")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Linearization fidelity diagnostics")
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_5_linearization_fidelity")


def fig_representative_prompt(tracing: pd.DataFrame, out_dir: Path) -> list[str]:
    # Select a representative stable-wrong prompt with large delta variance.
    cand = (
        tracing[tracing["trajectory_type"] == "stable_wrong"]
        .groupby(["model_id", "prompt_uid"], as_index=False)["delta"]
        .agg(delta_std=lambda x: float(np.nanstd(pd.to_numeric(x, errors="coerce"))))
        .sort_values("delta_std", ascending=False)
    )
    if cand.empty:
        sub = tracing.sort_values(["model_id", "prompt_uid", "layer_index"]).head(1)
    else:
        pick = cand.iloc[0]
        sub = tracing[(tracing["model_id"] == pick["model_id"]) & (tracing["prompt_uid"] == pick["prompt_uid"])].sort_values(
            "layer_index"
        )
    model_id = str(sub["model_id"].iloc[0])
    prompt_uid = str(sub["prompt_uid"].iloc[0])
    x = pd.to_numeric(sub["layer_index"], errors="coerce").to_numpy(dtype=np.int64)
    delta = pd.to_numeric(sub["delta"], errors="coerce").to_numpy(dtype=np.float64)
    s_attn = pd.to_numeric(sub["s_attn"], errors="coerce").to_numpy(dtype=np.float64)
    s_mlp = pd.to_numeric(sub["s_mlp"], errors="coerce").to_numpy(dtype=np.float64)
    comp = sub["competitor_key"].astype(str).tolist()
    tail_entry = max(0, int(len(x) - 8))

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.6, 0.7]})
    axes[0].plot(x, delta, color="#111111", lw=2.1)
    axes[0].axhline(0.0, color="#666666", lw=0.9, alpha=0.8)
    axes[0].axvline(tail_entry, color="#555555", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel("Logit margin δ")
    axes[0].set_title(
        "Representative prompt journey: "
        f"{_short_model(model_id)} | {prompt_uid}",
        fontsize=10,
    )
    axes[0].grid(alpha=0.22)

    axes[1].plot(x, s_attn, color=COLORS["attention"], lw=2.0, label="attention")
    axes[1].plot(x, s_mlp, color=COLORS["mlp"], lw=2.0, label="mlp")
    axes[1].axhline(0.0, color="#666666", lw=0.8, alpha=0.8)
    axes[1].axvline(tail_entry, color="#555555", lw=1.0, ls="--", alpha=0.8)
    axes[1].set_ylabel("Contribution to δ")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=0.22)

    cmap = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b", "D": "#e45756"}
    colors = [cmap.get(c, "#999999") for c in comp]
    axes[2].bar(x, np.ones_like(x), color=colors, width=1.0)
    axes[2].set_yticks([])
    axes[2].set_ylabel("k*(l)")
    axes[2].set_xlabel("Layer index")
    axes[2].set_xlim(x.min() - 0.5, x.max() + 0.5)
    axes[2].axvline(tail_entry, color="#555555", lw=1.0, ls="--", alpha=0.8)
    fig.tight_layout()
    return _save(fig, out_dir / "fig_vnext_6_representative_prompt_journey")


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Generate readability-first vNext paper figures from cached artifacts.")
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--audit-dir", type=Path, default=paths.audit)
    ap.add_argument("--output-dir", type=Path, default=paths.figures_vnext)
    ap.add_argument("--strict", action="store_true", help="Fail if audit CSV dependencies are missing.")
    args = ap.parse_args()

    _setup_style()
    ensure_dir(args.output_dir)
    require_paths(
        [
            args.parquet_dir / "decision_metrics.parquet",
            args.parquet_dir / "prompt_types.parquet",
            args.parquet_dir / "tracing_scalars.parquet",
        ]
    )

    decision = read_parquet_required(args.parquet_dir / "decision_metrics.parquet")
    prompt_types = read_parquet_required(args.parquet_dir / "prompt_types.parquet")
    tracing = read_parquet_required(args.parquet_dir / "tracing_scalars.parquet")
    dm = decision.merge(prompt_types[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")
    ts = tracing.merge(prompt_types[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")

    summary_csv = args.audit_dir / "substitution_sensitivity_summary.csv"
    layer_csv = args.audit_dir / "drift_reconstruction_by_layer.csv"
    type_csv = args.audit_dir / "drift_reconstruction_by_type.csv"
    if args.strict:
        require_paths([summary_csv, layer_csv, type_csv])
    if summary_csv.exists():
        sensitivity = pd.read_csv(summary_csv)
    else:
        sensitivity = pd.DataFrame(
            columns=["setting_group", "setting_name", "component", "mean", "ci_lo", "ci_hi", "blocked_reason"]
        )
    if layer_csv.exists():
        by_layer = pd.read_csv(layer_csv)
    else:
        by_layer = pd.DataFrame(columns=["layer_index", "mae_unit", "mae_ols"])
    if type_csv.exists():
        by_type = pd.read_csv(type_csv)
    else:
        by_type = pd.DataFrame(columns=["trajectory_type", "mae_unit", "mae_ols"])

    generated: list[str] = []
    generated += fig_trajectory_primitives(dm, args.output_dir)
    generated += fig_stability_map(prompt_types, args.output_dir)
    generated += fig_decomposition(ts, args.output_dir)
    if not sensitivity.empty:
        generated += fig_substitution_sensitivity(sensitivity, args.output_dir)
    if (not by_layer.empty) and (not by_type.empty):
        generated += fig_linearization_fidelity(by_layer, by_type, args.output_dir)
    generated += fig_representative_prompt(ts, args.output_dir)

    payload = {"figures_generated": generated, "count": len(generated)}
    out_json = args.output_dir / "figures_vnext_manifest.json"
    write_json(out_json, payload)
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
