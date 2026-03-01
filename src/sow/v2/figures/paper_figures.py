#!/usr/bin/env python3
"""Generate all paper figures from stored parquet results.

Each function produces one vector-PDF figure for the NeurIPS submission.
All figures use only pre-computed parquet data — no GPU inference required.

Usage:
    python -m sow.v2.figures.paper_figures \\
        --parquet-dir results/parquet \\
        --output-dir paper/final_paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from sow.v2.figures.style import (
    TRAJ, TRAJ_LABELS, TRAJ_ORDER, COMP, COMP_LABELS, SPAN, SPAN_LABELS,
    NEUTRAL, COL_WIDTH, TEXT_WIDTH,
    FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
    LW_DATA, LW_REF,
    configure_matplotlib, remove_top_right_spines, add_zero_line,
    add_panel_label, bootstrap_ci, depth_ticks,
)
from sow.v2.figures.data_loaders import (
    MODEL_LAYERS, load_decision_metrics, load_prompt_types,
    load_tracing_scalars, load_attention_data, load_causal_data,
)

# ---------------------------------------------------------------------------
# Figure 1: The Three Primitives
# ---------------------------------------------------------------------------

def fig1_primitives(parquet_dir: Path, output_dir: Path) -> Path:
    """Mean δ, g, |δ| by trajectory type vs normalised depth.

    Data: decision_metrics.parquet (delta, drift, boundary, layer_index)
          prompt_types.parquet (trajectory_type)
    """
    dm = load_decision_metrics(parquet_dir)
    dm["depth_bin"] = dm["depth"].round(2)

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.2), constrained_layout=True)

    metrics = [("delta", "Decision margin δ (logits)"),
               ("drift", "Per-layer drift g (logits)"),
               ("boundary", "Boundary distance |δ| (logits)")]
    panel_labels = ["a", "b", "c"]
    panel_titles = ["State: δ(l)", "Motion: g(l)", "Boundary: |δ(l)|"]

    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        agg = dm.groupby(["trajectory_type", "depth_bin"])[col].agg(["mean", "std", "count"]).reset_index()

        for tt in TRAJ_ORDER:
            s = agg[agg["trajectory_type"] == tt].sort_values("depth_bin")
            ax.plot(s["depth_bin"], s["mean"], color=TRAJ[tt], lw=LW_DATA,
                    label=TRAJ_LABELS[tt] if idx == 0 else None)
            # Shaded band for stable types on δ and boundary panels
            if idx in (0, 2) and tt in ("stable_correct", "stable_wrong"):
                sem = s["std"] / np.sqrt(s["count"].clip(lower=1))
                ax.fill_between(s["depth_bin"], s["mean"] - sem, s["mean"] + sem,
                                color=TRAJ[tt], alpha=0.08)

        add_zero_line(ax)
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_titles[idx], fontsize=FONT_TITLE)
        add_panel_label(ax, panel_labels[idx])
        remove_top_right_spines(ax)

        if idx == 2:
            ax.axhline(0.3, color=NEUTRAL, lw=LW_REF, ls="--", zorder=0)
            ax.text(0.02, 0.35, "$b_{\\mathrm{min}}$", fontsize=FONT_ANNOT,
                    color=NEUTRAL, transform=ax.get_yaxis_transform())

    axes[0].legend(fontsize=FONT_LEGEND, loc="lower left", frameon=False)

    out = output_dir / "fig1_primitives.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Phase Diagram
# ---------------------------------------------------------------------------

def fig2_phase_diagram(parquet_dir: Path, output_dir: Path) -> Path:
    """Scatter: |δ(L−τ)| vs tail flip count, colored by trajectory type.

    Data: decision_metrics.parquet, prompt_types.parquet
    """
    dm = load_decision_metrics(parquet_dir)
    tau = 8

    records = []
    for (mid, puid), grp in dm.groupby(["model_id", "prompt_uid"]):
        grp = grp.sort_values("layer_index")
        L = MODEL_LAYERS.get(mid, int(grp["layer_index"].max()) + 1)
        tail_start = max(L - tau, 0)
        tail = grp[grp["layer_index"] >= tail_start]
        if tail.empty:
            continue
        entry = grp[grp["layer_index"] == tail_start]
        b_entry = float(entry["boundary"].iloc[0]) if not entry.empty else np.nan
        signs = np.sign(tail["delta"].values)
        signs = signs[signs != 0]
        flips = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        records.append({
            "boundary_at_entry": b_entry, "tail_flips": flips,
            "trajectory_type": grp["trajectory_type"].iloc[0],
        })
    pdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8), constrained_layout=True)
    rng = np.random.default_rng(42)

    for tt in TRAJ_ORDER:
        s = pdf[pdf["trajectory_type"] == tt]
        jitter = rng.uniform(-0.12, 0.12, size=len(s))
        ax.scatter(s["boundary_at_entry"], s["tail_flips"].values + jitter,
                   c=TRAJ[tt], s=6, alpha=0.35, edgecolors="none",
                   label=TRAJ_LABELS[tt], rasterized=True)

    # Threshold overlays
    ax.axvline(0.3, color=NEUTRAL, lw=LW_REF, ls="--", zorder=0)
    ax.axhline(0.5, color=NEUTRAL, lw=LW_REF, ls="--", zorder=0)

    ax.set_xlabel("|δ(L−τ)| at tail entry (logits)")
    ax.set_ylabel("Sign flips in last τ layers")
    ax.set_yticks(range(0, int(pdf["tail_flips"].max()) + 2))
    ax.legend(fontsize=FONT_LEGEND, loc="upper right", frameon=False, markerscale=2)
    remove_top_right_spines(ax)

    out = output_dir / "fig2_phase_diagram.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Drift Decomposition
# ---------------------------------------------------------------------------

def fig3_decomposition(parquet_dir: Path, output_dir: Path) -> Path:
    """Two-force decomposition: attention steers, MLP commits.

    Data: tracing_scalars.parquet (s_attn, s_mlp, drift)
          prompt_types.parquet (trajectory_type)

    Panels:
      (a) Stable-correct decomposition traces
      (b) Stable-wrong decomposition traces
      (c) Cumulative contributions for stable-correct
      (d) Dominance ratio |s_attn|/(|s_attn|+|s_mlp|) by type
    """
    ts = load_tracing_scalars(parquet_dir)
    ts["depth_bin"] = (ts["depth"] * 20).round() / 20

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.0), constrained_layout=True)

    # --- Panel (a): Stable-correct decomposition ---
    ax = axes[0, 0]
    sc = ts[ts["trajectory_type"] == "stable_correct"]
    agg = sc.groupby("depth_bin").agg(
        s_attn=("s_attn", "mean"), s_mlp=("s_mlp", "mean"), drift=("drift", "mean"),
        s_attn_sem=("s_attn", "sem"), s_mlp_sem=("s_mlp", "sem"),
    ).reset_index()
    x = agg["depth_bin"]
    ax.fill_between(x, 0, agg["s_attn"], color=COMP["attention"], alpha=0.3, label="Attention")
    ax.fill_between(x, 0, agg["s_mlp"], color=COMP["mlp"], alpha=0.3, label="MLP")
    ax.plot(x, agg["drift"], color=NEUTRAL, lw=1.2, ls="--", label="Observed drift")
    add_zero_line(ax)
    ax.set_ylabel("Drift contribution (logits)")
    ax.set_title("Stable-Correct", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper right")
    add_panel_label(ax, "a")
    remove_top_right_spines(ax)

    # Annotations
    peak_idx = agg["s_attn"].idxmax()
    if peak_idx is not None and not pd.isna(peak_idx):
        peak_x = float(agg.loc[peak_idx, "depth_bin"])
        peak_y = float(agg.loc[peak_idx, "s_attn"])
        if peak_y > 0.1:
            ax.annotate("Attn steers", xy=(peak_x, peak_y), fontsize=FONT_ANNOT,
                        color=COMP["attention"], ha="center", va="bottom")

    # --- Panel (b): Stable-wrong decomposition ---
    ax = axes[0, 1]
    sw = ts[ts["trajectory_type"] == "stable_wrong"]
    agg_sw = sw.groupby("depth_bin").agg(
        s_attn=("s_attn", "mean"), s_mlp=("s_mlp", "mean"), drift=("drift", "mean"),
    ).reset_index()
    x = agg_sw["depth_bin"]
    ax.fill_between(x, 0, agg_sw["s_attn"], color=COMP["attention"], alpha=0.3)
    ax.fill_between(x, 0, agg_sw["s_mlp"], color=COMP["mlp"], alpha=0.3)
    ax.plot(x, agg_sw["drift"], color=NEUTRAL, lw=1.2, ls="--")
    add_zero_line(ax)
    ax.set_ylabel("Drift contribution (logits)")
    ax.set_title("Stable-Wrong", fontsize=FONT_TITLE)
    add_panel_label(ax, "b")
    remove_top_right_spines(ax)

    # --- Panel (c): Cumulative contributions (stable-correct) ---
    ax = axes[1, 0]
    sc_sorted = sc.sort_values(["model_id", "prompt_uid", "layer_index"])
    cum_attn = sc_sorted.groupby(["model_id", "prompt_uid"])["s_attn"].cumsum()
    cum_mlp = sc_sorted.groupby(["model_id", "prompt_uid"])["s_mlp"].cumsum()
    sc_sorted = sc_sorted.copy()
    sc_sorted["cum_attn"] = cum_attn.values
    sc_sorted["cum_mlp"] = cum_mlp.values
    sc_sorted["depth_bin"] = (sc_sorted["depth"] * 20).round() / 20

    agg_cum = sc_sorted.groupby("depth_bin").agg(
        cum_attn=("cum_attn", "mean"), cum_mlp=("cum_mlp", "mean"),
    ).reset_index()
    ax.plot(agg_cum["depth_bin"], agg_cum["cum_attn"], color=COMP["attention"],
            lw=LW_DATA, label="Σ Attention")
    ax.plot(agg_cum["depth_bin"], agg_cum["cum_mlp"], color=COMP["mlp"],
            lw=LW_DATA, label="Σ MLP")
    add_zero_line(ax)
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Cumulative contribution (logits)")
    ax.set_title("Cumulative (Stable-Correct)", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, frameon=False)
    add_panel_label(ax, "c")
    remove_top_right_spines(ax)

    # --- Panel (d): Dominance ratio by trajectory type ---
    ax = axes[1, 1]
    ts["abs_attn"] = ts["s_attn"].abs()
    ts["abs_mlp"] = ts["s_mlp"].abs()
    ts["depth_bin_10"] = (ts["depth"] * 10).round() / 10

    for tt in TRAJ_ORDER:
        sub = ts[ts["trajectory_type"] == tt]
        agg_d = sub.groupby("depth_bin_10").agg(
            aa=("abs_attn", "mean"), am=("abs_mlp", "mean"),
        ).reset_index()
        denom = (agg_d["aa"] + agg_d["am"]).clip(lower=1e-8)
        ratio = agg_d["aa"] / denom
        ax.plot(agg_d["depth_bin_10"], ratio, color=TRAJ[tt], lw=1.4,
                label=TRAJ_LABELS[tt])

    ax.axhline(0.5, color=NEUTRAL, lw=LW_REF, ls="--", zorder=0)
    ax.text(0.02, 0.53, "Attention\ndominant", fontsize=FONT_ANNOT, color=NEUTRAL,
            transform=ax.get_yaxis_transform(), va="bottom")
    ax.text(0.02, 0.47, "MLP\ndominant", fontsize=FONT_ANNOT, color=NEUTRAL,
            transform=ax.get_yaxis_transform(), va="top")
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("|s_attn| / (|s_attn|+|s_mlp|)")
    ax.set_title("Dominance Ratio", fontsize=FONT_TITLE)
    ax.set_ylim(0.2, 0.8)
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper right", ncol=2)
    add_panel_label(ax, "d")
    remove_top_right_spines(ax)

    out = output_dir / "fig3_decomposition.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Attention Routing
# ---------------------------------------------------------------------------

def fig4_attention_routing(parquet_dir: Path, output_dir: Path) -> Path:
    """Attention mass and contribution difference (correct − competitor) by depth.

    Data: attention_mass_by_span.parquet, attention_contrib_by_span.parquet,
          decision_metrics.parquet (for correct_key/competitor), prompt_types.parquet
    """
    mass_m, contrib_m = load_attention_data(parquet_dir)
    mass_m["depth_bin"] = (mass_m["depth"] * 20).round() / 20
    contrib_m["depth_bin"] = (contrib_m["depth"] * 20).round() / 20

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    metrics = [
        (mass_m, "attention_mass", "Δ mass per token (correct − competitor)", "Mass difference"),
        (contrib_m, "attention_contribution", "Δ contribution (correct − competitor)", "Contribution difference"),
    ]

    for idx, (df, col, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for tt, color in [("stable_correct", TRAJ["stable_correct"]),
                          ("stable_wrong", TRAJ["stable_wrong"])]:
            sub = df[df["trajectory_type"] == tt]
            corr = sub[sub["span_rel"] == "option_correct"].groupby("depth_bin")[col].mean()
            comp = sub[sub["span_rel"] == "option_competitor"].groupby("depth_bin")[col].mean()
            common_idx = corr.index.intersection(comp.index)
            diff = corr.loc[common_idx] - comp.loc[common_idx]

            ax.plot(diff.index, diff.values, color=color, lw=LW_DATA,
                    label=TRAJ_LABELS[tt])
            ax.fill_between(diff.index, 0, diff.values, color=color, alpha=0.12)

        add_zero_line(ax)
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=FONT_TITLE)
        add_panel_label(ax, chr(ord("a") + idx))
        remove_top_right_spines(ax)

    # Annotations on panel (a)
    axes[0].text(0.55, 0.025, "Routes to correct", fontsize=FONT_ANNOT,
                 color=TRAJ["stable_correct"], ha="center")
    axes[0].text(0.55, -0.020, "Routes to competitor", fontsize=FONT_ANNOT,
                 color=TRAJ["stable_wrong"], ha="center")
    axes[0].legend(fontsize=FONT_LEGEND, frameon=False, loc="upper left")

    out = output_dir / "fig4_attention_routing.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: Counterfactual Validation
# ---------------------------------------------------------------------------

def fig5_counterfactuals(parquet_dir: Path, output_dir: Path) -> Path:
    """Counterfactual experiments: removal, substitution, span deletion, controls.

    Data: ablation_results.parquet, patching_results.parquet,
          span_labels.parquet, negative_controls.parquet
    """
    abl, pat, sl, nc = load_causal_data(parquet_dir)

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.8), constrained_layout=True)

    # --- Panel (a): Simulated removal (violin) ---
    ax = axes[0, 0]
    for i, comp in enumerate(["attention", "mlp"]):
        vals = abl[abl["component"] == comp]["delta_shift"].values
        parts = ax.violinplot([vals], positions=[i], showmeans=True, showextrema=False,
                              widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(COMP[comp])
            pc.set_alpha(0.4)
        parts["cmeans"].set_color(COMP[comp])
        # Add mean annotation
        m = vals.mean()
        ax.text(i, m + 0.3, f"{m:.2f}", ha="center", fontsize=FONT_ANNOT, color=COMP[comp])

    add_zero_line(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Attention", "MLP"])
    ax.set_ylabel("Δδ from removal (logits)")
    ax.set_title("Simulated Removal", fontsize=FONT_TITLE)
    add_panel_label(ax, "a")
    remove_top_right_spines(ax)

    # --- Panel (b): Substitution summary (lollipop) ---
    ax = axes[0, 1]
    y_pos = [1, 0]
    for i, comp in enumerate(["attention", "mlp"]):
        sub = pat[pat["component"] == comp]["delta_shift"]
        mean, ci_lo, ci_hi = bootstrap_ci(sub.values)
        frac_pos = float((sub > 0).mean())

        # Mean + CI on left half
        ax.errorbar(mean, y_pos[i] + 0.15, xerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt="o", color=COMP[comp], markersize=5, capsize=3, lw=1.5)
        ax.text(mean + 0.3, y_pos[i] + 0.15, f"μ={mean:.2f}", fontsize=FONT_ANNOT,
                va="center", color=COMP[comp])

        # Fraction positive on lower row
        ax.plot(frac_pos * 10, y_pos[i] - 0.15, "s", color=COMP[comp], markersize=5)
        ax.text(frac_pos * 10 + 0.3, y_pos[i] - 0.15,
                f"{frac_pos:.0%} pos.", fontsize=FONT_ANNOT,
                va="center", color=COMP[comp])

    ax.axvline(0, color=NEUTRAL, lw=LW_REF, ls="--", zorder=0)
    ax.axvline(5, color=NEUTRAL, lw=LW_REF * 0.5, ls=":", alpha=0.3, zorder=0)
    ax.text(5, 1.5, "← frac. pos. ×10", fontsize=FONT_ANNOT - 1, color=NEUTRAL, ha="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["MLP", "Attention"])
    ax.set_xlabel("Δδ from substitution (logits)")
    ax.set_title("Substitution Effects", fontsize=FONT_TITLE)
    add_panel_label(ax, "b")
    remove_top_right_spines(ax)

    # --- Panel (c): Span deletion effects ---
    ax = axes[1, 0]
    for lbl, color in [("evidence", SPAN["evidence"]),
                       ("distractor", SPAN["distractor"]),
                       ("neutral", SPAN["neutral"])]:
        subset = sl[sl["span_label"] == lbl]["effect_delta"]
        if len(subset) > 0:
            ax.hist(subset, bins=40, alpha=0.45, color=color, density=True,
                    label=SPAN_LABELS[lbl])

    # Negative control reference lines
    for _, row in nc.iterrows():
        ctrl = row["control"]
        val = row["mean_effect_delta"]
        ls = "--" if ctrl == "shuffled" else ":"
        ax.axvline(val, color=NEUTRAL, lw=LW_REF, ls=ls, zorder=5)
        ax.text(val + 0.3, ax.get_ylim()[1] * 0.85, ctrl, fontsize=FONT_ANNOT - 1,
                color=NEUTRAL, rotation=90, va="top")

    ax.axvline(0, color=NEUTRAL, lw=LW_REF)
    ax.set_xlabel("Δ = δ_full − δ_deleted (logits)")
    ax.set_ylabel("Density")
    ax.set_title("Span Deletion Effects", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, frameon=False)
    add_panel_label(ax, "c")
    remove_top_right_spines(ax)

    # --- Panel (d): Negative controls ---
    ax = axes[1, 1]
    ctrl_order = ["observed", "shuffled", "sign_flipped"]
    ctrl_colors = {"observed": SPAN["evidence"], "shuffled": NEUTRAL, "sign_flipped": "#6b7280"}
    ctrl_labels = {"observed": "Observed\nevidence", "shuffled": "Shuffled\nlabels",
                   "sign_flipped": "Sign-flipped\neffects"}

    for i, ctrl in enumerate(ctrl_order):
        row = nc[nc["control"] == ctrl]
        if row.empty:
            continue
        val = float(row["mean_effect_delta"].iloc[0])
        ax.barh(i, val, color=ctrl_colors.get(ctrl, NEUTRAL), height=0.6, alpha=0.7)
        ax.text(val + 0.1, i, f"{val:.2f}", fontsize=FONT_ANNOT, va="center")

    ax.axvline(0, color=NEUTRAL, lw=LW_REF)
    ax.set_yticks(range(len(ctrl_order)))
    ax.set_yticklabels([ctrl_labels[c] for c in ctrl_order], fontsize=FONT_TICK)
    ax.set_xlabel("Mean effect δ (logits)")
    ax.set_title("Negative Controls", fontsize=FONT_TITLE)
    add_panel_label(ax, "d")
    remove_top_right_spines(ax)

    out = output_dir / "fig5_counterfactuals.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: Prompt Journey
# ---------------------------------------------------------------------------

def fig6_prompt_journey(
    parquet_dir: Path,
    output_dir: Path,
    prompts_path: Optional[Path] = None,
    spans_path: Optional[Path] = None,
) -> Path:
    """A single prompt's trajectory through decision space.

    Data: decision_metrics.parquet, tracing_scalars.parquet,
          prompt_packs/ccc_baseline_v1_3000.jsonl (prompt text),
          span_labels.parquet (span roles)
    """
    dm = load_decision_metrics(parquet_dir)
    ts = load_tracing_scalars(parquet_dir)

    # Select a stable-correct prompt present in tracing data with clear trajectory
    sc_prompts = ts[ts["trajectory_type"] == "stable_correct"][
        ["model_id", "prompt_uid"]
    ].drop_duplicates()

    best_prompt = None
    best_final_delta = -np.inf
    for _, row in sc_prompts.iterrows():
        mid, puid = row["model_id"], row["prompt_uid"]
        pdm = dm[(dm["model_id"] == mid) & (dm["prompt_uid"] == puid)].sort_values("layer_index")
        if pdm.empty:
            continue
        final_delta = float(pdm["delta"].iloc[-1])
        # Prefer prompts with competitor switches for visual interest
        n_comp_switches = (pdm["competitor"].shift() != pdm["competitor"]).sum()
        score = final_delta + n_comp_switches * 0.5
        if score > best_final_delta:
            best_final_delta = score
            best_prompt = (mid, puid)

    if best_prompt is None:
        # Fallback to any prompt
        first = sc_prompts.iloc[0]
        best_prompt = (first["model_id"], first["prompt_uid"])

    mid, puid = best_prompt
    pdm = dm[(dm["model_id"] == mid) & (dm["prompt_uid"] == puid)].sort_values("layer_index")
    pts = ts[(ts["model_id"] == mid) & (ts["prompt_uid"] == puid)].sort_values("layer_index")
    L = MODEL_LAYERS.get(mid, int(pdm["layer_index"].max()) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    # --- Panel (a): Prompt schematic ---
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Load span labels if available
    span_labels_df = pd.read_parquet(parquet_dir / "span_labels.parquet")
    prompt_spans = span_labels_df[
        (span_labels_df["model_id"] == mid) & (span_labels_df["prompt_uid"] == puid)
    ]

    span_roles = {
        "instruction": ("#e5e7eb", "Instruction"),
        "question_stem": ("#dbeafe", "Question"),
        "option_correct": ("#d1fae5", "Correct opt."),
        "option_competitor": ("#fee2e2", "Competitor"),
        "option_other": ("#f3f4f6", "Other opt."),
        "post_options": ("#f3f4f6", "Post-options"),
    }

    # Map span_id/span_role to labels
    displayed_spans = []
    for _, sr in prompt_spans.iterrows():
        role = sr.get("span_role", sr.get("span_id", "unknown"))
        lbl = sr.get("span_label", "neutral")
        displayed_spans.append((role, lbl, float(sr.get("effect_delta", 0.0))))

    # If we have span data, show as stacked blocks
    if displayed_spans:
        y_start = 0.92
        y_step = 0.12
        for i, (role, lbl, eff) in enumerate(displayed_spans[:7]):
            y = y_start - i * y_step
            color = span_roles.get(role, ("#f3f4f6", role))[0]
            # Override with evidence/distractor color
            if lbl == "evidence":
                color = "#d1fae5"
                outline = SPAN["evidence"]
            elif lbl == "distractor":
                color = "#fee2e2"
                outline = SPAN["distractor"]
            else:
                outline = "#d1d5db"

            ax.add_patch(plt.Rectangle((0.05, y - 0.04), 0.65, 0.08,
                                       facecolor=color, edgecolor=outline, lw=0.8))
            label_text = span_roles.get(role, (None, role))[1]
            ax.text(0.38, y, label_text, fontsize=FONT_ANNOT, ha="center", va="center")
            sign = "+" if eff > 0 else ""
            ax.text(0.78, y, f"Δ={sign}{eff:.2f}", fontsize=FONT_ANNOT - 0.5,
                    ha="left", va="center", color=NEUTRAL)
    else:
        ax.text(0.5, 0.5, "Span data\nnot available", fontsize=FONT_AXIS,
                ha="center", va="center", color=NEUTRAL)

    ax.set_title("Prompt Structure", fontsize=FONT_TITLE)
    add_panel_label(ax, "a")

    # --- Panel (b): δ trajectory + competitor strip ---
    ax = axes[1]
    depth = pdm["layer_index"] / (L - 1)
    delta = pdm["delta"].values

    ax.plot(depth, delta, color=NEUTRAL, lw=LW_DATA, zorder=3)
    ax.fill_between(depth, 0, delta, where=delta >= 0, color=TRAJ["stable_correct"],
                    alpha=0.15, interpolate=True)
    ax.fill_between(depth, 0, delta, where=delta < 0, color=TRAJ["stable_wrong"],
                    alpha=0.15, interpolate=True)

    # Tail window shading
    tail_depth = (L - 8) / (L - 1)
    ax.axvspan(tail_depth, 1.0, color="#fef3c7", alpha=0.3, zorder=0)
    ax.text(tail_depth + 0.02, ax.get_ylim()[1] * 0.9 if len(delta) > 0 else 1.0,
            "tail", fontsize=FONT_ANNOT, color="#92400e")

    # Competitor identity strip below x-axis
    competitors = pdm["competitor"].values
    comp_depths = depth.values
    unique_comps = sorted(set(competitors))
    comp_cmap = {"A": "#60a5fa", "B": "#f87171", "C": "#34d399", "D": "#fbbf24"}
    strip_y = ax.get_ylim()[0] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    strip_h = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for i in range(len(comp_depths) - 1):
        ax.fill_between([comp_depths[i], comp_depths[i + 1]],
                        strip_y, strip_y + strip_h,
                        color=comp_cmap.get(competitors[i], NEUTRAL),
                        clip_on=False, zorder=5)

    # Competitor legend (tiny)
    for j, c in enumerate(unique_comps):
        ax.plot([], [], "s", color=comp_cmap.get(c, NEUTRAL), markersize=3,
                label=f"Comp={c}")
    ax.legend(fontsize=FONT_ANNOT - 1, frameon=False, loc="lower right", ncol=len(unique_comps))

    add_zero_line(ax)
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("δ (logits)")
    ax.set_title("Decision Trajectory", fontsize=FONT_TITLE)
    add_panel_label(ax, "b")
    remove_top_right_spines(ax)

    # --- Panel (c): Cumulative component path ---
    ax = axes[2]
    if not pts.empty:
        pts_sorted = pts.sort_values("layer_index")
        cum_attn = pts_sorted["s_attn"].cumsum().values
        cum_mlp = pts_sorted["s_mlp"].cumsum().values
        n_layers = len(cum_attn)

        # Color by depth
        norm = plt.Normalize(0, n_layers - 1)
        cmap = plt.cm.Blues

        for i in range(n_layers - 1):
            ax.annotate("", xy=(cum_attn[i + 1], cum_mlp[i + 1]),
                        xytext=(cum_attn[i], cum_mlp[i]),
                        arrowprops=dict(arrowstyle="->", color=cmap(norm(i)),
                                        lw=1.2, shrinkA=0, shrinkB=0))

        # Start and end markers
        ax.plot(cum_attn[0], cum_mlp[0], "o", color=cmap(norm(0)),
                markersize=6, markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        ax.plot(cum_attn[-1], cum_mlp[-1], "o", color=cmap(norm(n_layers - 1)),
                markersize=6, zorder=5)
        ax.text(cum_attn[0] + 0.2, cum_mlp[0], "L0", fontsize=FONT_ANNOT)
        ax.text(cum_attn[-1] + 0.2, cum_mlp[-1], f"L{n_layers - 1}",
                fontsize=FONT_ANNOT)

    ax.set_xlabel("Σ Attention contribution (logits)")
    ax.set_ylabel("Σ MLP contribution (logits)")
    ax.set_title("Component-Space Path", fontsize=FONT_TITLE)
    add_panel_label(ax, "c")
    remove_top_right_spines(ax)

    out = output_dir / "fig6_prompt_journey.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    parquet_dir: Path,
    output_dir: Path,
    prompts_path: Optional[Path] = None,
    spans_path: Optional[Path] = None,
) -> list[Path]:
    """Generate all six paper figures and return output paths."""
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    print("Generating Figure 1: Three Primitives...")
    paths.append(fig1_primitives(parquet_dir, output_dir))
    print(f"  → {paths[-1]}")

    print("Generating Figure 2: Phase Diagram...")
    paths.append(fig2_phase_diagram(parquet_dir, output_dir))
    print(f"  → {paths[-1]}")

    print("Generating Figure 3: Drift Decomposition...")
    paths.append(fig3_decomposition(parquet_dir, output_dir))
    print(f"  → {paths[-1]}")

    print("Generating Figure 4: Attention Routing...")
    paths.append(fig4_attention_routing(parquet_dir, output_dir))
    print(f"  → {paths[-1]}")

    print("Generating Figure 5: Counterfactual Validation...")
    paths.append(fig5_counterfactuals(parquet_dir, output_dir))
    print(f"  → {paths[-1]}")

    print("Generating Figure 6: Prompt Journey...")
    paths.append(fig6_prompt_journey(parquet_dir, output_dir, prompts_path, spans_path))
    print(f"  → {paths[-1]}")

    print(f"\nAll {len(paths)} figures generated successfully.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from parquet data.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("results/parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/final_paper/figures"))
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--spans", type=Path, default=None)
    args = parser.parse_args()

    generate_all_figures(args.parquet_dir, args.output_dir, args.prompts, args.spans)


if __name__ == "__main__":
    main()
