#!/usr/bin/env python3
"""Regenerate all 6 paper figures from existing parquet data.

Fixes applied:
  - Attention routing: re-index option columns to relative (correct/competitor/other)
  - Phase diagram: replace algebraically redundant Σg with tail_flips
  - Layer indexing: normalize to depth fraction l/(L-1) across models
  - Causal panel (c): show Δ distribution without "validation" framing
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
PARQUET = RESULTS / "parquet"
OUT_DIR = REPO / "paper" / "final_paper"

MODEL_LAYERS = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
}

OPTION_COLS = ["option_A", "option_B", "option_C", "option_D"]
KEY_TO_OPTION = {"A": "option_A", "B": "option_B", "C": "option_C", "D": "option_D"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET / name)


def _get_prompt_keys(dm: pd.DataFrame) -> pd.DataFrame:
    """Get correct_key and final-layer competitor for each (model, prompt)."""
    final = (
        dm.sort_values("layer_index")
        .groupby(["model_id", "prompt_uid"])
        .last()
        .reset_index()[["model_id", "prompt_uid", "correct_key", "competitor", "is_correct"]]
    )
    final = final.rename(columns={"competitor": "competitor_key"})
    return final


def _remap_option_to_relative(row_span_label: str, correct_key: str, competitor_key: str) -> str:
    """Map absolute span labels to relative correctness labels."""
    if row_span_label not in OPTION_COLS:
        return row_span_label
    letter = row_span_label.replace("option_", "")
    if letter == correct_key:
        return "option_correct"
    elif letter == competitor_key:
        return "option_competitor"
    else:
        return "option_other"


def _depth_frac(layer_index: pd.Series, model_id: pd.Series) -> pd.Series:
    """Normalize layer index to [0, 1] per model."""
    L = model_id.map(MODEL_LAYERS)
    return layer_index / (L - 1).clip(lower=1)


def _get_prompt_types(pt: pd.DataFrame) -> pd.DataFrame:
    return pt[["model_id", "prompt_uid", "trajectory_type"]].copy()


# ---------------------------------------------------------------------------
# Figure 1: Three primitives (conceptual – unchanged but with depth frac)
# ---------------------------------------------------------------------------

def fig1_three_primitives(dm: pd.DataFrame, pt: pd.DataFrame) -> Path:
    """Trajectory overview with depth-fraction x-axis."""
    merged = dm.merge(pt[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")
    merged["depth"] = _depth_frac(merged["layer_index"], merged["model_id"])

    types = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    colors = {"stable_correct": "#2ca02c", "stable_wrong": "#d62728",
              "unstable_correct": "#1f77b4", "unstable_wrong": "#ff7f0e"}
    labels = {"stable_correct": "Stable-Correct", "stable_wrong": "Stable-Wrong",
              "unstable_correct": "Unstable-Correct", "unstable_wrong": "Unstable-Wrong"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: δ(depth) by type
    agg = merged.groupby(["trajectory_type", merged["depth"].round(2)]).agg(
        delta_mean=("delta", "mean")).reset_index()
    for t in types:
        s = agg[agg["trajectory_type"] == t]
        axes[0].plot(s["depth"], s["delta_mean"], color=colors[t], lw=2, label=labels[t])
    axes[0].axhline(0, color="#444", lw=0.8, alpha=0.4)
    axes[0].set_xlabel("Normalized Depth l/(L−1)")
    axes[0].set_ylabel("Mean δ (Decision Margin)")
    axes[0].set_title("State: Decision Margin")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.2)

    # Center: drift g(depth) by type
    agg_g = merged.groupby(["trajectory_type", merged["depth"].round(2)]).agg(
        drift_mean=("drift", "mean")).reset_index()
    for t in types:
        s = agg_g[agg_g["trajectory_type"] == t]
        axes[1].plot(s["depth"], s["drift_mean"], color=colors[t], lw=2, label=labels[t])
    axes[1].axhline(0, color="#444", lw=0.8, alpha=0.4)
    axes[1].set_xlabel("Normalized Depth l/(L−1)")
    axes[1].set_ylabel("Mean g (Drift)")
    axes[1].set_title("Motion: Per-Layer Drift")
    axes[1].legend(fontsize=7, loc="best")
    axes[1].grid(alpha=0.2)

    # Right: boundary |δ| by type
    agg_b = merged.groupby(["trajectory_type", merged["depth"].round(2)]).agg(
        boundary_mean=("boundary", "mean")).reset_index()
    for t in types:
        s = agg_b[agg_b["trajectory_type"] == t]
        axes[2].plot(s["depth"], s["boundary_mean"], color=colors[t], lw=2, label=labels[t])
    axes[2].set_xlabel("Normalized Depth l/(L−1)")
    axes[2].set_ylabel("Mean |δ| (Boundary)")
    axes[2].set_title("Boundary: Decision Flip Proximity")
    axes[2].legend(fontsize=7, loc="best")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "fig1_three_primitives.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Phase diagram with non-redundant axes
# ---------------------------------------------------------------------------

def fig2_phase_diagram(dm: pd.DataFrame, pt: pd.DataFrame) -> Path:
    """Phase diagram: boundary proximity at L-8 vs tail flip count.

    Avoids the algebraic redundancy of δ(L) vs Σg (which telescopes).
    """
    merged = dm.merge(pt[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")

    records = []
    for (mid, puid), grp in merged.groupby(["model_id", "prompt_uid"]):
        grp = grp.sort_values("layer_index")
        L = int(MODEL_LAYERS.get(mid, grp["layer_index"].max() + 1))
        tau = 8
        tail_start = max(L - tau, 0)
        tail = grp[grp["layer_index"] >= tail_start]
        if tail.empty:
            continue

        # Axis 1: boundary proximity at L - tau  (|δ(L-τ)|)
        entry_row = grp[grp["layer_index"] == tail_start]
        boundary_at_entry = float(entry_row["boundary"].iloc[0]) if not entry_row.empty else np.nan

        # Axis 2: tail flips = number of sign changes of δ in tail window
        signs = np.sign(tail["delta"].values)
        signs = signs[signs != 0]  # ignore exact zeros
        tail_flips = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

        traj = grp["trajectory_type"].iloc[0] if "trajectory_type" in grp.columns else "unknown"
        records.append({
            "model_id": mid,
            "prompt_uid": puid,
            "boundary_at_entry": boundary_at_entry,
            "tail_flips": tail_flips,
            "trajectory_type": traj,
        })

    pdf = pd.DataFrame(records)

    colors = {"stable_correct": "#2ca02c", "stable_wrong": "#d62728",
              "unstable_correct": "#1f77b4", "unstable_wrong": "#ff7f0e"}
    labels = {"stable_correct": "Stable-Correct", "stable_wrong": "Stable-Wrong",
              "unstable_correct": "Unstable-Correct", "unstable_wrong": "Unstable-Wrong"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]:
        s = pdf[pdf["trajectory_type"] == t]
        # Jitter tail_flips slightly to see density
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(s))
        ax.scatter(s["boundary_at_entry"], s["tail_flips"].values + jitter,
                   c=colors.get(t, "#999"), s=14, alpha=0.45, label=labels.get(t, t))
    ax.set_xlabel("|δ(L−τ)|  (Boundary Proximity at Tail Entry)")
    ax.set_ylabel("Tail Flips  (Sign Changes in Last τ Layers)")
    ax.set_title("Phase Diagram — Non-Redundant Axes")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "fig2_phase_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Drift decomposition with depth-fraction axis
# ---------------------------------------------------------------------------

def fig3_decomposition(ts: pd.DataFrame, pt: pd.DataFrame) -> Path:
    """Per-layer drift decomposition (attention + MLP) with normalized depth."""
    merged = ts.merge(pt[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")
    merged["depth"] = _depth_frac(merged["layer_index"], merged["model_id"])
    merged["depth_bin"] = (merged["depth"] * 20).round() / 20  # 5% bins

    types = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    colors_attn = {"stable_correct": "#2ca02c", "stable_wrong": "#d62728",
                   "unstable_correct": "#1f77b4", "unstable_wrong": "#ff7f0e"}
    colors_mlp = {"stable_correct": "#98df8a", "stable_wrong": "#ff9896",
                  "unstable_correct": "#aec7e8", "unstable_wrong": "#ffbb78"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: drift decomposition averaged across all types
    agg_all = merged.groupby("depth_bin").agg(
        s_attn=("s_attn", "mean"), s_mlp=("s_mlp", "mean"), drift=("drift", "mean")
    ).reset_index()
    axes[0].plot(agg_all["depth_bin"], agg_all["s_attn"], color="#1f77b4", lw=2.5, label="Attention scalar")
    axes[0].plot(agg_all["depth_bin"], agg_all["s_mlp"], color="#d62728", lw=2.5, label="MLP scalar")
    axes[0].plot(agg_all["depth_bin"], agg_all["drift"], color="#444", lw=1.5, ls="--", label="Observed drift")
    axes[0].axhline(0, color="#444", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("Scalar Value")
    axes[0].set_title("Drift Decomposition: Attention vs MLP (All Trajectories)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # Bottom: mean |component| by trajectory type
    agg_type = merged.groupby(["trajectory_type", "depth_bin"]).agg(
        abs_attn=("s_attn", lambda x: np.mean(np.abs(x))),
        abs_mlp=("s_mlp", lambda x: np.mean(np.abs(x))),
    ).reset_index()
    for t in types:
        s = agg_type[agg_type["trajectory_type"] == t]
        label_t = t.replace("_", "-").title()
        axes[1].plot(s["depth_bin"], s["abs_attn"], color=colors_attn[t], lw=2, label=f"{label_t} attn")
        axes[1].plot(s["depth_bin"], s["abs_mlp"], color=colors_mlp[t], lw=2, ls="--", label=f"{label_t} MLP")
    axes[1].set_xlabel("Normalized Depth l/(L−1)")
    axes[1].set_ylabel("Mean |Component Scalar|")
    axes[1].set_title("Component Magnitudes by Trajectory Type")
    axes[1].legend(fontsize=6, ncol=2, loc="upper right")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "fig3_decomposition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Counterfactual accounting panel
# ---------------------------------------------------------------------------

def fig4_causal_panel(abl: pd.DataFrame, pat: pd.DataFrame, sl: pd.DataFrame, nc: pd.DataFrame) -> Path:
    """Counterfactual accounting: (a) ablation shifts, (b) patching shifts,
    (c) Δ distribution by span type (operational, not validation)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Ablation
    ab = abl.groupby("component", as_index=False).agg(mean=("delta_shift", "mean"), std=("delta_shift", "std"))
    axes[0].bar(ab["component"], ab["mean"], yerr=ab["std"], color=["#1f77b4", "#d62728"],
                alpha=0.8, capsize=4)
    axes[0].set_title("(a) Simulated Removal: Δδ")
    axes[0].set_ylabel("Mean delta shift")
    axes[0].axhline(0, color="#444", lw=0.5)
    axes[0].grid(alpha=0.2)

    # (b) Patching
    pa = pat.groupby("component", as_index=False).agg(mean=("delta_shift", "mean"), std=("delta_shift", "std"))
    axes[1].bar(pa["component"], pa["mean"], yerr=pa["std"], color=["#2ca02c", "#ff7f0e"],
                alpha=0.8, capsize=4)
    axes[1].set_title("(b) Simulated Substitution: Δδ")
    axes[1].set_ylabel("Mean delta shift")
    axes[1].axhline(0, color="#444", lw=0.5)
    axes[1].grid(alpha=0.2)

    # (c) Distribution of effect_delta by span_label (NOT validation)
    for lbl, color in [("evidence", "#2ca02c"), ("distractor", "#d62728"), ("neutral", "#999")]:
        subset = sl[sl["span_label"] == lbl]
        if not subset.empty:
            axes[2].hist(subset["effect_delta"], bins=40, alpha=0.55, color=color, label=lbl, density=True)
    axes[2].axvline(0, color="#444", lw=0.8)
    axes[2].set_title("(c) Counterfactual Effect Distribution\nby Operationally Defined Span Type")
    axes[2].set_xlabel("Δ = δ_full − δ_deleted")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "fig4_causal_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: Attention routing with RELATIVE option indexing
# ---------------------------------------------------------------------------

def fig5_attention_routing(
    mass: pd.DataFrame,
    contrib: pd.DataFrame,
    dm: pd.DataFrame,
    pt: pd.DataFrame,
) -> Path:
    """Attention mass & contribution heatmaps with option columns indexed
    relative to correctness: correct / competitor / other."""

    keys = _get_prompt_keys(dm)  # model_id, prompt_uid, correct_key, competitor_key

    # Merge keys into attention data
    mass_m = mass.merge(keys[["model_id", "prompt_uid", "correct_key", "competitor_key"]],
                        on=["model_id", "prompt_uid"], how="inner")
    contrib_m = contrib.merge(keys[["model_id", "prompt_uid", "correct_key", "competitor_key"]],
                              on=["model_id", "prompt_uid"], how="inner")

    # Remap span labels
    mass_m["span_rel"] = mass_m.apply(
        lambda r: _remap_option_to_relative(r["span_label"], r["correct_key"], r["competitor_key"]), axis=1)
    contrib_m["span_rel"] = contrib_m.apply(
        lambda r: _remap_option_to_relative(r["span_label"], r["correct_key"], r["competitor_key"]), axis=1)

    # Add depth fraction
    mass_m["depth"] = _depth_frac(mass_m["layer_index"], mass_m["model_id"])
    contrib_m["depth"] = _depth_frac(contrib_m["layer_index"], contrib_m["model_id"])

    # Add trajectory type
    traj = pt[["model_id", "prompt_uid", "trajectory_type"]]
    mass_m = mass_m.merge(traj, on=["model_id", "prompt_uid"], how="left")
    contrib_m = contrib_m.merge(traj, on=["model_id", "prompt_uid"], how="left")

    span_order = ["instruction", "question_stem", "option_correct", "option_competitor", "option_other", "post_options"]

    # Normalize attention mass per token using char length as proxy
    # (We don't have exact token counts, but char count from spans.jsonl is a reasonable proxy)
    # Load spans for approximate token normalization
    spans_path = RESULTS / "spans.jsonl"
    char_counts: dict[tuple[str, str, str], int] = {}
    if spans_path.exists():
        with spans_path.open() as f:
            for line in f:
                rec = json.loads(line)
                char_counts[(rec["model_id"], rec["prompt_uid"], rec["span_id"])] = rec["end_char"] - rec["start_char"]

    # Compute per-token mass (approximate: chars / 4 ≈ tokens for English)
    def _approx_tokens(model_id, prompt_uid, span_label):
        chars = char_counts.get((model_id, prompt_uid, span_label), 0)
        return max(chars / 4.0, 1.0)  # rough chars-to-tokens

    mass_m["approx_tokens"] = mass_m.apply(
        lambda r: _approx_tokens(r["model_id"], r["prompt_uid"], r["span_label"]), axis=1)
    mass_m["mass_per_token"] = mass_m["attention_mass"] / mass_m["approx_tokens"]

    # Build heatmaps by trajectory type
    traj_types = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    traj_labels = {"stable_correct": "Stable-Correct", "stable_wrong": "Stable-Wrong",
                   "unstable_correct": "Unstable-Correct", "unstable_wrong": "Unstable-Wrong"}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    depth_bins = np.linspace(0, 1.0, 21)
    depth_labels = [f"{(a+b)/2:.2f}" for a, b in zip(depth_bins[:-1], depth_bins[1:])]

    for col_idx, tt in enumerate(traj_types):
        # Top row: attention mass per token
        sub = mass_m[mass_m["trajectory_type"] == tt].copy()
        sub["depth_bin"] = pd.cut(sub["depth"], bins=depth_bins, labels=depth_labels, include_lowest=True)
        pivot = sub.groupby(["depth_bin", "span_rel"])["mass_per_token"].mean().unstack(fill_value=0)
        # Reorder columns
        pivot = pivot.reindex(columns=[c for c in span_order if c in pivot.columns])
        if not pivot.empty:
            im = axes[0, col_idx].imshow(pivot.values.T, aspect="auto", cmap="YlOrRd",
                                         extent=[0, 1, len(pivot.columns)-0.5, -0.5])
            axes[0, col_idx].set_yticks(range(len(pivot.columns)))
            axes[0, col_idx].set_yticklabels(pivot.columns, fontsize=7)
            axes[0, col_idx].set_xlabel("Depth", fontsize=8)
        axes[0, col_idx].set_title(f"{traj_labels[tt]}\nMass/Token", fontsize=9)

        # Bottom row: decision-aligned contribution
        sub_c = contrib_m[contrib_m["trajectory_type"] == tt].copy()
        sub_c["depth_bin"] = pd.cut(sub_c["depth"], bins=depth_bins, labels=depth_labels, include_lowest=True)
        pivot_c = sub_c.groupby(["depth_bin", "span_rel"])["attention_contribution"].mean().unstack(fill_value=0)
        pivot_c = pivot_c.reindex(columns=[c for c in span_order if c in pivot_c.columns])
        if not pivot_c.empty:
            vmax = max(abs(pivot_c.values.min()), abs(pivot_c.values.max())) or 1.0
            axes[1, col_idx].imshow(pivot_c.values.T, aspect="auto", cmap="RdBu_r",
                                    vmin=-vmax, vmax=vmax,
                                    extent=[0, 1, len(pivot_c.columns)-0.5, -0.5])
            axes[1, col_idx].set_yticks(range(len(pivot_c.columns)))
            axes[1, col_idx].set_yticklabels(pivot_c.columns, fontsize=7)
            axes[1, col_idx].set_xlabel("Depth", fontsize=8)
        axes[1, col_idx].set_title(f"{traj_labels[tt]}\nContribution", fontsize=9)

    fig.suptitle("Attention Routing: Mass per Token (top) vs Decision-Aligned Contribution (bottom)\n"
                 "Option columns indexed relative to correctness", fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig5_attention_routing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: Prompt flow (conceptual, with corrected labels)
# ---------------------------------------------------------------------------

def fig6_prompt_flow(dm: pd.DataFrame, ts: pd.DataFrame, pt: pd.DataFrame) -> Path:
    """A single prompt's journey with corrected labeling."""
    # Pick a stable-correct example prompt
    sc = pt[pt["trajectory_type"] == "stable_correct"]
    if sc.empty:
        sc = pt.head(1)
    example = sc.iloc[0]
    mid, puid = example["model_id"], example["prompt_uid"]

    prompt_dm = dm[(dm["model_id"] == mid) & (dm["prompt_uid"] == puid)].sort_values("layer_index")
    prompt_ts = ts[(ts["model_id"] == mid) & (ts["prompt_uid"] == puid)].sort_values("layer_index")
    L = MODEL_LAYERS.get(mid, int(prompt_dm["layer_index"].max()) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: δ trajectory
    depth = prompt_dm["layer_index"] / (L - 1)
    axes[0].plot(depth, prompt_dm["delta"], color="#1f77b4", lw=2.5)
    axes[0].axhline(0, color="#444", lw=0.8, alpha=0.4)
    axes[0].fill_between(depth, 0, prompt_dm["delta"],
                         where=prompt_dm["delta"] > 0, alpha=0.15, color="#2ca02c")
    axes[0].fill_between(depth, 0, prompt_dm["delta"],
                         where=prompt_dm["delta"] < 0, alpha=0.15, color="#d62728")
    axes[0].set_xlabel("Normalized Depth")
    axes[0].set_ylabel("δ (Decision Margin)")
    axes[0].set_title("Trajectory: State δ(l)")
    axes[0].grid(alpha=0.2)

    # Center: attention & MLP scalars
    if not prompt_ts.empty:
        depth_ts = prompt_ts["layer_index"] / (L - 1)
        axes[1].plot(depth_ts, prompt_ts["s_attn"], color="#1f77b4", lw=2, label="Attention")
        axes[1].plot(depth_ts, prompt_ts["s_mlp"], color="#d62728", lw=2, label="MLP")
        axes[1].axhline(0, color="#444", lw=0.5, alpha=0.4)
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Normalized Depth")
    axes[1].set_ylabel("Component Scalar")
    axes[1].set_title("Decomposition: Routing vs Injection")
    axes[1].grid(alpha=0.2)

    # Right: cumulative drift
    cumg = prompt_dm["drift"].cumsum()
    axes[2].plot(depth, cumg, color="#444", lw=2)
    axes[2].axhline(0, color="#444", lw=0.5, alpha=0.4)
    axes[2].set_xlabel("Normalized Depth")
    axes[2].set_ylabel("Cumulative Drift")
    axes[2].set_title("Convergence Trajectory")
    axes[2].grid(alpha=0.2)

    fig.suptitle(f"Example Prompt Journey — {example['trajectory_type'].replace('_',' ').title()}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig6_prompt_flow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    dm = _load("decision_metrics.parquet")
    pt = _load("prompt_types.parquet")
    ts = _load("tracing_scalars.parquet")
    mass = _load("attention_mass_by_span.parquet")
    contrib = _load("attention_contrib_by_span.parquet")
    abl = _load("ablation_results.parquet")
    pat = _load("patching_results.parquet")
    sl = _load("span_labels.parquet")
    nc = _load("negative_controls.parquet")

    print("Generating Figure 1: Three Primitives...")
    f1 = fig1_three_primitives(dm, pt)
    print(f"  -> {f1}")

    print("Generating Figure 2: Phase Diagram (non-redundant axes)...")
    f2 = fig2_phase_diagram(dm, pt)
    print(f"  -> {f2}")

    print("Generating Figure 3: Drift Decomposition...")
    f3 = fig3_decomposition(ts, pt)
    print(f"  -> {f3}")

    print("Generating Figure 4: Counterfactual Panel...")
    f4 = fig4_causal_panel(abl, pat, sl, nc)
    print(f"  -> {f4}")

    print("Generating Figure 5: Attention Routing (relative indexing)...")
    f5 = fig5_attention_routing(mass, contrib, dm, pt)
    print(f"  -> {f5}")

    print("Generating Figure 6: Prompt Flow...")
    f6 = fig6_prompt_flow(dm, ts, pt)
    print(f"  -> {f6}")

    print("\nAll figures regenerated successfully.")


if __name__ == "__main__":
    main()
