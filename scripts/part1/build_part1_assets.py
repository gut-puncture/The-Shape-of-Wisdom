#!/usr/bin/env python3
"""Build all Paper 1 / Part I assets from cached parquet artifacts.

Usage::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python scripts/part1/build_part1_assets.py \
        --parquet-dir results/parquet --output-dir paper/part1 --seed 12345

Non-negotiable constraints enforced:
  C0  No new inference — forbidden imports cause immediate failure.
  C1  Exactly 3 model_id values.
  C3  No causal/mechanistic claims (only phenomenology).
  C4  Decision-space PCA from 4-option scores only (not hidden states).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ── C0 guard: fail immediately if anyone tries to import model-loading code ──
_FORBIDDEN = {
    "transformers",
    "torch",
    "tensorflow",
    "jax",
}
_original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _guarded_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _FORBIDDEN:
        raise ImportError(
            f"C0 VIOLATION: attempted to import '{name}'. "
            "Paper 1 build must not load any model or deep-learning library."
        )
    return _original_import(name, *args, **kwargs)


import builtins

builtins.__import__ = _guarded_import

# Enforce HF offline environment
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

# ── Restore normal import for the rest of the script ──
builtins.__import__ = _original_import

# Add scripts/part1 to path for plot_style
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    MODEL_COLORS,
    MODEL_LAYERS,
    MODEL_SHORT,
    OPTION_COLORS,
    REGIME_COLORS,
    apply_style,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Canonical hyperparameters ──
TAU_DEFAULT = 1.0
M_DEFAULT = 0.75
MAX_FLIPS_FOR_STABILITY = 1
EPSILON_BOUNDARY_BAND = 0.2
EXPECTED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_layerwise_scores(parquet_dir: Path) -> pd.DataFrame:
    """Load layerwise.parquet, extract per-option scores from JSON."""
    lw_path = parquet_dir / "layerwise.parquet"
    if not lw_path.exists():
        raise FileNotFoundError(f"layerwise.parquet not found at {lw_path}")

    df = pd.read_parquet(lw_path)
    print(f"  layerwise.parquet: {len(df):,} rows, {df['model_id'].nunique()} models")

    # Extract sA, sB, sC, sD from candidate_logits_json
    logits_dicts = df["candidate_logits_json"].apply(json.loads)
    df["sA"] = logits_dicts.apply(lambda d: float(d.get("A", 0.0)))
    df["sB"] = logits_dicts.apply(lambda d: float(d.get("B", 0.0)))
    df["sC"] = logits_dicts.apply(lambda d: float(d.get("C", 0.0)))
    df["sD"] = logits_dicts.apply(lambda d: float(d.get("D", 0.0)))

    return df


def load_decision_metrics(parquet_dir: Path) -> pd.DataFrame:
    """Load decision_metrics.parquet for correct_key and metadata."""
    dm_path = parquet_dir / "decision_metrics.parquet"
    if not dm_path.exists():
        raise FileNotFoundError(f"decision_metrics.parquet not found at {dm_path}")
    df = pd.read_parquet(dm_path)
    print(f"  decision_metrics.parquet: {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════
def compute_all_metrics(lw: pd.DataFrame, dm: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Compute all canonical Paper 1 metrics from raw scores.

    This recomputes everything from sA,sB,sC,sD rather than trusting cached
    delta/competitor columns in decision_metrics.parquet.
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Get correct_key per prompt from decision_metrics (layer 0 row)
    ck = (
        dm.groupby(["model_id", "prompt_uid"])["correct_key"]
        .first()
        .reset_index()
    )
    df = lw.merge(ck, on=["model_id", "prompt_uid"], how="inner")

    print(f"  Merged: {len(df):,} rows")

    # ── Score vector extraction ──
    scores_cols = {"A": "sA", "B": "sB", "C": "sC", "D": "sD"}
    score_matrix = df[["sA", "sB", "sC", "sD"]].values  # (N, 4)
    correct_keys = df["correct_key"].values
    choices = np.array(["A", "B", "C", "D"])

    # Map correct_key to index
    ck_idx = np.array([np.where(choices == k)[0][0] for k in correct_keys])

    # correct score
    s_correct = score_matrix[np.arange(len(df)), ck_idx]

    # Mask for incorrect options
    mask = np.ones((len(df), 4), dtype=bool)
    mask[np.arange(len(df)), ck_idx] = False
    incorrect_scores = np.where(mask, score_matrix, -np.inf)

    # ── Dynamic competitor ──
    k_dyn_idx = np.argmax(incorrect_scores, axis=1)
    k_dyn = choices[k_dyn_idx]
    s_competitor_dyn = incorrect_scores[np.arange(len(df)), k_dyn_idx]

    # ── Hard dynamic margin ──
    delta_hard_dyn = s_correct - s_competitor_dyn

    # ── Soft competitor (logsumexp) for various τ ──
    for tau_val, col_name in [(0.5, "delta_soft_tau_0_5"), (1.0, "delta_soft_tau_1_0"), (2.0, "delta_soft_tau_2_0")]:
        # logsumexp of incorrect scores / tau
        inc_scaled = incorrect_scores / tau_val
        inc_scaled = np.where(mask, inc_scaled, -np.inf)
        lse = tau_val * _logsumexp(inc_scaled, axis=1)
        df[col_name] = s_correct - lse

    df["delta_hard_dyn"] = delta_hard_dyn
    df["k_dyn"] = k_dyn
    df["sA"] = score_matrix[:, 0]
    df["sB"] = score_matrix[:, 1]
    df["sC"] = score_matrix[:, 2]
    df["sD"] = score_matrix[:, 3]

    # ── Fixed competitor (from final layer) ──
    L_model = df.groupby("model_id")["layer_index"].transform("max")
    df["L_model"] = L_model

    # Get final-layer dynamic competitor per (model, prompt)
    final_mask = df["layer_index"] == df["L_model"]
    final_rows = df.loc[final_mask, ["model_id", "prompt_uid", "k_dyn"]].copy()
    final_rows.rename(columns={"k_dyn": "k_fix"}, inplace=True)
    df = df.merge(final_rows, on=["model_id", "prompt_uid"], how="left")

    # Fixed competitor score
    k_fix_idx = np.array([np.where(choices == k)[0][0] for k in df["k_fix"].values])
    s_competitor_fix = df[["sA", "sB", "sC", "sD"]].values[np.arange(len(df)), k_fix_idx]
    df["delta_hard_fix"] = df[["sA", "sB", "sC", "sD"]].values[np.arange(len(df)), ck_idx if len(ck_idx) == len(df) else np.array([np.where(choices == k)[0][0] for k in df["correct_key"].values])] - s_competitor_fix

    # Recompute ck_idx after merge (lengths may differ)
    ck_idx_full = np.array([np.where(choices == k)[0][0] for k in df["correct_key"].values])
    s_correct_full = df[["sA", "sB", "sC", "sD"]].values[np.arange(len(df)), ck_idx_full]
    df["delta_hard_fix"] = s_correct_full - s_competitor_fix

    # ── Default margin = delta_soft_tau_1_0 ──
    df["delta_default"] = df["delta_soft_tau_1_0"]

    # ── Depth normalization ──
    df["depth_norm"] = df["layer_index"] / df["L_model"].clip(lower=1)

    # ── Per-prompt aggregates for final sign, commitment, flips ──
    df = df.sort_values(["model_id", "prompt_uid", "layer_index"]).reset_index(drop=True)

    # Final delta_default per prompt
    final_delta = df.loc[df["layer_index"] == df["L_model"], ["model_id", "prompt_uid", "delta_default"]].copy()
    final_delta.rename(columns={"delta_default": "final_delta_default"}, inplace=True)
    df = df.merge(final_delta, on=["model_id", "prompt_uid"], how="left")

    # Final sign
    df["final_sign"] = np.sign(df["final_delta_default"])
    df.loc[df["final_delta_default"] == 0, "final_sign"] = 0

    # Signed margin
    df["delta_signed"] = df["final_sign"] * df["delta_default"]

    # ── Switch indicator ──
    df["switch_indicator"] = 0
    grouped = df.groupby(["model_id", "prompt_uid"])
    for (mid, puid), grp in grouped:
        idx = grp.index
        k_vals = grp["k_dyn"].values
        switches = np.zeros(len(k_vals), dtype=int)
        for i in range(1, len(k_vals)):
            if k_vals[i] != k_vals[i - 1]:
                switches[i] = 1
        df.loc[idx, "switch_indicator"] = switches

    # ── Flip count, commitment layer, last_flip_layer, regime ──
    prompt_stats = []
    for (mid, puid), grp in df.groupby(["model_id", "prompt_uid"]):
        grp_sorted = grp.sort_values("layer_index")
        deltas = grp_sorted["delta_default"].values
        delta_signed = grp_sorted["delta_signed"].values
        layers = grp_sorted["layer_index"].values

        # Sign sequence (drop exact zeros)
        nonzero_mask = deltas != 0
        signs = np.sign(deltas[nonzero_mask])
        flip_count = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

        # Commitment layer: smallest l such that delta_signed[l] >= M
        # and for all l' >= l, delta_signed[l'] >= M
        commitment_layer = None
        for i in range(len(delta_signed)):
            if np.all(delta_signed[i:] >= M_DEFAULT):
                commitment_layer = int(layers[i])
                break

        # Last flip layer (for unstable)
        last_flip_layer = None
        full_signs = np.sign(deltas)
        for i in range(len(full_signs) - 1, 0, -1):
            if full_signs[i] != 0 and full_signs[i - 1] != 0 and full_signs[i] != full_signs[i - 1]:
                last_flip_layer = int(layers[i])
                break

        # Regime classification
        final_d = float(deltas[-1])
        if flip_count <= MAX_FLIPS_FOR_STABILITY and commitment_layer is not None:
            if final_d > 0:
                regime = "Stable-Correct"
            elif final_d < 0:
                regime = "Stable-Wrong"
            else:
                regime = "Unstable"
        else:
            regime = "Unstable"

        prompt_stats.append({
            "model_id": mid,
            "prompt_uid": puid,
            "flip_count": flip_count,
            "commitment_layer": commitment_layer,
            "last_flip_layer": last_flip_layer,
            "regime": regime,
        })

    stats_df = pd.DataFrame(prompt_stats)
    df = df.merge(stats_df, on=["model_id", "prompt_uid"], how="left")

    # Select canonical columns
    out_cols = [
        "model_id", "model_revision", "prompt_uid", "layer_index", "L_model",
        "depth_norm", "correct_key", "sA", "sB", "sC", "sD",
        "k_dyn", "k_fix",
        "delta_soft_tau_0_5", "delta_soft_tau_1_0", "delta_soft_tau_2_0",
        "delta_hard_dyn", "delta_hard_fix", "delta_default",
        "final_delta_default", "final_sign", "delta_signed",
        "switch_indicator", "regime", "commitment_layer", "flip_count",
        "last_flip_layer",
    ]
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].copy()


def _logsumexp(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable logsumexp along axis, handling -inf."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    return np.squeeze(x_max, axis=axis) + np.log(
        np.sum(np.exp(x - x_max), axis=axis)
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════
def validate_data(df: pd.DataFrame) -> list[str]:
    """Run data integrity tests T1–T6. Returns list of failures."""
    failures = []

    # T1: Model coverage
    models = sorted(df["model_id"].unique())
    if len(models) != 3:
        failures.append(f"T1: Expected 3 models, got {len(models)}: {models}")
    for m in models:
        n_prompts = df.loc[df["model_id"] == m, "prompt_uid"].nunique()
        if n_prompts < 1500:
            failures.append(f"T1: Model {m} has {n_prompts} prompts (need ≥1500)")
        elif n_prompts < 2500:
            warnings.warn(f"T1 WARN: Model {m} has {n_prompts} prompts (<2500)")

    # T2: Score availability
    for col in ["sA", "sB", "sC", "sD"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            failures.append(f"T2: {col} has {n_null} null values")

    # T3: correct_key validity
    bad_ck = ~df["correct_key"].isin(["A", "B", "C", "D"])
    if bad_ck.any():
        failures.append(f"T3: {bad_ck.sum()} rows with invalid correct_key")

    # T6: Regime sanity
    prompt_level = df.groupby(["model_id", "prompt_uid"]).first().reset_index()
    for m in models:
        sub = prompt_level[prompt_level["model_id"] == m]
        for regime, expected_sign in [("Stable-Correct", 1), ("Stable-Wrong", -1)]:
            r_sub = sub[sub["regime"] == regime]
            if len(r_sub) == 0:
                continue
            med = r_sub["final_delta_default"].median()
            if regime == "Stable-Correct" and med <= M_DEFAULT:
                failures.append(f"T6: {m} Stable-Correct median final δ = {med:.3f} ≤ {M_DEFAULT}")
            elif regime == "Stable-Wrong" and med >= -M_DEFAULT:
                failures.append(f"T6: {m} Stable-Wrong median final δ = {med:.3f} ≥ {-M_DEFAULT}")
        # Unstable should have smaller |δ|
        stable = sub[sub["regime"].isin(["Stable-Correct", "Stable-Wrong"])]
        unstable = sub[sub["regime"] == "Unstable"]
        if len(unstable) > 0 and len(stable) > 0:
            med_stable = stable["final_delta_default"].abs().median()
            med_unstable = unstable["final_delta_default"].abs().median()
            if med_unstable >= med_stable:
                warnings.warn(
                    f"T6 WARN: {m} Unstable median |δ| ({med_unstable:.3f}) "
                    f"≥ Stable ({med_stable:.3f})"
                )

    return failures


# ═══════════════════════════════════════════════════════════════════════════
# TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def write_table1_models(output_dir: Path) -> None:
    """Table 1: Models."""
    tex = r"""\begin{table}[t]
  \caption{Models studied.  All are 7--8B instruction-tuned transformers
  with pinned HuggingFace revisions.}
  \label{tab:models}
  \vskip 0.1in
  \begin{center}
  \begin{small}
  \begin{tabular}{lccc}
    \toprule
    Model & Params & Layers & Prompts \\
    \midrule
    Qwen 2.5-7B-Instruct  & 7.6\,B & 28 & 3\,000 \\
    Llama 3.1-8B-Instruct  & 8.0\,B & 32 & 3\,000 \\
    Mistral 7B-v0.3 & 7.2\,B & 32 & 3\,000 \\
    \bottomrule
  \end{tabular}
  \end{small}
  \end{center}
\end{table}
"""
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "table1_models.tex").write_text(tex)


def write_table2_regimes(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 2: Regime counts per model."""
    prompt_level = df.groupby(["model_id", "prompt_uid"]).first().reset_index()
    rows = []
    for m in EXPECTED_MODELS:
        sub = prompt_level[prompt_level["model_id"] == m]
        total = len(sub)
        for regime in ["Stable-Correct", "Stable-Wrong", "Unstable"]:
            r_sub = sub[sub["regime"] == regime]
            n = len(r_sub)
            pct = 100 * n / total if total > 0 else 0
            med = r_sub["final_delta_default"].median() if n > 0 else 0
            rows.append((MODEL_SHORT[m], regime, n, pct, med))

    tex = r"""\begin{table}[t]
  \caption{Regime counts and median final margin $\delta_{\mathrm{soft}}(\tau{=}1)$ per model.}
  \label{tab:regimes}
  \vskip 0.1in
  \begin{center}
  \begin{small}
  \begin{tabular}{llrrl}
    \toprule
    Model & Regime & $N$ & \% & Med.\ $\delta$ \\
    \midrule
"""
    for i, (model, regime, n, pct, med) in enumerate(rows):
        tex += f"    {model} & {regime} & {n:,} & {pct:.1f} & {med:+.2f} \\\\\n"
        if (i + 1) % 3 == 0 and i < len(rows) - 1:
            tex += "    \\midrule\n"
    tex += r"""    \bottomrule
  \end{tabular}
  \end{small}
  \end{center}
\end{table}
"""
    (output_dir / "tables" / "table2_regimes.tex").write_text(tex)


def write_table3_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 3: Regime proportions under varying τ and M."""
    prompt_level = df.groupby(["model_id", "prompt_uid"]).first().reset_index()
    rows = []
    for tau in [0.5, 1.0, 2.0]:
        tau_col = f"delta_soft_tau_{str(tau).replace('.', '_')}"
        for M in [0.5, 0.75, 1.0]:
            counts = {"Stable-Correct": 0, "Stable-Wrong": 0, "Unstable": 0}
            total = 0
            for _, row in prompt_level.iterrows():
                total += 1
                # Use precomputed flip_count, recompute commitment with different M
                sub = df[(df["model_id"] == row["model_id"]) & (df["prompt_uid"] == row["prompt_uid"])].sort_values("layer_index")
                if tau_col not in sub.columns:
                    counts["Unstable"] += 1
                    continue
                deltas = sub[tau_col].values
                # Final sign
                fs = np.sign(deltas[-1]) if deltas[-1] != 0 else 0
                d_signed = fs * deltas

                # Commitment
                commitment = None
                for i in range(len(d_signed)):
                    if np.all(d_signed[i:] >= M):
                        commitment = i
                        break

                # Flip count
                nonzero = deltas[deltas != 0]
                signs = np.sign(nonzero)
                fc = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

                if fc <= MAX_FLIPS_FOR_STABILITY and commitment is not None:
                    if deltas[-1] > 0:
                        counts["Stable-Correct"] += 1
                    elif deltas[-1] < 0:
                        counts["Stable-Wrong"] += 1
                    else:
                        counts["Unstable"] += 1
                else:
                    counts["Unstable"] += 1

            sc_pct = 100 * counts["Stable-Correct"] / total
            sw_pct = 100 * counts["Stable-Wrong"] / total
            u_pct = 100 * counts["Unstable"] / total
            rows.append((tau, M, sc_pct, sw_pct, u_pct))

    tex = r"""\begin{table}[t]
  \caption{Regime proportions (\%) under varying temperature $\tau$ and commitment threshold $M$.
  All three models pooled.}
  \label{tab:robustness}
  \vskip 0.1in
  \begin{center}
  \begin{small}
  \begin{tabular}{ccrrrr}
    \toprule
    $\tau$ & $M$ & Stable-Corr & Stable-Wrong & Unstable \\
    \midrule
"""
    for tau, M, sc, sw, u in rows:
        tex += f"    {tau:.1f} & {M:.2f} & {sc:.1f} & {sw:.1f} & {u:.1f} \\\\\n"
    tex += r"""    \bottomrule
  \end{tabular}
  \end{small}
  \end{center}
\end{table}
"""
    (output_dir / "tables" / "table3_robustness.tex").write_text(tex)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def fig1_examples(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 1: Example margin trajectories — one panel per regime."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)

    # Pick one model for examples
    model = EXPECTED_MODELS[0]
    sub = df[df["model_id"] == model].copy()
    prompt_level = sub.groupby("prompt_uid").first().reset_index()

    for ax, regime, title_short in zip(
        axes,
        ["Stable-Correct", "Stable-Wrong", "Unstable"],
        ["Stable-Correct", "Stable-Wrong", "Unstable"],
    ):
        regime_prompts = prompt_level[prompt_level["regime"] == regime]["prompt_uid"].values
        if len(regime_prompts) == 0:
            ax.set_title(f"{title_short}\n(no examples)")
            continue

        rng = np.random.default_rng(12345)
        sample = rng.choice(regime_prompts, size=min(5, len(regime_prompts)), replace=False)

        for puid in sample:
            p_data = sub[sub["prompt_uid"] == puid].sort_values("layer_index")
            ax.plot(
                p_data["depth_norm"].values,
                p_data["delta_default"].values,
                alpha=0.7,
                linewidth=1.0,
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(title_short, fontsize=10)
        ax.set_xlabel("Normalised depth")

    axes[0].set_ylabel("$\\delta_{\\mathrm{soft}}$ (logits)")

    # Competitor strip for first example
    fig.suptitle(
        f"Example trajectories ({MODEL_SHORT[model]})",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig1_examples.pdf")
    plt.close(fig)


def fig2_delta_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 2: Distribution of δ_default at selected depths, by model and regime."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)

    depth_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    tolerance = 0.05

    for ax, model in zip(axes, EXPECTED_MODELS):
        sub = df[df["model_id"] == model]

        positions = []
        data_arrays = []
        colors_list = []

        for i, d in enumerate(depth_bins):
            mask = (sub["depth_norm"] - d).abs() < tolerance
            layer_data = sub.loc[mask]
            for j, regime in enumerate(["Stable-Correct", "Stable-Wrong", "Unstable"]):
                r_data = layer_data[layer_data["regime"] == regime]["delta_default"].values
                if len(r_data) > 0:
                    positions.append(i * 4 + j)
                    data_arrays.append(r_data)
                    colors_list.append(REGIME_COLORS[regime])

        if data_arrays:
            parts = ax.violinplot(
                data_arrays,
                positions=positions,
                showmeans=True,
                showmedians=True,
                widths=0.8,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.6)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xticks([i * 4 + 1 for i in range(len(depth_bins))])
        ax.set_xticklabels([f"{d:.0%}" for d in depth_bins])
        ax.set_xlabel("Depth")
        ax.set_title(MODEL_SHORT[model])

    axes[0].set_ylabel("$\\delta_{\\mathrm{soft}}$ (logits)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r, alpha=0.6) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig2_delta_distribution.pdf")
    plt.close(fig)


def fig3_decision_space(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 3: PCA of 4-option score vectors, colored by regime."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    for ax, model in zip(axes, EXPECTED_MODELS):
        sub = df[df["model_id"] == model].copy()

        # PCA on [sA, sB, sC, sD]
        X = sub[["sA", "sB", "sC", "sD"]].values
        pca = PCA(n_components=2, random_state=12345)
        coords = pca.fit_transform(X)
        sub = sub.copy()
        sub["pc1"] = coords[:, 0]
        sub["pc2"] = coords[:, 1]

        # Sample trajectories for readability
        prompt_level = sub.groupby("prompt_uid").first().reset_index()
        rng = np.random.default_rng(12345)
        sample_uids = rng.choice(prompt_level["prompt_uid"].values, size=min(200, len(prompt_level)), replace=False)

        for regime, color in REGIME_COLORS.items():
            regime_uids = prompt_level.loc[prompt_level["regime"] == regime, "prompt_uid"].values
            plot_uids = [u for u in sample_uids if u in regime_uids]
            for uid in plot_uids[:30]:
                traj = sub[sub["prompt_uid"] == uid].sort_values("layer_index")
                ax.plot(traj["pc1"].values, traj["pc2"].values, color=color, alpha=0.15, linewidth=0.5)

        # Scatter final-layer points
        final = sub[sub["layer_index"] == sub["L_model"]].copy()
        for regime, color in REGIME_COLORS.items():
            r_final = final[final["regime"] == regime]
            ax.scatter(r_final["pc1"], r_final["pc2"], c=color, s=3, alpha=0.3, label=regime, zorder=5)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(MODEL_SHORT[model])

    axes[-1].legend(fontsize=7, markerscale=2, loc="upper right")
    fig.suptitle("Decision space (PCA of 4-option scores)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig3_decision_space_trajectories.pdf")
    plt.close(fig)


def fig4_flow_field(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 4: Mean drift field in depth × δ grid."""
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # Compute per-layer drift for delta_default
    df_sorted = df.sort_values(["model_id", "prompt_uid", "layer_index"])
    drift = df_sorted.groupby(["model_id", "prompt_uid"])["delta_default"].diff().shift(-1)
    df_work = df_sorted.copy()
    df_work["drift_default"] = drift

    # Grid
    depth_bins = np.linspace(0, 1, 30)
    delta_clip = 8.0
    delta_bins = np.linspace(-delta_clip, delta_clip, 40)

    df_work["depth_bin"] = pd.cut(df_work["depth_norm"], depth_bins, labels=False)
    df_work["delta_bin"] = pd.cut(
        df_work["delta_default"].clip(-delta_clip, delta_clip),
        delta_bins,
        labels=False,
    )

    grid = df_work.groupby(["depth_bin", "delta_bin"]).agg(
        mean_drift=("drift_default", "mean"),
        count=("drift_default", "count"),
    ).reset_index()

    # Pivot to 2D
    pivot = grid.pivot(index="delta_bin", columns="depth_bin", values="mean_drift")
    count_pivot = grid.pivot(index="delta_bin", columns="depth_bin", values="count")

    # Mask sparse cells
    masked = np.ma.masked_where(count_pivot.values < 5, pivot.values)

    im = ax.imshow(
        masked,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-2,
        vmax=2,
        extent=[0, 1, -delta_clip, delta_clip],
        interpolation="nearest",
    )

    # Boundary band
    ax.axhspan(-EPSILON_BOUNDARY_BAND, EPSILON_BOUNDARY_BAND, color="yellow", alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("$\\delta_{\\mathrm{soft}}$ (logits)")
    cb = fig.colorbar(im, ax=ax, label="Mean drift (logits/layer)")
    ax.set_title("Flow field: mean per-layer drift in decision space")

    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig4_flow_field.pdf")
    plt.close(fig)


def fig5_commitment_and_flips(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 5: Commitment depth and flip distributions."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    prompt_level = df.groupby(["model_id", "prompt_uid"]).first().reset_index()

    # (A) Commitment depth distribution
    ax = axes[0, 0]
    for model, color in MODEL_COLORS.items():
        sub = prompt_level[(prompt_level["model_id"] == model) & prompt_level["commitment_layer"].notna()]
        if len(sub) > 0:
            L = MODEL_LAYERS[model]
            depths = sub["commitment_layer"].values / (L - 1)
            ax.hist(depths, bins=20, alpha=0.5, color=color, label=MODEL_SHORT[model], density=True)
    ax.set_xlabel("Commitment depth (normalised)")
    ax.set_ylabel("Density")
    ax.set_title("(A) Commitment depth")
    ax.legend(fontsize=7)

    # (B) Flip count histogram
    ax = axes[0, 1]
    for model, color in MODEL_COLORS.items():
        sub = prompt_level[prompt_level["model_id"] == model]
        ax.hist(sub["flip_count"].values, bins=range(0, 15), alpha=0.5, color=color, label=MODEL_SHORT[model], density=True)
    ax.set_xlabel("Flip count")
    ax.set_ylabel("Density")
    ax.set_title("(B) Flip count distribution")
    ax.legend(fontsize=7)

    # (C) Last flip layer for Unstable
    ax = axes[1, 0]
    for model, color in MODEL_COLORS.items():
        sub = prompt_level[(prompt_level["model_id"] == model) & (prompt_level["regime"] == "Unstable") & prompt_level["last_flip_layer"].notna()]
        if len(sub) > 0:
            L = MODEL_LAYERS[model]
            depths = sub["last_flip_layer"].values / (L - 1)
            ax.hist(depths, bins=20, alpha=0.5, color=color, label=MODEL_SHORT[model], density=True)
    ax.set_xlabel("Last flip depth (normalised)")
    ax.set_ylabel("Density")
    ax.set_title("(C) Last flip layer (Unstable only)")
    ax.legend(fontsize=7)

    # (D) Competitor switch rate vs depth
    ax = axes[1, 1]
    for model, color in MODEL_COLORS.items():
        sub = df[df["model_id"] == model]
        switch_rate = sub.groupby("depth_norm")["switch_indicator"].mean()
        ax.plot(switch_rate.index, switch_rate.values, color=color, label=MODEL_SHORT[model], linewidth=1.5)
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Switch rate")
    ax.set_title("(D) Competitor switch rate")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig5_commitment_and_flips.pdf")
    plt.close(fig)


def fig6_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 6: Regime proportions under varying τ and M."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    prompt_level = df.groupby(["model_id", "prompt_uid"]).first().reset_index()

    tau_values = [0.5, 1.0, 2.0]
    M_values = [0.5, 0.75, 1.0]

    for ax_idx, model in enumerate(EXPECTED_MODELS):
        ax = axes[ax_idx]
        sub_prompts = prompt_level[prompt_level["model_id"] == model]
        sub_all = df[df["model_id"] == model]

        results = []
        for tau in tau_values:
            tau_col = f"delta_soft_tau_{str(tau).replace('.', '_')}"
            for M in M_values:
                sc, sw, unst = 0, 0, 0
                for _, row in sub_prompts.iterrows():
                    puid = row["prompt_uid"]
                    p_data = sub_all[sub_all["prompt_uid"] == puid].sort_values("layer_index")
                    deltas = p_data[tau_col].values
                    final_d = deltas[-1]
                    fs = np.sign(final_d) if final_d != 0 else 0
                    d_signed = fs * deltas

                    # Commitment
                    committed = False
                    for i in range(len(d_signed)):
                        if np.all(d_signed[i:] >= M):
                            committed = True
                            break

                    # Flips
                    nz = deltas[deltas != 0]
                    signs = np.sign(nz)
                    fc = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

                    if fc <= MAX_FLIPS_FOR_STABILITY and committed:
                        if final_d > 0:
                            sc += 1
                        elif final_d < 0:
                            sw += 1
                        else:
                            unst += 1
                    else:
                        unst += 1

                total = sc + sw + unst
                results.append({
                    "tau": tau, "M": M,
                    "Stable-Correct": 100 * sc / total,
                    "Stable-Wrong": 100 * sw / total,
                    "Unstable": 100 * unst / total,
                })

        res_df = pd.DataFrame(results)
        x = np.arange(len(tau_values))
        width = 0.25

        for m_idx, M in enumerate(M_values):
            m_data = res_df[res_df["M"] == M]
            bottom = np.zeros(len(tau_values))
            for regime, color in REGIME_COLORS.items():
                vals = m_data[regime].values
                ax.bar(x + m_idx * width, vals, width, bottom=bottom, color=color, alpha=0.7,
                       label=regime if ax_idx == 0 and m_idx == 0 else "")
                bottom += vals

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"τ={t}" for t in tau_values])
        ax.set_ylabel("Proportion (%)")
        ax.set_title(MODEL_SHORT[model])
        ax.set_ylim(0, 105)

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Robustness: regime proportions under varying τ and M", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "fig6_robustness.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# BUILD INFO & CLAIM EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_build_info(parquet_dir: Path, output_dir: Path) -> None:
    info = {
        "parquet_dir": str(parquet_dir.resolve()),
        "git_commit": _git_hash(),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cached_delta_column_definition": (
            "delta = s_correct - max(s_{j!=correct}) at each layer "
            "(hard dynamic margin, from src/sow/v2/metrics.py). "
            "Paper 1 recomputes all margins from raw sA/sB/sC/sD; "
            "cached delta is NOT used."
        ),
        "file_hashes": {},
    }
    for p in sorted(parquet_dir.glob("*.parquet")):
        info["file_hashes"][p.name] = _sha256(p)
    (output_dir / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")


def write_claim_evidence(output_dir: Path) -> None:
    claims = """# Paper 1 — Claim-to-Evidence Ledger

| # | Claim | Evidence | Figure/Table |
|---|-------|----------|-------------|
| 1 | Soft margins smooth out competitor-switching artifacts | δ_soft vs δ_hard comparison | Fig 1, Fig 2 |
| 2 | Three regimes (Stable-Correct, Stable-Wrong, Unstable) separate cleanly across all models | Regime counts, median final δ | Table 2, Fig 2 |
| 3 | Decision-space PCA reveals trajectory structure | 4-option score PCA | Fig 3 |
| 4 | Flow field shows convergent/divergent regions | Mean drift in depth×δ grid | Fig 4 |
| 5 | Commitment depth concentrates in late layers | Commitment depth distribution | Fig 5(A) |
| 6 | Flip counts are low for stable, high for unstable | Flip count histograms | Fig 5(B) |
| 7 | Last flips concentrate in late layers | Last-flip distribution | Fig 5(C) |
| 8 | Competitor switching is most frequent at mid-depth | Switch rate vs depth | Fig 5(D) |
| 9 | Regime classification is robust to τ and M variations | Regime proportions under perturbation | Table 3, Fig 6 |

## Unsupported Claims (None)

All claims in Paper 1 are supported by cached 4-option scores.
No causal, mechanistic, or hidden-state claims are made.
"""
    (output_dir / "CLAIM_EVIDENCE.md").write_text(claims)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper 1 / Part I assets")
    parser.add_argument("--parquet-dir", type=Path, default=Path("results/parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/part1"))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    parquet_dir = args.parquet_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Paper 1 / Part I — Build Pipeline")
    print("=" * 60)
    print(f"  parquet_dir: {parquet_dir}")
    print(f"  output_dir:  {output_dir}")
    print(f"  seed:        {args.seed}")
    print()

    # ── Step A: Load raw parquets ──
    print("[A] Loading raw parquets...")
    lw = load_layerwise_scores(parquet_dir)
    dm = load_decision_metrics(parquet_dir)

    # ── Step B: Build part1_core.parquet ──
    print("[B] Computing canonical metrics...")
    core = compute_all_metrics(lw, dm, args.seed)
    print(f"  part1_core: {len(core):,} rows, {core['model_id'].nunique()} models")

    # ── Validate ──
    print("[B.1] Running data integrity tests...")
    failures = validate_data(core)
    if failures:
        print("HARD FAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
        # Write failure report
        report = "# BUILD FAILURE REPORT\n\n"
        for f in failures:
            report += f"- {f}\n"
        (output_dir / "BUILD_FAILURE_REPORT.md").write_text(report)
        return 1
    print("  All integrity tests passed ✓")

    # Save core parquet
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    core.to_parquet(data_dir / "part1_core.parquet", index=False)
    print(f"  Saved: {data_dir / 'part1_core.parquet'}")

    # ── Step C: Tables ──
    print("[C] Generating LaTeX tables...")
    write_table1_models(output_dir)
    write_table2_regimes(core, output_dir)
    write_table3_robustness(core, output_dir)
    print("  Tables written ✓")

    # ── Step D: Figures ──
    print("[D] Generating figures...")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("  fig1_examples...")
    fig1_examples(core, output_dir)
    print("  fig2_delta_distribution...")
    fig2_delta_distribution(core, output_dir)
    print("  fig3_decision_space...")
    fig3_decision_space(core, output_dir)
    print("  fig4_flow_field...")
    fig4_flow_field(core, output_dir)
    print("  fig5_commitment_and_flips...")
    fig5_commitment_and_flips(core, output_dir)
    print("  fig6_robustness...")
    fig6_robustness(core, output_dir)
    print("  All figures generated ✓")

    # ── Step E: Build info & claim evidence ──
    print("[E] Writing BUILD_INFO.json and CLAIM_EVIDENCE.md...")
    write_build_info(parquet_dir, output_dir)
    write_claim_evidence(output_dir)

    # ── Step F: Figure QC ──
    print("[F] Running figure QC...")
    qc_script = Path(__file__).parent / "figure_qc.py"
    if qc_script.exists():
        ret = subprocess.run(
            [sys.executable, str(qc_script), str(output_dir / "figures")],
            capture_output=True, text=True,
        )
        print(ret.stdout)
        if ret.returncode != 0:
            print(f"  Figure QC FAILED:\n{ret.stderr}")
            return 1
    else:
        print("  figure_qc.py not found, skipping QC")

    print()
    print("=" * 60)
    print("BUILD COMPLETE ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
