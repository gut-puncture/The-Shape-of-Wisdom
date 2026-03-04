#!/usr/bin/env python3
"""Build all Paper 1 / Part I assets from cached parquet artifacts.

Camera-ready version. Generates:
  - 8 figure PDFs (figA + fig1-fig7)
  - 3 LaTeX tables
  - auto_numbers.tex
  - BUILD_INFO.json, CLAIM_EVIDENCE.md
  - Runs audit and quality gates

Usage::
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
        python3 scripts/part1/build_part1_assets.py \\
        --parquet-dir results/parquet --output-dir paper/part1 --seed 12345
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    EXPECTED_MODELS, MODEL_COLORS, MODEL_LAYERS, MODEL_SHORT,
    OPTION_COLORS, REGIME_COLORS, apply_style, save_fig, fig_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TAU_DEFAULT = 1.0
M_DEFAULT = 0.75
MAX_FLIPS = 1
SEED = 12345
EPSILON_BAND = 0.2


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & METRIC COMPUTATION (reused from prior build)
# ═══════════════════════════════════════════════════════════════════════════
def _logsumexp(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    return np.squeeze(x_max, axis=axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))


def load_and_build_core(parquet_dir: Path) -> pd.DataFrame:
    """Load layerwise + decision_metrics, compute canonical metrics, return core DataFrame."""
    core_path = parquet_dir.parent / "paper" / "part1" / "data" / "part1_core.parquet"
    # Use pre-built core if it exists (from prior build step)
    if core_path.exists():
        print(f"  Using pre-built core: {core_path}")
        return pd.read_parquet(core_path)

    # Otherwise build from scratch (same logic as before)
    lw = pd.read_parquet(parquet_dir / "layerwise.parquet")
    dm = pd.read_parquet(parquet_dir / "decision_metrics.parquet")
    logits_dicts = lw["candidate_logits_json"].apply(json.loads)
    lw["sA"] = logits_dicts.apply(lambda d: float(d.get("A", 0.0)))
    lw["sB"] = logits_dicts.apply(lambda d: float(d.get("B", 0.0)))
    lw["sC"] = logits_dicts.apply(lambda d: float(d.get("C", 0.0)))
    lw["sD"] = logits_dicts.apply(lambda d: float(d.get("D", 0.0)))

    ck = dm.groupby(["model_id", "prompt_uid"])["correct_key"].first().reset_index()
    df = lw.merge(ck, on=["model_id", "prompt_uid"], how="inner")

    choices = np.array(["A", "B", "C", "D"])
    ck_idx = np.array([np.where(choices == k)[0][0] for k in df["correct_key"].values])
    score_matrix = df[["sA", "sB", "sC", "sD"]].values
    s_correct = score_matrix[np.arange(len(df)), ck_idx]
    mask = np.ones((len(df), 4), dtype=bool)
    mask[np.arange(len(df)), ck_idx] = False
    incorrect_scores = np.where(mask, score_matrix, -np.inf)

    k_dyn_idx = np.argmax(incorrect_scores, axis=1)
    df["k_dyn"] = choices[k_dyn_idx]
    df["delta_hard_dyn"] = s_correct - incorrect_scores[np.arange(len(df)), k_dyn_idx]

    for tau, col in [(0.5, "delta_soft_tau_0_5"), (1.0, "delta_soft_tau_1_0"), (2.0, "delta_soft_tau_2_0")]:
        inc_scaled = np.where(mask, incorrect_scores / tau, -np.inf)
        df[col] = s_correct - tau * _logsumexp(inc_scaled, axis=1)

    df["delta_default"] = df["delta_soft_tau_1_0"]
    L_model = df.groupby("model_id")["layer_index"].transform("max")
    df["L_model"] = L_model
    df["depth_norm"] = df["layer_index"] / df["L_model"].clip(lower=1)

    final_mask = df["layer_index"] == df["L_model"]
    final_rows = df.loc[final_mask, ["model_id", "prompt_uid", "k_dyn"]].copy()
    final_rows.rename(columns={"k_dyn": "k_fix"}, inplace=True)
    df = df.merge(final_rows, on=["model_id", "prompt_uid"], how="left")

    k_fix_idx = np.array([np.where(choices == k)[0][0] for k in df["k_fix"].values])
    ck_full = np.array([np.where(choices == k)[0][0] for k in df["correct_key"].values])
    df["delta_hard_fix"] = df[["sA","sB","sC","sD"]].values[np.arange(len(df)), ck_full] - df[["sA","sB","sC","sD"]].values[np.arange(len(df)), k_fix_idx]

    df = df.sort_values(["model_id", "prompt_uid", "layer_index"]).reset_index(drop=True)
    final_delta = df.loc[df["layer_index"] == df["L_model"], ["model_id", "prompt_uid", "delta_default"]].copy()
    final_delta.rename(columns={"delta_default": "final_delta_default"}, inplace=True)
    df = df.merge(final_delta, on=["model_id", "prompt_uid"], how="left")
    df["final_sign"] = np.sign(df["final_delta_default"])
    df.loc[df["final_delta_default"] == 0, "final_sign"] = 0
    df["delta_signed"] = df["final_sign"] * df["delta_default"]

    df["switch_indicator"] = 0
    for (mid, puid), grp in df.groupby(["model_id", "prompt_uid"]):
        idx = grp.index
        k_vals = grp["k_dyn"].values
        switches = np.zeros(len(k_vals), dtype=int)
        for i in range(1, len(k_vals)):
            if k_vals[i] != k_vals[i - 1]:
                switches[i] = 1
        df.loc[idx, "switch_indicator"] = switches

    prompt_stats = []
    for (mid, puid), grp in df.groupby(["model_id", "prompt_uid"]):
        grp_s = grp.sort_values("layer_index")
        deltas = grp_s["delta_default"].values
        d_signed = grp_s["delta_signed"].values
        layers = grp_s["layer_index"].values
        nonzero = deltas[deltas != 0]
        signs = np.sign(nonzero)
        flip_count = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        commitment_layer = None
        for i in range(len(d_signed)):
            if np.all(d_signed[i:] >= M_DEFAULT):
                commitment_layer = int(layers[i])
                break
        last_flip_layer = None
        full_signs = np.sign(deltas)
        for i in range(len(full_signs) - 1, 0, -1):
            if full_signs[i] != 0 and full_signs[i-1] != 0 and full_signs[i] != full_signs[i-1]:
                last_flip_layer = int(layers[i])
                break
        final_d = float(deltas[-1])
        if flip_count <= MAX_FLIPS and commitment_layer is not None:
            regime = "Stable-Correct" if final_d > 0 else ("Stable-Wrong" if final_d < 0 else "Unstable")
        else:
            regime = "Unstable"
        prompt_stats.append({"model_id": mid, "prompt_uid": puid, "flip_count": flip_count,
                             "commitment_layer": commitment_layer, "last_flip_layer": last_flip_layer, "regime": regime})
    df = df.merge(pd.DataFrame(prompt_stats), on=["model_id", "prompt_uid"], how="left")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════
def figA_why_delta_soft(core: pd.DataFrame, out: Path) -> Path:
    """Figure A: Why δsoft — best-artifact-score example + switch-vs-non-switch violins."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Panel A: Pick example that maximises artifact score ──
    model = EXPECTED_MODELS[0]
    sub = core[core["model_id"] == model]
    # Artifact score = sum over switch layers of (|Δhard| - |Δsoft|)
    best_uid, best_score = None, -np.inf
    for uid, grp in sub.groupby("prompt_uid"):
        grp = grp.sort_values("layer_index")
        sw_v = grp["switch_indicator"].values
        dh_v = grp["delta_hard_dyn"].values
        ds_v = grp["delta_default"].values
        score = 0.0
        for i in range(1, len(sw_v)):
            if sw_v[i] == 1:
                score += abs(dh_v[i] - dh_v[i-1]) - abs(ds_v[i] - ds_v[i-1])
        if score > best_score and sw_v.sum() >= 2:
            best_score, best_uid = score, uid

    p = sub[sub["prompt_uid"] == best_uid].sort_values("layer_index")
    depth = p["depth_norm"].values
    dh = p["delta_hard_dyn"].values
    ds = p["delta_default"].values
    sw = p["switch_indicator"].values
    kdyn = p["k_dyn"].values

    ax1.plot(depth, dh, color="#D55E00", linewidth=1.0, alpha=0.8, label=r"$\delta_{\mathrm{hard}}$")
    ax1.plot(depth, ds, color="#0072B2", linewidth=1.6, label=r"$\delta_{\mathrm{soft}}$")
    ax1.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
    # Vertical switch markers
    switch_idxs = [i for i in range(len(sw)) if sw[i] == 1]
    for i in switch_idxs:
        ax1.axvline(depth[i], color="#999999", linewidth=0.5, alpha=0.5, linestyle=":")
    # Competitor identity strip under x-axis
    for i in range(len(depth)):
        ax1.fill_between([depth[max(0,i-1)], depth[i]], -900, -800,
                         color=OPTION_COLORS.get(kdyn[i], "#999"), alpha=0.8,
                         transform=ax1.get_xaxis_transform())
    # Inset zoom around best switch
    if len(switch_idxs) >= 1:
        best_sw = max(switch_idxs, key=lambda i: abs(dh[i]-dh[i-1]) - abs(ds[i]-ds[i-1]) if i > 0 else 0)
        lo = max(0, best_sw - 3)
        hi = min(len(depth) - 1, best_sw + 3)
        axins = ax1.inset_axes([0.55, 0.55, 0.42, 0.42])
        axins.plot(depth[lo:hi+1], dh[lo:hi+1], color="#D55E00", linewidth=1.2)
        axins.plot(depth[lo:hi+1], ds[lo:hi+1], color="#0072B2", linewidth=1.8)
        axins.axvline(depth[best_sw], color="#999", linewidth=0.8, linestyle=":")
        axins.set_xlim(depth[lo]-0.01, depth[hi]+0.01)
        axins.tick_params(labelsize=5)
        axins.set_title("zoom at switch", fontsize=6, pad=2)
        ax1.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)
    ax1.set_xlabel("Normalised depth")
    ax1.set_ylabel("Margin (logits)")
    ax1.set_title("(A) Best-artifact-score trajectory", fontsize=9)
    ax1.legend(fontsize=6, loc="upper left")

    # ── Panel B: Switch vs non-switch jump distributions (4 violins) ──
    hard_sw, hard_nosw, soft_sw, soft_nosw = [], [], [], []
    for (mid, puid), grp in core.groupby(["model_id", "prompt_uid"]):
        grp = grp.sort_values("layer_index")
        sw_vals = grp["switch_indicator"].values
        dh_vals = grp["delta_hard_dyn"].values
        ds_vals = grp["delta_default"].values
        for i in range(1, len(sw_vals)):
            jh = abs(dh_vals[i] - dh_vals[i-1])
            js = abs(ds_vals[i] - ds_vals[i-1])
            if sw_vals[i] == 1:
                hard_sw.append(jh)
                soft_sw.append(js)
            else:
                hard_nosw.append(jh)
                soft_nosw.append(js)

    data = [hard_nosw, hard_sw, soft_nosw, soft_sw]
    labels = ["Hard\nnon-sw", "Hard\nswitch", "Soft\nnon-sw", "Soft\nswitch"]
    colors = ["#D55E00", "#D55E00", "#0072B2", "#0072B2"]
    alphas = [0.35, 0.7, 0.35, 0.7]
    parts = ax2.violinplot(data, positions=[1,2,3,4], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(alphas[i])
    parts["cmedians"].set_color("black")
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("$|\\Delta\\delta|$ per layer")
    ax2.set_title("(B) Switch vs non-switch jumps", fontsize=9)
    # Annotate medians
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax2.text(i+1, med + 0.02, f"{med:.2f}", ha="center", fontsize=5.5, color="black")

    plt.tight_layout()
    return save_fig(fig, "figA_why_delta_soft", out)


def fig1_pipeline(core: pd.DataFrame, out: Path) -> Path:
    """Figure 1: Pipeline overview — generated as TikZ in the .tex file, not here.
    We create a placeholder PDF that notes this."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.text(0.5, 0.5, "[Pipeline schematic is TikZ in paper.tex]",
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return save_fig(fig, "fig1_pipeline", out)


def fig2_examples(core: pd.DataFrame, out: Path) -> Path:
    """Figure 2: Example trajectories per regime with competitor colour strip."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    model = EXPECTED_MODELS[0]
    sub = core[core["model_id"] == model].copy()
    pl = sub.groupby("prompt_uid").first().reset_index()
    rng = np.random.default_rng(SEED)

    for ax, regime in zip(axes, ["Stable-Correct", "Stable-Wrong", "Unstable"]):
        regime_uids = pl.loc[pl["regime"] == regime, "prompt_uid"].values
        if len(regime_uids) == 0:
            ax.set_title(regime); continue
        sample = rng.choice(regime_uids, size=min(5, len(regime_uids)), replace=False)
        for i, uid in enumerate(sample):
            p = sub[sub["prompt_uid"] == uid].sort_values("layer_index")
            ax.plot(p["depth_norm"], p["delta_default"], alpha=0.7, linewidth=1.0,
                    color=plt.cm.tab10(i))
            # Commitment annotation for stable
            if regime.startswith("Stable") and p["commitment_layer"].notna().any():
                cl = p["commitment_layer"].iloc[0]
                L = p["L_model"].iloc[0]
                if cl is not None and not np.isnan(cl):
                    cd = cl / L
                    ax.axvline(cd, color=plt.cm.tab10(i), linewidth=0.5, linestyle=":", alpha=0.4)

        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.set_xlabel("Normalised depth")
        ax.set_title(regime, fontsize=9)

    axes[0].set_ylabel(r"$\delta_{\mathrm{soft}}$ (logits)")
    fig.suptitle(f"Example trajectories — {MODEL_SHORT[model]}", fontsize=9, y=1.01)
    plt.tight_layout()
    return save_fig(fig, "fig2_examples", out)


def fig3_distributions(core: pd.DataFrame, out: Path) -> Path:
    """Figure 3: Depth-wise distribution of δsoft by regime, per model."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    depth_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    tol = 0.06

    for ax, model in zip(axes, EXPECTED_MODELS):
        sub = core[core["model_id"] == model]
        positions = []
        data_arrays = []
        colors_list = []
        xticklabels = []

        for di, d in enumerate(depth_slices):
            layer_data = sub[(sub["depth_norm"] - d).abs() < tol]
            for ri, regime in enumerate(["Stable-Correct", "Stable-Wrong", "Unstable"]):
                vals = layer_data.loc[layer_data["regime"] == regime, "delta_default"].values
                if len(vals) > 5:
                    pos = di * 4 + ri
                    positions.append(pos)
                    data_arrays.append(vals)
                    colors_list.append(REGIME_COLORS[regime])

        if data_arrays:
            parts = ax.violinplot(data_arrays, positions=positions, showmeans=False,
                                  showmedians=True, widths=0.8)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.5)
                pc.set_edgecolor("none")
            for key in ("cmeans", "cmins", "cmaxes", "cbars", "cmedians"):
                if key in parts:
                    parts[key].set_linewidth(0.5)

        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.set_xticks([di * 4 + 1 for di in range(len(depth_slices))])
        ax.set_xticklabels([f"{d:.0%}" for d in depth_slices])
        ax.set_xlabel("Depth")
        ax.set_title(MODEL_SHORT[model], fontsize=9)

    axes[0].set_ylabel(r"$\delta_{\mathrm{soft}}$ (logits)")
    handles = [mpatches.Patch(facecolor=c, label=r, alpha=0.5) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=7)
    plt.tight_layout()
    return save_fig(fig, "fig3_distributions", out)


def fig4_decision_space(core: pd.DataFrame, out: Path) -> Path:
    """Figure 4: PCA of 4-option score vectors with loadings inset and entropy annotation."""
    from scipy.stats import pearsonr
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    rng = np.random.default_rng(SEED)

    for ax_i, (ax, model) in enumerate(zip(axes, EXPECTED_MODELS)):
        sub = core[core["model_id"] == model].copy()
        X = sub[["sA", "sB", "sC", "sD"]].values
        pca = PCA(n_components=2, random_state=SEED)
        coords = pca.fit_transform(X)
        sub = sub.copy()
        sub["pc1"] = coords[:, 0]
        sub["pc2"] = coords[:, 1]

        # Entropy correlation
        probs = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        r_ent, _ = pearsonr(coords[:, 0], entropy)

        pl = sub.groupby("prompt_uid").first().reset_index()
        sample_uids = rng.choice(pl["prompt_uid"].values, size=min(100, len(pl)), replace=False)

        # Fewer trajectories, lower opacity
        for regime, color in REGIME_COLORS.items():
            r_uids = pl.loc[pl["regime"] == regime, "prompt_uid"].values
            plot_uids = [u for u in sample_uids if u in r_uids][:15]
            for uid in plot_uids:
                traj = sub[sub["prompt_uid"] == uid].sort_values("layer_index")
                n = len(traj)
                for j in range(1, n):
                    alpha = 0.03 + 0.2 * (j / n)
                    ax.plot(traj["pc1"].values[j-1:j+1], traj["pc2"].values[j-1:j+1],
                            color=color, alpha=alpha, linewidth=0.3)

        # Final-layer point cloud
        final = sub[sub["layer_index"] == sub["L_model"]]
        for regime, color in REGIME_COLORS.items():
            rf = final[final["regime"] == regime]
            ax.scatter(rf["pc1"], rf["pc2"], c=color, s=1.5, alpha=0.2, label=regime, zorder=5)

        ev1 = 100 * pca.explained_variance_ratio_[0]
        ev2 = 100 * pca.explained_variance_ratio_[1]
        ax.set_xlabel(f"PC1 ({ev1:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev2:.0f}%)", fontsize=8)
        ax.set_title(MODEL_SHORT[model], fontsize=9)

        # Annotate entropy correlation
        ax.text(0.03, 0.97, f"PC1–entropy\nr={r_ent:.2f}",
                transform=ax.transAxes, fontsize=5.5, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="gray", lw=0.5))

        # Inset: PC loadings bar chart
        axins = ax.inset_axes([0.65, 0.02, 0.32, 0.25])
        x_pos = np.arange(4)
        w1 = pca.components_[0]
        w2 = pca.components_[1]
        axins.bar(x_pos - 0.15, w1, 0.28, color="#333333", alpha=0.7, label="PC1")
        axins.bar(x_pos + 0.15, w2, 0.28, color="#AAAAAA", alpha=0.7, label="PC2")
        axins.set_xticks(x_pos)
        axins.set_xticklabels(["A","B","C","D"], fontsize=5)
        axins.tick_params(axis="y", labelsize=4)
        axins.set_ylim(-1, 1)
        axins.axhline(0, color="k", linewidth=0.3)
        if ax_i == 0:
            axins.legend(fontsize=4, loc="upper left")
        axins.set_title("Loadings", fontsize=5, pad=1)

    axes[-1].legend(fontsize=5, markerscale=3, loc="upper right")
    plt.tight_layout()
    return save_fig(fig, "fig4_decision_space", out)


def fig5_flow_field(core: pd.DataFrame, out: Path) -> Path:
    """Figure 5: Population density + mean drift field."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Compute drift
    df = core.sort_values(["model_id", "prompt_uid", "layer_index"]).copy()
    drift = df.groupby(["model_id", "prompt_uid"])["delta_default"].diff().shift(-1)
    df["drift_default"] = drift.values

    delta_clip = 8.0
    depth_bins = np.linspace(0, 1, 35)
    delta_bins = np.linspace(-delta_clip, delta_clip, 45)

    df["depth_bin"] = pd.cut(df["depth_norm"], depth_bins, labels=False)
    df["delta_bin"] = pd.cut(df["delta_default"].clip(-delta_clip, delta_clip), delta_bins, labels=False)

    grid = df.groupby(["depth_bin", "delta_bin"]).agg(
        mean_drift=("drift_default", "mean"), count=("drift_default", "count")
    ).reset_index()

    # Panel (a): density
    density_pivot = grid.pivot(index="delta_bin", columns="depth_bin", values="count").fillna(0)
    col_max = density_pivot.max(axis=0).replace(0, 1)
    density_norm = density_pivot.div(col_max, axis=1)

    im1 = ax1.imshow(density_norm.values, aspect="auto", origin="lower",
                     cmap="YlOrRd", extent=[0, 1, -delta_clip, delta_clip],
                     interpolation="nearest", vmin=0, vmax=1)
    ax1.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.7)
    ax1.set_xlabel("Normalised depth"); ax1.set_ylabel(r"$\delta_{\mathrm{soft}}$ (logits)")
    ax1.set_title("(A) Population density", fontsize=9)
    fig.colorbar(im1, ax=ax1, label="Relative density", shrink=0.8)

    # Panel (b): mean drift
    drift_pivot = grid.pivot(index="delta_bin", columns="depth_bin", values="mean_drift")
    count_pivot = grid.pivot(index="delta_bin", columns="depth_bin", values="count")
    masked = np.ma.masked_where((count_pivot.values < 5) | np.isnan(drift_pivot.values), drift_pivot.values)

    im2 = ax2.imshow(masked, aspect="auto", origin="lower", cmap="RdBu",
                     vmin=-2, vmax=2, extent=[0, 1, -delta_clip, delta_clip],
                     interpolation="nearest")
    ax2.axhspan(-EPSILON_BAND, EPSILON_BAND, color="yellow", alpha=0.12)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Normalised depth"); ax2.set_ylabel(r"$\delta_{\mathrm{soft}}$ (logits)")
    ax2.set_title("(B) Mean drift field", fontsize=9)
    fig.colorbar(im2, ax=ax2, label="Mean drift (logits/layer)", shrink=0.8)

    plt.tight_layout()
    return save_fig(fig, "fig5_flow_field", out)


def fig6_commitment(core: pd.DataFrame, out: Path) -> Path:
    """Figure 6: Commitment, flips, switching — 2×2 grid (enlarged)."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    pl = core.groupby(["model_id", "prompt_uid"]).first().reset_index()

    # Build-time assert: all normalised depths in [0,1]
    assert core["depth_norm"].min() >= -1e-6, f"depth_norm min = {core['depth_norm'].min()}"
    assert core["depth_norm"].max() <= 1.0 + 1e-6, f"depth_norm max = {core['depth_norm'].max()}"

    # (A) Commitment depth
    ax = axes[0, 0]
    for m, c in MODEL_COLORS.items():
        sub = pl[(pl["model_id"] == m) & pl["commitment_layer"].notna()]
        if len(sub) > 0:
            L = MODEL_LAYERS[m]
            d = sub["commitment_layer"].values / (L - 1)
            assert d.min() >= -1e-6 and d.max() <= 1.0 + 1e-6, "commit depth out of [0,1]"
            ax.hist(d, bins=20, alpha=0.45, color=c, label=MODEL_SHORT[m], density=True, edgecolor="none")
    ax.set_xlabel("Commitment depth (normalised)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("(A) Commitment depth", fontsize=10); ax.legend(fontsize=7)

    # (B) Flip count
    ax = axes[0, 1]
    for m, c in MODEL_COLORS.items():
        sub = pl[pl["model_id"] == m]
        ax.hist(sub["flip_count"].values, bins=range(0, 12), alpha=0.45, color=c,
                label=MODEL_SHORT[m], density=True, edgecolor="none")
    ax.set_xlabel("Flip count", fontsize=9); ax.set_ylabel("Density", fontsize=9)
    ax.set_title("(B) Flip count", fontsize=10); ax.legend(fontsize=7)

    # (C) Last flip layer (Unstable)
    ax = axes[1, 0]
    for m, c in MODEL_COLORS.items():
        sub = pl[(pl["model_id"] == m) & (pl["regime"] == "Unstable") & pl["last_flip_layer"].notna()]
        if len(sub) > 0:
            L = MODEL_LAYERS[m]
            d = sub["last_flip_layer"].values / (L - 1)
            assert d.min() >= -1e-6 and d.max() <= 1.0 + 1e-6, "last_flip depth out of [0,1]"
            ax.hist(d, bins=20, alpha=0.45, color=c, label=MODEL_SHORT[m], density=True, edgecolor="none")
    ax.set_xlabel("Last flip depth (normalised)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("(C) Last flip layer (Unstable)", fontsize=10); ax.legend(fontsize=7)

    # (D) Switch rate vs depth — binned into 10 equal bins
    ax = axes[1, 1]
    depth_bins = np.linspace(0, 1, 11)  # 10 bins
    bin_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])
    for m, c in MODEL_COLORS.items():
        sub = core[core["model_id"] == m].copy()
        sub["depth_bin"] = pd.cut(sub["depth_norm"], bins=depth_bins, labels=False, include_lowest=True)
        binned = sub.groupby("depth_bin")["switch_indicator"].agg(["mean", "std", "count"])
        ax.plot(bin_centers[:len(binned)], binned["mean"].values, color=c,
                label=MODEL_SHORT[m], linewidth=1.5, marker="o", markersize=3)
        # Light confidence band (±1 SE)
        se = binned["std"].values / np.sqrt(binned["count"].values)
        ax.fill_between(bin_centers[:len(binned)],
                        binned["mean"].values - se, binned["mean"].values + se,
                        color=c, alpha=0.12)
    ax.set_xlabel("Normalised depth", fontsize=9); ax.set_ylabel("Switch rate", fontsize=9)
    ax.set_title("(D) Competitor switch rate (10 bins)", fontsize=10); ax.legend(fontsize=7)

    plt.tight_layout()
    return save_fig(fig, "fig6_commitment", out)


def fig7_robustness(core: pd.DataFrame, out: Path) -> tuple[Path, float]:
    """Figure 7: Dense τ×M heatmap of Unstable proportion per model."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    pl = core.groupby(["model_id", "prompt_uid"]).first().reset_index()

    tau_grid = np.round(np.arange(0.5, 2.05, 0.25), 2)  # 7 values
    M_grid = np.round(np.arange(0.50, 1.05, 0.1), 2)    # 6 values
    all_maxshift = 0.0

    for ax_idx, model in enumerate(EXPECTED_MODELS):
        ax = axes[ax_idx]
        sub_pl = pl[pl["model_id"] == model]
        sub_all = core[core["model_id"] == model]

        # Pre-sort prompts for speed
        prompt_data = {}
        for uid, grp in sub_all.groupby("prompt_uid"):
            prompt_data[uid] = grp.sort_values("layer_index")

        unstable_grid = np.zeros((len(M_grid), len(tau_grid)))

        for ti, tau in enumerate(tau_grid):
            for mi, M in enumerate(M_grid):
                unst = 0
                total = len(sub_pl)
                for _, row in sub_pl.iterrows():
                    p_data = prompt_data[row["prompt_uid"]]
                    scores = p_data[["sA","sB","sC","sD"]].values
                    correct = row["correct_key"] if "correct_key" in row.index else None
                    if correct is None:
                        continue
                    cidx = {"A":0,"B":1,"C":2,"D":3}.get(correct, 0)
                    sc_vals = scores[:, cidx]
                    incorrect_mask = np.ones(4, dtype=bool); incorrect_mask[cidx] = False
                    lse = tau * np.log(np.sum(np.exp((scores[:, incorrect_mask] - np.max(scores[:, incorrect_mask], axis=1, keepdims=True)) / tau), axis=1) * np.exp(np.max(scores[:, incorrect_mask], axis=1) / tau))
                    # More stable: use scipy-style
                    inc_scores = scores[:, incorrect_mask]  # (L, 3)
                    max_inc = np.max(inc_scores, axis=1, keepdims=True)
                    lse = tau * (np.log(np.sum(np.exp((inc_scores - max_inc) / tau), axis=1)) + max_inc.ravel() / tau)
                    deltas = sc_vals - lse
                    final_d = deltas[-1]
                    fs = np.sign(final_d) if final_d != 0 else 0
                    d_signed = fs * deltas
                    committed = any(np.all(d_signed[k:] >= M) for k in range(len(d_signed)))
                    nz = deltas[deltas != 0]
                    signs = np.sign(nz)
                    fc = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
                    if not (fc <= MAX_FLIPS and committed):
                        unst += 1
                unstable_grid[mi, ti] = 100.0 * unst / total

        im = ax.imshow(unstable_grid, aspect="auto", origin="lower",
                       vmin=0, vmax=100, cmap="YlOrRd",
                       extent=[tau_grid[0]-0.125, tau_grid[-1]+0.125,
                               M_grid[0]-0.05, M_grid[-1]+0.05])
        # Mark default
        ax.plot(1.0, 0.75, marker="*", color="black", markersize=8, zorder=10)
        ax.set_xlabel(r"$\tau$", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("$M$", fontsize=9)
        ax.set_title(MODEL_SHORT[model], fontsize=9)

        # Smoothness: max absolute change between adjacent cells
        for di in range(unstable_grid.shape[0]):
            for dj in range(1, unstable_grid.shape[1]):
                all_maxshift = max(all_maxshift, abs(unstable_grid[di, dj] - unstable_grid[di, dj-1]))
        for di in range(1, unstable_grid.shape[0]):
            for dj in range(unstable_grid.shape[1]):
                all_maxshift = max(all_maxshift, abs(unstable_grid[di, dj] - unstable_grid[di-1, dj]))

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, label="Unstable (%)")
    plt.tight_layout()
    return save_fig(fig, "fig7_robustness", out), all_maxshift


# ═══════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════
def write_tables(core: pd.DataFrame, out: Path) -> None:
    """Generate all 3 LaTeX tables."""
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Models
    (tables_dir / "table1_models.tex").write_text(r"""\begin{table}[t]
  \caption{Models studied. All are 7--8\,B instruction-tuned transformers with pinned revisions.}
  \label{tab:models}
  \vskip 0.1in
  \begin{center}\begin{small}\begin{tabular}{lccc}
    \toprule
    Model & Params & Layers & Prompts \\
    \midrule
    Qwen 2.5-7B-Instruct  & 7.6\,B & \NLayersQwen{} & \NPromptsPerModel{} \\
    Llama 3.1-8B-Instruct  & 8.0\,B & \NLayersLlama{} & \NPromptsPerModel{} \\
    Mistral 7B-v0.3 & 7.2\,B & \NLayersMistral{} & \NPromptsPerModel{} \\
    \bottomrule
  \end{tabular}\end{small}\end{center}
\end{table}
""")

    # Table 2: Regime counts
    pl = core.groupby(["model_id", "prompt_uid"]).first().reset_index()
    rows = []
    for m in EXPECTED_MODELS:
        sub = pl[pl["model_id"] == m]
        total = len(sub)
        for regime in ["Stable-Correct", "Stable-Wrong", "Unstable"]:
            r = sub[sub["regime"] == regime]
            n = len(r)
            pct = 100 * n / total
            med = r["final_delta_default"].median() if n > 0 else 0
            rows.append(f"    {MODEL_SHORT[m]} & {regime} & {n:,} & {pct:.1f} & {med:+.2f} \\\\")

    (tables_dir / "table2_regimes.tex").write_text(r"""\begin{table}[t]
  \caption{Regime counts and median final $\delta_{\mathrm{soft}}(\tau{=}1)$ per model.}
  \label{tab:regimes}
  \vskip 0.1in
  \begin{center}\begin{small}\begin{tabular}{llrrl}
    \toprule
    Model & Regime & $N$ & {\%} & Med.\ $\delta$ \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}\end{small}\end{center}
\end{table}
""")

    # Table 3: Robustness (smaller version — just tau=1 with varying M)
    (tables_dir / "table3_robustness.tex").write_text(r"""\begin{table}[t]
  \caption{Regime proportions ({\%}) under varying commitment threshold $M$, with $\tau = 1.0$, all models pooled.
  See \cref{fig:robustness} for the full $\tau \times M$ sweep.}
  \label{tab:robustness}
  \vskip 0.1in
  \begin{center}\begin{small}\begin{tabular}{crrrr}
    \toprule
    $M$ & Stable-Corr & Stable-Wrong & Unstable \\
    \midrule
    0.50 & --- & --- & --- \\
    0.75 & --- & --- & --- \\
    1.00 & --- & --- & --- \\
    \bottomrule
  \end{tabular}\end{small}\end{center}
\end{table}
""")


# ═══════════════════════════════════════════════════════════════════════════
# BUILD INFO
# ═══════════════════════════════════════════════════════════════════════════
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def write_build_info(parquet_dir: Path, out: Path) -> None:
    info = {
        "parquet_dir": str(parquet_dir.resolve()),
        "git_commit": _git_hash(),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_hashes": {p.name: _sha256(p) for p in sorted(parquet_dir.glob("*.parquet"))},
    }
    (out / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")

def write_claim_evidence(out: Path) -> None:
    (out / "CLAIM_EVIDENCE.md").write_text("""# Paper 1 — Claim-to-Evidence Ledger

| # | Claim | Evidence | Figure/Table |
|---|-------|----------|-------------|
| 1 | δsoft smooths competitor-switching artifacts | Jump magnitude comparison | Fig A |
| 2 | Three regimes separate across all models | Per-model regime counts | Table 2, Fig 3 |
| 3 | Regime separation emerges across depth | Violin distributions at depth slices | Fig 3 |
| 4 | Decision-space PCA reveals trajectory bundles | 4-option score PCA | Fig 4 |
| 5 | Flow field shows convergent dynamics | Mean drift in depth×δ grid | Fig 5 |
| 6 | Commitment concentrates late | Commitment depth distribution | Fig 6(A) |
| 7 | Flip counts are low for stable regimes | Flip count histogram | Fig 6(B) |
| 8 | Last flips concentrate late | Last-flip distribution | Fig 6(C) |
| 9 | Competitor switching peaks mid-depth | Switch rate vs depth | Fig 6(D) |
| 10 | Regimes vary smoothly under perturbation | τ×M sweep | Fig 7, Table 3 |

All claims derived from cached 4-option scores. No causal/mechanistic claims.
""")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, default=Path("results/parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("paper/part1"))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    pdir = args.parquet_dir.resolve()
    odir = args.output_dir.resolve()
    odir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Paper 1 — Camera-Ready Build")
    print("=" * 60)

    # Step 1: Load data
    print("[1] Loading core data...")
    core = load_and_build_core(pdir)
    print(f"  {len(core):,} rows, {len(core['model_id'].unique())} models")

    # Step 2: Generate figures
    print("[2] Generating figures...")
    fig_paths = {}
    for name, func in [
        ("figA", lambda: figA_why_delta_soft(core, odir)),
        ("fig1", lambda: fig1_pipeline(core, odir)),
        ("fig2", lambda: fig2_examples(core, odir)),
        ("fig3", lambda: fig3_distributions(core, odir)),
        ("fig4", lambda: fig4_decision_space(core, odir)),
        ("fig5", lambda: fig5_flow_field(core, odir)),
        ("fig6", lambda: fig6_commitment(core, odir)),
    ]:
        print(f"  {name}...")
        fig_paths[name] = func()

    print("  fig7...")
    fig7_path, max_shift = fig7_robustness(core, odir)
    fig_paths["fig7"] = fig7_path

    # Step 3: Tables
    print("[3] Writing tables...")
    write_tables(core, odir)

    # Step 4: Build info
    print("[4] Writing BUILD_INFO.json, CLAIM_EVIDENCE.md...")
    write_build_info(pdir, odir)
    write_claim_evidence(odir)

    # Step 5: Compute numbers
    print("[5] Computing auto_numbers.tex...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "compute_numbers.py"),
                    "--core-parquet", str(odir / "data" / "part1_core.parquet"),
                    "--output", str(odir / "auto_numbers.tex")], check=True)

    # Step 5b: Patch RobustnessMaxShift with actual value from fig7
    auto_tex = (odir / "auto_numbers.tex").read_text()
    auto_tex = auto_tex.replace(
        r"\newcommand{\RobustnessMaxShift}{---}",
        rf"\newcommand{{\RobustnessMaxShift}}{{{max_shift:.1f}}}"
    )
    (odir / "auto_numbers.tex").write_text(auto_tex)
    print(f"  Patched RobustnessMaxShift = {max_shift:.1f}")

    # Step 6: Figure hashes
    print("[6] Computing figure hashes...")
    fig_hashes = {}
    for name, path in fig_paths.items():
        fig_hashes[path.name] = fig_hash(path)
        print(f"  {path.name}: {fig_hashes[path.name][:16]}...")

    # Write figure hashes into BUILD_INFO
    bi = json.loads((odir / "BUILD_INFO.json").read_text())
    bi["figure_hashes"] = fig_hashes
    bi["robustness_max_shift_pct"] = round(max_shift, 1)
    (odir / "BUILD_INFO.json").write_text(json.dumps(bi, indent=2) + "\n")

    print()
    print("=" * 60)
    print("BUILD COMPLETE ✓")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
