#!/usr/bin/env python3
"""Generate all paper figures from stored parquet results.

Each function produces one vector-PDF figure for the NeurIPS submission.
All figures use only pre-computed parquet data; no inference is run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Annotation
try:
    from scipy.ndimage import uniform_filter1d
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    uniform_filter1d = None

from sow.v2.figures.style import (
    COMP,
    COMP_LABELS,
    FILL,
    FONT_ANNOT,
    FONT_AXIS,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    LW_DATA,
    LW_REF,
    NEUTRAL,
    SPAN,
    SPAN_LABELS,
    TEXT_WIDTH,
    TRAJ,
    TRAJ_LABELS,
    TRAJ_ORDER,
    add_panel_label,
    add_zero_line,
    annot_arrow,
    bootstrap_ci,
    configure_matplotlib,
    cov_ellipse,
    glow_line,
    gradient_fill,
    remove_top_right_spines,
    shade_threshold_region,
    shared_legend,
    style_card,
)
from sow.v2.figures.data_loaders import (
    MODEL_LAYERS,
    load_attention_data,
    load_causal_data,
    load_decision_metrics,
    load_tracing_scalars,
)
from sow.v2.span_parser import parse_prompt_spans


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[4]


def _short_model_id(model_id: str) -> str:
    mid = str(model_id)
    if "Qwen" in mid:
        return "Qwen2.5-7B"
    if "Llama" in mid:
        return "Llama-3.1-8B"
    if "Mistral" in mid:
        return "Mistral-7B"
    return mid.split("/")[-1]


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5]).tolist()
    return float(np.mean(vals)), float(lo), float(hi)


def _load_prompt_lookup(prompts_path: Optional[Path]) -> dict[str, str]:
    path = prompts_path
    if path is None:
        default = _repo_root_from_this_file() / "prompt_packs" / "ccc_baseline_v1_3000.jsonl"
        path = default if default.exists() else None
    if path is None or not path.exists():
        return {}
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            try:
                rec = json.loads(row)
            except Exception:
                continue
            uid = str(rec.get("prompt_uid") or "")
            txt = str(rec.get("prompt_text") or "")
            if uid and txt:
                lookup[uid] = txt
    return lookup


def _assert_layout_integrity(fig: plt.Figure, *, fig_name: str) -> None:
    """Fail fast on overlap, off-canvas text, and unsafe annotation arrows."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    tol = 2.0

    text_boxes: list[tuple[object, any, object]] = []
    fig_level_artists: list[tuple[str, object]] = []

    if getattr(fig, "_suptitle", None) is not None:
        fig_level_artists.append(("suptitle", fig._suptitle))
    for leg in fig.legends:
        for t in leg.get_texts():
            if t.get_visible():
                fig_level_artists.append(("figure_legend", t))
    for ax_idx, ax in enumerate(fig.axes):
        artists: list = [ax.title, ax.xaxis.label, ax.yaxis.label]
        artists.extend([t for t in ax.texts if t.get_visible()])
        legend = ax.get_legend()
        if legend is not None:
            artists.extend([t for t in legend.get_texts() if t.get_visible()])

        for art in artists:
            if hasattr(art, "get_text") and not str(art.get_text() or "").strip():
                continue
            try:
                bb = art.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bb.width <= 2.0 or bb.height <= 2.0:
                continue

            if bb.x0 < fb.x0 - tol or bb.y0 < fb.y0 - tol or bb.x1 > fb.x1 + tol or bb.y1 > fb.y1 + tol:
                label = str(getattr(art, "get_text", lambda: type(art).__name__)())
                raise ValueError(f"{fig_name}: off-canvas label in axis {ax_idx}: {label!r}")
            text_boxes.append((art, bb, ax))

            if isinstance(art, Annotation) and art.arrow_patch is not None:
                if art.get_annotation_clip() is not True:
                    label = str(art.get_text() or "")
                    raise ValueError(f"{fig_name}: annotation_clip must be True for arrow annotation {label!r}")
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x, y = art.xy
                if art.xycoords == "data":
                    if not (min(xlim) - 1e-9 <= float(x) <= max(xlim) + 1e-9 and min(ylim) - 1e-9 <= float(y) <= max(ylim) + 1e-9):
                        label = str(art.get_text() or "")
                        raise ValueError(f"{fig_name}: arrow target out of axis bounds in axis {ax_idx}: {label!r}")

    for kind, art in fig_level_artists:
        if hasattr(art, "get_text") and not str(art.get_text() or "").strip():
            continue
        try:
            bb = art.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 2.0 or bb.height <= 2.0:
            continue
        if bb.x0 < fb.x0 - tol or bb.y0 < fb.y0 - tol or bb.x1 > fb.x1 + tol or bb.y1 > fb.y1 + tol:
            label = str(getattr(art, "get_text", lambda: type(art).__name__)())
            raise ValueError(f"{fig_name}: off-canvas {kind}: {label!r}")
        text_boxes.append((art, bb, None))

    for i in range(len(text_boxes)):
        _, bb1, ax1 = text_boxes[i]
        for j in range(i + 1, len(text_boxes)):
            _, bb2, ax2 = text_boxes[j]
            x0 = max(bb1.x0, bb2.x0)
            y0 = max(bb1.y0, bb2.y0)
            x1 = min(bb1.x1, bb2.x1)
            y1 = min(bb1.y1, bb2.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            inter = float((x1 - x0) * (y1 - y0))
            min_area = float(min(max(bb1.width * bb1.height, 1.0), max(bb2.width * bb2.height, 1.0)))
            overlap = inter / min_area
            same_axis = (ax1 is not None) and (ax1 is ax2)
            area_thr = 40.0 if same_axis else 55.0
            if inter >= area_thr and overlap >= 0.45:
                t1 = str(getattr(text_boxes[i][0], "get_text", lambda: type(text_boxes[i][0]).__name__)())
                t2 = str(getattr(text_boxes[j][0], "get_text", lambda: type(text_boxes[j][0]).__name__)())
                raise ValueError(f"{fig_name}: overlapping labels detected: {t1!r} vs {t2!r}")


def _save_figure(fig: plt.Figure, out: Path, *, fig_name: str) -> Path:
    _assert_layout_integrity(fig, fig_name=fig_name)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def _new_previews_dir() -> Path:
    return _repo_root_from_this_file() / "tmp" / "fig_previews"


# ---------------------------------------------------------------------------
# Figure 1: Three primitives
# ---------------------------------------------------------------------------

def fig1_primitives(parquet_dir: Path, output_dir: Path) -> Path:
    dm = load_decision_metrics(parquet_dir)
    dm["depth_bin"] = dm["depth"].round(2)

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.30), constrained_layout=False)
    fig.subplots_adjust(left=0.090, right=0.970, bottom=0.14, top=0.80, wspace=0.30)
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    specs = [
        ("delta", r"Decision margin $\delta$ (logits)", "STATE"),
        ("drift", r"Per-layer drift $g$ (logits)", "MOTION"),
        ("boundary", r"Boundary $|\delta|$ (logits)", "BOUNDARY"),
    ]

    # Accent colors for panel title underlines
    _accent = {"STATE": "#0a6abf", "MOTION": "#139e8c", "BOUNDARY": "#c93545"}

    handles_legend = []
    labels_legend = []

    rng_ghost = np.random.default_rng(42)

    for idx, (col, ylabel, title) in enumerate(specs):
        ax = axes[idx]
        style_card(ax)
        agg = dm.groupby(["trajectory_type", "depth_bin"], sort=True)[col].agg(["mean", "std", "count"]).reset_index()

        if col in {"delta", "drift"}:
            q = float(np.nanpercentile(np.abs(dm[col].to_numpy(dtype=np.float64)), 99))
            lim = min(max(q * 1.25, 1.8), 6.0)
            y_clip = (-lim, lim)
        else:
            q = float(np.nanpercentile(dm[col].to_numpy(dtype=np.float64), 99))
            y_hi = min(max(q * 1.15, 1.4), 8.5)
            y_clip = (0.0, y_hi)

        # Ghost individual trajectories — fewer, smoother, fainter
        for tt in ["stable_correct", "stable_wrong"]:
            sub_raw = dm[dm["trajectory_type"] == tt]
            pairs = sub_raw[["model_id", "prompt_uid"]].drop_duplicates().reset_index(drop=True)
            if pairs.empty:
                continue
            chosen_idx = rng_ghost.choice(pairs.index.to_numpy(), size=min(20, len(pairs)), replace=False)
            for i_pair in chosen_idx:
                pair = pairs.iloc[int(i_pair)]
                traj = sub_raw[
                    (sub_raw["model_id"] == pair["model_id"]) & (sub_raw["prompt_uid"] == pair["prompt_uid"])
                ].sort_values("depth_bin")
                if len(traj) >= 3:
                    y_ghost = np.clip(traj[col].to_numpy(dtype=np.float64), y_clip[0], y_clip[1])
                    # Light Gaussian smoothing on ghosts
                    if uniform_filter1d is not None and len(y_ghost) > 3:
                        y_ghost = uniform_filter1d(y_ghost, size=3, mode="nearest")
                    ax.plot(
                        traj["depth_bin"].to_numpy(dtype=np.float64),
                        y_ghost,
                        color=TRAJ[tt], lw=0.5, alpha=0.05, zorder=0,
                        solid_capstyle="round",
                    )

        for tt in TRAJ_ORDER:
            sub = agg[agg["trajectory_type"] == tt].sort_values("depth_bin")
            x = sub["depth_bin"].to_numpy(dtype=np.float64)
            y = sub["mean"].to_numpy(dtype=np.float64)
            sem = (sub["std"] / np.sqrt(sub["count"].clip(lower=1))).fillna(0.0).to_numpy(dtype=np.float64)

            stable = tt in {"stable_correct", "stable_wrong"}
            ls = "-" if stable else "--"

            if stable:
                # Glow line for stable trajectories
                line = glow_line(ax, x, y, color=TRAJ[tt], lw=2.2, glow_alpha=0.12,
                                 glow_width=4.0, zorder=4, ls=ls, label=TRAJ_LABELS[tt])
                # Gradient SEM fill
                gradient_fill(ax, x, y + sem, color=TRAJ[tt], alpha_max=0.18,
                              n_steps=5, reference=y, zorder=1)
                gradient_fill(ax, x, y - sem, color=TRAJ[tt], alpha_max=0.18,
                              n_steps=5, reference=y, zorder=1)
            else:
                line, = ax.plot(x, y, color=TRAJ[tt], lw=1.4, ls=ls, alpha=0.60,
                                label=TRAJ_LABELS[tt], zorder=3)

            if idx == 0:
                handles_legend.append(line)
                labels_legend.append(TRAJ_LABELS[tt])

        add_zero_line(ax)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_clip[0], y_clip[1])
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(ylabel)

        # Bold uppercase title with colored accent underline
        ax.set_title(title, fontsize=FONT_TITLE + 0.5, fontweight="bold",
                     color="#1a202c", pad=8)
        # Accent underline via a thin colored line below title
        ax.annotate("", xy=(0.15, 1.02), xytext=(0.85, 1.02),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-", lw=2.0, color=_accent[title], alpha=0.6),
                     annotation_clip=False)
        add_panel_label(ax, chr(ord("a") + idx))

        if col == "boundary":
            # Gradient threshold shading instead of flat fill
            for step_i in range(8):
                y_lo = 0.30 * step_i / 8
                y_hi_step = 0.30 * (step_i + 1) / 8
                a = 0.22 * (1.0 - step_i / 8)
                ax.axhspan(y_lo, y_hi_step, color=FILL["wrong_bg"], alpha=a, zorder=-1)
            ax.axhline(0.3, color=NEUTRAL, lw=LW_REF, ls="--", zorder=1, alpha=0.7)
            ax.text(0.03, 0.33, r"$b_{\min}=0.3$", color=NEUTRAL, fontsize=FONT_ANNOT,
                    transform=ax.get_yaxis_transform())

    shared_legend(fig, handles_legend, labels_legend, ncol=4, bbox_to_anchor=(0.5, 0.97))

    out = output_dir / "fig1_primitives.pdf"
    return _save_figure(fig, out, fig_name="fig1_primitives")


# ---------------------------------------------------------------------------
# Figure 2: Phase diagram
# ---------------------------------------------------------------------------

def fig2_phase_diagram(parquet_dir: Path, output_dir: Path) -> Path:
    dm = load_decision_metrics(parquet_dir)
    tau = 8

    rows = []
    for (_, _), grp in dm.groupby(["model_id", "prompt_uid"], sort=False):
        sg = grp.sort_values("layer_index")
        model_id = str(sg["model_id"].iloc[0])
        L = MODEL_LAYERS.get(model_id, int(sg["layer_index"].max()) + 1)
        tail_start = max(L - tau, 0)
        tail = sg[sg["layer_index"] >= tail_start]
        if tail.empty:
            continue
        entry = sg[sg["layer_index"] == tail_start]
        boundary_entry = float(entry["boundary"].iloc[0]) if not entry.empty else np.nan
        signs = np.sign(tail["delta"].to_numpy(dtype=np.float64))
        signs = signs[signs != 0]
        flips = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
        rows.append(
            {
                "boundary_at_entry": boundary_entry,
                "tail_flips": flips,
                "trajectory_type": str(sg["trajectory_type"].iloc[0]),
            }
        )

    pdf = pd.DataFrame.from_records(rows)
    if pdf.empty:
        raise ValueError("fig2_phase_diagram: no records after aggregation")

    fig = plt.figure(figsize=(3.50, 3.20), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.1, 0.95], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

    style_card(ax)
    style_card(ax_hist)

    # Scatter points colored by trajectory type — the data IS the figure
    for tt in TRAJ_ORDER:
        sub = pdf[pdf["trajectory_type"] == tt]
        if sub.empty:
            continue
        ax.scatter(
            sub["boundary_at_entry"], sub["tail_flips"],
            s=12, c=TRAJ[tt], alpha=0.50, zorder=3, edgecolors="none",
            rasterized=True, label=TRAJ_LABELS[tt],
        )

    # Covariance ellipses — dashed, refined
    legend_handles = []
    for tt in TRAJ_ORDER:
        sub = pdf[pdf["trajectory_type"] == tt]
        if sub.empty:
            continue
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor=TRAJ[tt], linewidth=1.4, label=TRAJ_LABELS[tt])
        )
        ell = cov_ellipse(
            sub["boundary_at_entry"].to_numpy(dtype=np.float64),
            sub["tail_flips"].to_numpy(dtype=np.float64),
            edgecolor=TRAJ[tt],
            facecolor=TRAJ[tt],
            alpha=0.10,
            lw=1.6,
        )
        if ell is not None:
            ell.set_linestyle("--")
            ax.add_patch(ell)

    # Threshold lines
    ax.axvline(0.3, color=NEUTRAL, lw=1.2, ls="--", alpha=0.7)
    ax.axhline(0.5, color=NEUTRAL, lw=1.2, ls="--", alpha=0.7)
    ax.text(0.31, 0.03, r"$b_{\min}$", fontsize=FONT_ANNOT, color=NEUTRAL,
            transform=ax.get_xaxis_transform())
    ax.text(0.99, 0.52, "flip threshold", fontsize=FONT_ANNOT, color=NEUTRAL,
            ha="right", transform=ax.get_yaxis_transform())

    # Region labels — clean text, no badges
    _region_labels = [
        (0.05, 0.06, "Stable-Wrong",     "stable_wrong"),
        (0.55, 0.06, "Stable-Correct",   "stable_correct"),
        (0.05, 0.72, "Unstable-Wrong",   "unstable_wrong"),
        (0.55, 0.72, "Unstable-Correct", "unstable_correct"),
    ]
    for rx, ry, rtxt, rtt in _region_labels:
        ax.text(
            rx, ry, rtxt,
            transform=ax.transAxes,
            fontsize=FONT_ANNOT - 0.3,
            fontstyle="italic",
            color=TRAJ[rtt],
            alpha=0.85,
        )

    ax.set_xlabel(r"$|\delta(L{-}\tau)|$ at tail entry (logits)")
    ax.set_ylabel(r"Tail sign flips (last $\tau{=}8$ layers)")
    ax.set_title("Phase Diagram", fontsize=FONT_TITLE + 0.4, fontweight="bold",
                 color="#1a202c")
    add_panel_label(ax, "a")
    ax.legend(frameon=False, fontsize=FONT_LEGEND - 0.2,
              loc="upper right")

    # Styled marginal histogram
    bins = np.arange(-0.5, float(pdf["tail_flips"].max()) + 1.5, 1.0)
    ax_hist.hist(pdf["tail_flips"], bins=bins, orientation="horizontal",
                 color="#a0aec0", edgecolor="#718096", lw=0.5, alpha=0.8)
    ax_hist.set_xlabel("Count")
    ax_hist.set_title("Marginal", fontsize=FONT_TITLE - 0.1, color="#4a5568")
    ax_hist.tick_params(axis="y", labelleft=False)

    out = output_dir / "fig2_phase_diagram.pdf"
    return _save_figure(fig, out, fig_name="fig2_phase_diagram")


# ---------------------------------------------------------------------------
# Figure 3: Drift decomposition and diagnostics
# ---------------------------------------------------------------------------

def fig3_decomposition(parquet_dir: Path, output_dir: Path) -> Path:
    ts = load_tracing_scalars(parquet_dir)
    dm = load_decision_metrics(parquet_dir)
    ts["depth_bin"] = (ts["depth"] * 24).round() / 24

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.90), constrained_layout=False)
    fig.subplots_adjust(left=0.092, right=0.996, bottom=0.12, top=0.86, wspace=0.44, hspace=0.72)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.18, 1.0], wspace=0.44, hspace=0.68)
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        style_card(ax)

    def _trace_panel(
        ax: plt.Axes,
        *,
        traj_type: str,
        label: str,
        title: str,
        show_ylabel: bool,
    ) -> tuple[list, list]:
        sub = ts[ts["trajectory_type"] == traj_type]
        agg = sub.groupby("depth_bin", sort=True).agg(
            s_attn=("s_attn", "mean"),
            s_mlp=("s_mlp", "mean"),
            drift=("drift", "mean"),
            s_attn_sem=("s_attn", "sem"),
            s_mlp_sem=("s_mlp", "sem"),
        ).reset_index()

        x = agg["depth_bin"].to_numpy(dtype=np.float64)
        attn = agg["s_attn"].to_numpy(dtype=np.float64)
        mlp = agg["s_mlp"].to_numpy(dtype=np.float64)
        drift = agg["drift"].to_numpy(dtype=np.float64)
        attn_sem = agg["s_attn_sem"].fillna(0.0).to_numpy(dtype=np.float64)
        mlp_sem = agg["s_mlp_sem"].fillna(0.0).to_numpy(dtype=np.float64)

        if uniform_filter1d is not None:
            _smooth = lambda v: uniform_filter1d(v, size=3, mode="nearest")
        else:
            _kernel = np.ones(3, dtype=np.float64) / 3.0
            _smooth = lambda v: np.convolve(v, _kernel, mode="same")
        attn = _smooth(attn)
        mlp = _smooth(mlp)
        drift = _smooth(drift)

        # Glow lines for component traces
        l1 = glow_line(ax, x, drift, color=NEUTRAL, lw=1.4, glow_alpha=0.08,
                        glow_width=3.0, zorder=3, ls="--", label=r"Observed drift $g(l)$")
        l2 = glow_line(ax, x, attn, color=COMP["attention"], lw=2.0, glow_alpha=0.12,
                        glow_width=4.0, zorder=4, label=r"Attention $s_{\mathrm{attn}}(l)$")
        l3 = glow_line(ax, x, mlp, color=COMP["mlp"], lw=2.0, glow_alpha=0.12,
                        glow_width=4.0, zorder=4, label=r"MLP $s_{\mathrm{mlp}}(l)$")
        net = attn + mlp
        net_line = glow_line(ax, x, net, color="#2d3748", lw=2.2, glow_alpha=0.10,
                              glow_width=3.5, zorder=5, label="Net force")
        # Gradient fills toward zero
        gradient_fill(ax, x, attn, color=COMP["attention"], alpha_max=0.22, n_steps=5, zorder=1)
        gradient_fill(ax, x, mlp, color=COMP["mlp"], alpha_max=0.22, n_steps=5, zorder=1)
        # Tug-of-war ribbon: color by which force dominates
        ax.fill_between(x, attn, mlp,
                        where=(attn >= mlp),
                        color=COMP["attention"], alpha=0.12, interpolate=True, lw=0, zorder=0)
        ax.fill_between(x, attn, mlp,
                        where=(attn < mlp),
                        color=COMP["mlp"], alpha=0.12, interpolate=True, lw=0, zorder=0)
        # SEM bands
        ax.fill_between(x, attn - attn_sem, attn + attn_sem, color=COMP["attention"], alpha=0.15, lw=0)
        ax.fill_between(x, mlp - mlp_sem, mlp + mlp_sem, color=COMP["mlp"], alpha=0.15, lw=0)
        add_zero_line(ax)
        _yw = float(np.nanpercentile(np.abs(np.concatenate([attn, mlp, drift])), 98))
        _yw = min(max(_yw * 1.4, 1.2), 4.0)
        ax.set_ylim(-_yw, _yw)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalised depth", labelpad=2.0)
        if show_ylabel:
            ax.set_ylabel("Delta margin / layer (logits)")
        else:
            ax.set_ylabel("")
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="semibold", color="#1a202c")
        add_panel_label(ax, label)
        return [l2, l3, net_line, l1], ["Attention", "MLP", "Net force", "Observed drift"]

    handles, labels = _trace_panel(
        ax_a,
        traj_type="stable_correct",
        label="a",
        title="Stable-Correct Decomposition",
        show_ylabel=True,
    )
    _trace_panel(
        ax_b,
        traj_type="stable_wrong",
        label="b",
        title="Stable-Wrong Decomposition",
        show_ylabel=False,
    )
    shared_legend(fig, handles, labels, ncol=4, bbox_to_anchor=(0.5, 0.995))

    sc = ts[ts["trajectory_type"] == "stable_correct"].copy()
    sc["window"] = np.where(sc["depth"] <= 0.30, "early", np.where(sc["depth"] >= 0.70, "late", "mid"))
    sc = sc[sc["window"].isin(["early", "late"])].copy()
    sc["attn_pos"] = sc["s_attn"].clip(lower=0.0)
    sc["mlp_pos"] = sc["s_mlp"].clip(lower=0.0)
    per_prompt = sc.groupby(["model_id", "prompt_uid", "window"], sort=False)[["attn_pos", "mlp_pos"]].mean().reset_index()
    denom = (per_prompt["attn_pos"] + per_prompt["mlp_pos"]).clip(lower=1e-12)
    per_prompt["attn_share"] = per_prompt["attn_pos"] / denom
    per_prompt["mlp_share"] = 1.0 - per_prompt["attn_share"]

    windows = ["early", "late"]
    xpos = np.arange(2, dtype=np.float64)
    width = 0.30
    for comp_idx, comp_col in enumerate(["attn_share", "mlp_share"]):
        color = COMP["attention"] if comp_col == "attn_share" else COMP["mlp"]
        offset = -width / 2 if comp_col == "attn_share" else width / 2
        means, low, high = [], [], []
        for win in windows:
            vals = per_prompt.loc[per_prompt["window"] == win, comp_col].to_numpy(dtype=np.float64)
            m, lo, hi = _bootstrap_mean_ci(vals)
            means.append(m)
            low.append(lo)
            high.append(hi)
        marr = np.asarray(means, dtype=np.float64)
        err = np.vstack([marr - np.asarray(low), np.asarray(high) - marr])
        ax_c.bar(xpos + offset, marr, width=width, color=color, alpha=0.86, label="Attention" if comp_col == "attn_share" else "MLP")
        ax_c.errorbar(xpos + offset, marr, yerr=err, fmt="none", ecolor=color, lw=1.0, capsize=2.6)

    ax_c.axhline(0.5, color=NEUTRAL, lw=LW_REF, ls="--")
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(["Early\n(0.00-0.30)", "Late\n(0.70-1.00)"])
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel("Positive steering share")
    ax_c.set_title("Dominance", fontsize=FONT_TITLE, pad=10)
    add_panel_label(ax_c, "c")
    ax_c.legend(frameon=False, fontsize=FONT_LEGEND - 0.2, loc="upper right")

    align_rows = []
    for (_, _), grp in ts.groupby(["model_id", "prompt_uid"], sort=False):
        s1 = np.sign(grp["s_attn"].to_numpy(dtype=np.float64))
        s2 = np.sign(grp["s_mlp"].to_numpy(dtype=np.float64))
        mask = (s1 != 0.0) & (s2 != 0.0)
        if int(np.sum(mask)) == 0:
            continue
        align_rows.append(
            {
                "trajectory_type": str(grp["trajectory_type"].iloc[0]),
                "alignment_frac": float(np.mean(s1[mask] == s2[mask])),
            }
        )
    adf = pd.DataFrame.from_records(align_rows)

    x = np.arange(len(TRAJ_ORDER), dtype=np.float64)
    means, ylo, yhi = [], [], []
    for tt in TRAJ_ORDER:
        vals = adf.loc[adf["trajectory_type"] == tt, "alignment_frac"].to_numpy(dtype=np.float64)
        m, lo, hi = _bootstrap_mean_ci(vals)
        means.append(m)
        ylo.append(m - lo)
        yhi.append(hi - m)
    means_arr = np.asarray(means, dtype=np.float64)
    yerr = np.vstack([np.asarray(ylo), np.asarray(yhi)])
    ax_d.bar(x, means_arr, color=[TRAJ[t] for t in TRAJ_ORDER], alpha=0.86, width=0.65,
             edgecolor="none", linewidth=0)
    ax_d.errorbar(x, means_arr, yerr=yerr, fmt="none", ecolor=NEUTRAL, lw=1.0, capsize=2.6)
    traj_ticks = ["S-C", "S-W", "U-C", "U-W"]
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(traj_ticks)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_ylabel("Align. frac")
    ax_d.set_title("Sign Alignment", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c", pad=10)
    add_panel_label(ax_d, "d")
    # Value labels on bars
    for i_bar, m_bar in enumerate(means_arr):
        ax_d.text(i_bar, m_bar + 0.03, f"{m_bar:.2f}", ha="center", va="bottom",
                  fontsize=FONT_ANNOT - 0.3, color=NEUTRAL)

    osc_rows = []
    for (_, _), grp in dm.groupby(["model_id", "prompt_uid"], sort=False):
        arr = grp.sort_values("layer_index")["drift"].to_numpy(dtype=np.float64)
        signs = np.sign(arr)
        nz = signs[signs != 0.0]
        if nz.size <= 1:
            osc_rate = 0.0
        else:
            osc_rate = float(np.sum(nz[1:] != nz[:-1])) / float(nz.size - 1)
        osc_rows.append({"trajectory_type": str(grp["trajectory_type"].iloc[0]), "osc_rate": osc_rate})
    odf = pd.DataFrame.from_records(osc_rows)

    means, ylo, yhi = [], [], []
    for tt in TRAJ_ORDER:
        vals = odf.loc[odf["trajectory_type"] == tt, "osc_rate"].to_numpy(dtype=np.float64)
        m, lo, hi = _bootstrap_mean_ci(vals)
        means.append(m)
        ylo.append(m - lo)
        yhi.append(hi - m)
    means_arr = np.asarray(means, dtype=np.float64)
    yerr = np.vstack([np.asarray(ylo), np.asarray(yhi)])
    ax_e.bar(x, means_arr, color=[TRAJ[t] for t in TRAJ_ORDER], alpha=0.86, width=0.65,
             edgecolor="none", linewidth=0)
    ax_e.errorbar(x, means_arr, yerr=yerr, fmt="none", ecolor=NEUTRAL, lw=1.0, capsize=2.6)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(traj_ticks)
    ax_e.set_ylim(0.0, 1.0)
    ax_e.set_ylabel("Flip rate")
    ax_e.set_title("Trajectory Oscillation", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c", pad=10)
    # Value labels on bars
    for i_bar, m_bar in enumerate(means_arr):
        ax_e.text(i_bar, m_bar + 0.03, f"{m_bar:.2f}", ha="center", va="bottom",
                  fontsize=FONT_ANNOT - 0.3, color=NEUTRAL)
    add_panel_label(ax_e, "e")

    out = output_dir / "fig3_decomposition.pdf"
    return _save_figure(fig, out, fig_name="fig3_decomposition")


# ---------------------------------------------------------------------------
# Figure 4: Attention routing
# ---------------------------------------------------------------------------

def fig4_attention_routing(parquet_dir: Path, output_dir: Path) -> Path:
    mass_m, contrib_m = load_attention_data(parquet_dir)
    mass_m["depth_bin"] = (mass_m["depth"] * 24).round() / 24

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.10), constrained_layout=False)
    fig.subplots_adjust(left=0.072, right=0.975, bottom=0.10, top=0.84, wspace=0.36)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 0.98], wspace=0.34)
    left = gs[0, 0].subgridspec(2, 1, hspace=0.18)
    ax_l1 = fig.add_subplot(left[0, 0])
    ax_l2 = fig.add_subplot(left[1, 0], sharex=ax_l1)
    ax_r = fig.add_subplot(gs[0, 1])

    for ax in [ax_l1, ax_l2, ax_r]:
        style_card(ax)

    role_order = ["instruction", "question_stem", "option_correct", "option_competitor", "option_other", "post_options"]
    role_names = {
        "instruction": "Instruction",
        "question_stem": "Question stem",
        "option_correct": "Correct option",
        "option_competitor": "Competitor option",
        "option_other": "Other options",
        "post_options": "Post-options",
    }
    role_colors = {
        "instruction": "#7fb3d8",
        "question_stem": "#6ec5b8",
        "option_correct": "#17876c",
        "option_competitor": "#c43a4b",
        "option_other": "#e8c96a",
        "post_options": "#a8b2c1",
    }

    def _stack_panel(ax: plt.Axes, tt: str, panel_label: str, title: str) -> list:
        sub = mass_m[mass_m["trajectory_type"] == tt]
        agg = sub.groupby(["depth_bin", "span_rel"], sort=True)["attention_mass"].mean().reset_index()
        piv = agg.pivot(index="depth_bin", columns="span_rel", values="attention_mass").fillna(0.0)
        for role in role_order:
            if role not in piv.columns:
                piv[role] = 0.0
        piv = piv[role_order].sort_index()
        x = piv.index.to_numpy(dtype=np.float64)
        y = piv.to_numpy(dtype=np.float64)
        denom = np.sum(y, axis=1, keepdims=True)
        denom[denom <= 1e-12] = 1.0
        frac = y / denom
        polys = ax.stackplot(
            x,
            frac.T,
            colors=[role_colors[r] for r in role_order],
            alpha=0.92,
            linewidth=0.3,
            edgecolor="none",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Attention mass fraction")
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="semibold", color="#1a202c")
        add_panel_label(ax, panel_label)
        return polys

    polys = _stack_panel(ax_l1, "stable_correct", "a", "Mass Routing: Stable-Correct")
    _stack_panel(ax_l2, "stable_wrong", "b", "Mass Routing: Stable-Wrong")
    ax_l2.set_xlabel("Normalised depth")
    ax_l1.tick_params(axis="x", labelbottom=False)
    shared_legend(fig, polys, [role_names[r] for r in role_order], ncol=3, bbox_to_anchor=(0.34, 0.985))

    cagg = contrib_m.groupby(["span_rel", "trajectory_type"], sort=True)["attention_contribution"].mean().reset_index()
    heat_roles = role_order
    heat_types = TRAJ_ORDER
    mat = np.zeros((len(heat_roles), len(heat_types)), dtype=np.float64)
    for i, role in enumerate(heat_roles):
        for j, tt in enumerate(heat_types):
            vals = cagg[(cagg["span_rel"] == role) & (cagg["trajectory_type"] == tt)]["attention_contribution"]
            mat[i, j] = float(vals.iloc[0]) if not vals.empty else 0.0

    vmax = float(np.nanmax(np.abs(mat)))
    vmax = vmax if vmax > 1e-9 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax_r.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    heat_type_ticks = ["Stable-\nCorrect", "Stable-\nWrong", "Unstable-\nCorrect", "Unstable-\nWrong"]
    ax_r.set_xticks(np.arange(len(heat_types)))
    ax_r.set_xticklabels(heat_type_ticks)
    ax_r.set_yticks(np.arange(len(heat_roles)))
    ax_r.set_yticklabels(["Instr.", "Q stem", "Correct", "Comp.", "Other", "Post"])
    ax_r.tick_params(axis="y", pad=2)
    ax_r.set_title("Decision-Aligned Contribution", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c", pad=7)
    add_panel_label(ax_r, "c")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat[i, j])
            txt_color = "white" if abs(val) > 0.45 * vmax else "#111827"
            ax_r.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=FONT_ANNOT + 0.2, color=txt_color)

    cbar = fig.colorbar(im, ax=ax_r, fraction=0.040, pad=0.014)
    cbar.ax.set_title("logit\nunits", fontsize=FONT_ANNOT)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    out = output_dir / "fig4_attention_routing.pdf"
    return _save_figure(fig, out, fig_name="fig4_attention_routing")


# ---------------------------------------------------------------------------
# Figure 5: Counterfactual validation
# ---------------------------------------------------------------------------

def fig5_counterfactuals(parquet_dir: Path, output_dir: Path) -> Path:
    abl, pat, sl, nc = load_causal_data(parquet_dir)

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.55), constrained_layout=False)
    fig.subplots_adjust(left=0.110, right=0.996, bottom=0.10, top=0.94, wspace=0.26, hspace=0.56)
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax in [ax_a, ax_b, ax_d]:
        style_card(ax)

    model_order = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    y_base = np.arange(len(model_order), dtype=np.float64)
    y_offsets = {"attention": -0.13, "mlp": +0.13}

    for comp in ["attention", "mlp"]:
        for idx, model_id in enumerate(model_order):
            vals = abl[(abl["component"] == comp) & (abl["model_id"] == model_id)]["delta_shift"].to_numpy(dtype=np.float64)
            mean, lo, hi = bootstrap_ci(vals)
            y = y_base[idx] + y_offsets[comp]
            ax_a.errorbar(
                mean,
                y,
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=COMP[comp],
                lw=1.2,
                capsize=2.8,
                markersize=4.6,
                label=COMP_LABELS[comp] if idx == 0 else None,
            )

    ax_a.axvline(0, color=NEUTRAL, lw=LW_REF, ls="--")
    annot_arrow(ax_a, text="higher final margin", xy=(0.6, -0.30), xytext=(1.45, -0.48), color=NEUTRAL)
    ax_a.set_yticks(y_base)
    ax_a.set_yticklabels([_short_model_id(m) for m in model_order])
    ax_a.set_xlabel("Delta delta from simulated removal (logits)")
    ax_a.set_title("(a) Simulated Removal", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c", pad=5)
    add_panel_label(ax_a, "a")
    ax_a.margins(x=0.08)
    ax_a.legend(frameon=False, fontsize=FONT_LEGEND - 0.2, loc="upper left", bbox_to_anchor=(0.01, 0.99))

    base = pat["delta_final_base"].to_numpy(dtype=np.float64)
    patched = pat["delta_final_patched"].to_numpy(dtype=np.float64)
    all_vals = np.concatenate([base, patched])
    lo = float(np.nanpercentile(all_vals, 1))
    hi = float(np.nanpercentile(all_vals, 99))
    bins = np.linspace(lo, hi, 30)
    ax_b.hist(base, bins=bins, density=True, color="#5a7fa8", alpha=0.55,
              label="Fail trajectory (base)", edgecolor="none", linewidth=0)
    ax_b.hist(patched, bins=bins, density=True, color="#d96b55", alpha=0.50,
              label="After patching", edgecolor="none", linewidth=0)
    mean_base = float(np.nanmean(base))
    mean_patched = float(np.nanmean(patched))
    ax_b.axvline(mean_base, color="#4b5f86", lw=1.1)
    ax_b.axvline(mean_patched, color="#c95f47", lw=1.1)
    ymax = float(ax_b.get_ylim()[1])
    x_text = lo + 0.56 * (hi - lo)
    annot_arrow(ax_b, text="mean shift", xy=(mean_patched, ymax * 0.73), xytext=(x_text, ymax * 0.88), color=NEUTRAL)
    ax_b.set_xlabel("Final delta (logits)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("(b) Activation Patching", fontsize=FONT_TITLE,
                   fontweight="semibold", color="#1a202c")
    add_panel_label(ax_b, "b")
    ax_b.legend(frameon=False, fontsize=FONT_LEGEND - 0.1, loc="lower left")

    ax_c.axis("off")
    ax_ct = ax_c.inset_axes([0.10, 0.51, 0.88, 0.47])
    ax_cb = ax_c.inset_axes([0.10, 0.04, 0.88, 0.40])
    style_card(ax_ct)
    style_card(ax_cb)
    add_panel_label(ax_ct, "c")

    comps = ["attention", "mlp"]
    x = np.arange(2, dtype=np.float64)
    means, ylo, yhi = [], [], []
    fracs = []
    for comp in comps:
        vals = pat[pat["component"] == comp]["delta_shift"].to_numpy(dtype=np.float64)
        m, lo_ci, hi_ci = bootstrap_ci(vals)
        means.append(m)
        ylo.append(m - lo_ci)
        yhi.append(hi_ci - m)
        fracs.append(float(np.mean(vals > 0)))

    means_arr = np.asarray(means, dtype=np.float64)
    yerr = np.vstack([np.asarray(ylo), np.asarray(yhi)])
    ax_ct.errorbar(x, means_arr, yerr=yerr, fmt="o", capsize=3.0, lw=1.4, color=NEUTRAL)
    for i, comp in enumerate(comps):
        ax_ct.plot(i, means_arr[i], "o", color=COMP[comp], markersize=5.0)
    ax_ct.axhline(0, color=NEUTRAL, lw=LW_REF, ls="--")
    ax_ct.set_xticks([])
    ax_ct.set_ylabel(r"Mean $\Delta\delta$ (logits)")
    ax_ct.set_title("(c) Substitution", fontsize=FONT_TITLE,
                    fontweight="semibold", color="#1a202c")
    top_labels = ["Attention", "MLP"]
    y_min = float(np.min(means_arr - yerr[0]))
    y_max = float(np.max(means_arr + yerr[1]))
    y_span = max(y_max - y_min, 1.0)
    ax_ct.set_ylim(y_min - 0.20 * y_span, y_max + 0.22 * y_span)
    for i, m in enumerate(means_arr):
        near_top = m > (y_min + 0.78 * y_span)
        dy_pts = -8 if near_top else 8
        va = "top" if near_top else "bottom"
        ax_ct.annotate(
            f"{top_labels[i]} {m:+.2f}",
            xy=(i, m),
            xytext=(0, dy_pts),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=FONT_ANNOT - 0.05,
        )

    ax_cb.bar(x, np.asarray(fracs), color=[COMP[c] for c in comps], alpha=0.86, width=0.55,
              edgecolor="none", linewidth=0)
    for i, frac in enumerate(fracs):
        ax_cb.text(i, frac + 0.03, f"{frac:.0%}", ha="center", va="bottom", fontsize=FONT_ANNOT)
    ax_cb.set_xticks(x)
    ax_cb.set_xticklabels(["Attention", "MLP"])
    ax_cb.set_ylim(0.0, 1.0)
    ax_cb.set_ylabel("Frac. > 0")

    labels = ["evidence", "distractor", "neutral"]
    xpos = np.arange(len(labels), dtype=np.float64)
    bar_means, bar_lo, bar_hi = [], [], []
    for label in labels:
        vals = sl[sl["span_label"] == label]["effect_delta"].to_numpy(dtype=np.float64)
        m, lo_ci, hi_ci = bootstrap_ci(vals)
        bar_means.append(m)
        bar_lo.append(m - lo_ci)
        bar_hi.append(hi_ci - m)

    bar = ax_d.bar(
        xpos,
        np.asarray(bar_means),
        yerr=np.vstack([np.asarray(bar_lo), np.asarray(bar_hi)]),
        capsize=3.0,
        color=[SPAN[l] for l in labels],
        alpha=0.82,
    )
    ax_d.axhline(0, color=NEUTRAL, lw=LW_REF, ls="--")
    ctrl_styles = {"observed": "-", "shuffled": "--", "sign_flipped": ":"}
    ctrl_label = {
        "observed": "Observed baseline",
        "shuffled": "Shuffled control",
        "sign_flipped": "Sign-flipped control",
    }
    for _, row in nc.iterrows():
        c_name = str(row["control"])
        val = float(row["mean_effect_delta"])
        ax_d.axhline(val, color=NEUTRAL, lw=0.95, ls=ctrl_styles.get(c_name, "-"), alpha=0.9, label=ctrl_label.get(c_name, c_name))

    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels(["Evidence", "Distractor", "Neutral"])
    ax_d.set_ylabel("Mean deletion effect (logits)")
    ax_d.set_title("(d) Span Deletion", fontsize=FONT_TITLE,
                   fontweight="semibold", color="#1a202c")
    add_panel_label(ax_d, "d")

    handles, labels_l = ax_d.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for hh, ll in zip(handles, labels_l):
        dedup[ll] = hh
    ax_d.legend(dedup.values(), dedup.keys(), fontsize=FONT_LEGEND - 0.3, frameon=False, loc="upper right")

    out = output_dir / "fig5_counterfactuals.pdf"
    return _save_figure(fig, out, fig_name="fig5_counterfactuals")


# ---------------------------------------------------------------------------
# Figure 6: Prompt journey flagship
# ---------------------------------------------------------------------------

def _choose_prompt_for_journey(dm: pd.DataFrame, ts: pd.DataFrame, span_labels_df: pd.DataFrame) -> tuple[str, str]:
    stats = []
    sc_prompts = ts[ts["trajectory_type"] == "stable_correct"][["model_id", "prompt_uid"]].drop_duplicates()
    for _, row in sc_prompts.iterrows():
        mid = str(row["model_id"])
        uid = str(row["prompt_uid"])
        pdm = dm[(dm["model_id"] == mid) & (dm["prompt_uid"] == uid)].sort_values("layer_index")
        pts = ts[(ts["model_id"] == mid) & (ts["prompt_uid"] == uid)].sort_values("layer_index")
        sl = span_labels_df[(span_labels_df["model_id"] == mid) & (span_labels_df["prompt_uid"] == uid)]
        if pdm.empty or pts.empty or sl.empty:
            continue

        comp = pdm["competitor"].astype(str).to_numpy()
        switches = int(np.sum(comp[1:] != comp[:-1])) if comp.size > 1 else 0
        delta_arr = pdm["delta"].to_numpy(dtype=np.float64)
        delta_signs = np.sign(delta_arr)
        delta_sign_flips = int(np.sum(np.diff(delta_signs) != 0))
        final_delta = float(pdm["delta"].iloc[-1])
        delta_span = float(np.nanmax(pdm["delta"]) - np.nanmin(pdm["delta"]))
        cum_attn = pts["s_attn"].cumsum().to_numpy(dtype=np.float64)
        cum_mlp = pts["s_mlp"].cumsum().to_numpy(dtype=np.float64)
        step = np.abs(np.diff(cum_attn)) + np.abs(np.diff(cum_mlp))
        max_step = float(np.nanmax(step)) if step.size > 0 else 0.0
        has_ev = bool((sl["span_label"] == "evidence").any())
        has_dis = bool((sl["span_label"] == "distractor").any())

        stats.append(
            {
                "model_id": mid,
                "prompt_uid": uid,
                "switches": switches,
                "delta_sign_flips": delta_sign_flips,
                "final_delta": final_delta,
                "delta_span": delta_span,
                "max_step": max_step,
                "has_ev": has_ev,
                "has_dis": has_dis,
            }
        )

    if not stats:
        first = sc_prompts.iloc[0]
        return str(first["model_id"]), str(first["prompt_uid"])

    df = pd.DataFrame.from_records(stats)
    cand = df[
        (df["delta_sign_flips"] >= 2)
        & (df["final_delta"] > 0.5)
        & (df["has_ev"])
    ].copy()
    if cand.empty:
        cand = df[(df["delta_sign_flips"] >= 1) & (df["final_delta"] > 0.0)].copy()
    if cand.empty:
        cand = df[df["delta_sign_flips"] >= 1].copy()
    if cand.empty:
        cand = df.copy()

    step_cap = float(cand["max_step"].quantile(0.75))
    span_cap = float(cand["delta_span"].quantile(0.85))
    trimmed = cand[(cand["max_step"] <= step_cap) & (cand["delta_span"] <= span_cap)].copy()
    if trimmed.empty:
        trimmed = cand

    chosen = trimmed.sort_values(
        ["delta_sign_flips", "switches", "final_delta"],
        ascending=[False, False, False],
    ).iloc[0]
    return str(chosen["model_id"]), str(chosen["prompt_uid"])


def fig6_prompt_journey(
    parquet_dir: Path,
    output_dir: Path,
    prompts_path: Optional[Path] = None,
    spans_path: Optional[Path] = None,
) -> Path:
    dm = load_decision_metrics(parquet_dir)
    ts = load_tracing_scalars(parquet_dir)
    span_labels_df = pd.read_parquet(parquet_dir / "span_labels.parquet")

    mid, uid = _choose_prompt_for_journey(dm, ts, span_labels_df)

    pdm = dm[(dm["model_id"] == mid) & (dm["prompt_uid"] == uid)].sort_values("layer_index").copy()
    pts = ts[(ts["model_id"] == mid) & (ts["prompt_uid"] == uid)].sort_values("layer_index").copy()
    sl = span_labels_df[(span_labels_df["model_id"] == mid) & (span_labels_df["prompt_uid"] == uid)].copy()
    if pdm.empty or pts.empty:
        raise ValueError("fig6_prompt_journey: selected prompt has no rows")
    delta_sign_flips = int(np.sum(np.diff(np.sign(pdm["delta"].to_numpy(dtype=np.float64))) != 0))
    print(
        f"  [fig6] selected model_id={mid}, prompt_uid={uid}, delta_sign_flips={delta_sign_flips}"
    )

    L = MODEL_LAYERS.get(mid, int(pdm["layer_index"].max()) + 1)
    prompt_lookup = _load_prompt_lookup(prompts_path)
    prompt_text = str(prompt_lookup.get(uid, "Prompt text unavailable in manifest lookup."))
    parsed_spans = parse_prompt_spans(prompt_text)

    span_meta = {}
    for _, row in sl.drop_duplicates(subset=["span_role"], keep="last").iterrows():
        span_meta[str(row["span_role"])] = {
            "span_label": str(row.get("span_label") or "neutral"),
            "effect_delta": float(row.get("effect_delta") or 0.0),
        }

    ck = str(pdm["correct_key"].iloc[-1]) if "correct_key" in pdm.columns else "A"
    kp = str(pdm["competitor"].iloc[-1])

    def rel_role(role: str) -> str:
        if role.startswith("option_"):
            letter = role.replace("option_", "").upper()
            if letter == ck:
                return "option_correct"
            if letter == kp:
                return "option_competitor"
            return "option_other"
        return role

    role_fill = {
        "instruction": "#edf2f7",
        "question_stem": "#e2edf7",
        "option_correct": "#c5e8d9",
        "option_competitor": "#f6d5d5",
        "option_other": "#eef1f5",
        "post_options": "#fff4dc",
    }
    role_label = {
        "instruction": "Instruction",
        "question_stem": "Question",
        "option_correct": "Correct option",
        "option_competitor": "Competitor option",
        "option_other": "Other option",
        "post_options": "Post-options",
    }

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.85), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.994, bottom=0.10, top=0.88, wspace=0.27, hspace=0.38)
    fig.suptitle("A Prompt\u2019s Journey Through Decision Space", fontsize=FONT_TITLE + 1.0,
                 y=0.975, fontweight="bold", color="#1a202c")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.72], height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.36)
    top_right = gs[0, 1].subgridspec(2, 1, height_ratios=[0.82, 0.18], hspace=0.02)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(top_right[0, 0])
    ax_b_strip = fig.add_subplot(top_right[1, 0], sharex=ax_b)
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) structured prompt spans
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axis("off")
    ax_a.set_title("Structured Prompt", fontsize=FONT_TITLE + 0.1, fontweight="semibold",
                   color="#1a202c", pad=5)
    add_panel_label(ax_a, "a")

    rows = []
    for span in parsed_spans:
        raw_role = str(span.label)
        rr = rel_role(raw_role)
        meta = span_meta.get(raw_role, {"span_label": "neutral", "effect_delta": 0.0})
        span_label = str(meta["span_label"])
        eff = float(meta["effect_delta"])
        outline = SPAN["evidence"] if span_label == "evidence" else (SPAN["distractor"] if span_label == "distractor" else "#9aa5b5")
        text_line = textwrap.shorten(" ".join(str(span.text).split()), width=56, placeholder=" ...")
        rows.append(
            {
                "role": rr,
                "role_name": role_label.get(rr, rr),
                "line": text_line,
                "span_label": span_label,
                "effect": eff,
                "outline": outline,
            }
        )
    rows = rows[:5]

    if rows:
        n = len(rows)
        h = 0.92 / float(n)
        top = 0.96
        evidence_idx = []
        distractor_idx = []
        for i, row in enumerate(rows):
            y1 = top - i * h
            y0 = y1 - 0.86 * h
            patch = FancyBboxPatch(
                (0.02, y0),
                0.95,
                0.84 * h,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                linewidth=1.00,
                edgecolor=row["outline"],
                facecolor=role_fill.get(row["role"], "#f8fafc"),
            )
            ax_a.add_patch(patch)
            sign = "+" if row["effect"] > 0 else ""
            tag = "E" if row["span_label"] == "evidence" else ("D" if row["span_label"] == "distractor" else "N")
            head = f"{row['role_name']} [{tag}, {sign}{row['effect']:.2f}]"
            body = textwrap.shorten(row["line"], width=36, placeholder=" ...")
            ax_a.text(0.045, y0 + 0.54 * h, head, fontsize=FONT_AXIS + 0.1, fontweight="semibold", va="center", ha="left")
            ax_a.text(0.045, y0 + 0.19 * h, body, fontsize=FONT_ANNOT + 0.4, color="#344054", va="center", ha="left")

            if row["span_label"] == "evidence":
                evidence_idx.append(i)
            if row["span_label"] == "distractor":
                distractor_idx.append(i)

        ax_a.text(
            0.03,
            0.01,
            "E=evidence, D=distractor, N=neutral",
            transform=ax_a.transAxes,
            fontsize=FONT_ANNOT + 0.15,
            color=NEUTRAL,
            ha="left",
            va="bottom",
        )
    else:
        ax_a.text(0.5, 0.5, "Prompt spans unavailable", ha="center", va="center", fontsize=FONT_AXIS)

    # (b) center: decision margin δ(l) vs real layer index
    style_card(ax_b)
    add_panel_label(ax_b, "b")
    ax_b.set_title("Decision Margin by Layer", fontsize=FONT_TITLE + 0.2,
                   fontweight="semibold", color="#1a202c", pad=2)

    layer_idx_b = pdm["layer_index"].to_numpy(dtype=np.float64)
    delta = pdm["delta"].to_numpy(dtype=np.float64)
    competitors = pdm["competitor"].astype(str).to_numpy()
    tail_layer = L - 8

    # LineCollection colored by sign of delta
    if len(layer_idx_b) >= 2:
        pts_lc = np.array([layer_idx_b, delta]).T.reshape(-1, 1, 2)
        segments_lc = np.concatenate([pts_lc[:-1], pts_lc[1:]], axis=1)
        seg_delta_mid = (delta[:-1] + delta[1:]) / 2.0
        seg_colors_lc = [
            TRAJ["stable_correct"] if v >= 0 else TRAJ["stable_wrong"]
            for v in seg_delta_mid
        ]
        lc = LineCollection(segments_lc, colors=seg_colors_lc, lw=2.2, zorder=4)
        ax_b.add_collection(lc)
    dot_colors = [
        TRAJ["stable_correct"] if v >= 0 else TRAJ["stable_wrong"]
        for v in delta
    ]
    ax_b.scatter(
        layer_idx_b, delta, s=9, c=dot_colors, zorder=5,
        marker="o",
    )
    ax_b.fill_between(layer_idx_b, 0, delta, where=delta >= 0, color=FILL["correct_bg"], alpha=0.75, interpolate=True)
    ax_b.fill_between(layer_idx_b, 0, delta, where=delta < 0, color=FILL["wrong_bg"], alpha=0.70, interpolate=True)
    shade_threshold_region(ax_b, x_from=float(tail_layer), x_to=float(L - 1), color=FILL["tail_bg"], alpha=0.45)
    tail_x_frac = (float(tail_layer) - (float(layer_idx_b[0]) - 0.5)) / (float(layer_idx_b[-1]) - float(layer_idx_b[0]) + 1.0)
    ax_b.text(
        min(0.99, tail_x_frac + 0.01),
        0.97,
        "tail (τ=8)",
        transform=ax_b.transAxes,
        fontsize=FONT_ANNOT,
        color="#8a5517",
        va="top",
        ha="left",
    )

    # Mark sign flips with small downward triangles
    signs = np.sign(delta)
    flip_mask = np.where(np.diff(signs) != 0)[0] + 1
    for fi in flip_mask:
        if 0 <= fi < len(layer_idx_b):
            ax_b.plot(layer_idx_b[fi], delta[fi], "v", color=NEUTRAL, ms=4.5, zorder=5)

    add_zero_line(ax_b)
    ax_b.set_xlim(float(layer_idx_b[0]) - 0.5, float(layer_idx_b[-1]) + 0.5)
    ax_b.set_ylabel(r"Decision margin $\delta$ (logits)")
    ax_b.tick_params(axis="x", labelbottom=False)

    # Competitor-identity strip in dedicated axis
    cmap_comp = {"A": "#60a5fa", "B": "#f87171", "C": "#34d399", "D": "#fbbf24"}
    ax_b_strip.set_xlim(float(layer_idx_b[0]) - 0.5, float(layer_idx_b[-1]) + 0.5)
    ax_b_strip.set_ylim(0.0, 1.0)
    for i in range(len(layer_idx_b) - 1):
        ax_b_strip.axvspan(
            float(layer_idx_b[i]),
            float(layer_idx_b[i + 1]),
            color=cmap_comp.get(competitors[i], "#9aa5b1"),
            ymin=0.0,
            ymax=1.0,
        )
    ax_b_strip.text(
        0.01,
        0.85,
        "competitor k*(l)",
        fontsize=FONT_ANNOT,
        color=NEUTRAL,
        ha="left",
        va="top",
        transform=ax_b_strip.transAxes,
    )
    ax_b_strip.set_xlabel("Layer")
    ax_b_strip.set_yticks([])
    for spine in ax_b_strip.spines.values():
        spine.set_visible(False)
    ax_b_strip.tick_params(axis="x", labelsize=FONT_TICK)

    # (c) right: component forces — attention and MLP scalars per layer
    style_card(ax_c)
    add_panel_label(ax_c, "c")
    ax_c.set_title("Component Forces", fontsize=FONT_TITLE + 0.2, fontweight="semibold",
                   color="#1a202c", pad=3)

    layer_idx_c = pts["layer_index"].to_numpy(dtype=np.float64)
    s_attn = pts["s_attn"].to_numpy(dtype=np.float64)
    s_mlp = pts["s_mlp"].to_numpy(dtype=np.float64)
    if uniform_filter1d is not None:
        s_attn = uniform_filter1d(s_attn, size=3, mode="nearest")
        s_mlp = uniform_filter1d(s_mlp, size=3, mode="nearest")

    ax_c.plot(layer_idx_c, s_attn, color=COMP["attention"], lw=1.6, label="Attention", zorder=3)
    ax_c.plot(layer_idx_c, s_mlp, color=COMP["mlp"], lw=1.6, label="MLP", zorder=3)
    net_c = s_attn + s_mlp
    ax_c.plot(layer_idx_c, net_c, color="#2d3748", lw=2.2, ls="-", label="Net", zorder=5, alpha=0.90)
    ax_c.fill_between(layer_idx_c, 0, s_attn, where=s_attn >= 0, color=COMP["attention"], alpha=0.26, interpolate=True)
    ax_c.fill_between(layer_idx_c, 0, s_attn, where=s_attn < 0, color=COMP["attention"], alpha=0.12, interpolate=True)
    ax_c.fill_between(layer_idx_c, 0, s_mlp, where=s_mlp >= 0, color=COMP["mlp"], alpha=0.26, interpolate=True)
    ax_c.fill_between(layer_idx_c, 0, s_mlp, where=s_mlp < 0, color=COMP["mlp"], alpha=0.12, interpolate=True)
    shade_threshold_region(ax_c, x_from=float(tail_layer), x_to=float(L - 1), color=FILL["tail_bg"], alpha=0.40)
    tail_x_frac_c = (float(tail_layer) - (float(layer_idx_c[0]) - 0.5)) / (float(layer_idx_c[-1]) - float(layer_idx_c[0]) + 1.0)
    ax_c.text(
        min(0.99, tail_x_frac_c + 0.01),
        0.97,
        "tail",
        transform=ax_c.transAxes,
        fontsize=FONT_ANNOT,
        color="#8a5517",
        va="top",
        ha="left",
    )
    add_zero_line(ax_c)
    ax_c.set_xlim(float(layer_idx_c[0]) - 0.5, float(layer_idx_c[-1]) + 0.5)
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel(r"Component scalar (logits)")
    yy = float(np.nanpercentile(np.abs(np.concatenate([s_attn, s_mlp, net_c])), 99))
    yy = min(max(yy * 1.2, 1.5), 5.5)
    ax_c.set_ylim(-yy, yy)
    ax_c.legend(frameon=False, fontsize=FONT_LEGEND - 0.2, loc="upper left")

    out = output_dir / "fig6_prompt_journey.pdf"
    return _save_figure(fig, out, fig_name="fig6_prompt_journey")


# ---------------------------------------------------------------------------
# Appendix Figure A1: Proxy validation — delta_bucket vs delta_single
# ---------------------------------------------------------------------------

def fig_appendix_proxy_validation(parquet_dir: Path, output_dir: Path) -> Path:
    """Hexbin scatter of delta_bucket vs delta_single for tracing subset."""
    dm = pd.read_parquet(parquet_dir / "decision_metrics.parquet")
    ts = pd.read_parquet(parquet_dir / "tracing_scalars.parquet")

    merged = ts[["model_id", "prompt_uid", "layer_index", "delta"]].merge(
        dm[["model_id", "prompt_uid", "layer_index", "delta"]],
        on=["model_id", "prompt_uid", "layer_index"],
        suffixes=("_single", "_bucket"),
        how="inner",
    )

    x = merged["delta_bucket"].to_numpy(dtype=np.float64)
    y = merged["delta_single"].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    r = float(np.corrcoef(x, y)[0, 1])
    sign_agree = float(np.mean(np.sign(x) == np.sign(y)))

    from sow.v2.figures.style import COL_WIDTH

    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.06, COL_WIDTH * 1.00), constrained_layout=True)
    style_card(ax)

    qx = float(np.nanpercentile(np.abs(x), 97.0))
    qy = float(np.nanpercentile(np.abs(y), 97.0))
    x_lim = min(max(qx * 1.10, 1.2), 6.0)
    y_lim = min(max(qy * 1.10, 2.0), 10.0)
    x_clip = np.clip(x, -x_lim, x_lim)
    y_clip = np.clip(y, -y_lim, y_lim)

    hb = ax.hexbin(
        x_clip,
        y_clip,
        gridsize=66,
        cmap="Blues",
        mincnt=1,
        linewidths=0.2,
        extent=[-x_lim, x_lim, -y_lim, y_lim],
    )
    fig.colorbar(hb, ax=ax, label="Count", shrink=0.82, pad=0.02)

    # y=x reference line
    xx = np.linspace(-x_lim, x_lim, 200)
    ax.plot(xx, xx, color=NEUTRAL, lw=1.4, ls="--", label="y = x")
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    add_zero_line(ax)
    ax.axvline(0, color=NEUTRAL, lw=0.6, ls=":")

    ax.set_xlabel(r"$\delta_{\mathrm{bucket}}(l)$ (logits)")
    ax.set_ylabel(r"$\delta_{\mathrm{single}}(l)$ (logits)")
    ax.set_title("Tracing Readout vs Bucket Readout", fontsize=FONT_TITLE)

    annot_text = f"$r = {r:.2f}$\nsign agree = {sign_agree:.1%}"
    ax.text(
        0.04, 0.96, annot_text,
        transform=ax.transAxes,
        fontsize=FONT_ANNOT + 0.5,
        va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85, "ec": "#b0b8c4"},
    )
    ax.text(
        0.98,
        0.02,
        "axes clipped at 99.5th pct",
        transform=ax.transAxes,
        fontsize=FONT_ANNOT - 0.1,
        color=NEUTRAL,
        ha="right",
        va="bottom",
    )

    out = output_dir / "figA1_proxy_validation.pdf"
    return _save_figure(fig, out, fig_name="figA1_proxy_validation")


# ---------------------------------------------------------------------------
# Figure 7: Decision Landscape
# ---------------------------------------------------------------------------

def fig7_decision_landscape(parquet_dir: Path, output_dir: Path) -> Path:
    dm = load_decision_metrics(parquet_dir)

    # --- Panel A: column-max-normalized density ---
    N_DEPTH_BINS_A = 32
    N_DELTA_BINS_A = 80
    DELTA_CLIP_A = 8.0

    depth_a = dm["depth"].to_numpy(dtype=np.float64)
    delta_a = dm["delta"].clip(-DELTA_CLIP_A, DELTA_CLIP_A).to_numpy(dtype=np.float64)

    density_raw, x_edges, y_edges = np.histogram2d(
        depth_a,
        delta_a,
        bins=[N_DEPTH_BINS_A, N_DELTA_BINS_A],
        range=[[0.0, 1.0], [-DELTA_CLIP_A, DELTA_CLIP_A]],
    )
    col_max = density_raw.max(axis=1, keepdims=True)
    col_max[col_max == 0] = 1.0
    density_norm = density_raw / col_max
    # Aggressively mask sparse bins: any bin with < 3 raw counts → invisible
    density_norm_masked = np.ma.masked_where(density_raw < 3, density_norm)

    # --- Panel B: count-masked drift field ---
    N_DEPTH_BINS_B = 20
    N_DELTA_BINS_B = 28
    DELTA_CLIP_B = 8.0

    depth_bins_b = np.linspace(0, 1, N_DEPTH_BINS_B + 1)
    delta_bins_b = np.linspace(-DELTA_CLIP_B, DELTA_CLIP_B, N_DELTA_BINS_B + 1)

    dm2 = dm.copy()
    dm2["delta_clipped_b"] = dm2["delta"].clip(-DELTA_CLIP_B, DELTA_CLIP_B)
    dm2["depth_bin_b"] = pd.cut(dm2["depth"], bins=depth_bins_b, labels=False)
    dm2["delta_bin_b"] = pd.cut(dm2["delta_clipped_b"], bins=delta_bins_b, labels=False)

    drift_agg_b = (
        dm2.groupby(["depth_bin_b", "delta_bin_b"])["drift"]
        .agg(["mean", "count"])
        .reset_index()
    )

    drift_grid = np.full((N_DEPTH_BINS_B, N_DELTA_BINS_B), np.nan)
    count_grid = np.zeros((N_DEPTH_BINS_B, N_DELTA_BINS_B), dtype=int)
    for _, row in drift_agg_b.iterrows():
        i = row["depth_bin_b"]
        j = row["delta_bin_b"]
        if pd.notna(i) and pd.notna(j):
            ii, jj = int(i), int(j)
            if 0 <= ii < N_DEPTH_BINS_B and 0 <= jj < N_DELTA_BINS_B:
                drift_grid[ii, jj] = float(row["mean"])
                count_grid[ii, jj] = int(row["count"])
    drift_grid[count_grid < 5] = np.nan

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.95), constrained_layout=False)
    fig.subplots_adjust(left=0.088, right=0.965, bottom=0.11, top=0.87, wspace=0.30)
    fig.suptitle("The Decision Landscape", fontsize=FONT_TITLE + 1.2, y=0.985,
                 fontweight="bold", color="#1a202c")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.05], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    for ax in [ax_a, ax_b]:
        style_card(ax)
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(r"Decision margin $\delta$ (logits)")
        add_zero_line(ax)
        ax.set_xlim(0.0, 1.0)
    ax_a.set_ylim(-DELTA_CLIP_A, DELTA_CLIP_A)
    ax_b.set_ylim(-DELTA_CLIP_B, DELTA_CLIP_B)

    # Custom colormap: white → warm amber → hot — no dark start, empty=white
    _cmap_pop = mcolors.LinearSegmentedColormap.from_list(
        "pop_heat",
        ["#ffffff", "#fff3e0", "#ffcc80", "#ff9800", "#e65100", "#b71c1c"],
        N=256,
    )
    _cmap_pop.set_bad("white")
    im_a = ax_a.pcolormesh(
        x_edges,
        y_edges,
        density_norm_masked.T,
        cmap=_cmap_pop,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    # Clean zero line
    ax_a.axhline(0, color="white", lw=0.6, ls="-", alpha=0.45, zorder=3)
    cb_a = plt.colorbar(im_a, ax=ax_a, label="Relative density (col-normalised)", shrink=0.72, pad=0.02)
    cb_a.ax.tick_params(labelsize=FONT_TICK - 0.5)
    ax_a.set_title("Population Flow", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c")
    add_panel_label(ax_a, "a")

    drift_masked = np.ma.masked_invalid(drift_grid)
    valid_vals = drift_grid[~np.isnan(drift_grid)]
    abs_max = float(np.nanpercentile(np.abs(valid_vals), 95)) if valid_vals.size > 0 else 1.0
    abs_max = min(max(abs_max, 0.2), 3.0)
    drift_norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    # Custom diverging colormap: deep blue → white → deep orange
    _cmap_drift = mcolors.LinearSegmentedColormap.from_list(
        "drift_div",
        ["#08306b", "#3a7ebf", "#a6c8e5", "white", "#fdc38d", "#d4662a", "#7f2704"],
        N=256,
    )
    im_b = ax_b.pcolormesh(
        depth_bins_b,
        delta_bins_b,
        drift_masked.T,
        cmap=_cmap_drift,
        norm=drift_norm,
        rasterized=True,
    )

    depth_mids_b = (depth_bins_b[:-1] + depth_bins_b[1:]) / 2.0
    delta_mids_b = (delta_bins_b[:-1] + delta_bins_b[1:]) / 2.0

    # Clean zero line only — no contour overlays
    ax_b.axhline(0, color=NEUTRAL, lw=0.6, ls="--", alpha=0.5, zorder=2)

    cb_b = plt.colorbar(im_b, ax=ax_b, label="Mean per-layer drift (logits)", shrink=0.72, pad=0.02)
    cb_b.ax.tick_params(labelsize=FONT_TICK - 0.5)
    ax_b.set_title("Mean Drift Field", fontsize=FONT_TITLE, fontweight="semibold",
                   color="#1a202c")
    add_panel_label(ax_b, "b")

    out = output_dir / "fig7_decision_landscape.pdf"
    return _save_figure(fig, out, fig_name="fig7_decision_landscape")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def generate_all_figures(
    parquet_dir: Path,
    output_dir: Path,
    prompts_path: Optional[Path] = None,
    spans_path: Optional[Path] = None,
) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    print("Generating Figure 1: Three Primitives...")
    paths.append(fig1_primitives(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 2: Phase Diagram...")
    paths.append(fig2_phase_diagram(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 3: Drift Decomposition...")
    paths.append(fig3_decomposition(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 4: Attention Routing...")
    paths.append(fig4_attention_routing(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 5: Counterfactual Validation...")
    paths.append(fig5_counterfactuals(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 6: Prompt Journey...")
    paths.append(fig6_prompt_journey(parquet_dir, output_dir, prompts_path, spans_path))
    print(f"  -> {paths[-1]}")

    print("Generating Figure 7: Decision Landscape...")
    paths.append(fig7_decision_landscape(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print("Generating Appendix Figure A1: Proxy Validation...")
    paths.append(fig_appendix_proxy_validation(parquet_dir, output_dir))
    print(f"  -> {paths[-1]}")

    print(f"\nAll {len(paths)} figures generated successfully.")
    return paths


def render_preview_pngs(*, figure_paths: list[Path], out_dir: Optional[Path] = None) -> list[Path]:
    """Render figure previews to PNG using available system tool."""
    dest = out_dir or _new_previews_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdftoppm"):
        created = []
        for pdf in figure_paths:
            prefix = dest / pdf.name
            cmd = f"pdftoppm -png -f 1 -singlefile '{pdf}' '{prefix}'"
            rc = int(os.system(cmd))
            if rc == 0:
                png = Path(str(prefix) + ".png")
                if png.exists():
                    created.append(png)
        return created

    if shutil.which("qlmanage"):
        quoted = " ".join(f"'{p}'" for p in figure_paths)
        cmd = f"qlmanage -t -s 1800 -o '{dest}' {quoted} >/dev/null 2>&1"
        os.system(cmd)
        return sorted(dest.glob("*.png"))

    return []


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
