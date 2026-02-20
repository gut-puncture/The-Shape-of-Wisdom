#!/usr/bin/env python3
"""
Data-faithful convergence visuals (no forced vortex geometry).

Core idea:
- x = normalized layer depth (left -> right)
- y = signed coordinate in a single linear attractor-centered axis
- each line = one prompt trajectory in the majority final-answer basin

Outputs:
- faithful_convergence_funnels.png
- faithful_convergence_funnels.gif
- beautiful_convergence_density_fields.png
- faithful_convergence_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
}


def _model_label(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def _normalize_monotonic_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be 1D with at least 2 points")
    x = np.maximum.accumulate(x)
    lo = float(x[0])
    hi = float(x[-1])
    if (not math.isfinite(lo)) or (not math.isfinite(hi)) or (hi - lo <= 1e-12):
        return np.linspace(0.0, 1.0, num=x.size, dtype=np.float64)
    x = (x - lo) / (hi - lo)
    x[0] = 0.0
    x[-1] = 1.0
    return x


def _build_display_x(
    xg: np.ndarray,
    *,
    mode: str,
    late_log_alpha: float,
    late_power_gamma: float,
) -> np.ndarray:
    x = np.asarray(xg, dtype=np.float64)
    if mode == "linear":
        xw = x
    elif mode == "late_log":
        alpha = max(1e-6, float(late_log_alpha))
        # Compresses early layers and stretches late layers while preserving order.
        xw = 1.0 - np.log1p(alpha * (1.0 - x)) / np.log1p(alpha)
    elif mode == "late_power":
        gamma = max(1.0001, float(late_power_gamma))
        # gamma>1 compresses early layers and spreads the late segment near x=1.
        xw = np.power(x, gamma)
    elif mode == "change_cdf":
        # Built later from trajectory change statistics.
        xw = x
    else:
        raise ValueError(f"unknown x-warp mode: {mode}")
    return _normalize_monotonic_unit(xw)


def _x_edges_from_centers(xc: np.ndarray) -> np.ndarray:
    xc = np.asarray(xc, dtype=np.float64)
    if xc.ndim != 1 or xc.size < 2:
        raise ValueError("xc must be 1D with at least 2 points")
    xe = np.empty((xc.size + 1,), dtype=np.float64)
    xe[1:-1] = 0.5 * (xc[:-1] + xc[1:])
    xe[0] = xc[0] - (xe[1] - xc[0])
    xe[-1] = xc[-1] + (xc[-1] - xe[-2])
    xe = np.clip(xe, 0.0, 1.0)
    xe[0] = 0.0
    xe[-1] = 1.0
    xe = np.maximum.accumulate(xe)
    return xe


def _smooth_display_curves(yi: np.ndarray, sigma: float) -> np.ndarray:
    s = float(sigma)
    if (not math.isfinite(s)) or s <= 0.0:
        return yi
    radius = max(1, int(math.ceil(3.0 * s)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / s) * (t / s))
    k /= float(k.sum())
    ypad = np.pad(yi, ((0, 0), (radius, radius)), mode="edge")
    ys = np.apply_along_axis(lambda z: np.convolve(z, k, mode="valid"), axis=1, arr=ypad)
    return ys.astype(np.float64, copy=False)


def _final_choice_idx(layer: Dict[str, Any]) -> int:
    tc = layer.get("top_candidate")
    if tc in ("A", "B", "C", "D"):
        return {"A": 0, "B": 1, "C": 2, "D": 3}[tc]
    cp = layer.get("candidate_probs") or {}
    vals = np.array([float(cp.get(c, 0.0)) for c in ("A", "B", "C", "D")], dtype=np.float64)
    return int(np.argmax(vals))


def _load_model(path: Path) -> Dict[str, Any]:
    hidden_rows: List[np.ndarray] = []
    final_choice: List[int] = []
    model_id: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lw = row.get("layerwise") or []
            if not isinstance(lw, list) or not lw:
                continue
            H = []
            valid = True
            for layer in lw:
                ph = layer.get("projected_hidden_128")
                if not isinstance(ph, list) or len(ph) == 0:
                    valid = False
                    break
                v = np.asarray(ph, dtype=np.float32)
                if v.ndim != 1 or not np.all(np.isfinite(v)):
                    valid = False
                    break
                H.append(v)
            if not valid:
                continue

            if model_id is None:
                model_id = str(row.get("model_id") or path.parent.name)

            hidden_rows.append(np.stack(H, axis=0).astype(np.float32))
            final_choice.append(_final_choice_idx(lw[-1]))

    if not hidden_rows:
        raise RuntimeError(f"no valid rows in {path}")

    hidden = np.stack(hidden_rows, axis=0)  # [N, L, D]
    final = np.asarray(final_choice, dtype=np.int16)
    return {
        "model_id": model_id or path.parent.name,
        "hidden": hidden,
        "final_choice": final,
    }


def _prepare_faithful_coords(
    d: Dict[str, Any],
    *,
    interp_steps: int,
    x_warp: str,
    late_log_alpha: float,
    late_power_gamma: float,
    x_left: float,
    x_right: float,
    population_mode: str,
    attractor_mode: str,
    display_smooth_sigma: float,
    focus_quantile: float,
    change_cdf_source: str,
) -> Dict[str, Any]:
    hidden = d["hidden"]  # [N,L,D]
    final = d["final_choice"]  # [N]
    n_rows, n_layers, _ = hidden.shape

    counts = np.bincount(final, minlength=4)
    target = int(np.argmax(counts))
    mask_majority = final == target
    Hb = hidden[mask_majority]  # [M,L,D]
    if Hb.shape[0] < 20:
        raise RuntimeError(f"not enough rows in majority basin for {d['model_id']}")

    if attractor_mode == "majority":
        H_attr = Hb
    elif attractor_mode == "global":
        H_attr = hidden
    else:
        raise ValueError(f"unknown attractor_mode: {attractor_mode}")

    if population_mode == "all":
        H_use = hidden
        final_use = final
    elif population_mode == "majority":
        H_use = Hb
        final_use = final[mask_majority]
    else:
        raise ValueError(f"unknown population_mode: {population_mode}")

    # Attractor = mean final hidden of this basin.
    attractor = H_attr[:, -1, :].mean(axis=0)  # [D]
    Z = H_use - attractor[None, None, :]  # [M,L,D]

    # Data-faithful linear map: first principal axis of basin trajectories.
    X = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0].astype(np.float64)

    y = np.einsum("nld,d->nl", Z, axis).astype(np.float64)  # [M,L]
    x = np.linspace(0.0, 1.0, num=n_layers, dtype=np.float64)
    xg = np.linspace(0.0, 1.0, num=(n_layers - 1) * int(interp_steps) + 1, dtype=np.float64)

    yi = np.empty((y.shape[0], xg.shape[0]), dtype=np.float64)
    for i in range(y.shape[0]):
        yi[i] = np.interp(xg, x, y[i])

    early_w = max(2, xg.shape[0] // 3)
    scale = float(np.percentile(np.abs(yi[:, :early_w]), 98.0))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.percentile(np.abs(yi), 98.0) + 1e-8)
    yi = yi / scale
    yi_metric = yi.copy()

    # Optional temporal-angle diagnostics (used only as annotation, not geometry transform).
    X2 = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
    _, _, vh2 = np.linalg.svd(X2, full_matrices=False)
    a1 = vh2[0].astype(np.float64)
    a2 = vh2[1].astype(np.float64) if vh2.shape[0] > 1 else vh2[0].astype(np.float64)
    p1 = np.einsum("nld,d->nl", Z, a1).astype(np.float64)
    p2 = np.einsum("nld,d->nl", Z, a2).astype(np.float64)
    theta = np.unwrap(np.arctan2(p2, p1), axis=1)
    mean_turn = float(np.mean(np.abs(theta[:, -1] - theta[:, 0])))

    # Beautiful-but-rigorous map:
    # full-space distance-to-attractor (128D norm) x fixed initial phase rank in [-1,1].
    # Phase rank only sets display ordering; radial magnitude is true high-D distance.
    r = np.linalg.norm(Z, axis=2).astype(np.float64)
    order = np.argsort(theta[:, 0])
    phase_rank = np.empty((theta.shape[0],), dtype=np.float64)
    phase_rank[order] = np.linspace(-1.0, 1.0, num=theta.shape[0], endpoint=True)
    y_rr = r * phase_rank[:, None]
    yi_rr = np.empty((y_rr.shape[0], xg.shape[0]), dtype=np.float64)
    for i in range(y_rr.shape[0]):
        yi_rr[i] = np.interp(xg, x, y_rr[i])
    scale_rr = float(np.percentile(np.abs(yi_rr[:, :early_w]), 98.0))
    if not math.isfinite(scale_rr) or scale_rr <= 1e-8:
        scale_rr = float(np.percentile(np.abs(yi_rr), 98.0) + 1e-8)
    yi_rr = yi_rr / scale_rr
    y_lim_rr = float(np.percentile(np.abs(yi_rr[:, :early_w]), 99.5))
    y_lim_rr = max(0.8, min(y_lim_rr, 2.4))
    yi_rr_plot = np.clip(yi_rr, -y_lim_rr * 1.20, y_lim_rr * 1.20)
    yi_rr_plot = _smooth_display_curves(yi_rr_plot, sigma=float(display_smooth_sigma))
    yi_rr_plot = np.clip(yi_rr_plot, -y_lim_rr * 1.20, y_lim_rr * 1.20).astype(np.float32)

    collapse = 1.0 - float(np.mean(np.abs(yi_metric[:, -1])) / (np.mean(np.abs(yi_metric[:, 0])) + 1e-12))
    collapse_rr = 1.0 - float(np.mean(np.abs(yi_rr[:, -1])) / (np.mean(np.abs(yi_rr[:, 0])) + 1e-12))
    left_p90 = float(np.percentile(np.abs(yi_metric[:, 0]), 90.0))
    right_p90 = float(np.percentile(np.abs(yi_metric[:, -1]), 90.0))

    # "focus" trajectories are those closest to attractor at final layer.
    q = float(np.clip(focus_quantile, 0.01, 0.99))
    final_abs = np.abs(yi_metric[:, -1])
    focus_thr = float(np.quantile(final_abs, q))
    focus_mask = final_abs <= focus_thr

    # Stable view bounds.
    y_lim = float(np.percentile(np.abs(yi_metric[:, :early_w]), 99.5))
    y_lim = max(0.6, min(y_lim, 2.2))
    yi_plot = np.clip(yi_metric, -y_lim * 1.20, y_lim * 1.20)
    yi_plot = _smooth_display_curves(yi_plot, sigma=float(display_smooth_sigma))
    yi_plot = np.clip(yi_plot, -y_lim * 1.20, y_lim * 1.20)
    if str(x_warp) == "change_cdf":
        if str(change_cdf_source) == "radial":
            base_for_change = yi_rr
        elif str(change_cdf_source) == "faithful":
            base_for_change = yi_metric
        else:
            raise ValueError(f"unknown change_cdf_source: {change_cdf_source}")
        med_change = np.median(np.abs(np.diff(base_for_change, axis=1)), axis=0)
        floor = max(1e-6, float(np.quantile(med_change, 0.10)) * 0.25)
        w = med_change + floor
        c = np.concatenate([[0.0], np.cumsum(w)])
        x_plot = _normalize_monotonic_unit(c)
    else:
        x_plot = _build_display_x(
            xg,
            mode=str(x_warp),
            late_log_alpha=float(late_log_alpha),
            late_power_gamma=float(late_power_gamma),
        )
    xl = float(np.clip(x_left, 0.0, 0.35))
    xr = float(np.clip(x_right, 0.65, 1.0))
    if xr <= xl + 0.08:
        xr = min(1.0, xl + 0.08)
    x_plot = xl + (xr - xl) * x_plot

    return {
        "model_id": d["model_id"],
        "target_choice": ["A", "B", "C", "D"][target],
        "n_basin": int(Hb.shape[0]),
        "n_used": int(H_use.shape[0]),
        "population_mode": str(population_mode),
        "attractor_mode": str(attractor_mode),
        "xg": xg,
        "x_plot": x_plot.astype(np.float32),
        "yi": yi_plot.astype(np.float32),
        "yi_metric": yi_metric.astype(np.float32),
        "collapse_fraction": float(collapse),
        "collapse_fraction_radial_rank": float(collapse_rr),
        "left_p90_abs": left_p90,
        "right_p90_abs": right_p90,
        "mean_turn_radians": mean_turn,
        "y_lim": y_lim,
        "n_total": int(n_rows),
        "yi_radial_rank": yi_rr_plot,
        "phase_rank": phase_rank.astype(np.float32),
        "y_lim_radial_rank": y_lim_rr,
        "final_choice_used": final_use.astype(np.int16),
        "focus_mask": focus_mask.astype(bool),
        "focus_quantile": q,
        "focus_threshold_final_abs": focus_thr,
        "change_cdf_source": str(change_cdf_source),
    }


def _plot_funnels_static(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_lines_visible: int,
) -> None:
    rng = np.random.default_rng(seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.0 * len(bundles), 4.8), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        xg = b["x_plot"]
        yi = b["yi"]  # [N,F]
        focus = b["focus_mask"]
        N = yi.shape[0]
        k = min(int(max_lines_visible), int(N))
        idx = rng.choice(np.arange(N), size=k, replace=False) if k < N else np.arange(N)
        ys = yi[idx]
        focus_s = focus[idx]

        q10 = np.quantile(yi, 0.10, axis=0)
        q90 = np.quantile(yi, 0.90, axis=0)
        q25 = np.quantile(yi, 0.25, axis=0)
        q75 = np.quantile(yi, 0.75, axis=0)
        med = np.median(yi, axis=0)
        ax.fill_between(xg, q10, q90, color="#2d3a4d", alpha=0.12, lw=0)
        ax.fill_between(xg, q25, q75, color="#2d3a4d", alpha=0.19, lw=0)

        for tr, is_focus in zip(ys, focus_s):
            if bool(is_focus):
                ax.plot(xg, tr, color="#0b1b34", alpha=0.05, lw=0.62)
            else:
                ax.plot(xg, tr, color="#6f7885", alpha=0.018, lw=0.50)

        ax.plot(xg, med, color="#0f1a2a", lw=1.2, alpha=0.9)
        ax.axhline(0.0, color="#3b3b3b", lw=0.8, alpha=0.35)
        ax.scatter([float(xg[-1])], [0.0], s=32, c="#111111", zorder=5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-b["y_lim"], b["y_lim"])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
        ax.text(
            0.01,
            0.07,
            f"n={b['n_used']}  |  collapse={100.0*b['collapse_fraction']:.1f}%  |  focus={100.0*b['focus_quantile']:.0f}% nearest",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#3b3b3b",
        )
        ax.text(0.00, -0.04, "early layers", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#4c4c4c")
        ax.text(1.00, -0.04, "late layers", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="#4c4c4c")

    fig.suptitle("Data-Faithful Convergence Funnels (No Geometric Forcing)", fontsize=17, y=1.02)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_funnels_density(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.0 * len(bundles), 4.8), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        yi = b["yi"]
        xg = b["x_plot"]
        y_lim = b["y_lim"]
        y_bins = np.linspace(-y_lim, y_lim, num=220)
        x_edges = _x_edges_from_centers(xg)
        den = np.zeros((y_bins.size - 1, xg.size), dtype=np.float64)
        for j in range(xg.size):
            hist, _ = np.histogram(yi[:, j], bins=y_bins, density=False)
            s = float(hist.sum())
            if s > 0:
                den[:, j] = hist / s
            else:
                den[:, j] = 0.0
        den = den / (np.max(den) + 1e-12)

        ax.pcolormesh(
            x_edges,
            y_bins,
            den,
            shading="auto",
            cmap="magma",
            alpha=0.92,
            vmin=0.0,
            vmax=1.0,
        )
        med = np.median(yi, axis=0)
        q20 = np.quantile(yi, 0.20, axis=0)
        q80 = np.quantile(yi, 0.80, axis=0)
        ax.plot(xg, med, color="#fff4d8", lw=1.35, alpha=0.95)
        ax.fill_between(xg, q20, q80, color="#fff4d8", alpha=0.12, lw=0)
        ax.scatter([float(xg[-1])], [0.0], s=30, c="#ffffff", edgecolors="none", zorder=6, alpha=0.95)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12, color="#fff4d8")
        ax.text(
            0.01,
            0.07,
            f"collapse={100.0*b['collapse_fraction']:.1f}%  |  mean turn={b['mean_turn_radians']:.2f} rad",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#ffeecb",
        )
        ax.text(0.00, -0.04, "early", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#e2d6c0")
        ax.text(1.00, -0.04, "late", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="#e2d6c0")

    fig.suptitle("Convergence Density Fields (Attractor-Centered, Data-Faithful)", fontsize=17, y=1.02, color="#fff4d8")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def _animate_funnels(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_lines_visible: int,
    fps: int,
) -> None:
    rng = np.random.default_rng(seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )

    fig, axs = plt.subplots(1, len(bundles), figsize=(6.0 * len(bundles), 4.9), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]

    # Fixed sampled trajectories for animation.
    sampled: Dict[str, np.ndarray] = {}
    for model_id, b in bundles.items():
        yi = b["yi"]
        N = yi.shape[0]
        k = min(int(max_lines_visible), int(N))
        idx = rng.choice(np.arange(N), size=k, replace=False) if k < N else np.arange(N)
        sampled[model_id] = yi[idx]

    n_frames = max(b["xg"].size for b in bundles.values())
    def _update(frame: int) -> None:
        for ax, model_id in zip(axs, bundles.keys()):
            b = bundles[model_id]
            xg = b["xg"]
            xp = b["x_plot"]
            yi = sampled[model_id]
            yi_metric = b["yi_metric"]
            li = min(frame, xg.size - 1)
            x = xp[: li + 1]

            ax.cla()
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-b["y_lim"], b["y_lim"])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            # Quantile envelope from full basin.
            all_y = b["yi"][:, : li + 1]
            q10 = np.quantile(all_y, 0.10, axis=0)
            q90 = np.quantile(all_y, 0.90, axis=0)
            q25 = np.quantile(all_y, 0.25, axis=0)
            q75 = np.quantile(all_y, 0.75, axis=0)
            ax.fill_between(x, q10, q90, color="#2d3a4d", alpha=0.11, lw=0)
            ax.fill_between(x, q25, q75, color="#2d3a4d", alpha=0.18, lw=0)

            for tr in yi:
                ax.plot(x, tr[: li + 1], color="#162132", alpha=0.03, lw=0.5)

            heads = yi[:, li]
            t = xg[li]
            ax.axhline(0.0, color="#3b3b3b", lw=0.8, alpha=0.35)
            ax.scatter([float(xp[-1])], [0.0], s=32, c="#111111", zorder=5)

            heads_metric = yi_metric[:, li]
            collapse_now = 1.0 - float(np.mean(np.abs(heads_metric)) / (np.mean(np.abs(yi_metric[:, 0])) + 1e-12))
            theta_now = 2.0 * np.pi * t
            ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
            ax.text(
                0.01,
                0.07,
                f"collapse={100.0*collapse_now:.1f}%  |  layer-angle θ={theta_now:.2f} rad  |  shown n={yi.shape[0]}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                color="#3b3b3b",
            )

        fig.suptitle("Convergence Through Depth (Attractor-Centered, No Forced Vortex)", fontsize=17, y=1.02)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=int(round(1000 / max(1, fps))))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _plot_radial_rank_static(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_lines_visible: int,
) -> None:
    rng = np.random.default_rng(seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.0 * len(bundles), 4.8), constrained_layout=False)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        xg = b["x_plot"]
        yi = b["yi_radial_rank"]
        pr = b["phase_rank"]
        N = yi.shape[0]
        k = min(int(max_lines_visible), int(N))
        idx = rng.choice(np.arange(N), size=k, replace=False) if k < N else np.arange(N)
        ys = yi[idx]
        pr_s = pr[idx]
        cmap = plt.get_cmap("turbo")
        init_p90 = float(np.percentile(np.abs(yi[:, 0]), 90.0))
        nonconv_thr = 0.35 * init_p90
        nonconv_mask_full = np.abs(yi[:, -1]) > nonconv_thr
        nonconv_pct = 100.0 * float(np.mean(nonconv_mask_full))

        q10 = np.quantile(yi, 0.10, axis=0)
        q90 = np.quantile(yi, 0.90, axis=0)
        q25 = np.quantile(yi, 0.25, axis=0)
        q75 = np.quantile(yi, 0.75, axis=0)
        ax.fill_between(xg, q10, q90, color="#233149", alpha=0.10, lw=0)
        ax.fill_between(xg, q25, q75, color="#233149", alpha=0.16, lw=0)

        for tr, rr in zip(ys, pr_s):
            is_nonconv = bool(abs(float(tr[-1])) > nonconv_thr)
            if is_nonconv:
                # Slightly emphasize trajectories that remain farther from the attractor.
                ax.plot(xg, tr, color=(0.38, 0.20, 0.15, 0.085), lw=0.62)
            else:
                c = cmap(0.5 * (rr + 1.0))
                ax.plot(xg, tr, color=(c[0], c[1], c[2], 0.032), lw=0.52)

        med = np.median(yi, axis=0)
        ax.plot(xg, med, color="#0f1a2a", lw=1.15, alpha=0.9)
        ax.axhline(0.0, color="#3b3b3b", lw=0.8, alpha=0.32)
        ax.scatter([float(xg[-1])], [0.0], s=32, c="#111111", zorder=5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-b["y_lim_radial_rank"], b["y_lim_radial_rank"])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
        ax.text(
            0.01,
            0.07,
            f"n={b['n_used']}  |  collapse={100.0*b['collapse_fraction_radial_rank']:.1f}%  |  non-converged={nonconv_pct:.1f}%",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#3b3b3b",
        )
        ax.text(0.00, -0.04, "early layers", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#4c4c4c")
        ax.text(1.00, -0.04, "late layers", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="#4c4c4c")

    fig.subplots_adjust(top=0.84, left=0.04, right=0.995, wspace=0.02)
    fig.suptitle("Lines Converging To One Attractor (Full-Space Distance View)", fontsize=17, y=0.97)
    fig.savefig(out_path, dpi=230, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _animate_radial_rank(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_lines_visible: int,
    fps: int,
) -> None:
    rng = np.random.default_rng(seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.0 * len(bundles), 4.9), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]

    sampled: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model_id, b in bundles.items():
        yi = b["yi_radial_rank"]
        pr = b["phase_rank"]
        N = yi.shape[0]
        k = min(int(max_lines_visible), int(N))
        idx = rng.choice(np.arange(N), size=k, replace=False) if k < N else np.arange(N)
        sampled[model_id] = (yi[idx], pr[idx])

    n_frames = max(b["xg"].size for b in bundles.values())
    cmap = plt.get_cmap("turbo")

    def _update(frame: int) -> None:
        for ax, model_id in zip(axs, bundles.keys()):
            b = bundles[model_id]
            xg = b["xg"]
            xp = b["x_plot"]
            yi, pr = sampled[model_id]
            li = min(frame, xg.size - 1)
            x = xp[: li + 1]

            ax.cla()
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-b["y_lim_radial_rank"], b["y_lim_radial_rank"])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            # Envelope from full set.
            all_y = b["yi_radial_rank"][:, : li + 1]
            q10 = np.quantile(all_y, 0.10, axis=0)
            q90 = np.quantile(all_y, 0.90, axis=0)
            q25 = np.quantile(all_y, 0.25, axis=0)
            q75 = np.quantile(all_y, 0.75, axis=0)
            ax.fill_between(x, q10, q90, color="#233149", alpha=0.10, lw=0)
            ax.fill_between(x, q25, q75, color="#233149", alpha=0.16, lw=0)

            for tr, rr in zip(yi, pr):
                c = cmap(0.5 * (rr + 1.0))
                ax.plot(x, tr[: li + 1], color=(c[0], c[1], c[2], 0.04), lw=0.5)

            heads = yi[:, li]
            # Temporal angle encoding only in head color.
            t = float(xg[li])
            ax.axhline(0.0, color="#3b3b3b", lw=0.8, alpha=0.32)
            ax.scatter([float(xp[-1])], [0.0], s=32, c="#111111", zorder=5)

            collapse_now = 1.0 - float(np.mean(np.abs(heads)) / (np.mean(np.abs(yi[:, 0])) + 1e-12))
            theta_now = 2.0 * np.pi * t
            ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
            ax.text(
                0.01,
                0.07,
                f"collapse={100.0*collapse_now:.1f}%  |  layer-angle θ={theta_now:.2f} rad  |  shown n={yi.shape[0]}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                color="#3b3b3b",
            )

        fig.suptitle("Lines Converging To One Attractor (Full-Space Distance View)", fontsize=17, y=1.02)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=int(round(1000 / max(1, fps))))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--interp-steps", type=int, default=6)
    ap.add_argument("--max-lines-static", type=int, default=1000)
    ap.add_argument("--max-lines-anim", type=int, default=650)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--x-warp", choices=["linear", "late_log", "late_power", "change_cdf"], default="linear")
    ap.add_argument("--late-log-alpha", type=float, default=18.0)
    ap.add_argument("--late-power-gamma", type=float, default=2.5)
    ap.add_argument("--x-left", type=float, default=0.03)
    ap.add_argument("--x-right", type=float, default=0.90)
    ap.add_argument("--population", choices=["all", "majority"], default="all")
    ap.add_argument("--attractor", choices=["majority", "global"], default="majority")
    ap.add_argument("--display-smooth-sigma", type=float, default=2.2)
    ap.add_argument("--focus-quantile", type=float, default=0.20)
    ap.add_argument("--change-cdf-source", choices=["faithful", "radial"], default="faithful")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "analysis" / "faithful")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_paths = sorted((run_dir / "outputs").glob("*/baseline_outputs.jsonl"))
    if not model_paths:
        raise RuntimeError(f"no baseline outputs in {run_dir / 'outputs'}")

    model_data = [_load_model(p) for p in model_paths]
    bundles = {
        d["model_id"]: _prepare_faithful_coords(
            d,
            interp_steps=int(args.interp_steps),
            x_warp=str(args.x_warp),
            late_log_alpha=float(args.late_log_alpha),
            late_power_gamma=float(args.late_power_gamma),
            x_left=float(args.x_left),
            x_right=float(args.x_right),
            population_mode=str(args.population),
            attractor_mode=str(args.attractor),
            display_smooth_sigma=float(args.display_smooth_sigma),
            focus_quantile=float(args.focus_quantile),
            change_cdf_source=str(args.change_cdf_source),
        )
        for d in model_data
    }

    static_png = out_dir / "faithful_convergence_funnels.png"
    anim_gif = out_dir / "faithful_convergence_funnels.gif"
    density_png = out_dir / "beautiful_convergence_density_fields.png"
    radial_rank_png = out_dir / "beautiful_radial_rank_funnels.png"
    radial_rank_gif = out_dir / "beautiful_radial_rank_funnels.gif"

    _plot_funnels_static(
        bundles=bundles,
        out_path=static_png,
        seed=int(args.seed),
        max_lines_visible=int(args.max_lines_static),
    )
    _animate_funnels(
        bundles=bundles,
        out_path=anim_gif,
        seed=int(args.seed) + 33,
        max_lines_visible=int(args.max_lines_anim),
        fps=int(args.fps),
    )
    _plot_funnels_density(bundles=bundles, out_path=density_png)
    _plot_radial_rank_static(
        bundles=bundles,
        out_path=radial_rank_png,
        seed=int(args.seed) + 99,
        max_lines_visible=int(args.max_lines_static),
    )
    _animate_radial_rank(
        bundles=bundles,
        out_path=radial_rank_gif,
        seed=int(args.seed) + 111,
        max_lines_visible=int(args.max_lines_anim),
        fps=int(args.fps),
    )

    summary = {
        "run_dir": str(run_dir),
        "outputs": {
            "faithful_convergence_funnels_png": str(static_png),
            "faithful_convergence_funnels_gif": str(anim_gif),
            "beautiful_convergence_density_fields_png": str(density_png),
            "beautiful_radial_rank_funnels_png": str(radial_rank_png),
            "beautiful_radial_rank_funnels_gif": str(radial_rank_gif),
        },
        "models": {
            k: {
                "n_total": int(v["n_total"]),
                "n_majority_basin": int(v["n_basin"]),
                "majority_basin_choice": str(v["target_choice"]),
                "collapse_fraction": float(v["collapse_fraction"]),
                "left_p90_abs": float(v["left_p90_abs"]),
                "right_p90_abs": float(v["right_p90_abs"]),
                "mean_turn_radians": float(v["mean_turn_radians"]),
                "collapse_fraction_radial_rank": float(v["collapse_fraction_radial_rank"]),
                "n_used": int(v["n_used"]),
                "population_mode": str(v["population_mode"]),
                "attractor_mode": str(v["attractor_mode"]),
                "focus_quantile": float(v["focus_quantile"]),
                "focus_threshold_final_abs": float(v["focus_threshold_final_abs"]),
                "change_cdf_source": str(v["change_cdf_source"]),
            }
            for k, v in bundles.items()
        },
        "mapping": {
            "x": "normalized layer depth",
            "x_display_plot": "display-only monotone warp of normalized layer depth",
            "y": "signed projection on first principal axis of attractor-centered hidden states",
            "y_alt_full_space": "full 128D distance-to-attractor multiplied by initial phase-rank sign for display ordering",
            "attractor": "mean final hidden state of majority final-answer basin",
            "transform_type": "linear, data-faithful",
            "forced_geometric_warp": False,
            "display_x_warp": str(args.x_warp),
            "display_x_warp_params": {
                "late_log_alpha": float(args.late_log_alpha),
                "late_power_gamma": float(args.late_power_gamma),
            },
            "display_x_window": {"x_left": float(args.x_left), "x_right": float(args.x_right)},
            "population_mode": str(args.population),
            "attractor_mode": str(args.attractor),
            "display_smoothing": {
                "method": "gaussian_1d_on_display_curves_only",
                "sigma": float(args.display_smooth_sigma),
            },
            "change_cdf_source": str(args.change_cdf_source),
        },
    }
    (out_dir / "faithful_convergence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
