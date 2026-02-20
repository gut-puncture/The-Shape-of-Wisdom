#!/usr/bin/env python3
"""
Deep baseline-only analysis for a completed run directory.

Outputs:
- deep_summary.json / deep_summary.md
- convergence_layer_metrics.csv
- topology_metrics.csv
- beautiful_convergence_overview.png
- beautiful_topology_constellation.png
- beautiful_convergence_simplex.gif
- beautiful_convergence_lines.gif
- beautiful_latent_vortex.gif
- beautiful_latent_vortex_clean.gif
- beautiful_horizontal_vortex_paths.png
- beautiful_horizontal_vortex_paths.gif
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
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel


CHOICES = ["A", "B", "C", "D"]
CHOICE_TO_IDX = {c: i for i, c in enumerate(CHOICES)}
CHOICE_COLORS = {
    "A": "#4C72B0",
    "B": "#DD8452",
    "C": "#55A868",
    "D": "#C44E52",
}

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
}

MODEL_COLORS = {
    "Qwen/Qwen2.5-7B-Instruct": "#4C72B0",
    "meta-llama/Llama-3.1-8B-Instruct": "#55A868",
    "mistralai/Mistral-7B-Instruct-v0.3": "#DD8452",
}


def _model_label(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def _safe_choice_idx(x: Any) -> int:
    if isinstance(x, str) and x in CHOICE_TO_IDX:
        return CHOICE_TO_IDX[x]
    return -1


def _extract_commitment_layer(commit_dict: Any, threshold: float) -> float:
    if not isinstance(commit_dict, dict) or not commit_dict:
        return float("nan")
    best_key = None
    best_gap = float("inf")
    for k in commit_dict.keys():
        try:
            kv = float(k)
        except Exception:
            continue
        gap = abs(kv - float(threshold))
        if gap < best_gap:
            best_gap = gap
            best_key = k
    if best_key is None:
        return float("nan")
    v = commit_dict.get(best_key)
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float) and math.isfinite(v):
        return float(v)
    return float("nan")


def _load_model_output(output_path: Path, commitment_threshold: float) -> Dict[str, Any]:
    probs_rows: List[np.ndarray] = []
    top_rows: List[np.ndarray] = []
    hidden_rows: List[np.ndarray] = []
    commitments: List[float] = []
    flips: List[int] = []
    correct: List[bool] = []
    domains: List[str] = []
    final_hidden: List[np.ndarray] = []
    model_id: str | None = None
    model_revision: str | None = None

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lw = row.get("layerwise") or []
            if not isinstance(lw, list) or not lw:
                continue

            if model_id is None:
                model_id = str(row.get("model_id") or "")
                model_revision = str(row.get("model_revision") or "")

            n_layers = len(lw)
            row_probs = np.zeros((n_layers, 4), dtype=np.float32)
            row_top = np.zeros((n_layers,), dtype=np.int16)
            row_hidden: List[np.ndarray] = []
            valid = True
            for li, layer in enumerate(lw):
                cp = layer.get("candidate_probs") or {}
                vals = np.array([float(cp.get(c, 0.0)) for c in CHOICES], dtype=np.float32)
                s = float(vals.sum())
                if not math.isfinite(s) or s <= 0:
                    vals = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                else:
                    vals = vals / s
                if not np.all(np.isfinite(vals)):
                    valid = False
                    break
                row_probs[li] = vals
                tc = _safe_choice_idx(layer.get("top_candidate"))
                if tc < 0:
                    tc = int(np.argmax(vals))
                row_top[li] = tc

                ph = layer.get("projected_hidden_128")
                if not isinstance(ph, list) or len(ph) == 0:
                    valid = False
                    break
                phv = np.asarray(ph, dtype=np.float32)
                if phv.ndim != 1 or not np.all(np.isfinite(phv)):
                    valid = False
                    break
                row_hidden.append(phv)

            if not valid:
                continue

            try:
                row_hidden_arr = np.stack(row_hidden, axis=0).astype(np.float32)
            except Exception:
                continue

            probs_rows.append(row_probs)
            top_rows.append(row_top)
            hidden_rows.append(row_hidden_arr)
            commitments.append(_extract_commitment_layer(row.get("commitment_layer_by_margin_threshold"), commitment_threshold))
            flips.append(int(row.get("flip_count") or 0))
            correct.append(bool(row.get("is_correct")))
            domains.append(str(row.get("coarse_domain") or "unknown"))
            final_hidden.append(row_hidden_arr[-1])

    if not probs_rows:
        raise RuntimeError(f"No valid baseline rows found in {output_path}")

    probs = np.stack(probs_rows, axis=0)  # [N, L, 4]
    top = np.stack(top_rows, axis=0)  # [N, L]
    hidden = np.stack(hidden_rows, axis=0)  # [N, L, D]
    X = np.stack(final_hidden, axis=0)  # [N, D]

    return {
        "model_id": model_id or output_path.parent.name,
        "model_revision": model_revision or "",
        "probs": probs,
        "top": top,
        "hidden": hidden,
        "commitment_layer": np.asarray(commitments, dtype=np.float64),
        "flip_count": np.asarray(flips, dtype=np.int32),
        "is_correct": np.asarray(correct, dtype=bool),
        "domain": np.asarray(domains, dtype=object),
        "final_hidden": X,
        "n_rows": int(probs.shape[0]),
        "n_layers": int(probs.shape[1]),
    }


def _convergence_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    probs: np.ndarray = d["probs"]
    top: np.ndarray = d["top"]
    commit: np.ndarray = d["commitment_layer"]
    flips: np.ndarray = d["flip_count"]
    correct: np.ndarray = d["is_correct"]
    n_rows, n_layers, _ = probs.shape

    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=2) / math.log(4.0)  # [N, L]
    convergence = 1.0 - entropy
    p_sorted = np.sort(probs, axis=2)
    margin = p_sorted[:, :, -1] - p_sorted[:, :, -2]
    agreement_with_final = (top == top[:, -1][:, None]).mean(axis=0)

    x = np.arange(n_layers, dtype=np.float64)
    x0 = x - x.mean()
    denom = float(np.sum(x0 * x0))
    slopes = (entropy * x0[None, :]).sum(axis=1) / denom

    first_entropy = entropy[:, 0]
    last_entropy = entropy[:, -1]
    first_conv = convergence[:, 0]
    last_conv = convergence[:, -1]
    first_margin = margin[:, 0]
    last_margin = margin[:, -1]
    t_res = ttest_rel(first_entropy, last_entropy)

    valid_commit = np.isfinite(commit)
    commit_valid = commit[valid_commit]

    correct_mask = correct
    incorrect_mask = ~correct_mask
    final_margin_correct = float(last_margin[correct_mask].mean()) if np.any(correct_mask) else float("nan")
    final_margin_incorrect = float(last_margin[incorrect_mask].mean()) if np.any(incorrect_mask) else float("nan")

    summary = {
        "n_rows": int(n_rows),
        "n_layers": int(n_layers),
        "entropy_start_mean": float(first_entropy.mean()),
        "entropy_end_mean": float(last_entropy.mean()),
        "entropy_delta_end_minus_start": float((last_entropy - first_entropy).mean()),
        "convergence_start_mean": float(first_conv.mean()),
        "convergence_end_mean": float(last_conv.mean()),
        "convergence_delta_end_minus_start": float((last_conv - first_conv).mean()),
        "margin_start_mean": float(first_margin.mean()),
        "margin_end_mean": float(last_margin.mean()),
        "margin_delta_end_minus_start": float((last_margin - first_margin).mean()),
        "entropy_ttest_stat": float(t_res.statistic),
        "entropy_ttest_pvalue": float(t_res.pvalue),
        "agreement_with_final_start": float(agreement_with_final[0]),
        "agreement_with_final_end": float(agreement_with_final[-1]),
        "agreement_with_final_gain": float(agreement_with_final[-1] - agreement_with_final[0]),
        "negative_entropy_slope_fraction": float(np.mean(slopes < 0.0)),
        "flip_count_mean": float(flips.mean()),
        "flip_count_median": float(np.median(flips)),
        "commitment_valid_fraction": float(valid_commit.mean()),
        "commitment_layer_median": float(np.nanmedian(commit_valid)) if commit_valid.size else float("nan"),
        "commitment_layer_p10": float(np.nanpercentile(commit_valid, 10)) if commit_valid.size else float("nan"),
        "commitment_layer_p90": float(np.nanpercentile(commit_valid, 90)) if commit_valid.size else float("nan"),
        "accuracy": float(correct_mask.mean()),
        "final_margin_correct_mean": final_margin_correct,
        "final_margin_incorrect_mean": final_margin_incorrect,
    }

    d["entropy"] = entropy
    d["convergence"] = convergence
    d["margin"] = margin
    d["agreement_with_final"] = agreement_with_final
    d["commitment_valid_mask"] = valid_commit
    d["summary"] = summary
    return summary


def _topology_metrics(d: Dict[str, Any], seed: int) -> Dict[str, Any]:
    X: np.ndarray = d["final_hidden"]
    domains: np.ndarray = d["domain"]
    labels, uniques = pd.factorize(domains)

    if X.ndim != 2 or X.shape[0] < 10 or len(uniques) < 2:
        return {
            "n_domains": int(len(uniques)),
            "silhouette_euclidean": float("nan"),
            "nearest_centroid_accuracy": float("nan"),
            "between_within_ratio": float("nan"),
            "mean_centroid_pairwise_distance": float("nan"),
            "min_centroid_pairwise_distance": float("nan"),
            "max_centroid_pairwise_distance": float("nan"),
        }

    sil = float(silhouette_score(X, labels, metric="euclidean"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    centroids = np.stack([X_train[y_train == c].mean(axis=0) for c in range(len(uniques))], axis=0)
    d2 = np.sum((X_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    y_pred = np.argmin(d2, axis=1)
    acc = float(np.mean(y_pred == y_test))

    global_mu = X.mean(axis=0)
    between = 0.0
    within = 0.0
    cls_centroids = []
    for c in range(len(uniques)):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        cls_centroids.append(mu)
        between += float(Xc.shape[0]) * float(np.sum((mu - global_mu) ** 2))
        within += float(np.sum((Xc - mu) ** 2))
    cls_centroids = np.stack(cls_centroids, axis=0)
    ratio = float(between / (within + 1e-12))

    cent_d = []
    for i in range(cls_centroids.shape[0]):
        for j in range(i + 1, cls_centroids.shape[0]):
            cent_d.append(float(np.linalg.norm(cls_centroids[i] - cls_centroids[j])))
    cent_d = np.asarray(cent_d, dtype=np.float64)

    return {
        "n_domains": int(len(uniques)),
        "silhouette_euclidean": sil,
        "nearest_centroid_accuracy": acc,
        "between_within_ratio": ratio,
        "mean_centroid_pairwise_distance": float(np.mean(cent_d)),
        "min_centroid_pairwise_distance": float(np.min(cent_d)),
        "max_centroid_pairwise_distance": float(np.max(cent_d)),
    }


def _interp01(y: np.ndarray, n: int = 201) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, y.shape[0], dtype=np.float64)
    g = np.linspace(0.0, 1.0, n, dtype=np.float64)
    yi = np.interp(g, x, y.astype(np.float64))
    return g, yi


def _plot_overview(model_data: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax_conv, ax_align, ax_commit = axs

    for model_id, d in model_data.items():
        color = MODEL_COLORS.get(model_id, "#333333")
        label = _model_label(model_id)
        g, conv_i = _interp01(d["convergence"].mean(axis=0))
        _, align_i = _interp01(d["agreement_with_final"])

        commit = d["commitment_layer"]
        L = d["n_layers"]
        committed_curve = np.array(
            [np.mean(np.isfinite(commit) & (commit <= li)) for li in range(L)], dtype=np.float64
        )
        _, committed_i = _interp01(committed_curve)

        ax_conv.plot(g, conv_i, lw=2.8, color=color, label=label)
        ax_align.plot(g, align_i, lw=2.8, color=color, label=label)
        ax_commit.plot(g, committed_i, lw=2.8, color=color, label=label)

    ax_conv.set_title("Convergence Index Through Depth")
    ax_conv.set_xlabel("Normalized Layer Depth")
    ax_conv.set_ylabel("Convergence Index (1 - normalized entropy)")
    ax_conv.set_xlim(0, 1)
    ax_conv.set_ylim(0, 1)

    ax_align.set_title("Agreement With Final Winner")
    ax_align.set_xlabel("Normalized Layer Depth")
    ax_align.set_ylabel("Fraction of Prompts")
    ax_align.set_xlim(0, 1)
    ax_align.set_ylim(0, 1)

    ax_commit.set_title("Commitment Accumulation (threshold=0.1)")
    ax_commit.set_xlabel("Normalized Layer Depth")
    ax_commit.set_ylabel("Fraction Committed")
    ax_commit.set_xlim(0, 1)
    ax_commit.set_ylim(0, 1)

    for ax in axs:
        ax.grid(alpha=0.18, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = ax_conv.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Baseline Run 1452_a: Decision Convergence and Commitment", fontsize=16, y=1.10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _make_convergence_lines_gif(model_data: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for model_id, d in model_data.items():
        g, conv_i = _interp01(d["convergence"].mean(axis=0), n=220)
        curves[model_id] = {"x": g, "y": conv_i}

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Convergence Index (1 - normalized entropy)")
    ax.set_title("Convergence Emergence Through Depth")
    ax.grid(alpha=0.20, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    line_handles: Dict[str, Any] = {}
    point_handles: Dict[str, Any] = {}
    text_handles: Dict[str, Any] = {}
    y_base = 0.06
    for i, model_id in enumerate(curves.keys()):
        color = MODEL_COLORS.get(model_id, "#333333")
        label = _model_label(model_id)
        (ln,) = ax.plot([], [], lw=3.1, color=color, alpha=0.95, label=label)
        (pt,) = ax.plot([], [], marker="o", color=color, markersize=6, alpha=0.95)
        txt = ax.text(
            0.72,
            y_base + 0.06 * i,
            f"{label}: --",
            color=color,
            fontsize=10,
            transform=ax.transAxes,
        )
        line_handles[model_id] = ln
        point_handles[model_id] = pt
        text_handles[model_id] = txt

    depth_vline = ax.axvline(0.0, color="#666666", lw=1.2, ls=":", alpha=0.7)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    n_frames = max(len(v["x"]) for v in curves.values())

    def _update(frame: int) -> None:
        for model_id, data in curves.items():
            x = data["x"]
            y = data["y"]
            li = min(frame, len(x) - 1)
            line_handles[model_id].set_data(x[: li + 1], y[: li + 1])
            point_handles[model_id].set_data([x[li]], [y[li]])
            text_handles[model_id].set_text(f"{_model_label(model_id)}: {y[li]:.2f}")
        x_now = list(curves.values())[0]["x"][min(frame, len(list(curves.values())[0]["x"]) - 1)]
        depth_vline.set_xdata([x_now, x_now])
        fig.suptitle("Convergence As an Order-Forming Process", fontsize=16, y=1.02)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=75)
    anim.save(out_path, writer=PillowWriter(fps=12))
    plt.close(fig)


def _smooth_track_nd(tr: np.ndarray, window: int = 5) -> np.ndarray:
    if tr.shape[0] < 3:
        return tr.copy()
    w = int(max(3, min(window, tr.shape[0] - (1 - tr.shape[0] % 2))))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return tr.copy()
    k = np.ones((w,), dtype=np.float64) / float(w)
    out = np.empty_like(tr, dtype=np.float64)
    for d in range(tr.shape[1]):
        out[:, d] = np.convolve(tr[:, d], k, mode="same")
    out[0] = tr[0]
    out[-1] = tr[-1]
    return out.astype(np.float32)


def _make_latent_vortex_gif(
    model_data: Dict[str, Dict[str, Any]],
    out_path: Path,
    sample_points_per_model: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed + 101)
    model_ids = list(model_data.keys())
    bundles: Dict[str, Dict[str, Any]] = {}
    max_layers = 0

    for model_id in model_ids:
        d = model_data[model_id]
        hidden = d["hidden"]  # [N, L, 128]
        top_final = d["top"][:, -1]  # [N]
        n_rows, n_layers, hdim = hidden.shape
        max_layers = max(max_layers, n_layers)

        counts = np.bincount(top_final, minlength=4)
        target_idx = int(np.argmax(counts))
        target_choice = CHOICES[target_idx]
        pool = np.flatnonzero(top_final == target_idx)
        if pool.size < 8:
            continue

        k = min(int(sample_points_per_model), int(pool.size))
        if pool.size > k:
            idx = rng.choice(pool, size=k, replace=False)
        else:
            idx = pool

        H = hidden[idx].astype(np.float32)  # [k, L, D]
        for i in range(H.shape[0]):
            H[i] = _smooth_track_nd(H[i], window=5)

        attractor = H[:, -1, :].mean(axis=0)  # [D]
        Hc = H - attractor[None, None, :]
        X = Hc.reshape(-1, hdim).astype(np.float64)

        # PCA basis without re-centering to keep attractor fixed at origin.
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        basis = vh[:2].T.astype(np.float64)  # [D, 2]
        uv = np.einsum("nld,dk->nlk", Hc, basis).astype(np.float32)  # [k, L, 2]

        # Color trajectories by initial angular position around attractor.
        theta0 = np.arctan2(uv[:, 0, 1], uv[:, 0, 0])  # [k]
        hue = (theta0 + np.pi) / (2.0 * np.pi)
        cmap = plt.get_cmap("hsv")
        point_colors = cmap(hue)
        point_colors[:, 3] = 0.42
        trail_colors = cmap(hue)
        trail_colors[:, 3] = 0.09

        radius = np.linalg.norm(uv, axis=2)  # [k, L]
        early = radius[:, : max(2, n_layers // 3)]
        lim = float(np.percentile(early, 99.5))
        lim = max(lim, 1e-3)
        mean_radius_curve = radius.mean(axis=0)

        trail_k = min(180, k)
        trail_idx = rng.choice(np.arange(k), size=trail_k, replace=False)

        bundles[model_id] = {
            "uv": uv,
            "n_layers": n_layers,
            "limit": lim,
            "point_colors": point_colors,
            "trail_colors": trail_colors,
            "trail_idx": trail_idx,
            "target_choice": target_choice,
            "target_count": int(pool.size),
            "mean_radius_curve": mean_radius_curve,
        }

    if not bundles:
        return {}

    plt.rcParams.update({"font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(1, len(model_ids), figsize=(5.4 * len(model_ids), 5), facecolor="#070B16")
    if len(model_ids) == 1:
        axs = [axs]

    def _draw_ax(ax: Any, model_id: str, li: int) -> None:
        b = bundles.get(model_id)
        ax.cla()
        ax.set_facecolor("#070B16")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        if b is None:
            return

        L = int(b["n_layers"])
        li = min(li, L - 1)
        uv = b["uv"]
        lim = float(b["limit"])
        tail = 6
        start = max(0, li - tail)

        # Concentric rings make the contraction field legible frame-by-frame.
        for rr, aa in [(1.00, 0.20), (0.75, 0.16), (0.50, 0.12), (0.25, 0.09)]:
            ax.add_patch(Circle((0.0, 0.0), lim * rr, fill=False, lw=1.0, ec="#4E5F89", alpha=aa))

        for ti in b["trail_idx"]:
            tr = uv[int(ti), start : li + 1, :]
            ax.plot(
                tr[:, 0],
                tr[:, 1],
                color=b["trail_colors"][int(ti)],
                lw=0.8,
                solid_capstyle="round",
            )

        pts = uv[:, li, :]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=b["point_colors"],
            s=9,
            linewidths=0,
        )
        ax.scatter([0.0], [0.0], marker="*", s=200, c="#FFE08A", edgecolors="none", alpha=0.95)

        mrc = b["mean_radius_curve"]
        collapse = 1.0 - float(mrc[li] / (mrc[0] + 1e-12))
        ax.text(
            0.02,
            0.97,
            f"{_model_label(model_id)}\nfinal answer basin: {b['target_choice']}\nlayer {li + 1}/{L}\nradial collapse: {100.0 * collapse:.1f}%",
            transform=ax.transAxes,
            color="#DDE6FF",
            fontsize=10,
            va="top",
            ha="left",
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")

    def _update(frame: int) -> None:
        for ax, model_id in zip(axs, model_ids):
            _draw_ax(ax, model_id, frame)
        fig.suptitle(
            "Latent Convergence Vortex: Many Prompt States Collapse Into One Decision Attractor",
            color="#F3F6FF",
            fontsize=16,
            y=0.98,
        )

    anim = FuncAnimation(fig, _update, frames=max_layers, interval=220)
    anim.save(out_path, writer=PillowWriter(fps=5))
    plt.close(fig)

    metrics: Dict[str, Dict[str, Any]] = {}
    for model_id, b in bundles.items():
        mr = np.asarray(b["mean_radius_curve"], dtype=np.float64)
        metrics[model_id] = {
            "target_choice": str(b["target_choice"]),
            "target_choice_count": int(b["target_count"]),
            "mean_radius_start": float(mr[0]),
            "mean_radius_end": float(mr[-1]),
            "mean_radius_delta": float(mr[-1] - mr[0]),
            "radial_collapse_fraction": float(1.0 - (mr[-1] / (mr[0] + 1e-12))),
        }
    return metrics


def _make_latent_vortex_clean_gif(
    model_data: Dict[str, Dict[str, Any]],
    out_path: Path,
    sample_points_per_model: int,
    seed: int,
    interp_steps: int = 4,
    fps: int = 5,
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed + 303)
    model_ids = list(model_data.keys())
    bundles: Dict[str, Dict[str, Any]] = {}

    for model_id in model_ids:
        d = model_data[model_id]
        hidden = d["hidden"]  # [N, L, 128]
        top_final = d["top"][:, -1]
        _, n_layers, hdim = hidden.shape

        counts = np.bincount(top_final, minlength=4)
        target_idx = int(np.argmax(counts))
        target_choice = CHOICES[target_idx]
        pool = np.flatnonzero(top_final == target_idx)
        if pool.size < 12:
            continue

        k = min(int(sample_points_per_model), int(pool.size))
        idx = rng.choice(pool, size=k, replace=False) if pool.size > k else pool

        H = hidden[idx].astype(np.float32)  # [k, L, D]
        for i in range(H.shape[0]):
            H[i] = _smooth_track_nd(H[i], window=5)

        attractor = H[:, -1, :].mean(axis=0)
        Hc = H - attractor[None, None, :]
        X = Hc.reshape(-1, hdim).astype(np.float64)
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        basis = vh[:2].T.astype(np.float64)
        uv = np.einsum("nld,dk->nlk", Hc, basis).astype(np.float32)  # [k, L, 2]

        # Quantify rotational structure in raw trajectories.
        theta_raw = np.unwrap(np.arctan2(uv[:, :, 1], uv[:, :, 0]), axis=1)
        net_turn = theta_raw[:, -1] - theta_raw[:, 0]
        sign_coherence = float(np.abs(np.mean(np.sign(net_turn))))
        mean_abs_turn = float(np.mean(np.abs(net_turn)))
        frac_turn_gt_quarter = float(np.mean(np.abs(net_turn) > (np.pi / 2.0)))

        # Phase-aligned + phase-spread view:
        # keep each trajectory's angular delta and radius, align cw/ccw direction,
        # and spread start phases to avoid severe overplotting.
        theta0 = theta_raw[:, :1]
        turn_dir = np.where(net_turn[:, None] >= 0.0, 1.0, -1.0)
        start = theta_raw[:, 0]
        order = np.argsort(start)
        phase_spread = np.empty_like(start)
        phase_spread[order] = np.linspace(0.0, 2.0 * np.pi, num=start.shape[0], endpoint=False)
        theta_vis = phase_spread[:, None] + turn_dir * (theta_raw - theta0)
        radius_layer = np.linalg.norm(uv, axis=2)
        uv_vis = np.stack(
            [radius_layer * np.cos(theta_vis), radius_layer * np.sin(theta_vis)],
            axis=2,
        ).astype(np.float32)

        # Interpolate with collapse-progress retiming so motion is visible across whole clip
        # (raw layer-depth motion is very back-loaded in this dataset).
        F = (n_layers - 1) * int(interp_steps) + 1
        uv_interp = np.empty((F, uv_vis.shape[0], 2), dtype=np.float32)
        rad_layer = np.linalg.norm(uv_vis, axis=2)  # [k, L]
        mr_layer = rad_layer.mean(axis=0)
        collapse_layer = 1.0 - (mr_layer / (mr_layer[0] + 1e-12))
        collapse_layer = np.maximum.accumulate(collapse_layer)
        # Ensure strictly non-decreasing x-grid for interpolation.
        xgrid = collapse_layer + np.arange(collapse_layer.shape[0], dtype=np.float64) * 1e-9
        c_end = float(collapse_layer[-1])
        targets = np.linspace(0.0, c_end, num=F, dtype=np.float64)
        layer_pos = np.interp(targets, xgrid, np.arange(n_layers, dtype=np.float64))
        for fi, x in enumerate(layer_pos):
            li = int(np.floor(x))
            if li >= n_layers - 1:
                uv_interp[fi] = uv_vis[:, -1, :]
                continue
            a = float(x - li)
            uv_interp[fi] = (1.0 - a) * uv_vis[:, li, :] + a * uv_vis[:, li + 1, :]

        radius = np.linalg.norm(uv_interp, axis=2)  # [F, k]
        mean_radius = radius.mean(axis=1)
        collapse_curve = 1.0 - (mean_radius / (mean_radius[0] + 1e-12))

        # Set limit from early spread to stabilize framing.
        lim = float(np.percentile(radius[: max(2, F // 3), :], 99.2))
        lim = max(lim, 1e-3)

        theta_start = np.arctan2(uv_interp[0, :, 1], uv_interp[0, :, 0])
        hue = (theta_start + np.pi) / (2.0 * np.pi)
        cmap = plt.get_cmap("hsv")
        colors = cmap(hue)
        colors[:, 3] = 0.50
        trail_colors = cmap(hue)
        trail_colors[:, 3] = 0.10
        trail_k = min(120, int(uv_interp.shape[1]))
        trail_idx = rng.choice(np.arange(uv_interp.shape[1]), size=trail_k, replace=False)

        bundles[model_id] = {
            "uv_interp": uv_interp,
            "collapse_curve": collapse_curve,
            "lim": lim,
            "colors": colors,
            "trail_colors": trail_colors,
            "trail_idx": trail_idx,
            "target_choice": target_choice,
            "target_count": int(pool.size),
            "n_layers": int(n_layers),
            "sign_coherence": sign_coherence,
            "mean_abs_turn": mean_abs_turn,
            "frac_turn_gt_quarter": frac_turn_gt_quarter,
        }

    if not bundles:
        return {}

    n_frames = max(b["uv_interp"].shape[0] for b in bundles.values())
    plt.rcParams.update({"font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(1, len(model_ids), figsize=(5.2 * len(model_ids), 5), facecolor="#F6F4EF")
    if len(model_ids) == 1:
        axs = [axs]

    def _draw_rings(ax: Any, lim: float) -> None:
        for rr, aa in [(1.0, 0.11), (0.72, 0.08), (0.48, 0.06), (0.26, 0.05)]:
            ax.add_patch(Circle((0.0, 0.0), lim * rr, fill=False, lw=0.9, ec="#2A3C5A", alpha=aa))

    def _update(frame: int) -> None:
        for ax, model_id in zip(axs, model_ids):
            b = bundles.get(model_id)
            ax.cla()
            ax.set_facecolor("#F6F4EF")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if b is None:
                continue

            uv = b["uv_interp"]
            li = min(frame, uv.shape[0] - 1)
            pts = uv[li]
            lim = float(b["lim"])
            collapse = float(b["collapse_curve"][li])
            _draw_rings(ax, lim)

            tail = 9
            st = max(0, li - tail)
            for ti in b["trail_idx"]:
                seg = uv[st : li + 1, int(ti), :]
                ax.plot(seg[:, 0], seg[:, 1], color=b["trail_colors"][int(ti)], lw=0.7, solid_capstyle="round")

            ax.scatter(pts[:, 0], pts[:, 1], c=b["colors"], s=10, linewidths=0)
            ax.scatter([0.0], [0.0], c="#111111", s=46, linewidths=0)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            ax.text(
                0.03,
                0.96,
                f"{_model_label(model_id)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                color="#111111",
            )
            ax.text(
                0.03,
                0.06,
                f"collapse {100.0 * collapse:.1f}%",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                color="#2B2B2B",
            )

        fig.suptitle(
            "Phase-Aligned Vortex View: Many States Converge To One Attractor",
            fontsize=15,
            y=0.98,
            color="#101010",
        )

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=int(round(1000 / max(1, fps))))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    metrics: Dict[str, Dict[str, Any]] = {}
    for model_id, b in bundles.items():
        cc = np.asarray(b["collapse_curve"], dtype=np.float64)
        metrics[model_id] = {
            "target_choice": str(b["target_choice"]),
            "target_choice_count": int(b["target_count"]),
            "collapse_start": float(cc[0]),
            "collapse_end": float(cc[-1]),
            "sign_coherence_raw": float(b["sign_coherence"]),
            "mean_abs_net_turn_raw_radians": float(b["mean_abs_turn"]),
            "fraction_large_turn_raw": float(b["frac_turn_gt_quarter"]),
        }
    return metrics


def _prepare_horizontal_vortex_paths(
    d: Dict[str, Any],
    seed: int,
    sample_points: int,
    interp_steps: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    hidden = d["hidden"]  # [N, L, D]
    top_final = d["top"][:, -1]
    _, n_layers, hdim = hidden.shape
    counts = np.bincount(top_final, minlength=4)
    target_idx = int(np.argmax(counts))
    target_choice = CHOICES[target_idx]
    pool = np.flatnonzero(top_final == target_idx)
    if pool.size < 12:
        raise RuntimeError(f"not enough rows for horizontal vortex: {d['model_id']}")

    # Use all available rows in the basin for density; this is usually 800-1000 rows.
    Hall = hidden[pool].astype(np.float32)  # [M, L, D]
    for i in range(Hall.shape[0]):
        Hall[i] = _smooth_track_nd(Hall[i], window=5)

    attractor = Hall[:, -1, :].mean(axis=0)
    Hc = Hall - attractor[None, None, :]
    X = Hc.reshape(-1, hdim).astype(np.float64)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    basis = vh[:2].T.astype(np.float64)
    uv = np.einsum("nld,dk->nlk", Hc, basis).astype(np.float32)  # [M, L, 2]

    theta = np.unwrap(np.arctan2(uv[:, :, 1], uv[:, :, 0]), axis=1)
    net_turn = theta[:, -1] - theta[:, 0]
    dsgn = np.sign(net_turn)
    dsgn[dsgn == 0] = 1.0
    predominant = np.sign(np.mean(dsgn))
    if predominant == 0:
        predominant = 1.0
    align = dsgn * predominant

    theta_vis = theta[:, :1] + align[:, None] * (theta - theta[:, :1])

    # Build a horizontal funnel from radius x phase:
    # 1) preserve each trajectory's radial collapse
    # 2) align cw/ccw direction
    # 3) spread start phase to avoid overplotting into a flat line
    # 4) add explicit layer-angle sweep so paths visibly twist through depth
    r = np.linalg.norm(uv, axis=2)
    delta = align[:, None] * (theta - theta[:, :1])
    start = theta[:, 0]
    order = np.argsort(start)
    phase0 = np.empty_like(start)
    phase0[order] = np.linspace(0.0, 2.0 * np.pi, num=start.shape[0], endpoint=False)
    x_layer = np.linspace(0.0, 1.0, num=n_layers, dtype=np.float64)
    layer_angle = (2.2 * np.pi) * x_layer[None, :]
    theta_vis = phase0[:, None] + layer_angle + (0.65 * delta)
    y = r * np.sin(theta_vis)

    scale = float(np.percentile(np.abs(y[:, : max(2, n_layers // 3)]), 99.5))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = float(np.max(np.abs(y)) + 1e-6)
    y = y / scale
    y = np.clip(y, -1.2, 1.2)

    xg = np.linspace(0.0, 1.0, num=(n_layers - 1) * int(interp_steps) + 1, dtype=np.float64)
    y_interp = np.empty((y.shape[0], xg.shape[0]), dtype=np.float32)
    theta_interp = np.empty((theta_vis.shape[0], xg.shape[0]), dtype=np.float32)
    for i in range(y.shape[0]):
        y_interp[i] = np.interp(xg, x_layer, y[i].astype(np.float64))
        theta_interp[i] = np.interp(xg, x_layer, theta_vis[i].astype(np.float64))

    # Rendering subset (animation); static uses all.
    k = min(int(sample_points), int(y_interp.shape[0]))
    idx_anim = rng.choice(np.arange(y_interp.shape[0]), size=k, replace=False) if k < y_interp.shape[0] else np.arange(y_interp.shape[0])
    y_anim = y_interp[idx_anim]
    theta_anim = theta_interp[idx_anim]

    theta0 = theta_anim[:, 0]
    hue0 = (theta0 + np.pi) / (2.0 * np.pi)
    cmap = plt.get_cmap("turbo")
    line_colors = cmap(hue0)
    line_colors[:, 3] = 0.085

    return {
        "xg": xg,
        "y_all": y_interp,
        "y_anim": y_anim,
        "theta_anim": theta_anim,
        "line_colors": line_colors,
        "target_choice": target_choice,
        "target_count": int(pool.size),
        "n_layers": int(n_layers),
    }


def _make_horizontal_vortex_paths_art(
    model_data: Dict[str, Dict[str, Any]],
    out_png: Path,
    out_gif: Path,
    seed: int,
    sample_points_per_model: int = 850,
    interp_steps: int = 5,
    fps: int = 7,
) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    for i, (model_id, d) in enumerate(model_data.items()):
        bundles[model_id] = _prepare_horizontal_vortex_paths(
            d=d,
            seed=seed + 700 + i,
            sample_points=sample_points_per_model,
            interp_steps=interp_steps,
        )

    # Static figure (all rows from each model).
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(5.7 * len(bundles), 4.4), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        xg = b["xg"]
        y_all = b["y_all"]
        q10 = np.quantile(y_all, 0.10, axis=0)
        q90 = np.quantile(y_all, 0.90, axis=0)
        q25 = np.quantile(y_all, 0.25, axis=0)
        q75 = np.quantile(y_all, 0.75, axis=0)

        ax.fill_between(xg, q10, q90, color="#293749", alpha=0.08, linewidth=0)
        ax.fill_between(xg, q25, q75, color="#293749", alpha=0.12, linewidth=0)

        for tr in y_all:
            ax.plot(xg, tr, color="#0f1520", alpha=0.020, lw=0.45)

        ax.scatter([1.0], [0.0], s=40, c="#000000", zorder=5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
        ax.text(0.01, 0.08, f"n={int(y_all.shape[0])} trajectories", transform=ax.transAxes, va="bottom", ha="left", fontsize=9, color="#3a3a3a")
        ax.text(0.00, -0.03, "early layers", transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#4a4a4a")
        ax.text(1.00, -0.03, "late layers", transform=ax.transAxes, va="top", ha="right", fontsize=9, color="#4a4a4a")

    fig.suptitle("Left-to-Right Convergence Funnel (Path-Mapped)", fontsize=16, y=1.02)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Animated reveal.
    fig, axs = plt.subplots(1, len(bundles), figsize=(5.7 * len(bundles), 4.6), constrained_layout=True)
    if len(bundles) == 1:
        axs = [axs]
    n_frames = max(b["xg"].shape[0] for b in bundles.values())

    def _update(frame: int) -> None:
        for ax, model_id in zip(axs, bundles.keys()):
            b = bundles[model_id]
            xg = b["xg"]
            ya = b["y_anim"]
            li = min(frame, xg.shape[0] - 1)
            x = xg[: li + 1]

            ax.cla()
            ax.set_facecolor("#f8f5ef")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-1.25, 1.25)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            # Envelope from all trajectories, clipped to revealed x.
            y_all = b["y_all"][:, : li + 1]
            q10 = np.quantile(y_all, 0.10, axis=0)
            q90 = np.quantile(y_all, 0.90, axis=0)
            q25 = np.quantile(y_all, 0.25, axis=0)
            q75 = np.quantile(y_all, 0.75, axis=0)
            ax.fill_between(x, q10, q90, color="#293749", alpha=0.08, linewidth=0)
            ax.fill_between(x, q25, q75, color="#293749", alpha=0.12, linewidth=0)

            for ti in range(ya.shape[0]):
                ax.plot(x, ya[ti, : li + 1], color=b["line_colors"][ti], lw=0.45)

            heads = ya[:, li]
            theta_now = b["theta_anim"][:, li]
            hue = np.mod(theta_now, 2.0 * np.pi) / (2.0 * np.pi)
            head_colors = plt.get_cmap("turbo")(hue)
            head_colors[:, 3] = 0.82
            ax.scatter(np.full((heads.shape[0],), xg[li]), heads, s=8, c=head_colors, linewidths=0, zorder=4)
            ax.scatter([1.0], [0.0], s=40, c="#000000", zorder=5)
            ax.axvline(x=xg[li], color="#445166", alpha=0.18, lw=0.9)

            collapse = 1.0 - float(np.mean(np.abs(heads)) / (np.mean(np.abs(ya[:, 0])) + 1e-12))
            ax.text(0.01, 0.96, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=12)
            ax.text(
                0.01,
                0.08,
                f"collapse {100.0 * collapse:.1f}%  |  shown n={ya.shape[0]}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                color="#333333",
            )

        fig.suptitle("Convergence Moves Left -> Right Through Layer Depth", fontsize=16, y=1.01)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=int(round(1000 / max(1, fps))))
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    metrics: Dict[str, Dict[str, Any]] = {}
    for model_id, b in bundles.items():
        y_all = b["y_all"]
        collapse = 1.0 - float(np.mean(np.abs(y_all[:, -1])) / (np.mean(np.abs(y_all[:, 0])) + 1e-12))
        metrics[model_id] = {
            "target_choice": str(b["target_choice"]),
            "target_choice_count": int(b["target_count"]),
            "all_trajectory_count": int(y_all.shape[0]),
            "animated_trajectory_count": int(b["y_anim"].shape[0]),
            "horizontal_collapse_fraction": float(collapse),
        }
    return metrics


def _plot_topology_constellation(run_analysis_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(run_analysis_dir / "domain_topology_centroids.csv")
    if df.empty:
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    models = list(dict.fromkeys(df["model_id"].tolist()))
    fig, axs = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), constrained_layout=True)
    if len(models) == 1:
        axs = [axs]

    all_domains = sorted(df["coarse_domain"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    dom_color = {d: cmap(i % 20) for i, d in enumerate(all_domains)}

    for ax, model_id in zip(axs, models):
        sub = df[df["model_id"] == model_id].copy()
        for _, r in sub.iterrows():
            dom = str(r["coarse_domain"])
            ax.scatter(float(r["pc1"]), float(r["pc2"]), s=70, color=dom_color[dom], alpha=0.90)
            ax.text(float(r["pc1"]), float(r["pc2"]), dom.replace("_", " "), fontsize=8, alpha=0.90)
        ax.set_title(_model_label(model_id))
        ax.set_xlabel("Centroid PC1")
        ax.set_ylabel("Centroid PC2")
        ax.grid(alpha=0.18, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Domain Topology Constellation (Final Layer Centroids)", fontsize=16, y=1.05)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_tetra_wire(ax: Any, vertices: np.ndarray) -> None:
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        p, q = vertices[i], vertices[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#888888", lw=1.0, alpha=0.85)
    for i, c in enumerate(CHOICES):
        v = vertices[i]
        ax.text(v[0], v[1], v[2], c, color=CHOICE_COLORS[c], fontsize=10, weight="bold")


def _make_simplex_gif(
    model_data: Dict[str, Dict[str, Any]],
    out_path: Path,
    sample_points_per_model: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    verts = np.array(
        [
            [1.0, 1.0, 1.0],     # A
            [1.0, -1.0, -1.0],   # B
            [-1.0, 1.0, -1.0],   # C
            [-1.0, -1.0, 1.0],   # D
        ],
        dtype=np.float64,
    ) / math.sqrt(3.0)

    model_ids = list(model_data.keys())
    sampled: Dict[str, Dict[str, Any]] = {}
    max_layers = 0
    for model_id in model_ids:
        d = model_data[model_id]
        probs = d["probs"]
        top = d["top"]
        n_rows, n_layers, _ = probs.shape
        max_layers = max(max_layers, n_layers)
        k = min(sample_points_per_model, n_rows)
        idx = rng.choice(n_rows, size=k, replace=False)
        probs_s = probs[idx]
        coords = np.einsum("nlk,kd->nld", probs_s, verts).astype(np.float32)
        final_choice = top[idx, -1]
        sampled[model_id] = {
            "coords": coords,  # [k, L, 3]
            "final_choice": final_choice,
            "n_layers": n_layers,
            "agreement": d["agreement_with_final"],
            "convergence": d["convergence"].mean(axis=0),
        }

    plt.rcParams.update({"font.family": "DejaVu Serif"})
    fig = plt.figure(figsize=(15, 5), facecolor="#f8f5ef")
    axes = [fig.add_subplot(1, len(model_ids), i + 1, projection="3d") for i in range(len(model_ids))]

    def _update(frame: int) -> None:
        for ax, model_id in zip(axes, model_ids):
            s = sampled[model_id]
            n_layers = int(s["n_layers"])
            li = min(frame, n_layers - 1)
            pts = s["coords"][:, li, :]
            final_choice = s["final_choice"]
            colors = [CHOICE_COLORS[CHOICES[int(i)]] for i in final_choice]

            ax.cla()
            _draw_tetra_wire(ax, verts)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=5, alpha=0.24, linewidths=0)
            ax.set_xlim(-0.72, 0.72)
            ax.set_ylim(-0.72, 0.72)
            ax.set_zlim(-0.72, 0.72)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=35 + frame * 1.6)

            align = float(s["agreement"][li])
            conv = float(s["convergence"][li])
            ax.set_title(
                f"{_model_label(model_id)}\nLayer {li + 1}/{n_layers}  |  align={align:.2f}  conv={conv:.2f}",
                fontsize=10,
            )

        fig.suptitle(
            "Convergence in the 4-Choice Belief Simplex (color = final chosen option)",
            fontsize=16,
            y=0.98,
        )

    anim = FuncAnimation(fig, _update, frames=max_layers, interval=220)
    anim.save(out_path, writer=PillowWriter(fps=4))
    plt.close(fig)


def _make_simplex_paths_art(
    model_data: Dict[str, Dict[str, Any]],
    out_png: Path,
    out_gif: Path,
) -> None:
    def _smooth_track(tr: np.ndarray, window: int = 5) -> np.ndarray:
        if tr.shape[0] < 3:
            return tr.copy()
        w = int(max(3, min(window, tr.shape[0] - (1 - tr.shape[0] % 2))))
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return tr.copy()
        k = np.ones((w,), dtype=np.float64) / float(w)
        out = np.empty_like(tr, dtype=np.float64)
        for d in range(tr.shape[1]):
            out[:, d] = np.convolve(tr[:, d], k, mode="same")
        out[0] = tr[0]
        out[-1] = tr[-1]
        return out.astype(np.float32)

    verts = np.array(
        [
            [1.0, 1.0, 1.0],     # A
            [1.0, -1.0, -1.0],   # B
            [-1.0, 1.0, -1.0],   # C
            [-1.0, -1.0, 1.0],   # D
        ],
        dtype=np.float64,
    ) / math.sqrt(3.0)

    model_ids = list(model_data.keys())
    tracks: Dict[str, Dict[str, np.ndarray]] = {}
    max_layers = 0
    for model_id in model_ids:
        d = model_data[model_id]
        probs = d["probs"]  # [N,L,4]
        top_final = d["top"][:, -1]
        L = int(d["n_layers"])
        max_layers = max(max_layers, L)
        tracks[model_id] = {}
        for ci, c in enumerate(CHOICES):
            m = top_final == ci
            if not np.any(m):
                continue
            mean_probs = probs[m].mean(axis=0)  # [L,4]
            coords = np.einsum("lk,kd->ld", mean_probs, verts).astype(np.float32)  # [L,3]
            coords = _smooth_track(coords, window=5)
            tracks[model_id][c] = coords

    plt.rcParams.update({"font.family": "DejaVu Serif"})
    fig = plt.figure(figsize=(15, 5), facecolor="#f8f5ef")
    axes = [fig.add_subplot(1, len(model_ids), i + 1, projection="3d") for i in range(len(model_ids))]

    for ax, model_id in zip(axes, model_ids):
        _draw_tetra_wire(ax, verts)
        for c in CHOICES:
            if c not in tracks[model_id]:
                continue
            tr = tracks[model_id][c]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=CHOICE_COLORS[c], lw=2.7, alpha=0.95)
            ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], color=CHOICE_COLORS[c], s=35, alpha=0.70, marker="o")
            ax.scatter(tr[-1, 0], tr[-1, 1], tr[-1, 2], color=CHOICE_COLORS[c], s=55, alpha=0.95, marker="*")
        ax.set_xlim(-0.72, 0.72)
        ax.set_ylim(-0.72, 0.72)
        ax.set_zlim(-0.72, 0.72)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=17, azim=38)
        ax.set_title(_model_label(model_id), fontsize=12)

    fig.suptitle("Choice-Attractor Trajectories Through Layers (start=o, end=*)", fontsize=16, y=0.98)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5), facecolor="#f8f5ef")
    axes = [fig.add_subplot(1, len(model_ids), i + 1, projection="3d") for i in range(len(model_ids))]

    def _update(frame: int) -> None:
        for ax, model_id in zip(axes, model_ids):
            ax.cla()
            _draw_tetra_wire(ax, verts)
            L = int(model_data[model_id]["n_layers"])
            li = min(frame, L - 1)
            for c in CHOICES:
                if c not in tracks[model_id]:
                    continue
                tr = tracks[model_id][c]
                ax.plot(tr[: li + 1, 0], tr[: li + 1, 1], tr[: li + 1, 2], color=CHOICE_COLORS[c], lw=2.6, alpha=0.95)
                ax.scatter(tr[li, 0], tr[li, 1], tr[li, 2], color=CHOICE_COLORS[c], s=38, alpha=0.98, marker="o")
            ax.set_xlim(-0.72, 0.72)
            ax.set_ylim(-0.72, 0.72)
            ax.set_zlim(-0.72, 0.72)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=17, azim=38 + frame * 1.4)
            ax.set_title(f"{_model_label(model_id)}  |  layer {li + 1}/{L}", fontsize=11)
        fig.suptitle("Convergence As Choice-Attractor Trajectories", fontsize=16, y=0.98)

    anim = FuncAnimation(fig, _update, frames=max_layers, interval=220)
    anim.save(out_gif, writer=PillowWriter(fps=4))
    plt.close(fig)


def _write_summary_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Deep Baseline Analysis Summary (run 1452_a)")
    lines.append("")
    lines.append("## Convergence Evidence")
    lines.append("")
    for model_id, s in summary["models"].items():
        lines.append(f"### {_model_label(model_id)}")
        lines.append(
            f"- Convergence index rose from `{s['convergence_start_mean']:.3f}` to `{s['convergence_end_mean']:.3f}` "
            f"(delta `{s['convergence_delta_end_minus_start']:+.3f}`)."
        )
        lines.append(
            f"- Entropy fell from `{s['entropy_start_mean']:.3f}` to `{s['entropy_end_mean']:.3f}` "
            f"(delta `{s['entropy_delta_end_minus_start']:+.3f}`, paired t-test p=`{s['entropy_ttest_pvalue']:.2e}`)."
        )
        lines.append(
            f"- Agreement with final winner rose from `{s['agreement_with_final_start']:.3f}` "
            f"to `{s['agreement_with_final_end']:.3f}` (gain `{s['agreement_with_final_gain']:+.3f}`)."
        )
        lines.append(
            f"- Fraction of prompts with decreasing entropy trend: `{100.0 * s['negative_entropy_slope_fraction']:.1f}%`."
        )
        lines.append("")
    lines.append("## Commitment")
    lines.append("")
    for model_id, s in summary["models"].items():
        lines.append(f"### {_model_label(model_id)}")
        lines.append(
            f"- Commit-detected fraction (threshold 0.1): `{100.0 * s['commitment_valid_fraction']:.1f}%` "
            f"(median layer `{s['commitment_layer_median']:.1f}`, P10 `{s['commitment_layer_p10']:.1f}`, "
            f"P90 `{s['commitment_layer_p90']:.1f}`)."
        )
        lines.append(
            f"- Mean flip count across layers: `{s['flip_count_mean']:.2f}` (median `{s['flip_count_median']:.1f}`)."
        )
        lines.append("")
    lines.append("## Domain Topology")
    lines.append("")
    for model_id, s in summary["topology"].items():
        lines.append(f"### {_model_label(model_id)}")
        lines.append(
            f"- Silhouette (euclidean): `{s['silhouette_euclidean']:.3f}`; nearest-centroid domain accuracy: "
            f"`{s['nearest_centroid_accuracy']:.3f}`."
        )
        lines.append(
            f"- Between/within separation ratio: `{s['between_within_ratio']:.3f}`; "
            f"centroid pairwise distance min/mean/max = `{s['min_centroid_pairwise_distance']:.2f}` / "
            f"`{s['mean_centroid_pairwise_distance']:.2f}` / `{s['max_centroid_pairwise_distance']:.2f}`."
        )
        lines.append("")
    lv = summary.get("latent_vortex") or {}
    if isinstance(lv, dict) and lv:
        lines.append("## Latent Convergence Vortex")
        lines.append("")
        for model_id, s in lv.items():
            lines.append(f"### {_model_label(model_id)}")
            lines.append(
                f"- Majority final-answer basin: `{s['target_choice']}` (`n={s['target_choice_count']}`)."
            )
            if "mean_radius_start" in s and "mean_radius_end" in s and "radial_collapse_fraction" in s:
                lines.append(
                    f"- Mean radius to final attractor dropped from `{s['mean_radius_start']:.3f}` "
                    f"to `{s['mean_radius_end']:.3f}` (collapse `{100.0 * s['radial_collapse_fraction']:.1f}%`)."
                )
            elif "collapse_end" in s:
                lines.append(
                    f"- Normalized radial collapse reached `{100.0 * float(s['collapse_end']):.1f}%`."
                )
            if "sign_coherence_raw" in s:
                lines.append(
                    f"- Raw-trajectory turn coherence: `{float(s['sign_coherence_raw']):.3f}`; "
                    f"fraction with large net turn: `{float(s['fraction_large_turn_raw']):.3f}`."
                )
            lines.append("")
    hv = summary.get("horizontal_vortex") or {}
    if isinstance(hv, dict) and hv:
        lines.append("## Horizontal Vortex Paths")
        lines.append("")
        for model_id, s in hv.items():
            lines.append(f"### {_model_label(model_id)}")
            lines.append(
                f"- Basin `{s['target_choice']}` with `{s['all_trajectory_count']}` total trajectories "
                f"(`{s['animated_trajectory_count']}` shown in animation)."
            )
            lines.append(
                f"- Left-to-right horizontal collapse: `{100.0 * float(s['horizontal_collapse_fraction']):.1f}%`."
            )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to run directory (rtx6000ada_baseline_20260216_1452_a)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <run-dir>/analysis/deep)")
    ap.add_argument("--commitment-threshold", type=float, default=0.1)
    ap.add_argument("--sample-points-per-model", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "analysis" / "deep")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = run_dir / "outputs"
    model_paths = sorted(outputs_dir.glob("*/baseline_outputs.jsonl"))
    if not model_paths:
        raise RuntimeError(f"No baseline outputs found under {outputs_dir}")

    model_data: Dict[str, Dict[str, Any]] = {}
    summary_models: Dict[str, Dict[str, Any]] = {}
    summary_topology: Dict[str, Dict[str, Any]] = {}

    layer_rows = []
    for p in model_paths:
        d = _load_model_output(p, commitment_threshold=float(args.commitment_threshold))
        model_id = str(d["model_id"])
        model_data[model_id] = d
        s = _convergence_metrics(d)
        summary_models[model_id] = s
        t = _topology_metrics(d, seed=int(args.seed))
        summary_topology[model_id] = t

        L = int(d["n_layers"])
        conv_mean = d["convergence"].mean(axis=0)
        align = d["agreement_with_final"]
        commit = d["commitment_layer"]
        committed = np.array([np.mean(np.isfinite(commit) & (commit <= li)) for li in range(L)], dtype=np.float64)
        for li in range(L):
            layer_rows.append(
                {
                    "model_id": model_id,
                    "model_label": _model_label(model_id),
                    "layer_index": li,
                    "n_layers": L,
                    "normalized_depth": float(li / max(L - 1, 1)),
                    "convergence_mean": float(conv_mean[li]),
                    "agreement_with_final": float(align[li]),
                    "commitment_fraction": float(committed[li]),
                }
            )

    layer_df = pd.DataFrame(layer_rows).sort_values(["model_id", "layer_index"])
    layer_df.to_csv(out_dir / "convergence_layer_metrics.csv", index=False)

    topo_df = pd.DataFrame(
        [
            {"model_id": m, "model_label": _model_label(m), **vals}
            for m, vals in summary_topology.items()
        ]
    )
    topo_df.to_csv(out_dir / "topology_metrics.csv", index=False)

    _plot_overview(model_data, out_dir / "beautiful_convergence_overview.png")
    _make_convergence_lines_gif(model_data, out_dir / "beautiful_convergence_lines.gif")
    _plot_topology_constellation(run_dir / "analysis", out_dir / "beautiful_topology_constellation.png")
    _make_simplex_gif(
        model_data=model_data,
        out_path=out_dir / "beautiful_convergence_simplex.gif",
        sample_points_per_model=int(args.sample_points_per_model),
        seed=int(args.seed),
    )
    _make_simplex_paths_art(
        model_data=model_data,
        out_png=out_dir / "beautiful_convergence_paths.png",
        out_gif=out_dir / "beautiful_convergence_paths.gif",
    )
    horizontal_vortex_metrics = _make_horizontal_vortex_paths_art(
        model_data=model_data,
        out_png=out_dir / "beautiful_horizontal_vortex_paths.png",
        out_gif=out_dir / "beautiful_horizontal_vortex_paths.gif",
        seed=int(args.seed),
        sample_points_per_model=max(700, int(args.sample_points_per_model)),
        interp_steps=5,
        fps=7,
    )
    per_model_horizontal_outputs: Dict[str, Dict[str, str]] = {}
    for i, model_id in enumerate(model_data.keys()):
        slug = model_id.replace("/", "__").replace(".", "-")
        out_png = out_dir / f"beautiful_horizontal_vortex_paths_{slug}.png"
        out_gif = out_dir / f"beautiful_horizontal_vortex_paths_{slug}.gif"
        _make_horizontal_vortex_paths_art(
            model_data={model_id: model_data[model_id]},
            out_png=out_png,
            out_gif=out_gif,
            seed=int(args.seed) + 900 + i,
            sample_points_per_model=max(700, int(args.sample_points_per_model)),
            interp_steps=5,
            fps=7,
        )
        per_model_horizontal_outputs[model_id] = {
            "png": str(out_png),
            "gif": str(out_gif),
        }
    vortex_metrics = _make_latent_vortex_clean_gif(
        model_data=model_data,
        out_path=out_dir / "beautiful_latent_vortex_clean.gif",
        sample_points_per_model=int(args.sample_points_per_model),
        seed=int(args.seed),
    )
    per_model_vortex_outputs: Dict[str, str] = {}
    for i, model_id in enumerate(model_data.keys()):
        slug = model_id.replace("/", "__").replace(".", "-")
        out_gif = out_dir / f"beautiful_latent_vortex_clean_{slug}.gif"
        _make_latent_vortex_clean_gif(
            model_data={model_id: model_data[model_id]},
            out_path=out_gif,
            sample_points_per_model=int(args.sample_points_per_model),
            seed=int(args.seed) + i + 1,
        )
        per_model_vortex_outputs[model_id] = str(out_gif)

    summary = {
        "run_dir": str(run_dir),
        "commitment_threshold": float(args.commitment_threshold),
        "models": summary_models,
        "topology": summary_topology,
        "horizontal_vortex": horizontal_vortex_metrics,
        "latent_vortex": vortex_metrics,
        "outputs": {
            "convergence_layer_metrics_csv": str(out_dir / "convergence_layer_metrics.csv"),
            "topology_metrics_csv": str(out_dir / "topology_metrics.csv"),
            "beautiful_convergence_overview_png": str(out_dir / "beautiful_convergence_overview.png"),
            "beautiful_topology_constellation_png": str(out_dir / "beautiful_topology_constellation.png"),
            "beautiful_convergence_simplex_gif": str(out_dir / "beautiful_convergence_simplex.gif"),
            "beautiful_convergence_lines_gif": str(out_dir / "beautiful_convergence_lines.gif"),
            "beautiful_convergence_paths_png": str(out_dir / "beautiful_convergence_paths.png"),
            "beautiful_convergence_paths_gif": str(out_dir / "beautiful_convergence_paths.gif"),
            "beautiful_horizontal_vortex_paths_png": str(out_dir / "beautiful_horizontal_vortex_paths.png"),
            "beautiful_horizontal_vortex_paths_gif": str(out_dir / "beautiful_horizontal_vortex_paths.gif"),
            "beautiful_horizontal_vortex_paths_per_model": per_model_horizontal_outputs,
            "beautiful_latent_vortex_clean_gif": str(out_dir / "beautiful_latent_vortex_clean.gif"),
            "beautiful_latent_vortex_clean_per_model": per_model_vortex_outputs,
        },
    }
    (out_dir / "deep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary, out_dir / "deep_summary.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
