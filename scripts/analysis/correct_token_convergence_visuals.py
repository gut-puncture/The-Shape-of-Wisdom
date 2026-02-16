#!/usr/bin/env python3
"""
Correct-token convergence visuals for baseline runs.

Data definition (rigorous):
- Per prompt, per layer, use candidate_probs[correct_key] from baseline manifest.
- Convergence target is the correct option token probability = 1.0.
- Plot distance to target as y = 1 - p_correct (lower is better).
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
import numpy as np


MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
}


def _model_label(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def _gaussian_smooth_rows(y: np.ndarray, sigma: float) -> np.ndarray:
    s = float(sigma)
    if (not math.isfinite(s)) or s <= 0.0:
        return y
    radius = max(1, int(math.ceil(3.0 * s)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / s) * (t / s))
    k /= float(k.sum())
    ypad = np.pad(y, ((0, 0), (radius, radius)), mode="edge")
    ys = np.apply_along_axis(lambda z: np.convolve(z, k, mode="valid"), axis=1, arr=ypad)
    return ys.astype(np.float64, copy=False)


def _load_correct_key_map(manifest_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r.get("prompt_uid") or "")
            ck = str(r.get("correct_key") or "").strip().upper()
            if uid and ck in ("A", "B", "C", "D"):
                out[uid] = ck
    if not out:
        raise RuntimeError(f"no prompt_uid->correct_key mapping in {manifest_path}")
    return out


def _load_model_rows(path: Path, correct_key_by_uid: Dict[str, str]) -> Dict[str, Any]:
    p_corr_rows: List[np.ndarray] = []
    margin_rows: List[np.ndarray] = []
    is_correct_rows: List[bool] = []
    model_id: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r.get("prompt_uid") or "")
            ck = correct_key_by_uid.get(uid)
            if ck is None:
                continue
            lw = r.get("layerwise") or []
            if not isinstance(lw, list) or not lw:
                continue

            pc: List[float] = []
            mg: List[float] = []
            ok = True
            for layer in lw:
                cp = layer.get("candidate_probs") or {}
                if not isinstance(cp, dict):
                    ok = False
                    break
                probs = {k: float(cp.get(k, 0.0)) for k in ("A", "B", "C", "D")}
                p = float(probs[ck])
                others = [probs[k] for k in ("A", "B", "C", "D") if k != ck]
                m = p - float(max(others))
                if (not math.isfinite(p)) or (not math.isfinite(m)):
                    ok = False
                    break
                pc.append(p)
                mg.append(m)
            if not ok:
                continue

            if model_id is None:
                model_id = str(r.get("model_id") or path.parent.name)
            p_corr_rows.append(np.asarray(pc, dtype=np.float64))
            margin_rows.append(np.asarray(mg, dtype=np.float64))
            is_correct_rows.append(bool(r.get("is_correct") is True))

    if not p_corr_rows:
        raise RuntimeError(f"no valid rows in {path}")

    P = np.stack(p_corr_rows, axis=0)
    M = np.stack(margin_rows, axis=0)
    C = np.asarray(is_correct_rows, dtype=bool)
    return {
        "model_id": model_id or path.parent.name,
        "p_correct": P,  # [N,L]
        "margin_correct_vs_other": M,  # [N,L]
        "is_correct": C,  # [N]
    }


def _earliest_commit_layer(
    p_correct: np.ndarray,
    margin: np.ndarray,
    *,
    p_thresh: float,
    margin_thresh: float,
) -> np.ndarray:
    # Earliest layer where both conditions hold and remain true through final layer.
    good = (p_correct >= float(p_thresh)) & (margin >= float(margin_thresh))
    # suffix_all[i, l] = all(good[i, l:])
    suffix_all = np.flip(np.cumprod(np.flip(good.astype(np.int8), axis=1), axis=1), axis=1).astype(bool)
    out = np.full((good.shape[0],), np.nan, dtype=np.float64)
    for i in range(good.shape[0]):
        idx = np.where(suffix_all[i])[0]
        if idx.size > 0:
            out[i] = float(idx[0])
    return out


def _prepare_bundle(
    d: Dict[str, Any],
    *,
    interp_steps: int,
    smooth_sigma: float,
    x_left: float,
    x_right: float,
    commit_p: float,
    commit_margin: float,
) -> Dict[str, Any]:
    P = d["p_correct"]  # [N,L]
    M = d["margin_correct_vs_other"]  # [N,L]
    C = d["is_correct"]  # [N]
    N, L = P.shape

    x = np.linspace(0.0, 1.0, num=L, dtype=np.float64)
    xg = np.linspace(0.0, 1.0, num=(L - 1) * int(interp_steps) + 1, dtype=np.float64)
    x_plot = float(x_left) + (float(x_right) - float(x_left)) * xg

    # y = distance to correct token target (1.0).
    Y = 1.0 - np.clip(P, 0.0, 1.0)
    Yi = np.empty((N, xg.size), dtype=np.float64)
    for i in range(N):
        Yi[i] = np.interp(xg, x, Y[i])
    Yi_display = _gaussian_smooth_rows(Yi, float(smooth_sigma))
    Yi_display = np.clip(Yi_display, 0.0, 1.0)

    commit_layer = _earliest_commit_layer(P, M, p_thresh=float(commit_p), margin_thresh=float(commit_margin))
    commit_norm = commit_layer / float(max(1, L - 1))

    frac_correct = float(np.mean(C))
    frac_strong_corr = float(np.mean(P[:, -1] >= 0.8))
    frac_nonconv = float(np.mean(P[:, -1] < 0.6))

    return {
        "model_id": d["model_id"],
        "x": x,
        "xg": xg,
        "x_plot": x_plot,
        "p_correct": P,
        "y_dist_raw": Y,
        "y_dist_interp": Yi_display.astype(np.float32),
        "is_correct": C,
        "commit_layer": commit_layer,
        "commit_norm": commit_norm,
        "n_rows": int(N),
        "n_layers": int(L),
        "accuracy": frac_correct,
        "strong_correct_prob_final": frac_strong_corr,
        "nonconverged_final_under_0_6": frac_nonconv,
    }


def _plot_sideways_convergence(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_lines_per_group: int,
    title: str,
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
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.2 * len(bundles), 5.2), constrained_layout=False)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        x = b["x_plot"]
        y = b["y_dist_interp"]
        corr = b["is_correct"]

        idx_corr = np.where(corr)[0]
        idx_inc = np.where(~corr)[0]
        kc = min(int(max_lines_per_group), int(idx_corr.size))
        ki = min(int(max_lines_per_group), int(idx_inc.size))
        pick_corr = rng.choice(idx_corr, size=kc, replace=False) if kc < idx_corr.size else idx_corr
        pick_inc = rng.choice(idx_inc, size=ki, replace=False) if ki < idx_inc.size else idx_inc

        # Quantile ribbons for both populations.
        for ids, col, a10, a25 in [
            (idx_inc, "#8a6b5a", 0.08, 0.12),
            (idx_corr, "#244064", 0.09, 0.14),
        ]:
            if ids.size > 0:
                q10 = np.quantile(y[ids], 0.10, axis=0)
                q90 = np.quantile(y[ids], 0.90, axis=0)
                q25 = np.quantile(y[ids], 0.25, axis=0)
                q75 = np.quantile(y[ids], 0.75, axis=0)
                ax.fill_between(x, q10, q90, color=col, alpha=a10, lw=0)
                ax.fill_between(x, q25, q75, color=col, alpha=a25, lw=0)

        for tr in y[pick_inc]:
            ax.plot(x, tr, color=(0.45, 0.28, 0.21, 0.06), lw=0.52)
        for tr in y[pick_corr]:
            ax.plot(x, tr, color=(0.06, 0.15, 0.31, 0.07), lw=0.58)

        # Median tracks.
        if idx_inc.size > 0:
            ax.plot(x, np.median(y[idx_inc], axis=0), color="#6f4f3f", lw=1.15, alpha=0.9)
        if idx_corr.size > 0:
            ax.plot(x, np.median(y[idx_corr], axis=0), color="#0d2542", lw=1.20, alpha=0.95)

        # Correct-token target point (distance=0).
        x_end = float(x[-1])
        ax.axhline(0.0, color="#2f2f2f", lw=0.9, alpha=0.45)
        ax.scatter([x_end], [0.0], s=36, c="#111111", zorder=6)

        # Final-layer scatter to show non-convergence spread.
        fy_corr = y[idx_corr, -1] if idx_corr.size else np.array([])
        fy_inc = y[idx_inc, -1] if idx_inc.size else np.array([])
        if fy_inc.size:
            ax.scatter(np.full((fy_inc.size,), x_end), fy_inc, s=5, c="#6f4f3f", alpha=0.28, linewidths=0, zorder=5)
        if fy_corr.size:
            ax.scatter(np.full((fy_corr.size,), x_end), fy_corr, s=5, c="#0d2542", alpha=0.28, linewidths=0, zorder=5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.02, -0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.95, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=13)
        ax.text(
            0.01,
            0.07,
            f"acc={100.0*b['accuracy']:.1f}% | final p(correct)>=0.8: {100.0*b['strong_correct_prob_final']:.1f}% | non-converged(p<0.6): {100.0*b['nonconverged_final_under_0_6']:.1f}%",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#3a3a3a",
        )
        ax.text(0.00, -0.04, "early layers", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#4c4c4c")
        ax.text(1.00, -0.04, "late layers", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="#4c4c4c")
        ax.text(0.985, 0.975, "target: correct token", transform=ax.transAxes, va="top", ha="right", fontsize=9, color="#262626")

    fig.subplots_adjust(top=0.84, left=0.03, right=0.995, wspace=0.02)
    fig.suptitle(title, fontsize=20, y=0.975)
    fig.savefig(out_path, dpi=230, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _plot_sideways_clean_convergence(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    max_texture_per_group: int,
    title: str,
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
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.2 * len(bundles), 4.9), constrained_layout=False)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        x = b["x_plot"]
        p = 1.0 - b["y_dist_interp"]  # probability on correct token
        corr = b["is_correct"]
        idx_corr = np.where(corr)[0]
        idx_inc = np.where(~corr)[0]

        # Quantile bands
        if idx_inc.size:
            q10 = np.quantile(p[idx_inc], 0.10, axis=0)
            q90 = np.quantile(p[idx_inc], 0.90, axis=0)
            q25 = np.quantile(p[idx_inc], 0.25, axis=0)
            q75 = np.quantile(p[idx_inc], 0.75, axis=0)
            ax.fill_between(x, q10, q90, color="#8a6b5a", alpha=0.10, lw=0)
            ax.fill_between(x, q25, q75, color="#8a6b5a", alpha=0.16, lw=0)
        if idx_corr.size:
            q10 = np.quantile(p[idx_corr], 0.10, axis=0)
            q90 = np.quantile(p[idx_corr], 0.90, axis=0)
            q25 = np.quantile(p[idx_corr], 0.25, axis=0)
            q75 = np.quantile(p[idx_corr], 0.75, axis=0)
            ax.fill_between(x, q10, q90, color="#244064", alpha=0.11, lw=0)
            ax.fill_between(x, q25, q75, color="#244064", alpha=0.18, lw=0)

        # Light texture trajectories.
        for ids, col in [(idx_inc, (0.45, 0.28, 0.21, 0.030)), (idx_corr, (0.06, 0.15, 0.31, 0.032))]:
            if ids.size == 0:
                continue
            k = min(int(max_texture_per_group), int(ids.size))
            pick = rng.choice(ids, size=k, replace=False) if k < ids.size else ids
            for tr in p[pick]:
                ax.plot(x, tr, color=col, lw=0.50)

        # Median curves.
        if idx_inc.size:
            ax.plot(x, np.median(p[idx_inc], axis=0), color="#6f4f3f", lw=1.35, alpha=0.95, label="final incorrect")
        if idx_corr.size:
            ax.plot(x, np.median(p[idx_corr], axis=0), color="#0d2542", lw=1.45, alpha=0.98, label="final correct")

        x_end = float(x[-1])
        # Final scatter (all prompts) to show converged vs non-converged spread.
        fy_inc = p[idx_inc, -1] if idx_inc.size else np.array([])
        fy_corr = p[idx_corr, -1] if idx_corr.size else np.array([])
        if fy_inc.size:
            ax.scatter(np.full((fy_inc.size,), x_end), fy_inc, s=5, c="#6f4f3f", alpha=0.26, linewidths=0, zorder=5)
        if fy_corr.size:
            ax.scatter(np.full((fy_corr.size,), x_end), fy_corr, s=5, c="#0d2542", alpha=0.26, linewidths=0, zorder=5)

        # Target point: correct token probability = 1.
        ax.scatter([x_end], [1.0], s=36, c="#111111", zorder=7)
        ax.axhline(1.0, color="#2f2f2f", lw=0.85, alpha=0.40)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.01, 0.95, _model_label(model_id), transform=ax.transAxes, va="top", ha="left", fontsize=13)
        ax.text(
            0.01,
            0.07,
            f"acc={100.0*b['accuracy']:.1f}% | final p(correct)>=0.8: {100.0*b['strong_correct_prob_final']:.1f}% | non-converged(p<0.6): {100.0*b['nonconverged_final_under_0_6']:.1f}%",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#3a3a3a",
        )
        ax.text(0.00, -0.05, "early layers", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#4c4c4c")
        ax.text(1.00, -0.05, "late layers", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="#4c4c4c")
        ax.text(0.99, 0.96, "target: correct token", transform=ax.transAxes, va="top", ha="right", fontsize=9, color="#262626")

    fig.subplots_adjust(top=0.84, left=0.03, right=0.995, wspace=0.02)
    fig.suptitle(title, fontsize=20, y=0.975)
    fig.savefig(out_path, dpi=230, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _plot_commitment_timing(
    bundles: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    commit_p: float,
    commit_margin: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f8f5ef",
            "axes.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
        }
    )
    fig, axs = plt.subplots(1, len(bundles), figsize=(6.2 * len(bundles), 4.2), constrained_layout=False)
    if len(bundles) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, bundles.keys()):
        b = bundles[model_id]
        corr = b["is_correct"]
        cm = b["commit_norm"]  # NaN if never commits under criterion
        x = np.linspace(0.0, 1.0, 220, dtype=np.float64)

        def cdf(mask: np.ndarray) -> np.ndarray:
            v = cm[mask]
            out = np.zeros_like(x)
            if v.size == 0:
                return out
            for i, t in enumerate(x):
                out[i] = float(np.mean(v <= t))
            return out

        yc = cdf(corr)
        yi = cdf(~corr)

        ax.plot(x, yc, color="#0d2542", lw=2.2, label="final correct")
        ax.plot(x, yi, color="#6f4f3f", lw=2.0, label="final incorrect")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.18, linestyle="-", linewidth=0.6)
        for s in ax.spines.values():
            s.set_alpha(0.3)
        ax.set_xlabel("normalized layer depth", fontsize=11)
        ax.set_ylabel("fraction committed", fontsize=11)
        ax.set_title(_model_label(model_id), fontsize=14)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.80, left=0.06, right=0.99, wspace=0.25)
    fig.suptitle(
        f"Commitment Timing To Correct Token (p_correct>={commit_p:.2f} and margin>={commit_margin:.2f}, stable through end)",
        fontsize=16,
        y=0.97,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--interp-steps", type=int, default=12)
    ap.add_argument("--display-smooth-sigma", type=float, default=3.0)
    ap.add_argument("--x-left", type=float, default=0.04)
    ap.add_argument("--x-right", type=float, default=0.90)
    ap.add_argument("--max-lines-per-group", type=int, default=900)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--commit-p-threshold", type=float, default=0.70)
    ap.add_argument("--commit-margin-threshold", type=float, default=0.15)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "analysis" / "correct_token_convergence")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifests" / "baseline_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    correct_key_by_uid = _load_correct_key_map(manifest_path)

    model_paths = sorted((run_dir / "outputs").glob("*/baseline_outputs.jsonl"))
    if not model_paths:
        raise RuntimeError(f"no outputs in {run_dir / 'outputs'}")

    model_data = [_load_model_rows(p, correct_key_by_uid) for p in model_paths]
    bundles = {
        d["model_id"]: _prepare_bundle(
            d,
            interp_steps=int(args.interp_steps),
            smooth_sigma=float(args.display_smooth_sigma),
            x_left=float(args.x_left),
            x_right=float(args.x_right),
            commit_p=float(args.commit_p_threshold),
            commit_margin=float(args.commit_margin_threshold),
        )
        for d in model_data
    }

    sideways_png = out_dir / "correct_token_convergence_sideways.png"
    sideways_clean_png = out_dir / "correct_token_convergence_sideways_clean.png"
    timing_png = out_dir / "correct_token_commitment_timing.png"
    summary_json = out_dir / "correct_token_convergence_summary.json"

    _plot_sideways_convergence(
        bundles=bundles,
        out_path=sideways_png,
        seed=int(args.seed),
        max_lines_per_group=int(args.max_lines_per_group),
        title="Convergence To Correct Answer Token (All Prompts, Sideways)",
    )
    _plot_sideways_clean_convergence(
        bundles=bundles,
        out_path=sideways_clean_png,
        seed=int(args.seed) + 17,
        max_texture_per_group=max(120, int(args.max_lines_per_group) // 3),
        title="Convergence To Correct Answer Token (All Prompts, Sideways, Clean View)",
    )
    _plot_commitment_timing(
        bundles=bundles,
        out_path=timing_png,
        commit_p=float(args.commit_p_threshold),
        commit_margin=float(args.commit_margin_threshold),
    )

    summary = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "outputs": {
            "sideways_png": str(sideways_png),
            "sideways_clean_png": str(sideways_clean_png),
            "timing_png": str(timing_png),
        },
        "config": {
            "interp_steps": int(args.interp_steps),
            "display_smooth_sigma": float(args.display_smooth_sigma),
            "x_left": float(args.x_left),
            "x_right": float(args.x_right),
            "commit_p_threshold": float(args.commit_p_threshold),
            "commit_margin_threshold": float(args.commit_margin_threshold),
            "max_lines_per_group": int(args.max_lines_per_group),
        },
        "models": {
            k: {
                "n_rows": int(v["n_rows"]),
                "n_layers": int(v["n_layers"]),
                "accuracy": float(v["accuracy"]),
                "final_p_correct_ge_0_8": float(v["strong_correct_prob_final"]),
                "nonconverged_final_p_correct_lt_0_6": float(v["nonconverged_final_under_0_6"]),
            }
            for k, v in bundles.items()
        },
        "definition": {
            "target_point": "correct option token probability=1.0 (distance=0.0)",
            "trajectory_y": "1 - p_correct(layer)",
            "all_prompts_included": True,
            "colors": {"blue": "final correct", "brown": "final incorrect"},
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
