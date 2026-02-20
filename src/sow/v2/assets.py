from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_phase_diagram(metrics_df: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    tail = metrics_df.sort_values("layer_index").groupby("prompt_uid").tail(8)
    agg = (
        tail.groupby(["prompt_uid", "model_id", "is_correct"], as_index=False)
        .agg(delta_last=("delta", "last"), drift_last8=("drift", "sum"), boundary_last=("boundary", "last"))
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for is_correct, color, label in [(True, "#1f77b4", "correct"), (False, "#d62728", "wrong")]:
        s = agg[agg["is_correct"] == is_correct]
        ax.scatter(s["delta_last"], s["drift_last8"], c=color, s=16, alpha=0.55, label=label)
    ax.axvline(0.0, color="#444", lw=0.8, alpha=0.4)
    ax.axhline(0.0, color="#444", lw=0.8, alpha=0.4)
    ax.set_xlabel("delta at final layer")
    ax.set_ylabel("sum drift (last 8 layers)")
    ax.set_title("Phase Diagram in Decision Space")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_trajectory_plots(metrics_df: pd.DataFrame, *, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if metrics_df.empty:
        return files

    for model_id, sub in metrics_df.groupby("model_id"):
        safe = model_id.replace("/", "__")
        layer = (
            sub.groupby(["layer_index", "is_correct"], as_index=False)
            .agg(delta_mean=("delta", "mean"), entropy_mean=("entropy", "mean"), flips=("drift", lambda x: float(np.mean(np.abs(x) > 0.0))))
        )

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        for ok, color, label in [(True, "#1f77b4", "correct"), (False, "#d62728", "wrong")]:
            s = layer[layer["is_correct"] == ok]
            ax1.plot(s["layer_index"], s["delta_mean"], color=color, lw=2, label=label)
        ax1.set_title(f"Delta Trajectories ({model_id})")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("mean delta")
        ax1.grid(alpha=0.2)
        ax1.legend(loc="best")
        p1 = out_dir / f"fig_trajectories_delta_{safe}.png"
        fig1.savefig(p1, dpi=170, bbox_inches="tight")
        plt.close(fig1)
        files.append(p1)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for ok, color, label in [(True, "#1f77b4", "correct"), (False, "#d62728", "wrong")]:
            s = layer[layer["is_correct"] == ok]
            ax2.plot(s["layer_index"], s["entropy_mean"], color=color, lw=2, label=label)
        ax2.set_title(f"Entropy Trajectories ({model_id})")
        ax2.set_xlabel("layer")
        ax2.set_ylabel("mean entropy")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="best")
        p2 = out_dir / f"fig_trajectories_entropy_{safe}.png"
        fig2.savefig(p2, dpi=170, bbox_inches="tight")
        plt.close(fig2)
        files.append(p2)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        for ok, color, label in [(True, "#1f77b4", "correct"), (False, "#d62728", "wrong")]:
            s = layer[layer["is_correct"] == ok]
            ax3.plot(s["layer_index"], s["flips"], color=color, lw=2, label=label)
        ax3.set_title(f"Flip Rate Proxy ({model_id})")
        ax3.set_xlabel("layer")
        ax3.set_ylabel("mean |drift|>0")
        ax3.grid(alpha=0.2)
        ax3.legend(loc="best")
        p3 = out_dir / f"fig_flips_{safe}.png"
        fig3.savefig(p3, dpi=170, bbox_inches="tight")
        plt.close(fig3)
        files.append(p3)

    return files


def write_sha_manifest(*, root_dir: Path, out_path: Path) -> None:
    records = []
    for p in sorted(root_dir.rglob("*")):
        if p.is_file():
            records.append({"path": str(p), "sha256": _sha256(p)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
