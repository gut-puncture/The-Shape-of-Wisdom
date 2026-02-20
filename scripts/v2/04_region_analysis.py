#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import base_parser, run_v2_root_for, write_json, write_parquet


def _loads_vec(x: str) -> np.ndarray:
    try:
        arr = np.asarray(json.loads(str(x)), dtype=np.float64)
    except Exception:
        arr = np.asarray([], dtype=np.float64)
    return arr


def main() -> int:
    ap = base_parser("V2: region and basin-gap analysis")
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    layer_path = out_root / "layerwise.parquet"
    types_path = out_root / "prompt_types.parquet"
    if not layer_path.exists() or not types_path.exists():
        raise SystemExit("missing required inputs: layerwise.parquet and prompt_types.parquet")

    layer = pd.read_parquet(layer_path)
    types = pd.read_parquet(types_path)
    df = layer.merge(types[["model_id", "prompt_uid", "trajectory_type"]], on=["model_id", "prompt_uid"], how="left")

    if args.model_name:
        df = df[df["model_id"].str.contains(args.model_name, na=False)]
    if args.max_prompts > 0 and not df.empty:
        keep = set(df["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        df = df[df["prompt_uid"].isin(keep)]

    if df.empty:
        out = pd.DataFrame(columns=["model_id", "prompt_uid", "layer_index", "trajectory_type", "basin_gap"])
        write_parquet(out_root / "basin_gap.parquet", out)
        write_json(out_root / "04_region_analysis.report.json", {"pass": True, "rows": 0})
        print(str(out_root / "basin_gap.parquet"))
        return 0

    df = df.copy()
    df["hidden_vec"] = df["projected_hidden_128_json"].map(_loads_vec)
    df = df[df["hidden_vec"].map(lambda x: x.size > 0)]

    centroid_rows = []
    grouped = df.groupby(["model_id", "layer_index", "trajectory_type"], sort=False)
    centroids = {}
    for key, g in grouped:
        mat = np.stack(g["hidden_vec"].tolist(), axis=0)
        cen = np.mean(mat, axis=0)
        centroids[key] = cen

    out_rows = []
    for _, row in df.iterrows():
        key = (row["model_id"], row["layer_index"], row["trajectory_type"])
        own = centroids.get(key)
        if own is None:
            continue
        vec = row["hidden_vec"]
        own_dist = float(np.linalg.norm(vec - own))
        other_dist = np.inf
        for (m, li, t), cen in centroids.items():
            if m != row["model_id"] or li != row["layer_index"] or t == row["trajectory_type"]:
                continue
            d = float(np.linalg.norm(vec - cen))
            if d < other_dist:
                other_dist = d
        if not np.isfinite(other_dist):
            other_dist = own_dist
        out_rows.append(
            {
                "model_id": row["model_id"],
                "prompt_uid": row["prompt_uid"],
                "layer_index": int(row["layer_index"]),
                "trajectory_type": row["trajectory_type"],
                "own_distance": own_dist,
                "other_distance": float(other_dist),
                "basin_gap": float(other_dist - own_dist),
            }
        )

    basin = pd.DataFrame.from_records(out_rows)
    write_parquet(out_root / "basin_gap.parquet", basin)

    fig_path = out_root / "fig_region_entry_exit.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    if not basin.empty:
        for t_type, color in [
            ("stable_correct", "#1f77b4"),
            ("stable_wrong", "#d62728"),
            ("unstable_correct", "#2ca02c"),
            ("unstable_wrong", "#9467bd"),
        ]:
            s = basin[basin["trajectory_type"] == t_type]
            if s.empty:
                continue
            layer_mean = s.groupby("layer_index")["basin_gap"].mean()
            ax.plot(layer_mean.index, layer_mean.values, label=t_type, color=color)
    ax.axhline(0.0, color="#333", lw=0.8, alpha=0.4)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean basin gap")
    ax.set_title("Region Entry/Exit Proxy")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    write_json(
        out_root / "04_region_analysis.report.json",
        {"pass": True, "rows": int(basin.shape[0]), "figure": str(fig_path), "out_path": str(out_root / "basin_gap.parquet")},
    )
    print(str(out_root / "basin_gap.parquet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
