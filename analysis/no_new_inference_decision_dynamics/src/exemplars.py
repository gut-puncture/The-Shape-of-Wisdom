from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "final_p_correct",
    "flip_count",
    "boundary_occupancy_prob_010",
    "last_flip_depth",
    "pooled_empirical_commitment_depth_005",
]


def select_exemplars(traj: pd.DataFrame) -> pd.DataFrame:
    frame = traj.copy()
    if "pooled_empirical_commitment_depth_005" not in frame.columns:
        frame["pooled_empirical_commitment_depth_005"] = np.nan
    frame["last_flip_depth_for_distance"] = frame["last_flip_depth"].fillna(1.05)
    frame["pooled_empirical_commitment_depth_005_for_distance"] = frame["pooled_empirical_commitment_depth_005"].fillna(1.05)
    selected_parts = [
        _select_group(
            frame[
                (frame["final_correct"])
                & (frame["flip_count"] <= 1)
                & frame["pooled_empirical_commitment_depth_005"].notna()
                & (frame["boundary_occupancy_prob_010"] <= frame["boundary_occupancy_prob_010"].median())
            ].copy(),
            label="stable_final_correct",
            n=3,
        ),
        _select_group(
            frame[
                (~frame["final_correct"])
                & (frame["flip_count"] <= 1)
                & frame["pooled_empirical_commitment_depth_005"].notna()
                & (frame["boundary_occupancy_prob_010"] <= frame["boundary_occupancy_prob_010"].median())
            ].copy(),
            label="stable_final_wrong",
            n=3,
        ),
        _select_group(
            frame[
                (
                    frame["boundary_occupancy_prob_010"]
                    >= frame["boundary_occupancy_prob_010"].quantile(0.9)
                )
                & (
                    (frame["last_flip_depth"] >= 0.8)
                    | frame["pooled_empirical_commitment_depth_005"].isna()
                )
            ].copy(),
            label="boundary_dwelling",
            n=3,
        ),
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["exemplar_rank"] = selected.groupby("group").cumcount() + 1
    return selected


def _select_group(group: pd.DataFrame, *, label: str, n: int) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame(columns=["model_id", "prompt_uid", "group"])
    features = pd.DataFrame(
        {
            "final_p_correct": group["final_p_correct"].to_numpy(dtype=np.float64),
            "flip_count": group["flip_count"].to_numpy(dtype=np.float64),
            "boundary_occupancy_prob_010": group["boundary_occupancy_prob_010"].to_numpy(dtype=np.float64),
            "last_flip_depth": group["last_flip_depth_for_distance"].to_numpy(dtype=np.float64),
            "pooled_empirical_commitment_depth_005": group["pooled_empirical_commitment_depth_005_for_distance"].to_numpy(dtype=np.float64),
        }
    )
    standardized = pd.DataFrame(index=features.index)
    for column in features.columns:
        values = features[column].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad if mad > 1e-12 else max(float(np.std(values)), 1.0)
        standardized[column] = (values - median) / scale
    center = standardized.median(axis=0).to_numpy(dtype=np.float64)
    distances = np.linalg.norm(standardized.to_numpy(dtype=np.float64) - center[None, :], axis=1)
    group = group.copy()
    group["medoid_distance"] = distances
    chosen_idx: list[int] = []
    for model_id in group.sort_values(["model_id"], kind="stable")["model_id"].drop_duplicates().tolist():
        model_rows = group[group["model_id"] == model_id].sort_values(["medoid_distance", "prompt_uid"], kind="stable")
        if not model_rows.empty and len(chosen_idx) < n:
            chosen_idx.append(int(model_rows.index[0]))
    if len(chosen_idx) < n:
        for idx in group.sort_values(["medoid_distance", "prompt_uid"], kind="stable").index.tolist():
            if idx not in chosen_idx:
                chosen_idx.append(int(idx))
            if len(chosen_idx) == n:
                break
    out = group.loc[chosen_idx].sort_values(["medoid_distance", "prompt_uid"], kind="stable").copy()
    out["group"] = label
    return out
