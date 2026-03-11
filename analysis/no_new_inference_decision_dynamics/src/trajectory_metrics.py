from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .constants import ANGULAR_Q_THRESHOLD, OLD_THRESHOLD_M
from .state_metrics import angular_step


def _sign_flip_count(delta: np.ndarray) -> tuple[int, float | None]:
    nonzero_indices = np.flatnonzero(delta != 0.0)
    if nonzero_indices.size <= 1:
        return 0, None
    signs = np.sign(delta[nonzero_indices])
    flips = np.flatnonzero(signs[1:] != signs[:-1])
    if flips.size == 0:
        return 0, None
    last_idx = int(nonzero_indices[int(flips[-1] + 1)])
    return int(flips.size), float(last_idx)


def derive_old_comparison_fields(old_core: pd.DataFrame | None) -> pd.DataFrame | None:
    if old_core is None:
        return None
    columns = ["model_id", "prompt_uid", "regime", "flip_count", "commitment_layer", "last_flip_layer", "L_model"]
    missing = [column for column in columns if column not in old_core.columns]
    if missing:
        return None
    prompt = old_core[columns].drop_duplicates(subset=["model_id", "prompt_uid"]).copy()
    prompt["old_regime"] = prompt["regime"].astype(str)
    prompt["old_flip_count"] = prompt["flip_count"].astype(np.int64)
    prompt["old_commitment_depth"] = prompt["commitment_layer"].astype(np.float64) / prompt["L_model"].clip(lower=1).astype(np.float64)
    prompt["old_last_flip_depth"] = prompt["last_flip_layer"].astype(np.float64) / prompt["L_model"].clip(lower=1).astype(np.float64)
    return prompt[["model_id", "prompt_uid", "old_regime", "old_flip_count", "old_commitment_depth", "old_last_flip_depth"]]


def compute_trajectory_metrics(layerwise: pd.DataFrame, old_core: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = layerwise.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").groupby(["model_id", "prompt_uid"], sort=False)
    for (model_id, prompt_uid), group in grouped:
        delta_soft = group["delta_soft"].to_numpy(dtype=np.float64)
        p_correct = group["p_correct"].to_numpy(dtype=np.float64)
        switch = group["switch"].to_numpy(dtype=np.float64)
        a = group["a"].to_numpy(dtype=np.float64)
        b = group["b"].to_numpy(dtype=np.float64)
        c = group["c"].to_numpy(dtype=np.float64)
        q = group["q"].to_numpy(dtype=np.float64)
        r = group["r"].to_numpy(dtype=np.float64)
        z = group["z"].to_numpy(dtype=np.float64)
        final_correct = bool(group["final_correct"].iloc[-1])
        flip_count, last_flip_layer_position = _sign_flip_count(delta_soft)
        last_flip_depth = None if last_flip_layer_position is None else float(z[int(last_flip_layer_position)])
        xyz = np.column_stack([a, b, c])
        diffs = np.diff(xyz, axis=0)
        path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
        radial_travel = float(np.abs(np.diff(r)).sum()) if len(r) > 1 else 0.0
        angular_steps = [
            angular_step(theta_prev=float(group["theta"].iloc[i - 1]), theta_curr=float(group["theta"].iloc[i]))
            for i in range(1, len(group))
            if float(q[i - 1]) > ANGULAR_Q_THRESHOLD and float(q[i]) > ANGULAR_Q_THRESHOLD
        ]
        angular_travel = float(np.sum(angular_steps)) if angular_steps else 0.0
        mean_angular_step = float(np.mean(angular_steps)) if angular_steps else float("nan")
        final_delta_soft = float(delta_soft[-1])
        final_p_correct = float(p_correct[-1])
        sign_final = 1.0 if final_delta_soft > 0 else (-1.0 if final_delta_soft < 0 else 0.0)
        delta_signed = sign_final * delta_soft
        old_commitment_depth = None
        for idx in range(len(delta_signed)):
            if bool(np.all(delta_signed[idx:] >= OLD_THRESHOLD_M)):
                old_commitment_depth = float(z[idx])
                break
        rows.append(
            {
                "model_id": model_id,
                "prompt_uid": prompt_uid,
                "subject": str(group["subject"].iloc[0]),
                "coarse_domain": str(group["coarse_domain"].iloc[0]),
                "question": str(group["question"].iloc[0]),
                "correct_option": str(group["correct_option"].iloc[0]),
                "final_predicted_option": str(group["final_predicted_option"].iloc[-1]),
                "final_argmax_tie": bool(group["final_argmax_tie"].iloc[-1]),
                "final_correct": final_correct,
                "final_delta_soft": final_delta_soft,
                "final_p_correct": final_p_correct,
                "flip_count": int(flip_count),
                "last_flip_depth": last_flip_depth,
                "total_switches": int(np.sum(switch)),
                "mean_switch_rate": float(np.mean(switch[1:])) if len(switch) > 1 else 0.0,
                "boundary_occupancy_prob_005": float(np.mean(np.abs(p_correct - 0.5) < 0.05)),
                "boundary_occupancy_prob_010": float(np.mean(np.abs(p_correct - 0.5) < 0.10)),
                "boundary_occupancy_margin_025": float(np.mean(np.abs(delta_soft) < 0.25)),
                "boundary_occupancy_margin_050": float(np.mean(np.abs(delta_soft) < 0.50)),
                "mean_q": float(np.mean(q)),
                "max_q": float(np.max(q)),
                "initial_q": float(q[0]),
                "final_q": float(q[-1]),
                "net_commitment_gain": float(a[-1] - a[0]),
                "net_decisiveness_gain": float(r[-1] - r[0]),
                "path_length": path_length,
                "radial_travel": radial_travel,
                "angular_travel": angular_travel,
                "mean_angular_step": mean_angular_step,
                "old_threshold_commitment_depth": old_commitment_depth,
            }
        )

    traj = pd.DataFrame(rows)
    if old_core is not None:
        old_fields = derive_old_comparison_fields(old_core)
        if old_fields is not None:
            traj = traj.merge(old_fields, on=["model_id", "prompt_uid"], how="left")
    else:
        traj["old_regime"] = pd.NA
        traj["old_flip_count"] = pd.NA
        traj["old_commitment_depth"] = pd.NA
        traj["old_last_flip_depth"] = pd.NA

    traj = _add_boundary_quartiles(traj)
    return traj


def _add_boundary_quartiles(traj: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model_id, group in traj.groupby("model_id", sort=False):
        ordered = group.sort_values(["boundary_occupancy_prob_010", "prompt_uid"], kind="stable").reset_index(drop=True).copy()
        ordered["boundary_occupancy_quartile"] = np.floor(np.arange(len(ordered)) * 4 / max(len(ordered), 1)).astype(int) + 1
        ordered.loc[ordered["boundary_occupancy_quartile"] > 4, "boundary_occupancy_quartile"] = 4
        parts.append(ordered)
    return pd.concat(parts, ignore_index=True) if parts else traj.copy()

