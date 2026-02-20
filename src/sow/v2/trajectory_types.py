from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectorySummary:
    prompt_uid: str
    model_id: str
    is_correct: bool
    trajectory_type: str
    sign_flip_count: int
    late_flip_count: int
    mean_abs_drift_last8: float


def _sign_flips(delta: np.ndarray) -> int:
    if delta.size <= 1:
        return 0
    s = np.sign(delta)
    flips = np.sum(s[1:] != s[:-1])
    return int(flips)


def _late_flips(delta: np.ndarray, *, tail_len: int = 8) -> int:
    if delta.size <= 1:
        return 0
    s = np.sign(delta)
    tail_start = max(0, int(delta.size) - int(tail_len))
    tail = s[tail_start:]
    if tail.size <= 1:
        return 0
    return int(np.sum(tail[1:] != tail[:-1]))


def classify_trajectory(delta: np.ndarray, *, is_correct: bool, drift: np.ndarray) -> str:
    if delta.size == 0:
        return "unstable_wrong" if not is_correct else "unstable_correct"

    late_flip_count = _late_flips(delta)
    tail = drift[max(0, drift.size - 8) :] if drift.size else np.asarray([], dtype=np.float64)
    mean_abs_drift_last8 = float(np.mean(np.abs(tail))) if tail.size else 0.0

    stable = (late_flip_count == 0) and (mean_abs_drift_last8 <= 0.15)
    if is_correct and stable:
        return "stable_correct"
    if (not is_correct) and stable:
        return "stable_wrong"
    if is_correct:
        return "unstable_correct"
    return "unstable_wrong"


def classify_trajectory_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "prompt_uid",
                "is_correct",
                "trajectory_type",
                "sign_flip_count",
                "late_flip_count",
                "mean_abs_drift_last8",
            ]
        )

    rows: List[Dict[str, object]] = []
    grouped = metrics_df.sort_values(["prompt_uid", "layer_index"]).groupby(["model_id", "prompt_uid"], sort=False)
    for (model_id, prompt_uid), g in grouped:
        delta = g["delta"].to_numpy(dtype=np.float64)
        drift = g["drift"].to_numpy(dtype=np.float64)
        is_correct = bool(g["is_correct"].iloc[0])
        t_type = classify_trajectory(delta, is_correct=is_correct, drift=drift)
        rows.append(
            {
                "model_id": str(model_id),
                "prompt_uid": str(prompt_uid),
                "is_correct": is_correct,
                "trajectory_type": t_type,
                "sign_flip_count": _sign_flips(delta),
                "late_flip_count": _late_flips(delta),
                "mean_abs_drift_last8": float(np.mean(np.abs(drift[max(0, drift.size - 8) :]))) if drift.size else 0.0,
            }
        )

    return pd.DataFrame.from_records(rows)


def type_counts(prompt_types_df: pd.DataFrame) -> Dict[str, int]:
    if prompt_types_df.empty:
        return {}
    vc = prompt_types_df["trajectory_type"].value_counts().to_dict()
    return {str(k): int(v) for k, v in vc.items()}
