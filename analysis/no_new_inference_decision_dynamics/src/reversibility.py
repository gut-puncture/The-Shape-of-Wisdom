from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import ALPHAS, FUTURE_FLIP_P_BINS, FUTURE_FLIP_Z_BINS


def _z_bin(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.minimum(np.floor(clipped * FUTURE_FLIP_Z_BINS).astype(np.int64), FUTURE_FLIP_Z_BINS - 1)


def _p_bin(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.minimum(np.floor(clipped * FUTURE_FLIP_P_BINS).astype(np.int64), FUTURE_FLIP_P_BINS - 1)


def alpha_label(alpha: float) -> str:
    return f"{int(round(alpha * 100)):03d}"


@dataclass
class BinEstimate:
    probability: float
    raw_bin_count: int
    raw_successes: int
    backoff_level: str


def compute_future_flip_metrics(layerwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = layerwise.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").copy()
    future_flip_values: list[float] = []
    for (_, _), group in df.groupby(["model_id", "prompt_uid"], sort=False):
        signs = np.sign(group["delta_soft"].to_numpy(dtype=np.float64))
        out = np.full(len(signs), np.nan, dtype=np.float64)
        for idx, sign in enumerate(signs):
            if sign == 0.0:
                continue
            later_nonzero = signs[idx + 1 :][signs[idx + 1 :] != 0.0]
            out[idx] = 1.0 if later_nonzero.size and np.any(later_nonzero != sign) else 0.0
        future_flip_values.extend(out.tolist())
    df["future_flip"] = future_flip_values
    df["z_bin"] = _z_bin(df["z"].to_numpy(dtype=np.float64))
    df["p_bin"] = _p_bin(df["p_correct"].to_numpy(dtype=np.float64))

    pooled_estimates = _build_estimator(df, pooled=True)
    model_estimates = {model_id: _build_estimator(sub, pooled=False) for model_id, sub in df.groupby("model_id", sort=False)}
    pooled_records = []
    model_records = []
    for row in df.itertuples(index=False):
        pooled = pooled_estimates.lookup(int(row.z_bin), int(row.p_bin))
        model = model_estimates[str(row.model_id)].lookup(int(row.z_bin), int(row.p_bin))
        pooled_records.append(pooled)
        model_records.append(model)

    df["pooled_future_flip_prob"] = [record.probability for record in pooled_records]
    df["pooled_backoff_level"] = [record.backoff_level for record in pooled_records]
    df["pooled_raw_bin_count"] = [record.raw_bin_count for record in pooled_records]
    df["pooled_raw_successes"] = [record.raw_successes for record in pooled_records]
    df["model_future_flip_prob"] = [record.probability for record in model_records]
    df["model_backoff_level"] = [record.backoff_level for record in model_records]
    df["model_raw_bin_count"] = [record.raw_bin_count for record in model_records]
    df["model_raw_successes"] = [record.raw_successes for record in model_records]

    for alpha in ALPHAS:
        label = alpha_label(alpha)
        df[f"pooled_empirically_committed_{label}"] = df["pooled_future_flip_prob"] <= alpha
        df[f"model_empirically_committed_{label}"] = df["model_future_flip_prob"] <= alpha

    trajectory_commitment = _empirical_commitment_depths(df)
    return df, trajectory_commitment


class _Estimator:
    def __init__(self, cell: pd.DataFrame, z_only: pd.DataFrame, global_mean: float, global_count: int, global_successes: int, *, pooled: bool) -> None:
        self.cell = cell.set_index(["z_bin", "p_bin"])
        self.z_only = z_only.set_index(["z_bin"])
        self.global_mean = float(global_mean)
        self.global_count = int(global_count)
        self.global_successes = int(global_successes)
        self.pooled = pooled

    def lookup(self, z_bin: int, p_bin: int) -> BinEstimate:
        key = (int(z_bin), int(p_bin))
        if key in self.cell.index:
            row = self.cell.loc[key]
            return BinEstimate(
                probability=float(row["probability"]),
                raw_bin_count=int(row["raw_count"]),
                raw_successes=int(row["successes"]),
                backoff_level="cell",
            )
        if int(z_bin) in self.z_only.index:
            row = self.z_only.loc[int(z_bin)]
            return BinEstimate(
                probability=float(row["probability"]),
                raw_bin_count=int(row["raw_count"]),
                raw_successes=int(row["successes"]),
                backoff_level="z_marginal",
            )
        return BinEstimate(
            probability=self.global_mean,
            raw_bin_count=self.global_count,
            raw_successes=self.global_successes,
            backoff_level="global_pooled" if self.pooled else "global_model",
        )


def _build_estimator(df: pd.DataFrame, *, pooled: bool) -> _Estimator:
    valid = df[df["future_flip"].notna()].copy()
    by_cell = valid.groupby(["z_bin", "p_bin"], sort=False)["future_flip"].agg(["sum", "size"]).reset_index()
    by_cell.rename(columns={"sum": "successes", "size": "raw_count"}, inplace=True)
    by_cell["probability"] = (by_cell["successes"] + 1.0) / (by_cell["raw_count"] + 2.0)
    by_z = valid.groupby(["z_bin"], sort=False)["future_flip"].agg(["sum", "size"]).reset_index()
    by_z.rename(columns={"sum": "successes", "size": "raw_count"}, inplace=True)
    by_z["probability"] = (by_z["successes"] + 1.0) / (by_z["raw_count"] + 2.0)
    global_successes = int(valid["future_flip"].sum())
    global_count = int(valid["future_flip"].shape[0])
    global_probability = (global_successes + 1.0) / (global_count + 2.0) if global_count else 0.5
    return _Estimator(by_cell, by_z, global_probability, global_count, global_successes, pooled=pooled)


def _empirical_commitment_depths(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").groupby(["model_id", "prompt_uid"], sort=False)
    for (model_id, prompt_uid), group in grouped:
        record: dict[str, Any] = {"model_id": model_id, "prompt_uid": prompt_uid}
        z = group["z"].to_numpy(dtype=np.float64)
        for alpha in ALPHAS:
            label = alpha_label(alpha)
            pooled_flag = group[f"pooled_empirically_committed_{label}"].to_numpy(dtype=bool)
            model_flag = group[f"model_empirically_committed_{label}"].to_numpy(dtype=bool)
            pooled_depth = _suffix_true_depth(z, pooled_flag)
            model_depth = _suffix_true_depth(z, model_flag)
            record[f"pooled_empirical_commitment_depth_{label}"] = pooled_depth
            record[f"model_empirical_commitment_depth_{label}"] = model_depth
        rows.append(record)
    return pd.DataFrame(rows)


def _suffix_true_depth(z: np.ndarray, flags: np.ndarray) -> float | None:
    if flags.size == 0:
        return None
    suffix = np.logical_and.accumulate(flags[::-1])[::-1]
    idx = np.flatnonzero(suffix)
    if idx.size == 0:
        return None
    return float(z[int(idx[0])])
