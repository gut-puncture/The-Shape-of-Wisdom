from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .constants import MODEL_ORDER, SUMMARY_Z_BINS


def rng_family(seed: int, n_children: int) -> list[np.random.Generator]:
    seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in seq.spawn(n_children)]


@dataclass
class CurveSummary:
    frame: pd.DataFrame
    pooled_frame: pd.DataFrame | None


def bootstrap_layerwise_curves(
    layerwise: pd.DataFrame,
    *,
    metric_specs: dict[str, Literal["mean", "median"]],
    group_col: str,
    bootstrap_reps: int,
    seed: int,
) -> CurveSummary:
    df = layerwise.copy()
    df["summary_z_bin"] = np.minimum(np.floor(np.clip(df["z"], 0.0, 1.0) * SUMMARY_Z_BINS).astype(int), SUMMARY_Z_BINS - 1)
    prompt_bin = (
        df.groupby(["model_id", "prompt_uid", group_col, "summary_z_bin"], sort=False)[list(metric_specs)]
        .mean()
        .reset_index()
    )

    per_model_rows: list[dict[str, object]] = []
    pooled_boot_store: dict[tuple[object, str, int], list[np.ndarray]] = {}
    pooled_center_store: dict[tuple[object, str, int], list[float]] = {}
    generators = rng_family(seed, len(MODEL_ORDER))

    for model_offset, model_id in enumerate(MODEL_ORDER):
        model_df = prompt_bin[prompt_bin["model_id"] == model_id]
        if model_df.empty:
            continue
        for group_value, group_df in model_df.groupby(group_col, sort=False):
            prompts = sorted(group_df["prompt_uid"].unique().tolist())
            for metric, reducer_name in metric_specs.items():
                pivot = group_df.pivot(index="prompt_uid", columns="summary_z_bin", values=metric).reindex(prompts)
                matrix = pivot.to_numpy(dtype=np.float64)
                curve = _bootstrap_single_metric(matrix, reducer_name, generators[model_offset], bootstrap_reps)
                for z_bin in range(matrix.shape[1]):
                    per_model_rows.append(
                        {
                            "model_id": model_id,
                            "group": group_value,
                            "metric": metric,
                            "summary_z_bin": z_bin,
                            "center": float(curve["center"][z_bin]),
                            "ci_low": float(curve["ci_low"][z_bin]),
                            "ci_high": float(curve["ci_high"][z_bin]),
                        }
                    )
                    key = (group_value, metric, z_bin)
                    pooled_boot_store.setdefault(key, []).append(curve["boot"][:, z_bin])
                    pooled_center_store.setdefault(key, []).append(float(curve["center"][z_bin]))

    pooled_rows = []
    for (group_value, metric, z_bin), arrays in pooled_boot_store.items():
        boot_matrix = np.stack(arrays, axis=0)
        pooled_boot = np.nanmean(boot_matrix, axis=0)
        pooled_rows.append(
            {
                "group": group_value,
                "metric": metric,
                "summary_z_bin": int(z_bin),
                "center": float(np.nanmean(pooled_center_store[(group_value, metric, z_bin)])),
                "ci_low": float(np.nanpercentile(pooled_boot, 2.5)),
                "ci_high": float(np.nanpercentile(pooled_boot, 97.5)),
            }
        )

    return CurveSummary(
        frame=pd.DataFrame(per_model_rows),
        pooled_frame=pd.DataFrame(pooled_rows) if pooled_rows else None,
    )


def _bootstrap_single_metric(
    matrix: np.ndarray,
    reducer_name: Literal["mean", "median"],
    rng: np.random.Generator,
    reps: int,
) -> dict[str, np.ndarray]:
    reducer = np.nanmedian if reducer_name == "median" else np.nanmean
    if matrix.size == 0:
        return {
            "center": np.full((SUMMARY_Z_BINS,), np.nan, dtype=np.float64),
            "ci_low": np.full((SUMMARY_Z_BINS,), np.nan, dtype=np.float64),
            "ci_high": np.full((SUMMARY_Z_BINS,), np.nan, dtype=np.float64),
            "boot": np.full((reps, SUMMARY_Z_BINS), np.nan, dtype=np.float64),
        }
    center = reducer(matrix, axis=0)
    boot = np.empty((reps, matrix.shape[1]), dtype=np.float64)
    for rep in range(reps):
        sample_idx = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        boot[rep] = reducer(matrix[sample_idx], axis=0)
    return {
        "center": center,
        "ci_low": np.nanpercentile(boot, 2.5, axis=0),
        "ci_high": np.nanpercentile(boot, 97.5, axis=0),
        "boot": boot,
    }
