from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def summarize_span_deletion_effects(span_effects_df: pd.DataFrame) -> pd.DataFrame:
    if span_effects_df.empty:
        return pd.DataFrame(columns=["span_label", "n", "mean_effect_delta", "median_effect_delta"])
    return (
        span_effects_df.groupby("span_label", as_index=False)
        .agg(
            n=("effect_delta", "count"),
            mean_effect_delta=("effect_delta", "mean"),
            median_effect_delta=("effect_delta", "median"),
        )
        .sort_values("span_label")
    )


def run_negative_controls(span_effects_df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    if span_effects_df.empty:
        return pd.DataFrame(columns=["control", "mean_effect_delta"])

    rng = np.random.default_rng(int(seed))
    values = span_effects_df["effect_delta"].to_numpy(dtype=np.float64)
    shuffled = rng.permutation(values)
    sign_flipped = values * rng.choice(np.asarray([-1.0, 1.0]), size=values.size, replace=True)

    return pd.DataFrame.from_records(
        [
            {"control": "observed", "mean_effect_delta": float(np.mean(values))},
            {"control": "shuffled", "mean_effect_delta": float(np.mean(shuffled))},
            {"control": "sign_flipped", "mean_effect_delta": float(np.mean(sign_flipped))},
        ]
    )


def compare_evidence_vs_distractor(span_effects_df: pd.DataFrame) -> Dict[str, float]:
    if span_effects_df.empty:
        return {"evidence_mean": 0.0, "distractor_mean": 0.0, "gap": 0.0}

    ev = span_effects_df.loc[span_effects_df["span_label"] == "evidence", "effect_delta"].to_numpy(dtype=np.float64)
    ds = span_effects_df.loc[span_effects_df["span_label"] == "distractor", "effect_delta"].to_numpy(dtype=np.float64)
    ev_m = float(np.mean(ev)) if ev.size else 0.0
    ds_m = float(np.mean(ds)) if ds.size else 0.0
    return {"evidence_mean": ev_m, "distractor_mean": ds_m, "gap": ev_m - ds_m}
