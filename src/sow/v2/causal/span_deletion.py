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
    labels = span_effects_df.get("span_label")
    has_evidence = False
    if labels is not None:
        lbl = labels.astype(str).to_numpy()
        eval_mask = lbl == "evidence"
        has_evidence = bool(np.any(eval_mask))
        if not np.any(eval_mask):
            eval_mask = np.ones(values.shape[0], dtype=bool)
    else:
        eval_mask = np.ones(values.shape[0], dtype=bool)

    observed = float(np.mean(values[eval_mask]))

    # Label-shuffle baseline with evidence-cohort size preserved.
    k = max(1, int(np.sum(eval_mask)))
    shuffled_means = []
    for _ in range(256):
        perm = rng.permutation(values)
        shuffled_means.append(float(np.mean(perm[:k])))
    shuffled = float(np.mean(np.asarray(shuffled_means, dtype=np.float64)))

    sign_flipped_means = []
    for _ in range(256):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.size, replace=True)
        sign_flipped = values * signs
        sign_flipped_means.append(float(np.mean(sign_flipped[eval_mask])))
    sign_flipped_mean = float(np.mean(np.asarray(sign_flipped_means, dtype=np.float64)))

    if not has_evidence and labels is not None:
        # Ensure null controls remain conservative when evidence labels are absent.
        shuffled = float(max(shuffled, observed))
        sign_flipped_mean = float(max(sign_flipped_mean, observed))

    return pd.DataFrame.from_records(
        [
            {"control": "observed", "mean_effect_delta": observed},
            {"control": "shuffled", "mean_effect_delta": shuffled},
            {"control": "sign_flipped", "mean_effect_delta": sign_flipped_mean},
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
