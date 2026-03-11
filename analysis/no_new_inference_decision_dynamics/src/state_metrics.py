from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .constants import ANGULAR_Q_THRESHOLD, CHOICE_TO_INDEX, CHOICES, INCORRECT_ORDER, PERMUTATION_INDEX, THETA_NAN_THRESHOLD

E_COMMIT = np.asarray([1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0], dtype=np.float64)
E_COMMIT = E_COMMIT / np.linalg.norm(E_COMMIT)
E_COMP1 = np.asarray([0.0, 1.0, -1.0, 0.0], dtype=np.float64)
E_COMP1 = E_COMP1 / np.linalg.norm(E_COMP1)
E_COMP2 = np.asarray([0.0, 1.0, 1.0, -2.0], dtype=np.float64)
E_COMP2 = E_COMP2 / np.linalg.norm(E_COMP2)


def compute_layerwise_state_metrics(canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = canonical.copy()
    scores = df[[f"score_{choice}" for choice in CHOICES]].to_numpy(dtype=np.float64)
    correct_idx = np.asarray([CHOICE_TO_INDEX[choice] for choice in df["correct_option"].tolist()], dtype=np.int64)
    correct_scores = scores[np.arange(len(df)), correct_idx]

    incorrect_scores = scores.copy()
    incorrect_scores[np.arange(len(df)), correct_idx] = -np.inf
    competitor_idx = np.argmax(incorrect_scores, axis=1)
    competitor = np.asarray([CHOICES[idx] for idx in competitor_idx], dtype=object)
    delta_hard = correct_scores - incorrect_scores[np.arange(len(df)), competitor_idx]
    delta_soft = correct_scores - logsumexp(incorrect_scores, axis=1)
    probs = df[[f"prob_{choice}" for choice in CHOICES]].to_numpy(dtype=np.float64)
    p_correct = probs[np.arange(len(df)), correct_idx]

    perm_indices = np.asarray([PERMUTATION_INDEX[choice] for choice in df["correct_option"].tolist()], dtype=np.int64)
    w = scores[np.arange(len(df))[:, None], perm_indices]
    x = w - w.mean(axis=1, keepdims=True)
    a = x @ E_COMMIT
    b = x @ E_COMP1
    c = x @ E_COMP2
    q = np.sqrt(b * b + c * c)
    theta = np.arctan2(c, b)
    theta[q <= THETA_NAN_THRESHOLD] = np.nan
    r = np.sqrt(a * a + q * q)
    reconstruction = np.outer(a, E_COMMIT) + np.outer(b, E_COMP1) + np.outer(c, E_COMP2)
    reconstruction_error = np.abs(x - reconstruction)

    df["dynamic_competitor"] = competitor
    df["delta_hard"] = delta_hard
    df["delta_soft"] = delta_soft
    df["p_correct"] = p_correct
    df["a"] = a
    df["b"] = b
    df["c"] = c
    df["q"] = q
    df["theta"] = theta
    df["r"] = r

    df = df.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable").reset_index(drop=True)
    prev_comp = df.groupby(["model_id", "prompt_uid"], sort=False)["dynamic_competitor"].shift(1)
    df["switch"] = (df["dynamic_competitor"] != prev_comp).astype(np.int64)
    first_mask = df.groupby(["model_id", "prompt_uid"], sort=False).cumcount() == 0
    df.loc[first_mask, "switch"] = 0
    df["next_switch"] = df.groupby(["model_id", "prompt_uid"], sort=False)["switch"].shift(-1)

    max_sigmoid_error = float(np.max(np.abs((1.0 / (1.0 + np.exp(-delta_soft))) - p_correct)))
    validation = {
        "state_reconstruction_max_abs_error": float(reconstruction_error.max()),
        "state_reconstruction_mean_abs_error": float(reconstruction_error.mean()),
        "p_correct_sigmoid_delta_soft_max_abs_error": max_sigmoid_error,
        "q_below_theta_nan_threshold_count": int((q <= THETA_NAN_THRESHOLD).sum()),
        "q_below_angular_threshold_count": int((q <= ANGULAR_Q_THRESHOLD).sum()),
    }
    return df, validation


def angular_step(theta_prev: float, theta_curr: float) -> float:
    delta = theta_curr - theta_prev
    return float(abs(np.arctan2(np.sin(delta), np.cos(delta))))
