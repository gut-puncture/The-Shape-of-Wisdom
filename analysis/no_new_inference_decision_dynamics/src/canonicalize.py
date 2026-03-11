from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .constants import CANONICAL_COLUMNS, CHOICE_TO_INDEX, CHOICES, MODEL_LAYERS, MODEL_SHORT
from .loaders import parse_json_column


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / denom


def _stack_scores(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(record.get(choice, float("nan"))) for choice in CHOICES] for record in records],
        dtype=np.float64,
    )


def _choice_indices(correct_options: pd.Series) -> np.ndarray:
    return np.asarray([CHOICE_TO_INDEX[str(choice)] for choice in correct_options.tolist()], dtype=np.int64)


def build_canonical_table(layerwise: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_use = manifest[
        [
            "prompt_uid",
            "example_id",
            "correct_key",
            "subject",
            "coarse_domain",
            "wrapper_id",
            "dataset",
            "split",
            "question",
        ]
    ].drop_duplicates(subset=["prompt_uid"])
    df = layerwise.merge(manifest_use, on=["prompt_uid"], how="left", suffixes=("", "_manifest"))
    logits_records = parse_json_column(df["candidate_logits_json"])
    scores = _stack_scores(logits_records)
    probs = _softmax(scores)
    top_idx = np.argmax(scores, axis=1)
    top_values = scores[np.arange(scores.shape[0]), top_idx][:, None]
    top_tie_count = (scores == top_values).sum(axis=1).astype(np.int64)
    correct_options = df["correct_key"].astype(str)
    correct_idx = _choice_indices(correct_options)

    incorrect_scores = scores.copy()
    incorrect_scores[np.arange(scores.shape[0]), correct_idx] = -np.inf
    competitor_idx = np.argmax(incorrect_scores, axis=1)
    competitor_max = incorrect_scores[np.arange(scores.shape[0]), competitor_idx][:, None]
    competitor_tie_count = ((incorrect_scores == competitor_max) & np.isfinite(incorrect_scores)).sum(axis=1).astype(np.int64)

    df = df.copy()
    for idx, choice in enumerate(CHOICES):
        df[f"score_{choice}"] = scores[:, idx]
        df[f"prob_{choice}"] = probs[:, idx]

    df["model_name"] = df["model_id"].astype(str)
    df["model_short"] = df["model_id"].map(MODEL_SHORT).fillna(df["model_id"])
    df["correct_option"] = correct_options
    df["top_candidate"] = [CHOICES[idx] for idx in top_idx]
    df["top_tie_count"] = top_tie_count
    df["top_is_tied"] = top_tie_count > 1
    df["competitor_tie_count"] = competitor_tie_count
    df["competitor_is_tied"] = competitor_tie_count > 1

    df["max_layer_index"] = df.groupby("model_id")["layer_index"].transform("max").astype(np.int64)
    df["n_layers_logged"] = df.groupby(["model_id", "prompt_uid"])["layer_index"].transform("size").astype(np.int64)
    df["z"] = df["layer_index"].astype(np.float64) / df["max_layer_index"].clip(lower=1).astype(np.float64)

    final_rows = (
        df.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable")
        .groupby(["model_id", "prompt_uid"], sort=False)
        .tail(1)
        .copy()
    )
    final_pred = final_rows["top_candidate"].astype(str)
    final_argmax_tie = final_rows["top_is_tied"].astype(bool)
    final_lookup = final_rows[["model_id", "prompt_uid"]].copy()
    final_lookup["final_predicted_option"] = final_pred.values
    final_lookup["final_argmax_tie"] = final_argmax_tie.values
    df = df.merge(final_lookup, on=["model_id", "prompt_uid"], how="left")
    df["final_correct"] = df["final_predicted_option"].astype(str) == df["correct_option"].astype(str)

    df = df[list(CANONICAL_COLUMNS) + ["question"]].copy()
    if df["model_id"].map(MODEL_LAYERS).notna().all():
        expected_max = df["model_id"].map(MODEL_LAYERS).astype(np.int64) - 1
        mismatch = expected_max != df["max_layer_index"].astype(np.int64)
        if bool(mismatch.any()):
            raise ValueError("observed max_layer_index does not match expected model layers")
    return df


def validation_rows(
    canonical: pd.DataFrame,
    manifest: pd.DataFrame,
    decision_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_dupes = int(canonical.duplicated(["model_id", "prompt_uid", "layer_index"]).sum())
    manifest_dupes = int(manifest.duplicated(["prompt_uid"]).sum())
    rows.append({"scope": "global", "metric": "canonical_rows", "actual": int(len(canonical)), "expected": 276000, "pass": int(len(canonical) == 276000), "details": ""})
    rows.append({"scope": "global", "metric": "manifest_rows", "actual": int(len(manifest)), "expected": 3000, "pass": int(len(manifest) == 3000), "details": ""})
    rows.append({"scope": "global", "metric": "manifest_prompt_uid_duplicates", "actual": manifest_dupes, "expected": 0, "pass": int(manifest_dupes == 0), "details": ""})
    rows.append({"scope": "global", "metric": "canonical_duplicate_rows", "actual": layer_dupes, "expected": 0, "pass": int(layer_dupes == 0), "details": ""})
    rows.append(
        {
            "scope": "global",
            "metric": "missing_values_total",
            "actual": int(canonical.isna().sum().sum()),
            "expected": 0,
            "pass": int(int(canonical.isna().sum().sum()) == 0),
            "details": "question may be empty string but not NA",
        }
    )
    per_model = canonical.groupby("model_id").agg(
        prompts=("prompt_uid", "nunique"),
        rows=("prompt_uid", "size"),
        min_layer=("layer_index", "min"),
        max_layer=("layer_index", "max"),
        layers=("layer_index", "nunique"),
    )
    for model_id, record in per_model.iterrows():
        expected_prompts = 3000
        rows.append({"scope": model_id, "metric": "prompt_count", "actual": int(record["prompts"]), "expected": expected_prompts, "pass": int(int(record["prompts"]) == expected_prompts), "details": ""})
        rows.append({"scope": model_id, "metric": "row_count", "actual": int(record["rows"]), "expected": int(record["prompts"] * record["layers"]), "pass": int(int(record["rows"]) == int(record["prompts"] * record["layers"])), "details": ""})
        rows.append({"scope": model_id, "metric": "min_layer_index", "actual": int(record["min_layer"]), "expected": 0, "pass": int(int(record["min_layer"]) == 0), "details": ""})
        rows.append({"scope": model_id, "metric": "max_layer_index", "actual": int(record["max_layer"]), "expected": MODEL_LAYERS[str(model_id)] - 1, "pass": int(int(record["max_layer"]) == MODEL_LAYERS[str(model_id)] - 1), "details": ""})

    complete = canonical.groupby(["model_id", "prompt_uid"]).agg(
        min_layer=("layer_index", "min"),
        max_layer=("layer_index", "max"),
        layers=("layer_index", "nunique"),
        max_layer_index=("max_layer_index", "max"),
        n_layers_logged=("n_layers_logged", "max"),
    )
    complete_ok = bool(
        (
            (complete["min_layer"] == 0)
            & (complete["max_layer"] == complete["max_layer_index"])
            & (complete["layers"] == complete["n_layers_logged"])
            & (complete["layers"] == complete["max_layer_index"] + 1)
        ).all()
    )
    rows.append({"scope": "global", "metric": "complete_layer_coverage", "actual": int(complete_ok), "expected": 1, "pass": int(complete_ok), "details": ""})

    probs_from_scores = canonical[[f"prob_{choice}" for choice in CHOICES]].to_numpy(dtype=np.float64)
    probs_sum = np.abs(probs_from_scores.sum(axis=1) - 1.0).max()
    rows.append({"scope": "global", "metric": "probability_sum_max_abs_error", "actual": float(probs_sum), "expected": 0.0, "pass": int(float(probs_sum) < 1e-12), "details": ""})

    dm_use = decision_metrics.rename(columns={"correct_key": "dm_correct_key"})
    cross = canonical.merge(dm_use, on=["model_id", "prompt_uid", "layer_index"], how="left")
    correct_key_mismatch = int((cross["correct_option"].astype(str) != cross["dm_correct_key"].astype(str)).sum())
    p_correct_expected = np.choose(
        np.asarray([CHOICE_TO_INDEX[choice] for choice in cross["correct_option"].tolist()], dtype=np.int64),
        [cross["prob_A"].to_numpy(dtype=np.float64), cross["prob_B"].to_numpy(dtype=np.float64), cross["prob_C"].to_numpy(dtype=np.float64), cross["prob_D"].to_numpy(dtype=np.float64)],
    )
    p_correct_max_abs_err = float(np.max(np.abs(p_correct_expected - cross["p_correct"].to_numpy(dtype=np.float64))))
    final_consistent = bool(
        (
            cross.sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable")
            .groupby(["model_id", "prompt_uid"], sort=False)
            .tail(1)["final_predicted_option"]
            .notna()
        ).all()
    )
    rows.append({"scope": "global", "metric": "decision_metrics_correct_key_mismatch", "actual": correct_key_mismatch, "expected": 0, "pass": int(correct_key_mismatch == 0), "details": ""})
    rows.append({"scope": "global", "metric": "decision_metrics_p_correct_max_abs_error", "actual": p_correct_max_abs_err, "expected": 0.0, "pass": int(p_correct_max_abs_err < 1e-12), "details": ""})
    rows.append({"scope": "global", "metric": "final_predicted_option_defined", "actual": int(final_consistent), "expected": 1, "pass": int(final_consistent), "details": ""})

    rows.append({"scope": "global", "metric": "top_tied_rows", "actual": int(canonical["top_is_tied"].sum()), "expected": 0, "pass": int(True), "details": "reported for sensitivity, not a failure"})
    rows.append({"scope": "global", "metric": "competitor_tied_rows", "actual": int(canonical["competitor_is_tied"].sum()), "expected": 0, "pass": int(True), "details": "reported for sensitivity, not a failure"})
    rows.append({"scope": "global", "metric": "final_argmax_tied_trajectories", "actual": int(canonical[["model_id", "prompt_uid", "final_argmax_tie"]].drop_duplicates()["final_argmax_tie"].sum()), "expected": 0, "pass": int(True), "details": "reported for sensitivity, not a failure"})

    report = {
        "canonical_rows": int(len(canonical)),
        "manifest_rows": int(len(manifest)),
        "layerwise_duplicates": layer_dupes,
        "manifest_duplicates": manifest_dupes,
        "decision_metrics_correct_key_mismatch": correct_key_mismatch,
        "decision_metrics_p_correct_max_abs_error": p_correct_max_abs_err,
        "complete_layer_coverage": complete_ok,
    }
    return pd.DataFrame(rows), report

