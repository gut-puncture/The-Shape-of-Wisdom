from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def delete_span(prompt_text: str, *, start_char: int, end_char: int) -> str:
    text = str(prompt_text or "")
    s = max(0, min(int(start_char), len(text)))
    e = max(s, min(int(end_char), len(text)))
    return text[:s] + text[e:]


def replace_span(prompt_text: str, *, start_char: int, end_char: int, replacement: str) -> str:
    text = str(prompt_text or "")
    s = max(0, min(int(start_char), len(text)))
    e = max(s, min(int(end_char), len(text)))
    return text[:s] + str(replacement) + text[e:]


def compute_span_effect(*, full_delta: float, mutated_delta: float) -> float:
    return float(full_delta) - float(mutated_delta)


def label_span_effect(effect: float, *, evidence_threshold: float = 0.05, distractor_threshold: float = -0.05) -> str:
    v = float(effect)
    if v >= float(evidence_threshold):
        return "evidence"
    if v <= float(distractor_threshold):
        return "distractor"
    return "neutral"


def label_span_effects(
    df: pd.DataFrame,
    *,
    evidence_threshold: float = 0.05,
    distractor_threshold: float = -0.05,
    output_col: str = "span_label",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out[str(output_col)] = [
        label_span_effect(x, evidence_threshold=evidence_threshold, distractor_threshold=distractor_threshold)
        for x in out["effect_delta"].astype(float).tolist()
    ]
    return out


def completed_span_keys_for_mode(df: pd.DataFrame, *, mode: str) -> set[tuple[str, str, str]]:
    """
    Return completed (model_id, prompt_uid, span_id) keys valid for the requested mode.
    This prevents proxy rows from blocking recomputation when mode='model'.
    """
    if df.empty:
        return set()
    m = str(mode)
    sub = df
    if "counterfactual_mode" in sub.columns:
        sub = sub[sub["counterfactual_mode"].astype(str) == m]
    elif m == "model" and "effect_source" in sub.columns:
        sub = sub[sub["effect_source"].astype(str) == "model_counterfactual"]
    cols = ["model_id", "prompt_uid", "span_id"]
    if not all(c in sub.columns for c in cols):
        return set()
    return {
        (str(r["model_id"]), str(r["prompt_uid"]), str(r["span_id"]))
        for _, r in sub[cols].drop_duplicates().iterrows()
    }


def aggregate_label_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    vc = df["span_label"].value_counts(normalize=True).to_dict()
    return {str(k): float(v) for k, v in vc.items()}
