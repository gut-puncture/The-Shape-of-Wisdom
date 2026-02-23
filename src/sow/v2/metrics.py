from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LayerDecisionMetrics:
    layer_index: int
    delta: float
    boundary: float
    drift: float
    competitor: str
    p_correct: float
    prob_margin: float
    entropy: float


def _normalize_probs(candidate_probs: Mapping[str, Any]) -> Dict[str, float]:
    vals = np.asarray([float(candidate_probs.get(c, 0.0)) for c in CHOICES], dtype=np.float64)
    s = float(vals.sum())
    if (not np.isfinite(s)) or s <= 0.0:
        vals = np.full((4,), 0.25, dtype=np.float64)
    else:
        vals = vals / s
    return {c: float(vals[i]) for i, c in enumerate(CHOICES)}


def _competitor_from_logits(candidate_logits: Mapping[str, Any], *, correct_key: str) -> str:
    best = None
    best_val = None
    for c in CHOICES:
        if c == correct_key:
            continue
        v = float(candidate_logits.get(c, float("-inf")))
        if (best is None) or (v > float(best_val)):
            best = c
            best_val = v
    return str(best) if best is not None else "A"


def _final_layer_top1_from_logits(row: Mapping[str, Any]) -> str | None:
    layerwise = row.get("layerwise") or []
    if not isinstance(layerwise, list) or not layerwise:
        return None
    last = layerwise[-1] or {}
    logits = last.get("candidate_logits") or {}
    best_choice = None
    best_value = None
    for c in CHOICES:
        try:
            v = float(logits.get(c, float("-inf")))
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if best_choice is None or v > float(best_value):
            best_choice = c
            best_value = v
    return str(best_choice) if best_choice is not None else None


def compute_row_decision_metrics(row: Mapping[str, Any], *, correct_key: str) -> List[LayerDecisionMetrics]:
    layerwise = row.get("layerwise") or []
    if not isinstance(layerwise, list) or not layerwise:
        return []

    ck = str(correct_key).strip().upper()
    if ck not in CHOICES:
        raise ValueError(f"invalid correct_key: {correct_key!r}")

    deltas: List[float] = []
    probs_margin: List[float] = []
    entropies: List[float] = []
    competitors: List[str] = []
    probs_correct: List[float] = []

    for layer in layerwise:
        logits = layer.get("candidate_logits") or {}
        probs = _normalize_probs(layer.get("candidate_probs") or {})
        comp = _competitor_from_logits(logits, correct_key=ck)

        delta = float(logits.get(ck, 0.0)) - float(logits.get(comp, 0.0))
        p_correct = float(probs[ck])
        p_comp = float(probs[comp])
        prob_margin = p_correct - p_comp

        ent = layer.get("candidate_entropy")
        if ent is None:
            p_arr = np.asarray([probs[c] for c in CHOICES], dtype=np.float64)
            ent = -float(np.sum(p_arr * np.log(np.clip(p_arr, 1e-12, 1.0))))

        deltas.append(delta)
        probs_margin.append(prob_margin)
        entropies.append(float(ent))
        competitors.append(comp)
        probs_correct.append(p_correct)

    out: List[LayerDecisionMetrics] = []
    for i, delta in enumerate(deltas):
        drift = 0.0 if i == (len(deltas) - 1) else float(deltas[i + 1] - delta)
        out.append(
            LayerDecisionMetrics(
                layer_index=i,
                delta=float(delta),
                boundary=float(abs(delta)),
                drift=float(drift),
                competitor=str(competitors[i]),
                p_correct=float(probs_correct[i]),
                prob_margin=float(probs_margin[i]),
                entropy=float(entropies[i]),
            )
        )
    return out


def build_decision_metrics_frame(
    rows: Iterable[Mapping[str, Any]],
    *,
    correct_key_by_prompt_uid: Mapping[str, str],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in rows:
        prompt_uid = str(row.get("prompt_uid") or "")
        if not prompt_uid:
            continue
        ck = str(correct_key_by_prompt_uid.get(prompt_uid) or "").strip().upper()
        if ck not in CHOICES:
            continue

        final_top1 = _final_layer_top1_from_logits(row)
        if final_top1 not in CHOICES:
            continue
        is_correct = bool(str(final_top1) == ck)

        layer_metrics = compute_row_decision_metrics(row, correct_key=ck)
        if not layer_metrics:
            continue

        for m in layer_metrics:
            records.append(
                {
                    "model_id": str(row.get("model_id") or ""),
                    "model_revision": str(row.get("model_revision") or ""),
                    "prompt_uid": prompt_uid,
                    "example_id": str(row.get("example_id") or ""),
                    "wrapper_id": str(row.get("wrapper_id") or ""),
                    "coarse_domain": str(row.get("coarse_domain") or "unknown"),
                    "is_correct": bool(is_correct),
                    "correct_key": ck,
                    "layer_index": int(m.layer_index),
                    "delta": float(m.delta),
                    "boundary": float(m.boundary),
                    "drift": float(m.drift),
                    "competitor": str(m.competitor),
                    "p_correct": float(m.p_correct),
                    "prob_margin": float(m.prob_margin),
                    "entropy": float(m.entropy),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_revision",
                "prompt_uid",
                "example_id",
                "wrapper_id",
                "coarse_domain",
                "is_correct",
                "correct_key",
                "layer_index",
                "delta",
                "boundary",
                "drift",
                "competitor",
                "p_correct",
                "prob_margin",
                "entropy",
            ]
        )
    return pd.DataFrame.from_records(records)
