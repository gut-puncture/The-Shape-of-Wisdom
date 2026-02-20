#!/usr/bin/env python3
"""
Deep convergence analysis for baseline run 1452_a.

Scope:
- Baseline-only (no robustness).
- Uses run directory as source of truth.
- Computes metric-space convergence and latent-dynamics metrics from existing outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CHOICES = ["A", "B", "C", "D"]
CHOICE_TO_IDX = {c: i for i, c in enumerate(CHOICES)}

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
}

MODEL_ORDER = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

MODEL_COLORS = {
    "Qwen/Qwen2.5-7B-Instruct": "#1f4e79",
    "meta-llama/Llama-3.1-8B-Instruct": "#7f3b08",
    "mistralai/Mistral-7B-Instruct-v0.3": "#2d6a4f",
}

KNOWN_METRICS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "accuracy": 0.673,
        "final_p_correct_ge_0_8": 0.465,
        "final_p_correct_lt_0_6": 0.411,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "accuracy": 0.607,
        "final_p_correct_ge_0_8": 0.346,
        "final_p_correct_lt_0_6": 0.544,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "accuracy": 0.542,
        "final_p_correct_ge_0_8": 0.435,
        "final_p_correct_lt_0_6": 0.510,
    },
}


@dataclass
class ModelArrays:
    model_id: str
    n_rows: int
    n_layers: int
    prompt_uid: np.ndarray
    example_id: np.ndarray
    coarse_domain: np.ndarray
    subject: np.ndarray
    correct_key_idx: np.ndarray
    is_correct: np.ndarray
    p_correct: np.ndarray
    margin_correct: np.ndarray
    entropy: np.ndarray
    top_idx: np.ndarray
    hidden: np.ndarray
    structure_df: pd.DataFrame


def _model_label(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def _safe_prob_vec(cp: Dict[str, Any]) -> np.ndarray:
    vals = np.array([float(cp.get(c, 0.0)) for c in CHOICES], dtype=np.float64)
    s = float(vals.sum())
    if (not math.isfinite(s)) or s <= 0.0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    vals = vals / s
    vals = np.clip(vals, 0.0, 1.0)
    vals = vals / float(vals.sum() + 1e-12)
    return vals


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _avg_pairwise_jaccard(option_texts: List[str]) -> float:
    sets = [set(_tokenize_words(t)) for t in option_texts]
    scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if not u:
                scores.append(0.0)
            else:
                scores.append(float(len(sets[i] & sets[j])) / float(len(u)))
    return float(np.mean(scores)) if scores else 0.0


def _extract_structure_features(question: str, options: Dict[str, str]) -> Dict[str, float]:
    q = question or ""
    option_texts = [str(options.get(c, "")) for c in CHOICES]
    q_words = _tokenize_words(q)
    opt_word_lens = np.array([len(_tokenize_words(x)) for x in option_texts], dtype=np.float64)
    opt_char_lens = np.array([len(x) for x in option_texts], dtype=np.float64)
    has_negation = int(
        bool(re.search(r"\b(not|except|least|incorrect|false|never|cannot|isn't|aren't|won't|doesn't|don't)\b", q.lower()))
    )
    numeric_option_count = int(sum(bool(re.search(r"\d", x)) for x in option_texts))
    return {
        "question_len_chars": float(len(q)),
        "question_len_words": float(len(q_words)),
        "option_len_words_mean": float(np.mean(opt_word_lens)) if opt_word_lens.size else 0.0,
        "option_len_words_std": float(np.std(opt_word_lens)) if opt_word_lens.size else 0.0,
        "option_len_chars_mean": float(np.mean(opt_char_lens)) if opt_char_lens.size else 0.0,
        "option_len_chars_std": float(np.std(opt_char_lens)) if opt_char_lens.size else 0.0,
        "numeric_option_count": float(numeric_option_count),
        "has_negation": float(has_negation),
        "option_pairwise_jaccard": _avg_pairwise_jaccard(option_texts),
    }


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r.get("prompt_uid") or "")
            ck = str(r.get("correct_key") or "").strip().upper()
            if not uid or ck not in CHOICE_TO_IDX:
                continue
            options = r.get("options") or {}
            rows[uid] = {
                "prompt_uid": uid,
                "example_id": str(r.get("example_id") or ""),
                "coarse_domain": str(r.get("coarse_domain") or "unknown"),
                "subject": str(r.get("subject") or "unknown"),
                "correct_key": ck,
                "question": str(r.get("question") or ""),
                "options": options if isinstance(options, dict) else {},
            }
    if not rows:
        raise RuntimeError(f"No valid manifest rows in {manifest_path}")
    return rows


def load_model(path: Path, manifest: Dict[str, Dict[str, Any]]) -> ModelArrays:
    prompt_uid: List[str] = []
    example_id: List[str] = []
    coarse_domain: List[str] = []
    subject: List[str] = []
    correct_key_idx: List[int] = []
    is_correct: List[bool] = []

    p_correct_rows: List[np.ndarray] = []
    margin_rows: List[np.ndarray] = []
    entropy_rows: List[np.ndarray] = []
    top_rows: List[np.ndarray] = []
    hidden_rows: List[np.ndarray] = []

    structure_records: List[Dict[str, Any]] = []
    model_id: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = str(row.get("prompt_uid") or "")
            m = manifest.get(uid)
            if m is None:
                continue

            lw = row.get("layerwise") or []
            if not isinstance(lw, list) or not lw:
                continue

            ck_idx = CHOICE_TO_IDX[m["correct_key"]]
            pc: List[float] = []
            mg: List[float] = []
            et: List[float] = []
            tp: List[int] = []
            hh: List[np.ndarray] = []
            valid = True

            for layer in lw:
                cp = layer.get("candidate_probs") or {}
                if not isinstance(cp, dict):
                    valid = False
                    break
                probs = _safe_prob_vec(cp)
                p = float(probs[ck_idx])
                others = [float(probs[j]) for j in range(4) if j != ck_idx]
                mrg = p - max(others)
                ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / math.log(4.0)
                top = int(np.argmax(probs))

                ph = layer.get("projected_hidden_128")
                if not isinstance(ph, list):
                    valid = False
                    break
                hv = np.asarray(ph, dtype=np.float32)
                if hv.ndim != 1 or hv.size == 0 or not np.all(np.isfinite(hv)):
                    valid = False
                    break

                if not (math.isfinite(p) and math.isfinite(mrg) and math.isfinite(ent)):
                    valid = False
                    break

                pc.append(p)
                mg.append(mrg)
                et.append(ent)
                tp.append(top)
                hh.append(hv)

            if not valid:
                continue

            if model_id is None:
                model_id = str(row.get("model_id") or path.parent.name)

            prompt_uid.append(uid)
            example_id.append(m["example_id"])
            coarse_domain.append(m["coarse_domain"])
            subject.append(m["subject"])
            correct_key_idx.append(ck_idx)
            is_correct.append(bool(row.get("is_correct") is True))

            p_correct_rows.append(np.asarray(pc, dtype=np.float64))
            margin_rows.append(np.asarray(mg, dtype=np.float64))
            entropy_rows.append(np.asarray(et, dtype=np.float64))
            top_rows.append(np.asarray(tp, dtype=np.int16))
            hidden_rows.append(np.stack(hh, axis=0).astype(np.float32))

            sf = _extract_structure_features(m["question"], m["options"])
            sf.update(
                {
                    "prompt_uid": uid,
                    "example_id": m["example_id"],
                    "coarse_domain": m["coarse_domain"],
                    "subject": m["subject"],
                }
            )
            structure_records.append(sf)

    if not p_correct_rows:
        raise RuntimeError(f"No valid baseline rows in {path}")

    P = np.stack(p_correct_rows, axis=0)
    M = np.stack(margin_rows, axis=0)
    E = np.stack(entropy_rows, axis=0)
    T = np.stack(top_rows, axis=0)
    H = np.stack(hidden_rows, axis=0)

    return ModelArrays(
        model_id=model_id or path.parent.name,
        n_rows=int(P.shape[0]),
        n_layers=int(P.shape[1]),
        prompt_uid=np.asarray(prompt_uid, dtype=object),
        example_id=np.asarray(example_id, dtype=object),
        coarse_domain=np.asarray(coarse_domain, dtype=object),
        subject=np.asarray(subject, dtype=object),
        correct_key_idx=np.asarray(correct_key_idx, dtype=np.int16),
        is_correct=np.asarray(is_correct, dtype=bool),
        p_correct=P,
        margin_correct=M,
        entropy=E,
        top_idx=T,
        hidden=H,
        structure_df=pd.DataFrame(structure_records),
    )


def first_pass_layer(x: np.ndarray, threshold: float) -> np.ndarray:
    hit = x >= float(threshold)
    idx = np.argmax(hit, axis=1)
    ok = hit[np.arange(hit.shape[0]), idx]
    return np.where(ok, idx, -1).astype(np.int32)


def stable_commit_layer(p: np.ndarray, m: np.ndarray, p_thr: float, m_thr: float) -> np.ndarray:
    good = (p >= float(p_thr)) & (m >= float(m_thr))
    suffix_all = np.flip(np.cumprod(np.flip(good.astype(np.int8), axis=1), axis=1), axis=1).astype(bool)
    idx = np.argmax(suffix_all, axis=1)
    ok = suffix_all[np.arange(suffix_all.shape[0]), idx]
    return np.where(ok, idx, -1).astype(np.int32)


def hazard_curve(event_layer: np.ndarray, n_layers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_layer = np.asarray(event_layer, dtype=np.int32)
    hazards = np.zeros((n_layers,), dtype=np.float64)
    at_risk = np.zeros((n_layers,), dtype=np.float64)
    events = np.zeros((n_layers,), dtype=np.float64)
    n = float(event_layer.shape[0])

    for l in range(n_layers):
        risk = np.sum(event_layer >= l)
        ev = np.sum(event_layer == l)
        at_risk[l] = float(risk)
        events[l] = float(ev)
        hazards[l] = float(ev) / float(risk) if risk > 0 else 0.0

    cum = np.cumsum(events) / max(1.0, n)
    return hazards, cum, at_risk


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(1e-12, ((a.size - 1) * va + (b.size - 1) * vb) / float(a.size + b.size - 2)))
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def top_nonoverlap_windows(effect: np.ndarray, window: int, top_k: int = 3) -> List[Tuple[int, int, float]]:
    e = np.asarray(effect, dtype=np.float64)
    n = e.size
    if n == 0:
        return []
    w = max(1, min(int(window), n))
    scores = []
    for s in range(0, n - w + 1):
        v = np.nanmean(np.abs(e[s : s + w]))
        scores.append((s, s + w - 1, float(v)))
    scores.sort(key=lambda x: x[2], reverse=True)

    selected: List[Tuple[int, int, float]] = []
    for s, t, v in scores:
        overlap = any(not (t < s2 or s > t2) for s2, t2, _ in selected)
        if not overlap:
            selected.append((s, t, v))
        if len(selected) >= top_k:
            break
    return selected


def compute_basin_gap(hidden: np.ndarray, key_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, l, d = hidden.shape
    centroids = np.zeros((l, 4, d), dtype=np.float64)

    global_cent = hidden.mean(axis=0).astype(np.float64)
    for c in range(4):
        mask = key_idx == c
        if np.any(mask):
            centroids[:, c, :] = hidden[mask].mean(axis=0)
        else:
            centroids[:, c, :] = global_cent

    basin_gap = np.zeros((n, l), dtype=np.float64)
    for li in range(l):
        h = hidden[:, li, :].astype(np.float64)
        c = centroids[li]  # [4, D]
        dist = np.linalg.norm(h[:, None, :] - c[None, :, :], axis=2)
        dc = dist[np.arange(n), key_idx]
        d_others = dist.copy()
        d_others[np.arange(n), key_idx] = np.inf
        dw = d_others.min(axis=1)
        basin_gap[:, li] = dw - dc

    return basin_gap, centroids


def compute_alignment(hidden: np.ndarray, centroids: np.ndarray, key_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, l, d = hidden.shape
    if l < 2:
        raise RuntimeError("Need at least 2 layers for displacement alignment")

    delta = hidden[:, 1:, :].astype(np.float64) - hidden[:, :-1, :].astype(np.float64)
    step_norm = np.linalg.norm(delta, axis=2)

    align = np.zeros((n, l - 1), dtype=np.float64)
    for li in range(l - 1):
        corr_cent = centroids[li, key_idx, :]  # [N, D]
        direction = corr_cent - hidden[:, li, :].astype(np.float64)
        num = np.sum(delta[:, li, :] * direction, axis=1)
        den = np.linalg.norm(delta[:, li, :], axis=1) * np.linalg.norm(direction, axis=1) + 1e-12
        align[:, li] = num / den

    start_to_end = hidden[:, -1, :].astype(np.float64) - hidden[:, 0, :].astype(np.float64)
    path_length = np.sum(step_norm, axis=1)
    straightness = np.linalg.norm(start_to_end, axis=1) / (path_length + 1e-12)

    if l - 1 >= 2:
        u = delta[:, :-1, :]
        v = delta[:, 1:, :]
        num = np.sum(u * v, axis=2)
        den = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=2) + 1e-12
        cos = np.clip(num / den, -1.0, 1.0)
        curvature = np.mean(np.arccos(cos), axis=1)
    else:
        curvature = np.zeros((n,), dtype=np.float64)

    return align, path_length, straightness, curvature


def build_prompt_level_table(d: ModelArrays, basin_gap: np.ndarray, align: np.ndarray, path_length: np.ndarray, straightness: np.ndarray, curvature: np.ndarray, stable_commit: np.ndarray, first_p80: np.ndarray, first_p60: np.ndarray, first_margin: np.ndarray) -> pd.DataFrame:
    n, l = d.p_correct.shape
    trans = d.top_idx[:, 1:] != d.top_idx[:, :-1]
    lq = int(round(0.25 * (l - 1)))
    early_t = max(1, (l - 1) // 3)

    late_slice = slice(max(0, trans.shape[1] - 2), trans.shape[1])
    late_flip_any = np.any(trans[:, late_slice], axis=1)

    df = pd.DataFrame(
        {
            "model_id": d.model_id,
            "prompt_uid": d.prompt_uid,
            "example_id": d.example_id,
            "coarse_domain": d.coarse_domain,
            "subject": d.subject,
            "is_correct": d.is_correct.astype(int),
            "final_p_correct": d.p_correct[:, -1],
            "final_margin_correct": d.margin_correct[:, -1],
            "final_entropy": d.entropy[:, -1],
            "nonconverged_final_p_lt_0_6": (d.p_correct[:, -1] < 0.6).astype(int),
            "strong_final_p_ge_0_8": (d.p_correct[:, -1] >= 0.8).astype(int),
            "p_correct_l0": d.p_correct[:, 0],
            "p_correct_lq": d.p_correct[:, lq],
            "margin_l0": d.margin_correct[:, 0],
            "margin_lq": d.margin_correct[:, lq],
            "entropy_l0": d.entropy[:, 0],
            "entropy_lq": d.entropy[:, lq],
            "basin_gap_l0": basin_gap[:, 0],
            "basin_gap_lq": basin_gap[:, lq],
            "basin_gap_final": basin_gap[:, -1],
            "alignment_mean": np.mean(align, axis=1),
            "alignment_l0": align[:, 0],
            "alignment_lq": align[:, min(lq, align.shape[1] - 1)],
            "path_length": path_length,
            "straightness": straightness,
            "curvature_proxy": curvature,
            "flip_count": np.sum(trans, axis=1),
            "early_flip_count": np.sum(trans[:, :early_t], axis=1),
            "late_flip_any": late_flip_any.astype(int),
            "stable_commit_layer": stable_commit,
            "first_pass_p80_layer": first_p80,
            "first_pass_p60_layer": first_p60,
            "first_pass_margin_layer": first_margin,
        }
    )

    s = d.structure_df.copy()
    keep_cols = [
        "prompt_uid",
        "question_len_chars",
        "question_len_words",
        "option_len_words_mean",
        "option_len_words_std",
        "option_len_chars_mean",
        "option_len_chars_std",
        "numeric_option_count",
        "has_negation",
        "option_pairwise_jaccard",
    ]
    s = s[keep_cols]
    df = df.merge(s, on="prompt_uid", how="left")
    return df


def compute_predictor_tables(prompt_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "p_correct_l0",
        "p_correct_lq",
        "margin_l0",
        "margin_lq",
        "entropy_l0",
        "entropy_lq",
        "basin_gap_l0",
        "basin_gap_lq",
        "early_flip_count",
        "question_len_words",
        "option_len_words_std",
        "numeric_option_count",
        "has_negation",
        "option_pairwise_jaccard",
    ]

    perf_rows: List[Dict[str, Any]] = []
    coef_rows: List[Dict[str, Any]] = []

    for model_id, sub in prompt_df.groupby("model_id"):
        y = sub["nonconverged_final_p_lt_0_6"].astype(int).to_numpy()
        X = sub[feature_cols].astype(float).to_numpy()

        if len(np.unique(y)) < 2:
            perf_rows.append(
                {
                    "model_id": model_id,
                    "n_rows": int(sub.shape[0]),
                    "class_balance_nonconverged": float(np.mean(y)),
                    "roc_auc": float("nan"),
                    "average_precision": float("nan"),
                }
            )
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        clf = LogisticRegression(max_iter=4000, solver="liblinear")
        clf.fit(x_train_s, y_train)
        prob = clf.predict_proba(x_test_s)[:, 1]

        perf_rows.append(
            {
                "model_id": model_id,
                "n_rows": int(sub.shape[0]),
                "class_balance_nonconverged": float(np.mean(y)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "average_precision": float(average_precision_score(y_test, prob)),
            }
        )

        coefs = clf.coef_[0]
        for feat, coef in zip(feature_cols, coefs):
            coef_rows.append(
                {
                    "model_id": model_id,
                    "feature": feat,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "direction": "higher_feature_increases_failure" if coef > 0 else "higher_feature_reduces_failure",
                }
            )

    perf_df = pd.DataFrame(perf_rows)
    coef_df = pd.DataFrame(coef_rows)
    if not coef_df.empty:
        coef_df = coef_df.sort_values(["model_id", "abs_coefficient"], ascending=[True, False])
    return perf_df, coef_df


def compute_domain_subject_tables(prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    domain = (
        prompt_df.groupby(["model_id", "coarse_domain"], as_index=False)
        .agg(
            n=("prompt_uid", "count"),
            accuracy=("is_correct", "mean"),
            strong_final_p_ge_0_8=("strong_final_p_ge_0_8", "mean"),
            nonconverged_final_p_lt_0_6=("nonconverged_final_p_lt_0_6", "mean"),
            late_flip_rate=("late_flip_any", "mean"),
            mean_final_p_correct=("final_p_correct", "mean"),
        )
        .sort_values(["model_id", "nonconverged_final_p_lt_0_6"], ascending=[True, False])
    )

    subject = (
        prompt_df.groupby(["model_id", "subject"], as_index=False)
        .agg(
            n=("prompt_uid", "count"),
            accuracy=("is_correct", "mean"),
            strong_final_p_ge_0_8=("strong_final_p_ge_0_8", "mean"),
            nonconverged_final_p_lt_0_6=("nonconverged_final_p_lt_0_6", "mean"),
            mean_final_p_correct=("final_p_correct", "mean"),
        )
        .query("n >= 20")
        .sort_values(["model_id", "nonconverged_final_p_lt_0_6"], ascending=[True, False])
    )
    return domain, subject


def compute_cross_model_difficulty(prompt_df: pd.DataFrame, manifest: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    pivot_nonconv = prompt_df.pivot_table(
        index="example_id",
        columns="model_id",
        values="nonconverged_final_p_lt_0_6",
        aggfunc="max",
    )
    pivot_finalp = prompt_df.pivot_table(
        index="example_id",
        columns="model_id",
        values="final_p_correct",
        aggfunc="mean",
    )

    out = pd.DataFrame(index=pivot_nonconv.index)
    for model_id in MODEL_ORDER:
        if model_id in pivot_nonconv.columns:
            out[f"nonconv__{model_id}"] = pivot_nonconv[model_id].fillna(0).astype(int)
        else:
            out[f"nonconv__{model_id}"] = 0

    out["nonconv_model_count"] = out.filter(like="nonconv__").sum(axis=1)
    out["mean_final_p_correct_across_models"] = pivot_finalp.mean(axis=1)

    meta_rows = []
    for uid, m in manifest.items():
        meta_rows.append(
            {
                "example_id": m["example_id"],
                "coarse_domain": m["coarse_domain"],
                "subject": m["subject"],
            }
        )
    meta_df = pd.DataFrame(meta_rows).drop_duplicates(subset=["example_id"])

    out = out.reset_index().merge(meta_df, on="example_id", how="left")
    out = out.sort_values(
        ["nonconv_model_count", "mean_final_p_correct_across_models"],
        ascending=[False, True],
    )
    return out


def compute_failure_modes(prompt_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for model_id, sub in prompt_df.groupby("model_id"):
        strong = sub[sub["strong_final_p_ge_0_8"] == 1]
        nonconv = sub[sub["nonconverged_final_p_lt_0_6"] == 1].copy()
        if nonconv.empty:
            continue

        path_thr = float(np.quantile(strong["path_length"], 0.25)) if not strong.empty else float(np.quantile(sub["path_length"], 0.25))
        align_thr = float(np.quantile(strong["alignment_mean"], 0.25)) if not strong.empty else float(np.quantile(sub["alignment_mean"], 0.25))

        nonconv["insufficient_drift"] = (nonconv["path_length"] < path_thr).astype(int)
        nonconv["wrong_direction"] = ((nonconv["alignment_mean"] < 0.0) | (nonconv["basin_gap_final"] < 0.0)).astype(int)
        nonconv["late_instability"] = (nonconv["late_flip_any"] == 1).astype(int)

        primary: List[str] = []
        for _, r in nonconv.iterrows():
            if int(r["late_instability"]) == 1:
                primary.append("late_instability")
            elif int(r["wrong_direction"]) == 1:
                primary.append("wrong_direction")
            elif int(r["insufficient_drift"]) == 1:
                primary.append("insufficient_drift")
            else:
                primary.append("mixed_other")
        nonconv["primary_failure_mode"] = primary

        n = float(nonconv.shape[0])
        primary_counts = nonconv["primary_failure_mode"].value_counts()
        for mode in ["insufficient_drift", "wrong_direction", "late_instability", "mixed_other"]:
            rows.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "is_primary": 1,
                    "count": int(primary_counts.get(mode, 0)),
                    "fraction_within_nonconverged": float(primary_counts.get(mode, 0) / n),
                }
            )

        for mode in ["insufficient_drift", "wrong_direction", "late_instability"]:
            cnt = int(nonconv[mode].sum())
            rows.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "is_primary": 0,
                    "count": cnt,
                    "fraction_within_nonconverged": float(cnt / n),
                }
            )

    return pd.DataFrame(rows)


def plot_correct_token_dynamics(layer_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )
    models = [m for m in MODEL_ORDER if m in set(layer_df["model_id"]) ]
    fig, axs = plt.subplots(2, len(models), figsize=(5.7 * len(models), 7.2), constrained_layout=True)
    if len(models) == 1:
        axs = np.array(axs).reshape(2, 1)

    for j, model_id in enumerate(models):
        sub = layer_df[layer_df["model_id"] == model_id]
        for outcome, color, label in [("final_correct", "#12355b", "Final correct"), ("final_incorrect", "#8d4f2f", "Final incorrect")]:
            s = sub[sub["outcome"] == outcome]
            axs[0, j].plot(s["normalized_depth"], s["p_correct_mean"], color=color, lw=2.1, label=label)
            axs[1, j].plot(s["normalized_depth"], s["margin_mean"], color=color, lw=2.1, label=label)

        axs[0, j].set_title(_model_label(model_id), fontsize=12)
        axs[0, j].set_ylim(0.0, 1.0)
        axs[1, j].set_ylim(-1.0, 1.0)
        axs[0, j].grid(alpha=0.2)
        axs[1, j].grid(alpha=0.2)
        axs[1, j].set_xlabel("Normalized layer depth")
        axs[0, j].set_ylabel("Mean p(correct)")
        axs[1, j].set_ylabel("Mean margin(correct - max other)")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Correct-Token Dynamics By Final Outcome (Metric Space)", fontsize=15, y=1.02)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_commitment_cumulative(hazard_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )
    models = [m for m in MODEL_ORDER if m in set(hazard_df["model_id"]) ]
    fig, axs = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.1), constrained_layout=True)
    if len(models) == 1:
        axs = [axs]

    for ax, model_id in zip(axs, models):
        sub = hazard_df[hazard_df["model_id"] == model_id]
        for outcome, color, label in [("final_correct", "#12355b", "Final correct"), ("final_incorrect", "#8d4f2f", "Final incorrect")]:
            s = sub[sub["outcome"] == outcome]
            ax.plot(s["normalized_depth"], s["cumulative_commit_fraction"], color=color, lw=2.2, label=label)

        ax.set_title(_model_label(model_id))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("Cumulative stable-commit fraction")
        ax.grid(alpha=0.2)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Stable Commit Emergence (p>=0.70, margin>=0.15)", fontsize=15, y=1.03)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_latent_dynamics(latent_layer_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )

    models = [m for m in MODEL_ORDER if m in set(latent_layer_df["model_id"]) ]
    fig, axs = plt.subplots(2, len(models), figsize=(5.7 * len(models), 7.2), constrained_layout=True)
    if len(models) == 1:
        axs = np.array(axs).reshape(2, 1)

    for j, model_id in enumerate(models):
        sub = latent_layer_df[latent_layer_df["model_id"] == model_id]
        for outcome, color, label in [("final_correct", "#12355b", "Final correct"), ("final_incorrect", "#8d4f2f", "Final incorrect")]:
            s = sub[sub["outcome"] == outcome]
            axs[0, j].plot(s["normalized_depth"], s["basin_gap_mean"], color=color, lw=2.1, label=label)
            axs[1, j].plot(s["normalized_depth"], s["alignment_mean"], color=color, lw=2.1, label=label)

        axs[0, j].axhline(0.0, color="#444", lw=0.8, alpha=0.4)
        axs[1, j].axhline(0.0, color="#444", lw=0.8, alpha=0.4)
        axs[0, j].set_title(_model_label(model_id), fontsize=12)
        axs[0, j].grid(alpha=0.2)
        axs[1, j].grid(alpha=0.2)
        axs[1, j].set_xlabel("Normalized layer depth")
        axs[0, j].set_ylabel("Mean basin gap (dist_wrong - dist_correct)")
        axs[1, j].set_ylabel("Mean alignment to correct-direction")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Latent Dynamics By Final Outcome (Metric Space)", fontsize=15, y=1.02)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_divergence_heatmap(effect_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )
    models = [m for m in MODEL_ORDER if m in set(effect_df["model_id"]) ]
    metrics = ["p_correct", "margin", "basin_gap", "alignment"]

    fig, axs = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.0), constrained_layout=True)
    if len(models) == 1:
        axs = [axs]

    vmax = float(np.nanmax(np.abs(effect_df["cohen_d"]))) if not effect_df.empty else 1.0
    vmax = max(vmax, 0.5)
    for ax, model_id in zip(axs, models):
        sub = effect_df[effect_df["model_id"] == model_id]
        pivot = (
            sub.pivot_table(index="metric", columns="layer_index", values="cohen_d", aggfunc="mean")
            .reindex(metrics)
        )
        arr = pivot.to_numpy(dtype=np.float64)
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_xticks([0, arr.shape[1] // 2, arr.shape[1] - 1])
        ax.set_xticklabels(["early", "mid", "late"])
        ax.set_title(_model_label(model_id), fontsize=12)
        ax.set_xlabel("Layer position")

    cbar = fig.colorbar(im, ax=axs, shrink=0.9)
    cbar.set_label("Cohen's d (strong-converged minus non-converged)")
    fig.suptitle("Divergence Effect Sizes Across Layers", fontsize=15, y=1.05)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_domain_nonconv(domain_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )

    dom_rank = (
        domain_df.groupby("coarse_domain", as_index=False)["nonconverged_final_p_lt_0_6"]
        .mean()
        .sort_values("nonconverged_final_p_lt_0_6", ascending=False)
        .head(12)
    )
    domains = dom_rank["coarse_domain"].tolist()
    sub = domain_df[domain_df["coarse_domain"].isin(domains)].copy()

    x = np.arange(len(domains))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 4.8), constrained_layout=True)
    for i, model_id in enumerate(MODEL_ORDER):
        s = (
            sub[sub["model_id"] == model_id]
            .set_index("coarse_domain")
            .reindex(domains)["nonconverged_final_p_lt_0_6"]
            .fillna(0.0)
            .to_numpy()
        )
        ax.bar(x + (i - 1) * width, s, width=width, label=_model_label(model_id), color=MODEL_COLORS.get(model_id, "#777"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ") for d in domains], rotation=30, ha="right")
    ax.set_ylabel("Non-converged fraction (final p(correct)<0.6)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False)
    ax.set_title("Systematically Hard Domains (Baseline, all 3000 prompts/model)")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_failure_modes(failure_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )
    sub = failure_df[failure_df["is_primary"] == 1].copy()
    modes = ["insufficient_drift", "wrong_direction", "late_instability", "mixed_other"]
    colors = {
        "insufficient_drift": "#7aa6c2",
        "wrong_direction": "#c17f59",
        "late_instability": "#b24a4a",
        "mixed_other": "#8a8a8a",
    }

    fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.6), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    bottom = np.zeros((len(MODEL_ORDER),), dtype=np.float64)

    for mode in modes:
        vals = []
        for model_id in MODEL_ORDER:
            row = sub[(sub["model_id"] == model_id) & (sub["mode"] == mode)]
            vals.append(float(row["fraction_within_nonconverged"].iloc[0]) if not row.empty else 0.0)
        vals_np = np.asarray(vals, dtype=np.float64)
        ax.bar(x, vals_np, bottom=bottom, color=colors[mode], width=0.55, label=mode.replace("_", " "))
        bottom += vals_np

    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in MODEL_ORDER], rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Primary failure-mode fraction within non-converged prompts")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Why Prompts Fail To Converge (Primary mode decomposition)")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_performance(perf_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "savefig.facecolor": "#f7f5ef",
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.3), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    roc_vals = []
    ap_vals = []
    for model_id in MODEL_ORDER:
        s = perf_df[perf_df["model_id"] == model_id]
        if s.empty:
            roc_vals.append(np.nan)
            ap_vals.append(np.nan)
        else:
            roc_vals.append(float(s["roc_auc"].iloc[0]))
            ap_vals.append(float(s["average_precision"].iloc[0]))

    ax.bar(x - width / 2, roc_vals, width=width, color="#1f4e79", alpha=0.85, label="ROC-AUC")
    ax.bar(x + width / 2, ap_vals, width=width, color="#7f3b08", alpha=0.85, label="Average Precision")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in MODEL_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False)
    ax.set_title("Early-layer Predictors Of Final Non-Convergence")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def write_executive_summary(out_path: Path, headline_df: pd.DataFrame, failure_df: pd.DataFrame, perf_df: pd.DataFrame, hard_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Executive Summary: Deep Convergence Analysis (Run 1452_a, Baseline Only)")
    lines.append("")
    lines.append("## Measured Facts")
    lines.append("")
    for _, r in headline_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: accuracy `{100.0*float(r['accuracy']):.1f}%`, final `p(correct)>=0.8` `{100.0*float(r['final_p_correct_ge_0_8']):.1f}%`, final non-converged `p(correct)<0.6` `{100.0*float(r['final_p_correct_lt_0_6']):.1f}%`."
        )
    lines.append("")
    if not hard_df.empty:
        n3 = int((hard_df["nonconv_model_count"] == 3).sum())
        n2 = int((hard_df["nonconv_model_count"] >= 2).sum())
        lines.append(f"- Cross-model hard prompts: `{n3}` prompts are non-converged in all 3 models; `{n2}` prompts are non-converged in at least 2 models.")
        lines.append("")

    lines.append("## Failure Pattern Snapshot")
    lines.append("")
    prim = failure_df[failure_df["is_primary"] == 1]
    for model_id in MODEL_ORDER:
        sub = prim[prim["model_id"] == model_id]
        if sub.empty:
            continue
        top = sub.sort_values("fraction_within_nonconverged", ascending=False).iloc[0]
        lines.append(
            f"- {_model_label(model_id)}: largest primary failure mode is `{str(top['mode']).replace('_', ' ')}` at `{100.0*float(top['fraction_within_nonconverged']):.1f}%` of non-converged prompts."
        )
    lines.append("")

    lines.append("## Predictability From Early Layers")
    lines.append("")
    for _, r in perf_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: early-layer predictor ROC-AUC `{float(r['roc_auc']):.3f}`, AP `{float(r['average_precision']):.3f}`."
        )
    lines.append("")

    lines.append("## Scope Note")
    lines.append("")
    lines.append("- This summary uses baseline run `1452_a` only. Robustness/wrapper analyses are intentionally excluded in this session.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_deep_report(
    out_path: Path,
    run_dir: Path,
    definitions: Dict[str, Any],
    headline_df: pd.DataFrame,
    verify_df: pd.DataFrame,
    commitment_df: pd.DataFrame,
    latent_summary_df: pd.DataFrame,
    divergence_windows_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    hard_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# DEEP_CONVERGENCE_REPORT")
    lines.append("")
    lines.append("## Study Scope")
    lines.append("")
    lines.append(f"- Run source of truth: `{run_dir}`")
    lines.append("- Data scope: baseline outputs only (`3000` prompts per model).")
    lines.append("- Robustness scope: excluded in this session by design.")
    lines.append("- Metric/display discipline: all core metrics are computed in raw metric space; figures use linear depth axes with no geometric warping.")
    lines.append("")

    lines.append("## Convergence Definitions (Formal)")
    lines.append("")
    lines.append("For each prompt `i` and layer `l` with 4-choice probabilities `p_i^A(l)...p_i^D(l)` and correct option `c_i`:")
    lines.append("")
    lines.append("1. Correct-token probability")
    lines.append("- `p_correct,i(l) = p_i^{c_i}(l)`")
    lines.append("2. Correct margin")
    lines.append("- `m_correct,i(l) = p_i^{c_i}(l) - max_{k != c_i} p_i^k(l)`")
    lines.append("3. Probability-target convergence")
    lines.append("- `C_prob^tau(i) = 1[p_correct,i(L-1) >= tau]`, with `tau in {0.6, 0.8}`")
    lines.append("4. Margin-target convergence")
    lines.append("- `C_margin^gamma(i) = 1[m_correct,i(L-1) >= gamma]`, with `gamma = 0.15`")
    lines.append("5. Stability-through-end commitment")
    lines.append("- `l*_i = min l such that for all t>=l, p_correct,i(t)>=0.70 and m_correct,i(t)>=0.15` (else undefined)")
    lines.append("6. Basin-distance convergence")
    lines.append("- Let `h_i(l)` be `projected_hidden_128` at layer `l`.")
    lines.append("- Let `mu_c(l)` be the layer-`l` centroid for prompts with correct key `c`.")
    lines.append("- `gap_i(l) = min_{k!=c_i} ||h_i(l)-mu_k(l)||_2 - ||h_i(l)-mu_{c_i}(l)||_2`.")
    lines.append("- Basin converged at end if `gap_i(L-1) >= 0`; stable basin commitment is the earliest layer where this remains true through end.")
    lines.append("")

    lines.append("## Verification Against Known Baseline Observations")
    lines.append("")
    for _, r in verify_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: recomputed accuracy `{100.0*float(r['accuracy']):.1f}%`, final `p(correct)>=0.8` `{100.0*float(r['final_p_correct_ge_0_8']):.1f}%`, final `p(correct)<0.6` `{100.0*float(r['final_p_correct_lt_0_6']):.1f}%`; absolute diffs vs known = `{float(r['max_abs_diff_vs_known']):.4f}`."
        )
    lines.append("")

    lines.append("## Correct-Token Dynamics")
    lines.append("")
    for _, r in commitment_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: stable-commit detected in `{100.0*float(r['stable_commit_valid_fraction']):.1f}%` of prompts; among final-correct prompts, `{100.0*float(r['final_correct_commit_before_last_two_layers']):.1f}%` commit before the last two layers; late-flip rate = `{100.0*float(r['late_flip_rate_all']):.1f}%` overall / `{100.0*float(r['late_flip_rate_nonconverged']):.1f}%` within non-converged."
        )
    lines.append("")

    lines.append("## Latent-Space Dynamics")
    lines.append("")
    for _, r in latent_summary_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: mean path length strong-converged vs non-converged = `{float(r['path_length_strong_mean']):.3f}` vs `{float(r['path_length_nonconv_mean']):.3f}`; mean alignment = `{float(r['alignment_strong_mean']):.3f}` vs `{float(r['alignment_nonconv_mean']):.3f}`; mean curvature = `{float(r['curvature_strong_mean']):.3f}` vs `{float(r['curvature_nonconv_mean']):.3f}`."
        )
    lines.append("")

    lines.append("### Highest-Divergence Layer Windows")
    lines.append("")
    for _, r in divergence_windows_df.sort_values(["model_id", "metric", "rank"]).iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))} | `{r['metric']}` window rank {int(r['rank'])}: layers `{int(r['start_layer'])}`-`{int(r['end_layer'])}` (mean |d|=`{float(r['window_abs_effect']):.3f}`)."
        )
    lines.append("")

    lines.append("## Prompt-Level Difficulty Structure")
    lines.append("")
    if not hard_df.empty:
        n3 = int((hard_df["nonconv_model_count"] == 3).sum())
        n2 = int((hard_df["nonconv_model_count"] >= 2).sum())
        lines.append(f"- `{n3}` prompts are non-converged in all 3 models; `{n2}` prompts are non-converged in at least 2 models.")
    hard_domains = (
        domain_df.groupby("coarse_domain", as_index=False)["nonconverged_final_p_lt_0_6"]
        .mean()
        .sort_values("nonconverged_final_p_lt_0_6", ascending=False)
        .head(8)
    )
    if not hard_domains.empty:
        dom_text = ", ".join(
            [f"{r.coarse_domain} ({100.0*float(r.nonconverged_final_p_lt_0_6):.1f}%)" for r in hard_domains.itertuples()]
        )
        lines.append(f"- Hardest domains by mean non-convergence rate: {dom_text}.")
    lines.append("")

    lines.append("## Early-Layer Predictors Of Failure")
    lines.append("")
    for _, r in perf_df.sort_values("model_id").iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: ROC-AUC `{float(r['roc_auc']):.3f}`, AP `{float(r['average_precision']):.3f}` from early-layer + structure features."
        )
    lines.append("")
    lines.append("Top positive coefficients (higher -> higher failure odds):")
    top_coef = (
        coef_df[coef_df["coefficient"] > 0]
        .sort_values(["model_id", "abs_coefficient"], ascending=[True, False])
        .groupby("model_id")
        .head(4)
    )
    for _, r in top_coef.iterrows():
        lines.append(
            f"- {_model_label(str(r['model_id']))}: `{str(r['feature'])}` coef `{float(r['coefficient']):.3f}`."
        )
    lines.append("")

    lines.append("## Failure Mechanism Decomposition")
    lines.append("")
    prim = failure_df[failure_df["is_primary"] == 1]
    for model_id in MODEL_ORDER:
        sub = prim[prim["model_id"] == model_id]
        if sub.empty:
            continue
        parts = []
        for mode in ["insufficient_drift", "wrong_direction", "late_instability", "mixed_other"]:
            rr = sub[sub["mode"] == mode]
            frac = float(rr["fraction_within_nonconverged"].iloc[0]) if not rr.empty else 0.0
            parts.append(f"{mode}={100.0*frac:.1f}%")
        lines.append(f"- {_model_label(model_id)}: " + ", ".join(parts) + ".")
    lines.append("")

    lines.append("## Assumptions And Caveats")
    lines.append("")
    lines.append("- Hidden geometry is measured in `projected_hidden_128`, so latent distances are PCA-subspace distances, not full hidden-state distances.")
    lines.append("- Centroids are computed from the same baseline dataset (no held-out centroid split), so basin metrics are descriptive, not causal proofs.")
    lines.append("- Predictor models are interpretable baselines, not optimized classifiers.")
    lines.append("- This report intentionally excludes robustness/wrapper effects.")
    lines.append("")

    lines.append("## What AI researchers can do now to increase convergence-to-correct")
    lines.append("")
    lines.append("1. Prompt-level lever: reduce early uncertainty on hard prompt families.")
    lines.append("- Mechanism: larger early entropy and lower early margin predict later non-convergence.")
    lines.append("- Expected effect: earlier entry into the correct basin and fewer late reversals.")
    lines.append("- Minimal validation: for top hard domains, compare baseline prompts vs clarity-edited prompts and measure first-pass/hazard shifts with the same analysis stack.")
    lines.append("2. Inference-time lever: stabilize after first strong commit, not only at final layer.")
    lines.append("- Mechanism: many final-correct prompts commit before the last two layers; non-converged prompts exhibit elevated late flips.")
    lines.append("- Expected effect: lower late-instability errors without retraining.")
    lines.append("- Minimal validation: run a two-pass decode policy that enforces consistency checks after stable-commit windows and compare non-convergence rate.")
    lines.append("3. Training/objective lever: optimize for correct-direction drift and late-layer stability.")
    lines.append("- Mechanism: non-converged prompts show weaker alignment toward correct-direction vectors and weaker basin-gap separation.")
    lines.append("- Expected effect: higher margin and basin separation by late-middle layers.")
    lines.append("- Minimal validation: fine-tune with auxiliary loss terms for correct-margin growth and reduced late flips; evaluate on baseline prompts with the same metrics.")
    lines.append("4. Representation-steering lever: target the highest-divergence windows.")
    lines.append("- Mechanism: converged vs non-converged trajectories diverge most in specific layer windows (reported above).")
    lines.append("- Expected effect: concentrated steering effort where decision pathways separate most.")
    lines.append("- Minimal validation: apply small activation steering only within top divergence windows and measure changes in `p_correct`, margin, and late flips.")
    lines.append("")

    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append("- E1: Prompt-clarity intervention on top-50 hardest prompts (`>=2/3` model failures).")
    lines.append("- E2: Commit-aware decoding guardrail focused on late-layer flip suppression.")
    lines.append("- E3: Objective-level fine-tuning with margin-growth and stability losses.")
    lines.append("- E4: Layer-window steering ablation on top divergence windows vs adjacent control windows.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifests" / "baseline_manifest.jsonl"
    model_paths = sorted((run_dir / "outputs").glob("*/baseline_outputs.jsonl"))
    if not manifest_path.exists():
        raise RuntimeError(f"Missing baseline manifest: {manifest_path}")
    if not model_paths:
        raise RuntimeError(f"No baseline outputs found in {run_dir / 'outputs'}")

    manifest = load_manifest(manifest_path)

    # Definitions and thresholds used throughout.
    definitions = {
        "probability_target_thresholds": [0.6, 0.8],
        "margin_target_threshold": 0.15,
        "stable_commit": {"p_threshold": 0.70, "margin_threshold": 0.15},
        "basin_distance_threshold": 0.0,
    }

    model_data: Dict[str, ModelArrays] = {}
    headline_rows: List[Dict[str, Any]] = []
    verify_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    hazard_rows: List[Dict[str, Any]] = []
    latent_layer_rows: List[Dict[str, Any]] = []
    latent_summary_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    divergence_window_rows: List[Dict[str, Any]] = []

    prompt_tables: List[pd.DataFrame] = []

    for path in model_paths:
        d = load_model(path, manifest)
        model_data[d.model_id] = d

        if d.n_rows != 3000:
            raise RuntimeError(f"Expected 3000 rows for {d.model_id}, found {d.n_rows}")

        final_p = d.p_correct[:, -1]
        accuracy = float(np.mean(d.is_correct))
        strong = float(np.mean(final_p >= 0.8))
        nonconv = float(np.mean(final_p < 0.6))

        headline_rows.append(
            {
                "model_id": d.model_id,
                "n_rows": d.n_rows,
                "n_layers": d.n_layers,
                "accuracy": accuracy,
                "final_p_correct_ge_0_8": strong,
                "final_p_correct_lt_0_6": nonconv,
            }
        )

        known = KNOWN_METRICS.get(d.model_id, {})
        diffs = []
        for k in ["accuracy", "final_p_correct_ge_0_8", "final_p_correct_lt_0_6"]:
            if k in known:
                diffs.append(abs(({"accuracy": accuracy, "final_p_correct_ge_0_8": strong, "final_p_correct_lt_0_6": nonconv}[k]) - known[k]))
        verify_rows.append(
            {
                "model_id": d.model_id,
                "accuracy": accuracy,
                "final_p_correct_ge_0_8": strong,
                "final_p_correct_lt_0_6": nonconv,
                "known_accuracy": known.get("accuracy", np.nan),
                "known_final_p_correct_ge_0_8": known.get("final_p_correct_ge_0_8", np.nan),
                "known_final_p_correct_lt_0_6": known.get("final_p_correct_lt_0_6", np.nan),
                "max_abs_diff_vs_known": float(max(diffs) if diffs else np.nan),
            }
        )

        # Convergence events.
        first_p80 = first_pass_layer(d.p_correct, 0.8)
        first_p60 = first_pass_layer(d.p_correct, 0.6)
        first_margin = first_pass_layer(d.margin_correct, 0.15)
        stable_commit = stable_commit_layer(
            d.p_correct,
            d.margin_correct,
            p_thr=float(definitions["stable_commit"]["p_threshold"]),
            m_thr=float(definitions["stable_commit"]["margin_threshold"]),
        )

        # Flip/oscillation diagnostics.
        trans = d.top_idx[:, 1:] != d.top_idx[:, :-1]
        flip_count = np.sum(trans, axis=1)
        late_flip_any = np.any(trans[:, max(0, trans.shape[1] - 2) :], axis=1)

        # Correct-token dynamics by layer.
        for outcome_name, mask in [("final_correct", d.is_correct), ("final_incorrect", ~d.is_correct)]:
            if not np.any(mask):
                continue
            for li in range(d.n_layers):
                layer_rows.append(
                    {
                        "model_id": d.model_id,
                        "outcome": outcome_name,
                        "layer_index": li,
                        "normalized_depth": float(li / max(1, d.n_layers - 1)),
                        "p_correct_mean": float(np.mean(d.p_correct[mask, li])),
                        "p_correct_median": float(np.median(d.p_correct[mask, li])),
                        "margin_mean": float(np.mean(d.margin_correct[mask, li])),
                        "margin_median": float(np.median(d.margin_correct[mask, li])),
                        "entropy_mean": float(np.mean(d.entropy[mask, li])),
                    }
                )

        # Commitment hazard/cumulative curves.
        for outcome_name, mask in [("final_correct", d.is_correct), ("final_incorrect", ~d.is_correct)]:
            if not np.any(mask):
                continue
            hz, cum, risk = hazard_curve(stable_commit[mask], d.n_layers)
            for li in range(d.n_layers):
                hazard_rows.append(
                    {
                        "model_id": d.model_id,
                        "outcome": outcome_name,
                        "layer_index": li,
                        "normalized_depth": float(li / max(1, d.n_layers - 1)),
                        "hazard": float(hz[li]),
                        "cumulative_commit_fraction": float(cum[li]),
                        "at_risk": float(risk[li]),
                    }
                )

        # Latent-space metrics.
        basin_gap, centroids = compute_basin_gap(d.hidden, d.correct_key_idx)
        align, path_length, straightness, curvature = compute_alignment(d.hidden, centroids, d.correct_key_idx)
        basin_stable_commit = first_pass_layer(basin_gap, 0.0)

        for outcome_name, mask in [("final_correct", d.is_correct), ("final_incorrect", ~d.is_correct)]:
            if not np.any(mask):
                continue
            for li in range(d.n_layers):
                ali = min(li, align.shape[1] - 1)
                latent_layer_rows.append(
                    {
                        "model_id": d.model_id,
                        "outcome": outcome_name,
                        "layer_index": li,
                        "normalized_depth": float(li / max(1, d.n_layers - 1)),
                        "basin_gap_mean": float(np.mean(basin_gap[mask, li])),
                        "alignment_mean": float(np.mean(align[mask, ali])),
                        "step_norm_mean": float(np.mean(np.linalg.norm(d.hidden[mask, min(li + 1, d.n_layers - 1), :] - d.hidden[mask, max(0, li - 1 if li > 0 else 0), :], axis=1))),
                    }
                )

        strong_mask = final_p >= 0.8
        nonconv_mask = final_p < 0.6

        latent_summary_rows.append(
            {
                "model_id": d.model_id,
                "path_length_strong_mean": float(np.mean(path_length[strong_mask])) if np.any(strong_mask) else np.nan,
                "path_length_nonconv_mean": float(np.mean(path_length[nonconv_mask])) if np.any(nonconv_mask) else np.nan,
                "straightness_strong_mean": float(np.mean(straightness[strong_mask])) if np.any(strong_mask) else np.nan,
                "straightness_nonconv_mean": float(np.mean(straightness[nonconv_mask])) if np.any(nonconv_mask) else np.nan,
                "curvature_strong_mean": float(np.mean(curvature[strong_mask])) if np.any(strong_mask) else np.nan,
                "curvature_nonconv_mean": float(np.mean(curvature[nonconv_mask])) if np.any(nonconv_mask) else np.nan,
                "alignment_strong_mean": float(np.mean(align[strong_mask])) if np.any(strong_mask) else np.nan,
                "alignment_nonconv_mean": float(np.mean(align[nonconv_mask])) if np.any(nonconv_mask) else np.nan,
                "basin_gap_final_strong_mean": float(np.mean(basin_gap[strong_mask, -1])) if np.any(strong_mask) else np.nan,
                "basin_gap_final_nonconv_mean": float(np.mean(basin_gap[nonconv_mask, -1])) if np.any(nonconv_mask) else np.nan,
            }
        )

        # Divergence analysis.
        if np.any(strong_mask) and np.any(nonconv_mask):
            metric_map = {
                "p_correct": d.p_correct,
                "margin": d.margin_correct,
                "basin_gap": basin_gap,
                "alignment": align,
            }
            for metric, arr in metric_map.items():
                arr = np.asarray(arr)
                for li in range(arr.shape[1]):
                    eff = cohen_d(arr[strong_mask, li], arr[nonconv_mask, li])
                    effect_rows.append(
                        {
                            "model_id": d.model_id,
                            "metric": metric,
                            "layer_index": int(li),
                            "normalized_depth": float(li / max(1, arr.shape[1] - 1)),
                            "cohen_d": float(eff),
                        }
                    )

                eff_vec = np.array([cohen_d(arr[strong_mask, li], arr[nonconv_mask, li]) for li in range(arr.shape[1])], dtype=np.float64)
                wins = top_nonoverlap_windows(eff_vec, window=max(2, arr.shape[1] // 6), top_k=3)
                for rank, (s, e, v) in enumerate(wins, start=1):
                    divergence_window_rows.append(
                        {
                            "model_id": d.model_id,
                            "metric": metric,
                            "rank": rank,
                            "start_layer": int(s),
                            "end_layer": int(e),
                            "window_abs_effect": float(v),
                        }
                    )

        # Commitment summary row.
        commit_valid = stable_commit >= 0
        correct_mask = d.is_correct
        correct_commit_before_last_two = np.mean((stable_commit[correct_mask] >= 0) & (stable_commit[correct_mask] <= d.n_layers - 3)) if np.any(correct_mask) else np.nan
        commitment_summary = {
            "model_id": d.model_id,
            "stable_commit_valid_fraction": float(np.mean(commit_valid)),
            "stable_commit_median_layer": float(np.median(stable_commit[commit_valid])) if np.any(commit_valid) else np.nan,
            "final_correct_commit_before_last_two_layers": float(correct_commit_before_last_two),
            "late_flip_rate_all": float(np.mean(late_flip_any)),
            "late_flip_rate_nonconverged": float(np.mean(late_flip_any[nonconv_mask])) if np.any(nonconv_mask) else np.nan,
            "flip_count_mean": float(np.mean(flip_count)),
            "flip_count_median": float(np.median(flip_count)),
        }

        # store as singleton frame for later concatenation
        prompt_df = build_prompt_level_table(
            d=d,
            basin_gap=basin_gap,
            align=align,
            path_length=path_length,
            straightness=straightness,
            curvature=curvature,
            stable_commit=stable_commit,
            first_p80=first_p80,
            first_p60=first_p60,
            first_margin=first_margin,
        )
        prompt_df["basin_stable_commit_layer"] = basin_stable_commit
        prompt_df["model_commit_summary_stable_commit_valid_fraction"] = commitment_summary["stable_commit_valid_fraction"]
        prompt_tables.append(prompt_df)

        hazard_rows.append(
            {
                "model_id": d.model_id,
                "outcome": "_summary_",
                "layer_index": -1,
                "normalized_depth": -1.0,
                "hazard": np.nan,
                "cumulative_commit_fraction": np.nan,
                "at_risk": np.nan,
                "stable_commit_valid_fraction": commitment_summary["stable_commit_valid_fraction"],
                "stable_commit_median_layer": commitment_summary["stable_commit_median_layer"],
                "final_correct_commit_before_last_two_layers": commitment_summary["final_correct_commit_before_last_two_layers"],
                "late_flip_rate_all": commitment_summary["late_flip_rate_all"],
                "late_flip_rate_nonconverged": commitment_summary["late_flip_rate_nonconverged"],
                "flip_count_mean": commitment_summary["flip_count_mean"],
                "flip_count_median": commitment_summary["flip_count_median"],
            }
        )

    headline_df = pd.DataFrame(headline_rows).sort_values("model_id")
    verify_df = pd.DataFrame(verify_rows).sort_values("model_id")
    layer_df = pd.DataFrame(layer_rows).sort_values(["model_id", "outcome", "layer_index"])
    hazard_df = pd.DataFrame(hazard_rows)
    hazard_layer_df = hazard_df[hazard_df["outcome"] != "_summary_"].copy()
    commitment_df = hazard_df[hazard_df["outcome"] == "_summary_"].copy()
    latent_layer_df = pd.DataFrame(latent_layer_rows).sort_values(["model_id", "outcome", "layer_index"])
    latent_summary_df = pd.DataFrame(latent_summary_rows).sort_values("model_id")
    effect_df = pd.DataFrame(effect_rows).sort_values(["model_id", "metric", "layer_index"])
    divergence_windows_df = pd.DataFrame(divergence_window_rows).sort_values(["model_id", "metric", "rank"])

    prompt_df = pd.concat(prompt_tables, axis=0, ignore_index=True)

    domain_df, subject_df = compute_domain_subject_tables(prompt_df)
    hard_df = compute_cross_model_difficulty(prompt_df, manifest)
    failure_df = compute_failure_modes(prompt_df)
    perf_df, coef_df = compute_predictor_tables(prompt_df, seed=int(args.seed))

    # Tables.
    (out_dir / "convergence_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    headline_df.to_csv(out_dir / "headline_metrics.csv", index=False)
    verify_df.to_csv(out_dir / "known_observation_verification.csv", index=False)
    layer_df.to_csv(out_dir / "correct_token_dynamics_by_layer.csv", index=False)
    hazard_layer_df.to_csv(out_dir / "commitment_hazard_curves.csv", index=False)
    commitment_df.to_csv(out_dir / "commitment_summary.csv", index=False)
    latent_layer_df.to_csv(out_dir / "latent_dynamics_by_layer.csv", index=False)
    latent_summary_df.to_csv(out_dir / "latent_summary_metrics.csv", index=False)
    effect_df.to_csv(out_dir / "divergence_effects_by_layer.csv", index=False)
    divergence_windows_df.to_csv(out_dir / "divergence_windows.csv", index=False)
    prompt_df.to_csv(out_dir / "prompt_level_metrics.csv", index=False)
    domain_df.to_csv(out_dir / "difficulty_by_domain.csv", index=False)
    subject_df.to_csv(out_dir / "difficulty_by_subject.csv", index=False)
    hard_df.to_csv(out_dir / "cross_model_hard_prompts.csv", index=False)
    failure_df.to_csv(out_dir / "failure_mode_breakdown.csv", index=False)
    perf_df.to_csv(out_dir / "early_failure_predictor_performance.csv", index=False)
    coef_df.to_csv(out_dir / "early_failure_predictor_coefficients.csv", index=False)

    # Figures.
    plot_correct_token_dynamics(layer_df, fig_dir / "correct_token_dynamics_by_outcome.png")
    plot_commitment_cumulative(hazard_layer_df, fig_dir / "commitment_cumulative_curves.png")
    plot_latent_dynamics(latent_layer_df, fig_dir / "latent_dynamics_by_outcome.png")
    if not effect_df.empty:
        plot_divergence_heatmap(effect_df, fig_dir / "divergence_effect_heatmap.png")
    plot_domain_nonconv(domain_df, fig_dir / "hard_domain_nonconvergence_rates.png")
    if not failure_df.empty:
        plot_failure_modes(failure_df, fig_dir / "failure_mode_primary_breakdown.png")
    if not perf_df.empty:
        plot_predictor_performance(perf_df, fig_dir / "early_predictor_performance.png")

    write_executive_summary(
        out_path=out_dir / "EXECUTIVE_SUMMARY.md",
        headline_df=headline_df,
        failure_df=failure_df,
        perf_df=perf_df,
        hard_df=hard_df,
    )

    write_deep_report(
        out_path=out_dir / "DEEP_CONVERGENCE_REPORT.md",
        run_dir=run_dir,
        definitions=definitions,
        headline_df=headline_df,
        verify_df=verify_df,
        commitment_df=commitment_df,
        latent_summary_df=latent_summary_df,
        divergence_windows_df=divergence_windows_df,
        domain_df=domain_df,
        perf_df=perf_df,
        coef_df=coef_df,
        failure_df=failure_df,
        hard_df=hard_df,
    )

    # Deep technical report alias for explicit output requirement.
    tech_alias = out_dir / "TECHNICAL_REPORT.md"
    tech_alias.write_text((out_dir / "DEEP_CONVERGENCE_REPORT.md").read_text(encoding="utf-8"), encoding="utf-8")

    # Run-level summary JSON.
    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "baseline_only": True,
        "n_models": int(len(headline_df)),
        "headline_metrics": headline_df.to_dict(orient="records"),
        "verification": verify_df.to_dict(orient="records"),
        "files": {
            "definitions": str(out_dir / "convergence_definitions.json"),
            "headline_metrics": str(out_dir / "headline_metrics.csv"),
            "verification": str(out_dir / "known_observation_verification.csv"),
            "prompt_level_metrics": str(out_dir / "prompt_level_metrics.csv"),
            "report": str(out_dir / "DEEP_CONVERGENCE_REPORT.md"),
            "executive_summary": str(out_dir / "EXECUTIVE_SUMMARY.md"),
            "figures_dir": str(fig_dir),
        },
        "metric_vs_display_space": {
            "metric_space": "all calculations use raw layer indices and raw probabilities/distances with no axis warping",
            "display_space": "figures use linear normalized depth only; no geometric forcing applied",
        },
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
