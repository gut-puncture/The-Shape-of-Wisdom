from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np

from sow.v2.span_counterfactuals import compute_span_effect, label_span_effect
from sow.v2.span_parser import parse_prompt_spans

_OPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:[\(\[]\s*[A-D]\s*[\)\]]|[A-D][\)\.:])\s+",
    flags=re.IGNORECASE,
)

_REPLACEMENTS = [
    (re.compile(r"\bQuestion\b", flags=re.IGNORECASE), "Prompt"),
    (re.compile(r"\bRead\b", flags=re.IGNORECASE), "Review"),
    (re.compile(r"\bAnswer\b", flags=re.IGNORECASE), "Final answer"),
    (re.compile(r"\bChoose\b", flags=re.IGNORECASE), "Select"),
]


def deterministic_paraphrase(prompt_text: str) -> str:
    lines = str(prompt_text or "").splitlines()
    out: List[str] = []
    for line in lines:
        if _OPTION_LINE_RE.match(line):
            out.append(line)
            continue
        rewritten = line
        for pat, rep in _REPLACEMENTS:
            rewritten = pat.sub(rep, rewritten)
        out.append(rewritten)
    return "\n".join(out)


def proxy_mutated_delta(*, full_delta: float, span_role: str, span_len: int, prompt_len: int, correct_key: str) -> float:
    ratio = float(span_len) / max(1.0, float(prompt_len))
    if span_role == "question_stem":
        effect = 0.30 * ratio + 0.02
    elif span_role.startswith("option_") and span_role.endswith(str(correct_key)):
        effect = 0.35 * ratio + 0.02
    elif span_role.startswith("option_"):
        effect = -0.20 * ratio
    elif span_role == "instruction":
        effect = 0.05 * ratio
    else:
        effect = -0.03 * ratio
    return float(full_delta - effect)


def proxy_span_effect_labels(
    *,
    prompt_text: str,
    full_delta: float,
    correct_key: str,
    evidence_threshold: float = 0.05,
    distractor_threshold: float = -0.05,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for span in parse_prompt_spans(prompt_text):
        mutated = proxy_mutated_delta(
            full_delta=float(full_delta),
            span_role=str(span.label),
            span_len=int(span.end_char - span.start_char),
            prompt_len=len(str(prompt_text or "")),
            correct_key=str(correct_key),
        )
        effect = compute_span_effect(full_delta=float(full_delta), mutated_delta=float(mutated))
        labels[str(span.label)] = label_span_effect(
            float(effect),
            evidence_threshold=float(evidence_threshold),
            distractor_threshold=float(distractor_threshold),
        )
    return labels


def span_label_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(str(x) for x in a)
    sb = set(str(x) for x in b)
    union = sa | sb
    if not union:
        return 1.0
    inter = sa & sb
    return float(len(inter) / len(union))


def score_prompt_paraphrase(
    *,
    prompt_text: str,
    full_delta: float,
    correct_key: str,
    paraphrased_text: str | None = None,
    evidence_threshold: float = 0.05,
    distractor_threshold: float = -0.05,
) -> Dict[str, float]:
    para = str(paraphrased_text) if paraphrased_text is not None else deterministic_paraphrase(prompt_text)
    orig = proxy_span_effect_labels(
        prompt_text=str(prompt_text),
        full_delta=float(full_delta),
        correct_key=str(correct_key),
        evidence_threshold=float(evidence_threshold),
        distractor_threshold=float(distractor_threshold),
    )
    par = proxy_span_effect_labels(
        prompt_text=str(para),
        full_delta=float(full_delta),
        correct_key=str(correct_key),
        evidence_threshold=float(evidence_threshold),
        distractor_threshold=float(distractor_threshold),
    )
    common = sorted(set(orig.keys()) & set(par.keys()))
    if common:
        agreement = float(np.mean(np.asarray([1.0 if orig[k] == par[k] else 0.0 for k in common], dtype=np.float64)))
    else:
        agreement = 0.0
    jaccard = span_label_jaccard(orig.keys(), par.keys())
    return {
        "label_agreement": float(agreement),
        "span_jaccard": float(jaccard),
        "n_common_labels": float(len(common)),
        "n_original_labels": float(len(orig)),
        "n_paraphrased_labels": float(len(par)),
    }
