from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

_OPTION_LINE_RE = re.compile(r"^\s*([A-D])[\)\.:]\s+", flags=re.MULTILINE)
_QUESTION_RE = re.compile(r"question\s*:\s*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PromptSpan:
    span_id: str
    label: str
    start_char: int
    end_char: int
    text: str


def _clip(start: int, end: int, n: int) -> tuple[int, int]:
    s = max(0, min(int(start), n))
    e = max(s, min(int(end), n))
    return s, e


def parse_prompt_spans(prompt_text: str) -> List[PromptSpan]:
    text = str(prompt_text or "")
    n = len(text)
    if n == 0:
        return []

    q_match = _QUESTION_RE.search(text)
    q_start = int(q_match.start()) if q_match else 0

    option_matches = list(_OPTION_LINE_RE.finditer(text))
    spans: List[PromptSpan] = []

    if option_matches:
        first_opt_start = int(option_matches[0].start())
        ins_s, ins_e = _clip(0, q_start if q_start < first_opt_start else first_opt_start, n)
        if ins_e > ins_s:
            spans.append(PromptSpan("instruction", "instruction", ins_s, ins_e, text[ins_s:ins_e]))

        stem_s, stem_e = _clip(q_start if q_match else ins_e, first_opt_start, n)
        if stem_e > stem_s:
            spans.append(PromptSpan("question_stem", "question_stem", stem_s, stem_e, text[stem_s:stem_e]))

        for i, m in enumerate(option_matches):
            key = str(m.group(1)).upper()
            s = int(m.start())
            e = int(option_matches[i + 1].start()) if i + 1 < len(option_matches) else n
            s, e = _clip(s, e, n)
            if e > s:
                spans.append(PromptSpan(f"option_{key}", f"option_{key}", s, e, text[s:e]))

        last_end = int(option_matches[-1].end())
        post_s, post_e = _clip(last_end, n, n)
        if post_e > post_s:
            spans.append(PromptSpan("post_options", "post_options", post_s, post_e, text[post_s:post_e]))
        return spans

    # Fallback: simple two-span split for prompts without option-line markers.
    mid = q_start if q_match else n
    a_s, a_e = _clip(0, mid, n)
    b_s, b_e = _clip(mid, n, n)
    if a_e > a_s:
        spans.append(PromptSpan("instruction", "instruction", a_s, a_e, text[a_s:a_e]))
    if b_e > b_s:
        spans.append(PromptSpan("question_stem", "question_stem", b_s, b_e, text[b_s:b_e]))
    return spans


def spans_to_records(*, prompt_uid: str, prompt_text: str, spans: Iterable[PromptSpan]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for span in spans:
        out.append(
            {
                "prompt_uid": str(prompt_uid),
                "span_id": str(span.span_id),
                "label": str(span.label),
                "start_char": int(span.start_char),
                "end_char": int(span.end_char),
                "text": str(span.text),
                "prompt_text": str(prompt_text),
            }
        )
    return out
