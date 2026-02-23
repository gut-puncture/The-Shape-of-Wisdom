from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set


_UNICODE_MINUS = {
    "\u2212",  # MINUS SIGN
    "\u2010",  # HYPHEN
    "\u2011",  # NON-BREAKING HYPHEN
    "\u2012",  # FIGURE DASH
    "\u2013",  # EN DASH
    "\u2014",  # EM DASH
    "\u2015",  # HORIZONTAL BAR
    "\ufe63",  # SMALL HYPHEN-MINUS
    "\uff0d",  # FULLWIDTH HYPHEN-MINUS
}


def _normalize_minus(text: str) -> str:
    for ch in _UNICODE_MINUS:
        text = text.replace(ch, "-")
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _normalize_minus(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_first_letter(token: str) -> Optional[str]:
    t = normalize_text(token)
    if not t:
        return None
    # Strip tokenizer boundary markers and leading punctuation often attached to pieces.
    t = re.sub(r"^[\u0120\u2581]+", "", t)
    t = re.sub(r"^\s*[-*•]+\s*", "", t)
    t = re.sub(r'^[\(\[\{<"\']+', "", t).lstrip()
    if not t:
        return None
    ch = t[0].upper()
    return ch if ch in {"A", "B", "C", "D"} else None


def _canonical_numeric_token(token: str) -> Optional[str]:
    t = normalize_text(token)
    if not t:
        return None

    # Strip surrounding punctuation like "(11)." -> "11"
    t = re.sub(r'^[\(\[\{<"\']+', "", t)
    t = re.sub(r'[\)\]\}>"\'\.,;:]+$', "", t)
    t = t.strip()
    if not t:
        return None

    tl = t.lower()
    if tl == "pi" or t == "π" or ("π" in t):
        return "pi"

    if re.fullmatch(r"-?\d+/\d+", t):
        return t

    # Leading decimal .5 / -.5
    if re.fullmatch(r"-?\.\d+", t):
        t = ("-0" + t[1:]) if t.startswith("-.") else ("0" + t)

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", t) or re.fullmatch(r"[+-]?\d*\.\d+", t):
        try:
            d = Decimal(t)
        except InvalidOperation:
            return None
        n = d.normalize()
        s = format(n, "f")
        return "0" if s == "-0" else s

    return None


def _leading_numeric_from_option(option_text: str) -> Optional[str]:
    if not isinstance(option_text, str):
        return None
    t = normalize_text(option_text)
    # Order matters: prefer fractions before integers so "3/4" isn't captured as "3".
    m = re.match(r"^([+-]?\d+/\d+|[+-]?(?:\d+\.\d+|\d+|\.\d+)|π|pi)\b", t, flags=re.IGNORECASE)
    if not m:
        return None
    return _canonical_numeric_token(m.group(1))


def _option_text_substring_hits(response_norm: str, options: Dict[str, str]) -> List[str]:
    hits: List[str] = []
    r = response_norm.lower()
    for k in ["A", "B", "C", "D"]:
        opt = options.get(k, "")
        if not isinstance(opt, str):
            continue
        opt_norm = normalize_text(opt)
        if len(opt_norm) < 3:
            continue
        if opt_norm.lower() in r:
            hits.append(k)
    return hits


def parse_choice(
    *,
    response_text: str,
    first_token: Optional[str],
    options: Dict[str, str],
) -> Dict[str, Any]:
    raw = response_text or ""
    text_nfkc = _normalize_minus(unicodedata.normalize("NFKC", raw))
    text_norm = normalize_text(raw)

    out: Dict[str, Any] = {
        "parsed_choice": None,
        "decision": "unresolved_no_signal",
        "debug": {
            "first_token_letter": None,
            "letter_candidates": [],
            "numeric_candidates": [],
            "option_text_hits": [],
            "conflicts": [],
        },
    }

    # --- Letters ---
    ft_letter = _extract_first_letter(first_token or "")
    out["debug"]["first_token_letter"] = ft_letter

    letter_candidates: List[str] = []
    m0 = re.match(r'^\s*(?:[-*•]+\s*)?[\(\[\{<"\']*\s*([ABCD])\b', text_nfkc, flags=re.IGNORECASE)
    if m0:
        letter_candidates.append(m0.group(1).upper())

    cue_letter_pat = re.compile(
        r"\b(?:final\s+answer|answer|correct\s+answer|option|choice)\b\s*(?:is|:|=)?\s*[\(\[\{<\"']*\s*([ABCD])\b",
        flags=re.IGNORECASE,
    )
    for m in cue_letter_pat.finditer(text_nfkc):
        letter_candidates.append(m.group(1).upper())

    choose_pat = re.compile(r"\b(?:choose|pick|select)\b\s*([ABCD])\b", flags=re.IGNORECASE)
    for m in choose_pat.finditer(text_nfkc):
        letter_candidates.append(m.group(1).upper())

    is_correct_pat = re.compile(r"\b([ABCD])\b\s+is\s+(?:the\s+)?correct\s+answer\b", flags=re.IGNORECASE)
    for m in is_correct_pat.finditer(text_nfkc):
        letter_candidates.append(m.group(1).upper())

    unique_letters: Set[str] = set([c for c in letter_candidates if c in {"A", "B", "C", "D"}])
    out["debug"]["letter_candidates"] = sorted(unique_letters)
    letter_choice: Optional[str] = None
    if ft_letter:
        letter_choice = ft_letter
        if len(unique_letters) > 1 or (len(unique_letters) == 1 and ft_letter not in unique_letters):
            out["debug"]["conflicts"].append("first_token_vs_response_letter")
    else:
        if len(unique_letters) > 1:
            # Regression wants this treated as no-signal (not conflicting signals).
            out["parsed_choice"] = None
            out["decision"] = "unresolved_no_signal"
            return out
        letter_choice = next(iter(unique_letters)) if len(unique_letters) == 1 else None

    # --- Numerics ---
    numeric_tokens: List[str] = []

    if first_token:
        c = _canonical_numeric_token(first_token)
        if c is not None:
            numeric_tokens.append(c)

    cue_num_pat = re.compile(
        r"\b(?:final\s+answer|answer|option|choice)\b\s*(?:is|:|=)?\s*([^\s,;]+)",
        flags=re.IGNORECASE,
    )
    for m in cue_num_pat.finditer(text_nfkc):
        c = _canonical_numeric_token(m.group(1))
        if c is not None:
            numeric_tokens.append(c)

    # Order matters: prefer fractions before integers so "3/4" isn't captured as "3".
    mlead = re.match(
        r'^\s*[\(\[\{<"\']*\s*([+-]?\d+/\d+|[+-]?(?:\d+\.\d+|\d+|\.\d+)|π|pi)\b',
        text_nfkc,
        flags=re.IGNORECASE,
    )
    if mlead:
        c = _canonical_numeric_token(mlead.group(1))
        if c is not None:
            numeric_tokens.append(c)

    if ("π" in text_nfkc) or re.search(r"\bpi\b", text_norm, flags=re.IGNORECASE):
        numeric_tokens.append("pi")

    unique_nums: Set[str] = set(numeric_tokens)
    out["debug"]["numeric_candidates"] = sorted(unique_nums)

    numeric_choice: Optional[str] = None
    if len(unique_nums) == 1:
        tok = next(iter(unique_nums))

        # 1) Full numeric match against option text.
        full_matches: List[str] = []
        for k in ["A", "B", "C", "D"]:
            opt = options.get(k, "")
            copt = _canonical_numeric_token(opt)
            if copt is not None and copt == tok:
                full_matches.append(k)
        if len(full_matches) == 1:
            numeric_choice = full_matches[0]

        # 2) Leading numeric match (e.g., "8 meters")
        if numeric_choice is None:
            lead_matches: List[str] = []
            for k in ["A", "B", "C", "D"]:
                lead = _leading_numeric_from_option(options.get(k, ""))
                if lead is not None and lead == tok:
                    lead_matches.append(k)
            if len(lead_matches) == 1:
                numeric_choice = lead_matches[0]

        # 3) Index fallback 1..4 -> A..D
        if numeric_choice is None and tok in {"1", "2", "3", "4"}:
            numeric_choice = {"1": "A", "2": "B", "3": "C", "4": "D"}[tok]

    elif len(unique_nums) > 1:
        # Keep numeric_choice = None; let a unique letter still resolve if present.
        numeric_choice = None

    # --- Option-text substring (last resort) ---
    option_hits = _option_text_substring_hits(text_norm, options)
    out["debug"]["option_text_hits"] = sorted(set(option_hits))
    option_choice: Optional[str] = None
    if len(set(option_hits)) == 1:
        option_choice = option_hits[0]

    # --- Conflicts ---
    if ft_letter and numeric_choice and ft_letter != numeric_choice:
        out["debug"]["conflicts"].append("first_token_vs_numeric")
        out["parsed_choice"] = None
        out["decision"] = "unresolved_conflicting_signals"
        return out
    elif letter_choice and numeric_choice and letter_choice != numeric_choice:
        out["parsed_choice"] = None
        out["decision"] = "unresolved_conflicting_signals"
        return out

    if ft_letter and option_choice and ft_letter != option_choice:
        out["debug"]["conflicts"].append("first_token_vs_option_text")
    elif numeric_choice and option_choice and numeric_choice != option_choice:
        out["parsed_choice"] = None
        out["decision"] = "unresolved_conflicting_signals"
        return out

    # --- Resolve ---
    if ft_letter:
        out["parsed_choice"] = ft_letter
        out["decision"] = "resolved_letter_first_token"
        return out

    if letter_choice:
        out["parsed_choice"] = letter_choice
        out["decision"] = "resolved_letter"
        return out

    if numeric_choice:
        out["parsed_choice"] = numeric_choice
        out["decision"] = "resolved_numeric"
        return out

    if option_choice:
        out["parsed_choice"] = option_choice
        out["decision"] = "resolved_option_text"
        return out

    out["parsed_choice"] = None
    out["decision"] = "unresolved_no_signal"
    return out
