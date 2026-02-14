from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sow.hashing import sha256_file


LETTER_OPTIONS = ["A", "B", "C", "D"]

# Minimal, fixed variant set (see docs/IMPLEMENTATION_SPEC.md Stage 5).
VARIANT_TEMPLATES = ["{L}", " {L}", "\n{L}", "({L})", "{L}.", "{L}:"]


def _normalize_piece_for_bucket(piece: str) -> str:
    # Spec: Unicode NFKC, strip whitespace, uppercase, remove surrounding punctuation.
    t = unicodedata.normalize("NFKC", piece or "")
    t = t.strip()
    t = t.upper()
    t = re.sub(r'^[\(\[\{<"\'\s]+', "", t)
    t = re.sub(r'[\)\]\}>"\'\s\.,;:]+$', "", t)
    t = t.strip()
    return t


def _piece_to_letter(piece: str) -> Optional[str]:
    t = _normalize_piece_for_bucket(piece)
    return t if t in {"A", "B", "C", "D"} else None


def variants_for_letter(letter: str) -> List[str]:
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError("letter must be one of A/B/C/D")
    return [tpl.format(L=letter) for tpl in VARIANT_TEMPLATES]


def build_buckets_from_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """
    Build option token buckets using a tokenizer object that provides:
      - encode(text, add_special_tokens=False) -> List[int]
      - decode([id], **kwargs) -> str
    """
    buckets: Dict[str, Set[int]] = {k: set() for k in LETTER_OPTIONS}
    token_pieces: Dict[int, str] = {}
    evidence: Dict[str, Dict[str, List[int]]] = {k: {} for k in LETTER_OPTIONS}

    for letter in LETTER_OPTIONS:
        for variant in variants_for_letter(letter):
            ids: List[int] = tokenizer.encode(variant, add_special_tokens=False)  # type: ignore[attr-defined]
            if not isinstance(ids, list):
                raise TypeError("tokenizer.encode must return a list of token ids")
            evidence[letter][variant] = list(ids)
            for tid in ids:
                # Decode a single token id to its surface form.
                piece = tokenizer.decode(  # type: ignore[attr-defined]
                    [tid],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                token_pieces[tid] = piece
                mapped = _piece_to_letter(piece)
                if mapped == letter:
                    buckets[letter].add(int(tid))

    # Detect overlaps after bucket assignment.
    overlaps: Dict[str, List[int]] = {}
    for i, a in enumerate(LETTER_OPTIONS):
        for b in LETTER_OPTIONS[i + 1 :]:
            inter = buckets[a] & buckets[b]
            if inter:
                overlaps[f"{a}{b}"] = sorted(inter)

    out = {
        "variant_templates": list(VARIANT_TEMPLATES),
        "evidence_token_ids_by_letter_and_variant": evidence,
        "token_pieces_by_id": {str(k): v for k, v in sorted(token_pieces.items(), key=lambda kv: kv[0])},
        "buckets": {k: sorted(v) for k, v in buckets.items()},
        "overlaps": overlaps,
        "normalization_policy": {
            "unicode": "NFKC",
            "strip_whitespace": True,
            "uppercase": True,
            "strip_surrounding_punctuation": True,
        },
    }
    return out


def validate_bucket_obj(obj: Dict[str, Any]) -> None:
    if "buckets" not in obj or not isinstance(obj["buckets"], dict):
        raise ValueError("bucket obj missing buckets")
    buckets = obj["buckets"]
    for k in LETTER_OPTIONS:
        if k not in buckets:
            raise ValueError(f"bucket obj missing letter {k}")
        if not isinstance(buckets[k], list) or not buckets[k]:
            raise ValueError(f"bucket for {k} must be a non-empty list")
        if any((not isinstance(x, int)) for x in buckets[k]):
            raise ValueError(f"bucket for {k} must contain ints")
    overlaps = obj.get("overlaps") or {}
    if overlaps:
        # We don't expect overlaps; fail fast so we don't silently bias scoring.
        raise ValueError(f"overlapping token buckets detected: keys={sorted(overlaps.keys())}")


def model_fs_id(model_id: str) -> str:
    # Filesystem-safe identifier.
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def write_token_buckets_file(
    *,
    out_path: Path,
    run_id: str,
    model_id: str,
    model_revision: str,
    tokenizer_class: str,
    transformers_version: str,
    bucket_obj: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_class": tokenizer_class,
        "transformers_version": transformers_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **bucket_obj,
    }
    validate_bucket_obj(payload)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_and_write_option_buckets_for_models(
    *,
    run_id: str,
    models: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Builds token buckets for all models and writes one JSON file per model.
    """
    from transformers import AutoTokenizer  # local import to avoid hard dependency at import time
    import transformers

    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Dict[str, Any]] = []
    for m in models:
        mid = m["model_id"]
        rev = m["revision"]
        tok = AutoTokenizer.from_pretrained(mid, revision=rev, use_fast=True, trust_remote_code=False)

        bucket_obj = build_buckets_from_tokenizer(tok)
        out_path = out_dir / f"{model_fs_id(mid)}.json"
        write_token_buckets_file(
            out_path=out_path,
            run_id=run_id,
            model_id=mid,
            model_revision=rev,
            tokenizer_class=type(tok).__name__,
            transformers_version=str(transformers.__version__),
            bucket_obj=bucket_obj,
        )
        files.append(
            {
                "model_id": mid,
                "model_revision": rev,
                "path": str(out_path),
                "sha256": sha256_file(out_path),
                "bucket_sizes": {k: len(bucket_obj["buckets"][k]) for k in LETTER_OPTIONS},
            }
        )

    return {
        "pass": True,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }

