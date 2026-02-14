from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from sow.constants import ANSWER_SUFFIX, BASELINE_WRAPPER_ID, EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2
from sow.manifest.canonicalize import compute_manifest_row_sha256


def validate_prompt_text_suffix(prompt_text: str) -> None:
    if not isinstance(prompt_text, str):
        raise ValueError("prompt_text must be a string")
    if not prompt_text.endswith(ANSWER_SUFFIX):
        raise ValueError("prompt_text must end with literal 'Answer: ' (including trailing space)")


def validate_options(options: Dict[str, Any]) -> None:
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    for k in ["A", "B", "C", "D"]:
        if k not in options:
            raise ValueError(f"options missing key {k}")
        if not isinstance(options[k], str) or options[k] == "":
            raise ValueError(f"options.{k} must be a non-empty string")


def validate_manifest_row(row: Dict[str, Any]) -> None:
    for k in ["example_id", "wrapper_id", "prompt_id", "prompt_uid", "prompt_text", "options", "correct_key", "manifest_sha256"]:
        if k not in row:
            raise ValueError(f"manifest row missing key: {k}")
    for k in ["example_id", "wrapper_id", "prompt_id", "prompt_uid", "prompt_text", "correct_key", "manifest_sha256"]:
        if not isinstance(row[k], str):
            raise ValueError(f"manifest row key {k} must be a string")
    if row["prompt_uid"] != row["prompt_id"]:
        raise ValueError("Milestone 1 requires prompt_uid == prompt_id")
    if row["correct_key"] not in {"A", "B", "C", "D"}:
        raise ValueError("correct_key must be one of A/B/C/D")
    validate_options(row["options"])
    validate_prompt_text_suffix(row["prompt_text"])
    exp_sha = compute_manifest_row_sha256(row)
    if row["manifest_sha256"] != exp_sha:
        raise ValueError("manifest_sha256 mismatch (row is not self-consistent)")


def validate_baseline_manifest(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != 3000:
        raise ValueError(f"baseline manifest must have 3000 rows; got {len(rows)}")
    ex_ids = set()
    prompt_uids = set()
    for r in rows:
        validate_manifest_row(r)
        if r["wrapper_id"] != BASELINE_WRAPPER_ID:
            raise ValueError(f"baseline wrapper_id must be {BASELINE_WRAPPER_ID}")
        ex = r["example_id"]
        puid = r["prompt_uid"]
        if ex in ex_ids:
            raise ValueError(f"duplicate example_id in baseline: {ex}")
        if puid in prompt_uids:
            raise ValueError(f"duplicate prompt_uid in baseline: {puid}")
        ex_ids.add(ex)
        prompt_uids.add(puid)


def validate_robustness_manifest(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != 60000:
        raise ValueError(f"robustness manifest must have 60000 rows; got {len(rows)}")
    expected_wrappers = set(EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2)
    prompt_uids = set()
    seen_pairs: Set[Tuple[str, str]] = set()
    by_example_count: Dict[str, int] = {}
    by_example_wrappers: Dict[str, Set[str]] = {}
    seen_wrappers: Set[str] = set()
    for r in rows:
        validate_manifest_row(r)
        wid = r["wrapper_id"]
        if wid not in expected_wrappers:
            raise ValueError(f"unexpected wrapper_id: {wid}")
        seen_wrappers.add(wid)
        puid = r["prompt_uid"]
        if puid in prompt_uids:
            raise ValueError(f"duplicate prompt_uid in robustness: {puid}")
        prompt_uids.add(puid)
        ex = r["example_id"]
        pair = (ex, wid)
        if pair in seen_pairs:
            raise ValueError(f"duplicate (example_id, wrapper_id) in robustness: {ex}|{wid}")
        seen_pairs.add(pair)
        by_example_count[ex] = by_example_count.get(ex, 0) + 1
        by_example_wrappers.setdefault(ex, set()).add(wid)
    if seen_wrappers != expected_wrappers:
        missing = sorted(expected_wrappers - seen_wrappers)
        extra = sorted(seen_wrappers - expected_wrappers)
        raise ValueError(f"wrapper set mismatch; missing={missing} extra={extra}")

    if len(by_example_count) != 3000:
        raise ValueError(f"robustness manifest must have exactly 3000 unique example_id; got {len(by_example_count)}")

    bad_examples: List[Dict[str, Any]] = []
    for ex, cnt in by_example_count.items():
        ws = by_example_wrappers.get(ex, set())
        if cnt != 20 or ws != expected_wrappers:
            bad_examples.append(
                {
                    "example_id": ex,
                    "row_count": cnt,
                    "n_wrappers": len(ws),
                    "missing_wrappers": sorted(expected_wrappers - ws)[:5],
                    "extra_wrappers": sorted(ws - expected_wrappers)[:5],
                }
            )
    if bad_examples:
        total_bad = len(bad_examples)
        bad_examples = sorted(bad_examples, key=lambda x: x["example_id"])[:10]
        raise ValueError(f"{total_bad} examples invalid (expected exactly 20 rows + full wrapper set); sample={bad_examples}")
