from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sow.constants import (
    ANSWER_SUFFIX,
    BASELINE_WRAPPER_ID,
    EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2,
    ROBUSTNESS_SUFFIX,
)
from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl, write_jsonl


def _canonical_row_for_sha(row: Dict[str, Any]) -> Dict[str, Any]:
    # Only include fields that must be stable and join-relevant.
    return {
        "example_id": row["example_id"],
        "wrapper_id": row["wrapper_id"],
        "prompt_id": row["prompt_id"],
        "prompt_uid": row["prompt_uid"],
        "prompt_text": row["prompt_text"],
        "options": row["options"],
        "correct_key": row["correct_key"],
    }


def compute_manifest_row_sha256(row: Dict[str, Any]) -> str:
    canon = _canonical_row_for_sha(row)
    payload = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def ensure_answer_suffix(text: str) -> str:
    if text.endswith(ANSWER_SUFFIX):
        return text
    if text.endswith("Answer:"):
        return text + " "
    # Strip trailing whitespace then append required suffix.
    return text.rstrip() + "\n\n" + ANSWER_SUFFIX


def enforce_robustness_suffix(text: str) -> str:
    # Always append instruction + Answer: boundary, even if similar text already exists.
    base = text.rstrip()
    return base + ROBUSTNESS_SUFFIX


def canonicalize_baseline_row(src: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = ensure_answer_suffix(src["prompt_text"])
    row = {
        "prompt_id": src["prompt_id"],
        "prompt_uid": src["prompt_id"],
        "module": src.get("module"),
        "dataset": src.get("dataset"),
        "split": src.get("split"),
        "example_id": src["example_id"],
        "subject": src.get("subject"),
        "coarse_domain": src.get("coarse_domain"),
        "question": src.get("question"),
        "options": src["options"],
        "correct_key": src["correct_key"],
        "wrapper_id": src["wrapper_id"],
        "wrapper_description": src.get("wrapper_description"),
        "prompt_text": prompt_text,
    }
    row["manifest_sha256"] = compute_manifest_row_sha256(row)
    return row


def canonicalize_robustness_row(src: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = enforce_robustness_suffix(src["prompt_text"])
    row = {
        "prompt_id": src["prompt_id"],
        "prompt_uid": src["prompt_id"],
        "module": src.get("module"),
        "dataset": src.get("dataset"),
        "split": src.get("split"),
        "example_id": src["example_id"],
        "subject": src.get("subject"),
        "coarse_domain": src.get("coarse_domain"),
        "question": src.get("question"),
        "options": src["options"],
        "correct_key": src["correct_key"],
        "wrapper_id": src["wrapper_id"],
        "wrapper_description": src.get("wrapper_description"),
        "prompt_text": prompt_text,
    }
    row["manifest_sha256"] = compute_manifest_row_sha256(row)
    return row


def _ascii_box_lines(question: str, options: Dict[str, str], *, width: int) -> List[str]:
    # Matches typical "ascii_box" style from the paid v2 file, but is deterministic and wrap-safe.
    def wrap(label: str, text: str) -> List[str]:
        # We want lines like "│ Question: ... │"
        prefix = f"{label}"
        avail = max(10, width - 2 - len(prefix))
        chunks = textwrap.wrap(text, width=avail, break_long_words=False, break_on_hyphens=False) or [""]
        return [prefix + chunks[0]] + [(" " * len(prefix)) + c for c in chunks[1:]]

    content = []
    content.extend(wrap("Question: ", question))
    content.append("Available options:")
    content.append("")
    for k in ["A", "B", "C", "D"]:
        content.extend(wrap(f"  {k}) ", options[k]))
    content.append("")
    content.append("Please select the correct answer by providing the letter (A-D).")

    # Now frame in box.
    def pad(s: str) -> str:
        inner = s[: width - 2]
        return inner + (" " * (width - 2 - len(inner)))

    top = "┌" + ("─" * (width - 2)) + "┐"
    sep = "├" + ("─" * (width - 2)) + "┤"
    bot = "└" + ("─" * (width - 2)) + "┘"

    out = [top]
    # First block: question (multi-line)
    for ln in content[: len(wrap("Question: ", question))]:
        out.append("│" + pad(ln) + "│")
    out.append(sep)
    # Rest content
    for ln in content[len(wrap("Question: ", question)) :]:
        out.append("│" + pad(ln) + "│")
    out.append(bot)
    return out


def generate_missing_ascii_box_prompt(*, question: str, options: Dict[str, str]) -> str:
    # Pick a width similar to existing prompts (65) but adapt if needed.
    # Compute a conservative width based on the longest unwrapped line prefix + a short slice of text.
    width = 65
    lines = _ascii_box_lines(question, options, width=width)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CanonicalizationReport:
    run_id: str
    input_path: str
    input_sha256: str
    output_path: str
    output_sha256: str
    total_input_rows: int
    total_output_rows: int
    expected_unique_keys: int
    duplicates_dropped: int
    filtered_out: int
    missing_repairs: int
    duplicate_events: List[Dict[str, Any]]
    filtered_events: List[Dict[str, Any]]
    repairs: List[Dict[str, Any]]
    generated_at_utc: str


def canonicalize_baseline_manifest(
    *,
    run_id: str,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    report_path: Path,
) -> Tuple[str, str]:
    rows: List[Dict[str, Any]] = []
    total = 0
    for src in iter_jsonl(input_path):
        total += 1
        row = canonicalize_baseline_row(src)
        rows.append(row)

    write_jsonl(output_path, rows)
    out_sha = sha256_file(output_path)
    in_sha = sha256_file(input_path)

    meta = {
        "run_id": run_id,
        "input_path": str(input_path),
        "input_sha256": in_sha,
        "output_path": str(output_path),
        "output_sha256": out_sha,
        "rows": len(rows),
        "prompt_uid_policy": "prompt_uid = prompt_id",
        "baseline_wrapper_id": BASELINE_WRAPPER_ID,
        "required_answer_suffix": ANSWER_SUFFIX,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Baseline canonicalization report is still written (even if no drops).
    report = {
        "run_id": run_id,
        "kind": "baseline_manifest",
        "input_path": str(input_path),
        "input_sha256": in_sha,
        "output_path": str(output_path),
        "output_sha256": out_sha,
        "total_input_rows": total,
        "total_output_rows": len(rows),
        "duplicates_dropped": 0,
        "filtered_out": 0,
        "missing_repairs": 0,
        "events": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return in_sha, out_sha


def canonicalize_robustness_manifest_v2(
    *,
    run_id: str,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    report_path: Path,
    repair_missing: bool,
) -> Tuple[str, str]:
    expected_wrappers = set(EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2)

    # Keep last-seen row per (example_id, wrapper_id).
    by_key: Dict[Tuple[str, str], Tuple[int, Dict[str, Any]]] = {}
    duplicate_events: List[Dict[str, Any]] = []
    filtered_events: List[Dict[str, Any]] = []
    total = 0
    filtered = 0
    seen_examples: set[str] = set()
    for ln_no, src in enumerate(iter_jsonl(input_path), start=1):
        total += 1
        wid = src.get("wrapper_id")
        ex = src.get("example_id")
        if isinstance(ex, str) and ex:
            seen_examples.add(ex)
        if wid not in expected_wrappers:
            filtered += 1
            filtered_events.append(
                {
                    "line": ln_no,
                    "example_id": ex,
                    "wrapper_id": wid,
                    "prompt_id": src.get("prompt_id"),
                    "reason": "wrapper_id_not_in_expected_v2_list",
                }
            )
            continue
        key = (ex, wid)
        if key in by_key:
            prev_ln, prev = by_key[key]
            duplicate_events.append(
                {
                    "example_id": ex,
                    "wrapper_id": wid,
                    "dropped_line": prev_ln,
                    "kept_line": ln_no,
                    "dropped_prompt_id": prev.get("prompt_id"),
                    "kept_prompt_id": src.get("prompt_id"),
                    "dropped_prompt_text_sha256": sha256_text(str(prev.get("prompt_text", ""))),
                    "kept_prompt_text_sha256": sha256_text(str(src.get("prompt_text", ""))),
                    "reason": "duplicate_key_keep_last_line",
                }
            )
        by_key[key] = (ln_no, src)

    # Detect missing wrappers per example.
    per_example: Dict[str, set[str]] = {}
    for (ex, wid) in by_key.keys():
        per_example.setdefault(ex, set()).add(wid)

    repairs: List[Dict[str, Any]] = []
    missing_pairs: List[Tuple[str, str]] = []
    for ex in sorted(seen_examples):
        ws = per_example.get(ex, set())
        missing = expected_wrappers - ws
        for wid in sorted(missing):
            missing_pairs.append((ex, wid))

    if missing_pairs and not repair_missing:
        # Validate-only mode: write a report for auditability, then fail fast.
        in_sha = sha256_file(input_path)
        report = {
            "run_id": run_id,
            "kind": "robustness_manifest_v2_validate_only",
            "input_path": str(input_path),
            "input_sha256": in_sha,
            "output_path": str(output_path),
            "output_sha256": None,
            "total_input_rows": total,
            "total_output_rows": None,
            "expected_unique_keys": 3000 * 20,
            "unique_keys_after_filter_and_dedupe": len(by_key),
            "duplicates_dropped": len(duplicate_events),
            "filtered_out": filtered,
            "missing_pairs_total": len(missing_pairs),
            "missing_pairs_sample": [{"example_id": ex, "wrapper_id": wid} for ex, wid in missing_pairs[:50]],
            "duplicate_events": duplicate_events[:5000],
            "filtered_events": filtered_events[:5000],
            "repairs": [],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "validate-only mode does not write output manifest; report captures detected issues",
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        raise ValueError(f"Missing (example_id, wrapper_id) pairs: {missing_pairs[:10]} (total {len(missing_pairs)})")

    # Repair only the known missing wrapper.
    if repair_missing:
        allowed = {("mmlu::test::12183", "ascii_box")}
        missing_set = set(missing_pairs)
        unexpected_missing = sorted(missing_set - allowed)
        if unexpected_missing:
            raise ValueError(
                "Unexpected missing wrapper pairs (refusing to guess): "
                + "; ".join([f"{ex}|{wid}" for ex, wid in unexpected_missing[:10]])
                + f" (total {len(unexpected_missing)})"
            )
        for ex, wid in sorted(missing_set):
            if (ex, wid) not in allowed:
                continue
            # Find any existing row for this example to pull question/options/correct_key.
            sample_key = next((k for k in by_key.keys() if k[0] == ex), None)
            if sample_key is None:
                raise ValueError(f"Cannot repair missing wrapper for {ex}: example not present in file")
            _, sample_src = by_key[sample_key]
            question = sample_src["question"]
            options = sample_src["options"]
            correct_key = sample_src["correct_key"]

            prompt_text = generate_missing_ascii_box_prompt(question=question, options=options)
            # Wrap it in the same suffix policy as other robustness prompts.
            prompt_text = enforce_robustness_suffix(prompt_text)

            prompt_id = f"robustness::{wid}::{ex}"
            repaired_src = {
                "prompt_id": prompt_id,
                "module": "robustness_v2_repaired",
                "dataset": sample_src.get("dataset"),
                "split": sample_src.get("split"),
                "example_id": ex,
                "subject": sample_src.get("subject"),
                "coarse_domain": sample_src.get("coarse_domain"),
                "question": question,
                "options": options,
                "correct_key": correct_key,
                "wrapper_id": wid,
                "wrapper_description": "Deterministic repaired ascii_box wrapper (generated locally)",
                "prompt_text": prompt_text,
            }
            repairs.append(
                {
                    "example_id": ex,
                    "wrapper_id": wid,
                    "new_prompt_id": prompt_id,
                    "reason": "repair_missing_wrapper_deterministic_template",
                }
            )
            by_key[(ex, wid)] = (total + 1, repaired_src)

    # Build output rows in deterministic order.
    out_rows: List[Dict[str, Any]] = []
    for (ex, wid) in sorted(by_key.keys()):
        _, src = by_key[(ex, wid)]
        row = canonicalize_robustness_row(src)
        out_rows.append(row)

    write_jsonl(output_path, out_rows)
    out_sha = sha256_file(output_path)
    in_sha = sha256_file(input_path)

    meta = {
        "run_id": run_id,
        "input_path": str(input_path),
        "input_sha256": in_sha,
        "output_path": str(output_path),
        "output_sha256": out_sha,
        "rows": len(out_rows),
        "expected_wrappers_v2": EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2,
        "prompt_uid_policy": "prompt_uid = prompt_id",
        "suffix_policy": {
            "appended_instruction": "Return only the letter (A, B, C, or D).",
            "required_answer_suffix": ANSWER_SUFFIX,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "run_id": run_id,
        "kind": "robustness_manifest_v2",
        "input_path": str(input_path),
        "input_sha256": in_sha,
        "output_path": str(output_path),
        "output_sha256": out_sha,
        "total_input_rows": total,
        "total_output_rows": len(out_rows),
        "expected_unique_keys": 3000 * 20,
        "duplicates_dropped": len(duplicate_events),
        "filtered_out": filtered,
        "missing_repairs": len(repairs),
        "missing_pairs_before_repair_total": len(missing_pairs),
        "missing_pairs_before_repair_sample": [{"example_id": ex, "wrapper_id": wid} for ex, wid in missing_pairs[:50]],
        "duplicate_events": duplicate_events[:5000],  # keep bounded; full info is in input file
        "filtered_events": filtered_events[:5000],
        "repairs": repairs,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "duplicate_events list is truncated for size; counts reflect full file",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return in_sha, out_sha
