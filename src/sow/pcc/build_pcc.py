from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.manifest.schema import validate_manifest_row


def _next_available_path(path: Path) -> Path:
    """
    Return `path` if it doesn't exist; otherwise append a deterministic attempt suffix.
    """
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def _write_jsonl_atomic_new(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write JSONL to a brand new path (refuse to overwrite) using a temp file + atomic rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def infer_context_length(model_id: str, revision: str) -> Dict[str, Any]:
    """
    Best-effort context length inference (used only for prompt-length safety filters).
    """
    from transformers import AutoConfig  # local import

    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)

    # Common config fields across decoder-only models.
    for attr in ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length"]:
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return {"context_length": int(v), "source": f"config.{attr}"}

    # Fallback: tokenizer.model_max_length is sometimes set, but can also be an absurd sentinel.
    try:
        from transformers import AutoTokenizer  # local import

        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
        v = getattr(tok, "model_max_length", None)
        if isinstance(v, int) and 0 < v < 1_000_000:
            return {"context_length": int(v), "source": "tokenizer.model_max_length"}
    except Exception:
        pass

    # Conservative default if nothing is discoverable.
    return {"context_length": 4096, "source": "default_4096"}


def validate_pcc_baseline_manifest(rows: List[Dict[str, Any]], *, expected_n: int) -> None:
    if len(rows) != expected_n:
        raise ValueError(f"PCC baseline manifest must have {expected_n} rows; got {len(rows)}")
    ex_ids: Set[str] = set()
    prompt_uids: Set[str] = set()
    for r in rows:
        validate_manifest_row(r)
        ex = r["example_id"]
        puid = r["prompt_uid"]
        if ex in ex_ids:
            raise ValueError(f"duplicate example_id in PCC baseline: {ex}")
        if puid in prompt_uids:
            raise ValueError(f"duplicate prompt_uid in PCC baseline: {puid}")
        ex_ids.add(ex)
        prompt_uids.add(puid)


def validate_pcc_robustness_manifest(
    rows: List[Dict[str, Any]],
    *,
    expected_n_examples: int,
    expected_wrapper_ids: List[str],
) -> None:
    expected_wrappers = set(expected_wrapper_ids)
    if len(rows) != expected_n_examples * len(expected_wrappers):
        raise ValueError(f"PCC robustness manifest must have {expected_n_examples * len(expected_wrappers)} rows; got {len(rows)}")

    prompt_uids: Set[str] = set()
    seen_pairs: Set[Tuple[str, str]] = set()
    by_example_count: Dict[str, int] = {}
    by_example_wrappers: Dict[str, Set[str]] = {}
    seen_wrappers: Set[str] = set()
    for r in rows:
        validate_manifest_row(r)
        wid = r["wrapper_id"]
        if wid not in expected_wrappers:
            raise ValueError(f"unexpected wrapper_id in PCC robustness: {wid}")
        seen_wrappers.add(wid)

        puid = r["prompt_uid"]
        if puid in prompt_uids:
            raise ValueError(f"duplicate prompt_uid in PCC robustness: {puid}")
        prompt_uids.add(puid)

        ex = r["example_id"]
        pair = (ex, wid)
        if pair in seen_pairs:
            raise ValueError(f"duplicate (example_id, wrapper_id) in PCC robustness: {ex}|{wid}")
        seen_pairs.add(pair)
        by_example_count[ex] = by_example_count.get(ex, 0) + 1
        by_example_wrappers.setdefault(ex, set()).add(wid)

    if seen_wrappers != expected_wrappers:
        missing = sorted(expected_wrappers - seen_wrappers)
        extra = sorted(seen_wrappers - expected_wrappers)
        raise ValueError(f"PCC wrapper set mismatch; missing={missing} extra={extra}")

    if len(by_example_count) != expected_n_examples:
        raise ValueError(f"PCC robustness manifest must have exactly {expected_n_examples} unique example_id; got {len(by_example_count)}")

    bad_examples: List[Dict[str, Any]] = []
    for ex, cnt in by_example_count.items():
        ws = by_example_wrappers.get(ex, set())
        if cnt != len(expected_wrappers) or ws != expected_wrappers:
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
        raise ValueError(f"{total_bad} examples invalid in PCC robustness; sample={bad_examples}")


def _alloc_uniform_per_domain(domains: List[str], *, target_size: int, seed: int, salt: str) -> Dict[str, int]:
    keys = sorted(set(domains))
    n = len(keys)
    if n == 0:
        raise ValueError("no coarse_domain values found")
    base = target_size // n
    rem = target_size % n

    rng = random.Random(int(sha256_text(f"{seed}|{salt}"), 16))
    keys_shuffled = list(keys)
    rng.shuffle(keys_shuffled)
    extra = set(keys_shuffled[:rem])

    return {k: int(base + (1 if k in extra else 0)) for k in keys}


def select_pcc_example_ids(
    *,
    baseline_rows: Iterable[Dict[str, Any]],
    robustness_rows: Iterable[Dict[str, Any]],
    expected_wrapper_ids: List[str],
    token_count_fn: Callable[[str], int],
    max_input_tokens: int,
    target_size: int,
    seed: int,
    salt: str,
) -> Dict[str, Any]:
    expected_wrappers = set(expected_wrapper_ids)

    domain_by_example: Dict[str, str] = {}
    max_len_by_example: Dict[str, int] = {}
    max_len_source_by_example: Dict[str, str] = {}

    baseline_examples: Set[str] = set()
    for r in baseline_rows:
        ex = str(r["example_id"])
        baseline_examples.add(ex)
        domain_by_example[ex] = str(r.get("coarse_domain") or "unknown")
        n_tok = int(token_count_fn(str(r["prompt_text"])))
        max_len_by_example[ex] = n_tok
        max_len_source_by_example[ex] = "baseline"

    wrappers_by_example: Dict[str, Set[str]] = defaultdict(set)
    for r in robustness_rows:
        ex = str(r["example_id"])
        wid = str(r["wrapper_id"])
        wrappers_by_example[ex].add(wid)
        n_tok = int(token_count_fn(str(r["prompt_text"])))
        prev = max_len_by_example.get(ex)
        if prev is None or n_tok > prev:
            max_len_by_example[ex] = n_tok
            max_len_source_by_example[ex] = wid

    robustness_examples = set(wrappers_by_example.keys())
    if robustness_examples != baseline_examples:
        missing_in_robust = sorted(baseline_examples - robustness_examples)[:10]
        missing_in_base = sorted(robustness_examples - baseline_examples)[:10]
        raise ValueError(
            "baseline/robustness example_id mismatch; "
            f"missing_in_robust_sample={missing_in_robust} missing_in_base_sample={missing_in_base}"
        )

    eligible_by_domain: Dict[str, List[str]] = defaultdict(list)
    dropped_too_long: List[Dict[str, Any]] = []
    dropped_wrapper_mismatch: List[Dict[str, Any]] = []

    for ex in sorted(baseline_examples):
        ws = wrappers_by_example.get(ex, set())
        if ws != expected_wrappers:
            dropped_wrapper_mismatch.append(
                {
                    "example_id": ex,
                    "n_wrappers": len(ws),
                    "missing_wrappers": sorted(expected_wrappers - ws)[:5],
                    "extra_wrappers": sorted(ws - expected_wrappers)[:5],
                }
            )
            continue

        ml = int(max_len_by_example.get(ex, 0))
        if ml > int(max_input_tokens):
            dropped_too_long.append(
                {
                    "example_id": ex,
                    "max_prompt_tokens": ml,
                    "max_prompt_source": max_len_source_by_example.get(ex, "unknown"),
                }
            )
            continue

        eligible_by_domain[domain_by_example.get(ex, "unknown")].append(ex)

    eligible_total = sum(len(v) for v in eligible_by_domain.values())
    if eligible_total < target_size:
        raise ValueError(f"eligible examples {eligible_total} < target_size {target_size}")

    all_domains = sorted(set(domain_by_example.values()))
    alloc = _alloc_uniform_per_domain(all_domains, target_size=target_size, seed=seed, salt=salt)
    selected: List[str] = []
    for domain in sorted(alloc.keys()):
        want = int(alloc[domain])
        bucket = list(eligible_by_domain.get(domain, []))
        bucket.sort()
        seed2 = int(sha256_text(f"{seed}|{salt}|{domain}"), 16)
        rng2 = random.Random(seed2)
        rng2.shuffle(bucket)
        if len(bucket) < want:
            raise ValueError(f"domain {domain} has only {len(bucket)} eligible examples, need {want}")
        selected.extend(bucket[:want])

    selected = sorted(set(selected))
    if len(selected) != target_size:
        raise RuntimeError(f"selected {len(selected)} examples, expected {target_size}")

    counts_eligible = {k: len(v) for k, v in sorted(eligible_by_domain.items(), key=lambda kv: kv[0])}
    counts_selected: Dict[str, int] = defaultdict(int)
    for ex in selected:
        counts_selected[domain_by_example.get(ex, "unknown")] += 1

    return {
        "target_size": int(target_size),
        "eligible_total": int(eligible_total),
        "selected_total": int(len(selected)),
        "max_input_tokens": int(max_input_tokens),
        "eligible_by_domain": counts_eligible,
        "selected_by_domain": {k: int(v) for k, v in sorted(counts_selected.items(), key=lambda kv: kv[0])},
        "dropped_too_long": dropped_too_long[:200],  # cap for report size
        "dropped_too_long_total": int(len(dropped_too_long)),
        "dropped_wrapper_mismatch": dropped_wrapper_mismatch[:50],
        "dropped_wrapper_mismatch_total": int(len(dropped_wrapper_mismatch)),
        "selected_example_ids": selected,
        "selected_example_ids_sha256": sha256_text("\n".join(selected) + "\n"),
    }


def build_pcc_manifests_for_model(
    *,
    run_id: str,
    model_name: str,
    model_id: str,
    model_revision: str,
    baseline_manifest_path: Path,
    robustness_manifest_path: Path,
    expected_wrapper_ids: List[str],
    generation_max_new_tokens: int,
    target_size: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer  # local import

    tok = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True, trust_remote_code=False)

    ctx = infer_context_length(model_id=model_id, revision=model_revision)
    context_length = int(ctx["context_length"])
    max_input_tokens = int(context_length - int(generation_max_new_tokens))
    if max_input_tokens <= 0:
        raise ValueError("invalid max_input_tokens computed from context_length and max_new_tokens")

    def token_count_fn(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=True))

    sel = select_pcc_example_ids(
        baseline_rows=iter_jsonl(baseline_manifest_path),
        robustness_rows=iter_jsonl(robustness_manifest_path),
        expected_wrapper_ids=expected_wrapper_ids,
        token_count_fn=token_count_fn,
        max_input_tokens=max_input_tokens,
        target_size=target_size,
        seed=seed,
        salt=f"{model_id}@{model_revision}",
    )
    selected_set = set(sel["selected_example_ids"])

    # Write manifests (subset only; rows are copied verbatim from canonical manifests).
    out_baseline = _next_available_path(out_dir / f"pcc_baseline.{model_name}.jsonl")
    out_robust = _next_available_path(out_dir / f"pcc_robustness.{model_name}.jsonl")

    def iter_baseline_out() -> Iterable[Dict[str, Any]]:
        for r in iter_jsonl(baseline_manifest_path):
            if str(r["example_id"]) in selected_set:
                yield r

    def iter_robust_out() -> Iterable[Dict[str, Any]]:
        for r in iter_jsonl(robustness_manifest_path):
            if str(r["example_id"]) in selected_set:
                yield r

    _write_jsonl_atomic_new(out_baseline, iter_baseline_out())
    _write_jsonl_atomic_new(out_robust, iter_robust_out())

    # Validate written manifests.
    baseline_rows = list(iter_jsonl(out_baseline))
    robust_rows = list(iter_jsonl(out_robust))
    validate_pcc_baseline_manifest(baseline_rows, expected_n=int(target_size))
    validate_pcc_robustness_manifest(robust_rows, expected_n_examples=int(target_size), expected_wrapper_ids=expected_wrapper_ids)

    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "model_id": model_id,
        "model_revision": model_revision,
        "baseline_manifest_path": str(baseline_manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
        "robustness_manifest_path": str(robustness_manifest_path),
        "robustness_manifest_sha256": sha256_file(robustness_manifest_path),
        "generation_max_new_tokens": int(generation_max_new_tokens),
        "target_size": int(target_size),
        "seed": int(seed),
        "context_length": int(context_length),
        "context_length_source": str(ctx["source"]),
        "max_input_tokens": int(max_input_tokens),
        "selection": sel,
        "outputs": {
            "pcc_baseline_path": str(out_baseline),
            "pcc_baseline_sha256": sha256_file(out_baseline),
            "pcc_robustness_path": str(out_robust),
            "pcc_robustness_sha256": sha256_file(out_robust),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_meta = _next_available_path(out_dir / f"pcc_meta.{model_name}.json")
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return {
        **meta["outputs"],
        "pcc_meta_path": str(out_meta),
        "pcc_meta_sha256": sha256_file(out_meta),
        "selection": sel,
    }
