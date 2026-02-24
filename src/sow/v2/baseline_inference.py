from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np

from sow.io_jsonl import iter_jsonl
from sow.judging.deterministic_parser import parse_choice
from sow.token_buckets.option_buckets import build_buckets_from_tokenizer
from sow.v2.model_nuances import apply_tokenizer_nuance, assert_transformers_version_floor


_CHOICES = ("A", "B", "C", "D")


class NonFiniteBatchError(RuntimeError):
    """Raised when a batch produced non-finite outputs and must be retried at smaller size."""


class ThermalCheckpointExit(RuntimeError):
    """Raised when thermal policy requests checkpoint+exit semantics."""

    def __init__(self, action: Mapping[str, Any]):
        super().__init__("thermal checkpoint exit requested")
        self.action = dict(action)


def resume_key_for(*, model_id: str, prompt_uid: str) -> str:
    raw = f"{str(model_id)}::{str(prompt_uid)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{str(model_id)}::{digest[:24]}"


def repair_trailing_partial_line(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if (not data) or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    if cut < 0:
        path.write_bytes(b"")
        return
    path.write_bytes(data[: cut + 1])


def append_jsonl_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_resume_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    for row in iter_jsonl(path):
        key = str(row.get("resume_key") or "").strip()
        if key:
            out.add(key)
    return out


def select_pending_manifest_rows(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    completed_keys: Set[str],
) -> tuple[List[Mapping[str, Any]], Dict[str, int]]:
    pending_rows: List[Mapping[str, Any]] = []
    resume_rows_total = 0
    resume_rows_skipped = 0
    completed = {str(x) for x in completed_keys if str(x)}

    for row in manifest_rows:
        prompt_uid = str(row.get("prompt_uid") or "").strip()
        if not prompt_uid:
            continue
        resume_rows_total += 1
        rk = resume_key_for(model_id=str(model_id), prompt_uid=prompt_uid)
        if rk in completed:
            resume_rows_skipped += 1
            continue
        pending_rows.append(row)

    return pending_rows, {
        "resume_rows_total": int(resume_rows_total),
        "resume_rows_skipped": int(resume_rows_skipped),
        "pending_rows": int(len(pending_rows)),
    }


def checkpoint_flush_required(*, rows_since_checkpoint: int, checkpoint_every: int) -> bool:
    return int(rows_since_checkpoint) >= max(1, int(checkpoint_every))


def _is_retryable_batch_error(exc: BaseException) -> bool:
    if isinstance(exc, NonFiniteBatchError):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "non-finite" in msg
        or "nan" in msg
        or "inf" in msg
    )


def execute_with_batch_backoff(
    *,
    items: Sequence[Any],
    batch_chain: Sequence[int],
    run_batch: Callable[[Sequence[Any]], None],
) -> Dict[str, Any]:
    chain = [int(x) for x in batch_chain if int(x) > 0]
    if not chain:
        raise ValueError("batch_chain must contain at least one positive batch size")

    offset = 0
    chain_idx = 0
    batch_sizes_used: List[int] = []

    while offset < len(items):
        current_bs = min(int(chain[chain_idx]), int(len(items) - offset))
        batch = items[offset : offset + current_bs]
        try:
            run_batch(batch)
            batch_sizes_used.append(int(current_bs))
            offset += int(current_bs)
        except BaseException as exc:
            if not _is_retryable_batch_error(exc):
                raise
            if chain_idx >= len(chain) - 1:
                raise
            chain_idx += 1

    return {
        "rows_processed": int(offset),
        "batch_sizes_used": batch_sizes_used,
        "final_batch_size": int(chain[chain_idx]),
    }


def _validate_layer(layer: Mapping[str, Any]) -> List[str]:
    errs: List[str] = []
    if "layer_index" not in layer:
        errs.append("missing_layer_index")
    logits = layer.get("candidate_logits")
    probs = layer.get("candidate_probs")
    if not isinstance(logits, Mapping):
        errs.append("missing_candidate_logits")
    if not isinstance(probs, Mapping):
        errs.append("missing_candidate_probs")

    for c in _CHOICES:
        try:
            float((logits or {}).get(c))
        except Exception:
            errs.append(f"invalid_candidate_logit_{c}")
        try:
            float((probs or {}).get(c))
        except Exception:
            errs.append(f"invalid_candidate_prob_{c}")

    try:
        float(layer.get("candidate_entropy"))
    except Exception:
        errs.append("invalid_candidate_entropy")

    top_candidate = str(layer.get("top_candidate") or "")
    if top_candidate not in _CHOICES:
        errs.append("invalid_top_candidate")

    try:
        float(layer.get("top2_margin_prob"))
    except Exception:
        errs.append("invalid_top2_margin_prob")

    proj = layer.get("projected_hidden_128")
    if not isinstance(proj, list) or len(proj) != 128:
        errs.append("invalid_projected_hidden_128")
    else:
        for idx, val in enumerate(proj):
            try:
                float(val)
            except Exception:
                errs.append(f"invalid_projected_hidden_128_{idx}")
                break

    return errs


def _layer_index_sequence_valid(layerwise: Sequence[Mapping[str, Any]]) -> bool:
    idxs: List[int] = []
    for layer in layerwise:
        try:
            idxs.append(int((layer or {}).get("layer_index")))
        except Exception:
            return False
    return idxs == list(range(len(idxs)))


def validate_baseline_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_model_id: str,
    expected_model_revision: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    seen_resume_keys: Set[str] = set()

    required_keys = {
        "run_id",
        "model_id",
        "model_revision",
        "prompt_uid",
        "example_id",
        "wrapper_id",
        "coarse_domain",
        "resume_key",
        "generated_text",
        "first_generated_token_text",
        "parsed_choice",
        "parser_status",
        "parser_signals",
        "is_correct",
        "layerwise",
    }

    for row in rows:
        missing = sorted(k for k in required_keys if k not in row)
        if missing:
            errors.append("missing_required_keys")
            continue

        model_id = str(row.get("model_id") or "")
        model_revision = str(row.get("model_revision") or "")
        if model_id != str(expected_model_id):
            errors.append("wrong_model_id")
        if model_revision != str(expected_model_revision):
            errors.append("wrong_model_revision")

        prompt_uid = str(row.get("prompt_uid") or "").strip()
        if not prompt_uid:
            errors.append("missing_prompt_uid")

        resume_key = str(row.get("resume_key") or "").strip()
        if not resume_key:
            errors.append("missing_resume_key")
        else:
            expected_resume_key = resume_key_for(model_id=model_id, prompt_uid=prompt_uid) if prompt_uid else None
            if expected_resume_key is None or resume_key != expected_resume_key:
                errors.append("invalid_resume_key")
        if resume_key and resume_key in seen_resume_keys:
            errors.append("duplicate_resume_key")
        elif resume_key:
            seen_resume_keys.add(resume_key)

        parsed_choice = row.get("parsed_choice")
        if parsed_choice is not None and str(parsed_choice) not in _CHOICES:
            errors.append("invalid_parsed_choice")

        parser_status = str(row.get("parser_status") or "")
        if parser_status not in {"resolved", "unresolved"}:
            errors.append("invalid_parser_status")

        if not isinstance(row.get("parser_signals"), Mapping):
            errors.append("invalid_parser_signals")

        layerwise = row.get("layerwise")
        if not isinstance(layerwise, list) or len(layerwise) == 0:
            errors.append("missing_layerwise")
        else:
            for layer in layerwise:
                errors.extend(_validate_layer(layer if isinstance(layer, Mapping) else {}))
            if not _layer_index_sequence_valid(layerwise if isinstance(layerwise, list) else []):
                errors.append("invalid_layer_index_sequence")

    uniq_errors = sorted(set(errors))
    return {
        "pass": len(uniq_errors) == 0,
        "rows": int(len(rows)),
        "errors": uniq_errors,
    }


def _candidate_metrics_from_vocab_logits(vocab_logits: np.ndarray, buckets: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    cand_logits: Dict[str, float] = {}
    vocab_size = int(vocab_logits.shape[0])
    for c in _CHOICES:
        ids = [int(i) for i in (buckets.get(c) or []) if 0 <= int(i) < vocab_size]
        if not ids:
            cand_logits[c] = float("-inf")
            continue
        vals = vocab_logits[ids]
        if vals.size == 0:
            cand_logits[c] = float("-inf")
            continue
        m = float(np.max(vals))
        cand_logits[c] = m

    arr = np.asarray([cand_logits[c] for c in _CHOICES], dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not bool(np.any(finite_mask)):
        raise NonFiniteBatchError("non-finite candidate logits")

    safe = np.where(finite_mask, arr, -1e9)
    mx = float(np.max(safe))
    expv = np.exp(np.clip(safe - mx, -80.0, 80.0))
    expv[~finite_mask] = 0.0
    den = float(np.sum(expv))
    if (not np.isfinite(den)) or den <= 0.0:
        raise NonFiniteBatchError("non-finite candidate probabilities")
    probs = expv / den

    if not bool(np.all(np.isfinite(probs))):
        raise NonFiniteBatchError("non-finite candidate probabilities")

    cand_probs = {c: float(probs[i]) for i, c in enumerate(_CHOICES)}
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    top_idx = int(np.argmax(probs))
    top_candidate = _CHOICES[top_idx]
    sorted_probs = sorted([float(x) for x in probs.tolist()], reverse=True)
    top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0

    return {
        "candidate_logits": {k: float(v) for k, v in cand_logits.items()},
        "candidate_probs": cand_probs,
        "candidate_entropy": float(entropy),
        "top_candidate": str(top_candidate),
        "top2_margin_prob": float(top2_margin),
    }


def _options_for_row(row: Mapping[str, Any]) -> Dict[str, str]:
    raw = row.get("options")
    out: Dict[str, str] = {}
    if isinstance(raw, Mapping):
        for c in _CHOICES:
            out[c] = str(raw.get(c) or "")
        return out

    # Fallback for manifests that store option_* columns.
    for c in _CHOICES:
        out[c] = str(row.get(f"option_{c.lower()}") or row.get(f"option_{c}") or "")
    return out


def run_baseline_for_model(
    *,
    run_id: str,
    model: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    resume: bool,
    checkpoint_every_prompts: int,
    batch_chain: Sequence[int],
    max_new_tokens: int = 8,
    deterministic_seed: int = 12345,
    thermal_check_fn: Callable[[], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(model.get("model_id") or "")
    model_revision = str(model.get("revision") or "")
    if not model_id:
        raise ValueError("model_id is required")
    assert_transformers_version_floor(model_id, str(transformers.__version__))

    checkpoint_every = max(1, int(checkpoint_every_prompts))

    if (not resume) and out_path.exists():
        out_path.unlink()
    repair_trailing_partial_line(out_path)
    completed_keys = load_completed_resume_keys(out_path) if resume else set()

    pending_rows, resume_meta = select_pending_manifest_rows(
        manifest_rows=manifest_rows,
        model_id=model_id,
        completed_keys=completed_keys,
    )
    resume_rows_total = int(resume_meta.get("resume_rows_total", 0))
    resume_rows_skipped = int(resume_meta.get("resume_rows_skipped", 0))
    pending_rows_count = int(resume_meta.get("pending_rows", len(pending_rows)))

    if not pending_rows:
        return {
            "pass": True,
            "stopped_early": False,
            "rows_written": 0,
            "output_path": str(out_path),
            "rows_per_second": 0.0,
            "batch_sizes_used": [],
            "pending_rows": int(pending_rows_count),
            "resume_rows_total": int(resume_rows_total),
            "resume_rows_skipped": int(resume_rows_skipped),
            "checkpoint_flush_count": 0,
        }

    if hasattr(torch, "manual_seed"):
        torch.manual_seed(int(deterministic_seed))
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(deterministic_seed))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        default_dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        default_dtype = torch.float16
    else:
        device = torch.device("cpu")
        default_dtype = torch.float32

    # Release stale MPS cache from prior model execution within the same process.
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision if model_revision else None,
        use_fast=True,
        trust_remote_code=False,
    )
    apply_tokenizer_nuance(tokenizer, model_id=model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision if model_revision else None,
        trust_remote_code=False,
        torch_dtype=default_dtype,
    )
    model_obj.to(device)
    model_obj.eval()

    bucket_obj = build_buckets_from_tokenizer(tokenizer)
    buckets = {c: [int(x) for x in (bucket_obj.get("buckets") or {}).get(c, [])] for c in _CHOICES}
    lm_head = model_obj.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model does not expose output embeddings")

    rows_buffer: List[Dict[str, Any]] = []
    batch_sizes_used: List[int] = []
    rows_written = 0
    rows_since_checkpoint = 0
    checkpoint_flush_count = 0

    def _flush_checkpoint() -> None:
        nonlocal rows_written, rows_since_checkpoint, rows_buffer, checkpoint_flush_count
        if not rows_buffer:
            return
        append_jsonl_rows(out_path, rows_buffer)
        rows_written += int(len(rows_buffer))
        checkpoint_flush_count += 1
        rows_since_checkpoint = 0
        rows_buffer = []

    def _run_batch(batch: Sequence[Mapping[str, Any]]) -> None:
        nonlocal rows_since_checkpoint
        if thermal_check_fn is not None:
            action = dict(thermal_check_fn() or {})
            if bool(action.get("checkpoint_exit")):
                raise ThermalCheckpointExit(action)
        prompts = [str(r.get("prompt_text") or "") for r in batch]
        if any((not p.strip()) for p in prompts):
            raise RuntimeError("manifest row missing prompt_text")

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")

        attention_mask = enc["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        input_ids = enc["input_ids"].to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        with torch.no_grad():
            outputs = model_obj(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            generate_ids = model_obj.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=False,
                max_new_tokens=max(1, int(max_new_tokens)),
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        hidden_states = list(outputs.hidden_states or [])
        transformer_states = hidden_states[1:] if len(hidden_states) > 1 else []
        if not transformer_states:
            raise RuntimeError("model did not return hidden states")

        bs = int(input_ids.shape[0])
        for i in range(bs):
            row = batch[i]
            prompt_uid = str(row.get("prompt_uid") or "")
            if not prompt_uid:
                continue
            correct_key = str(row.get("correct_key") or "A").strip().upper()
            if correct_key not in _CHOICES:
                correct_key = "A"

            first_token_id = int(torch.argmax(outputs.logits[i, -1, :]).item())
            first_token_text = tokenizer.decode(
                [first_token_id],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            prompt_len = int(attention_mask[i].sum().item())
            generated_tail = generate_ids[i, prompt_len:]
            generated_text = tokenizer.decode(generated_tail, skip_special_tokens=True).strip()
            if not generated_text:
                generated_text = str(first_token_text)

            parser = parse_choice(
                response_text=str(generated_text),
                first_token=str(first_token_text),
                options=_options_for_row(row),
            )

            layerwise: List[Dict[str, Any]] = []
            for layer_idx, hs in enumerate(transformer_states):
                hidden_last = hs[i, -1, :]
                vocab_logits = lm_head(hidden_last).detach().float().cpu().numpy()
                cand = _candidate_metrics_from_vocab_logits(vocab_logits=vocab_logits, buckets=buckets)

                projected = hidden_last.detach().float().cpu().numpy().astype(np.float64)
                if projected.shape[0] < 128:
                    pad = np.zeros((128 - int(projected.shape[0]),), dtype=np.float64)
                    proj128 = np.concatenate([projected, pad], axis=0)
                else:
                    proj128 = projected[:128]

                layerwise.append(
                    {
                        "layer_index": int(layer_idx),
                        "candidate_logits": cand["candidate_logits"],
                        "candidate_probs": cand["candidate_probs"],
                        "candidate_entropy": float(cand["candidate_entropy"]),
                        "top_candidate": str(cand["top_candidate"]),
                        "top2_margin_prob": float(cand["top2_margin_prob"]),
                        "projected_hidden_128": [float(x) for x in proj128.tolist()],
                    }
                )

            if not layerwise:
                raise RuntimeError("empty layerwise outputs")

            final_top = str(layerwise[-1].get("top_candidate") or "")
            is_correct = bool(final_top == correct_key)
            parser_choice = parser.get("parsed_choice")
            out_row = {
                "run_id": str(run_id),
                "model_id": str(model_id),
                "model_revision": str(model_revision),
                "prompt_uid": prompt_uid,
                "example_id": str(row.get("example_id") or prompt_uid),
                "wrapper_id": str(row.get("wrapper_id") or "plain_exam"),
                "coarse_domain": str(row.get("coarse_domain") or "unknown"),
                "resume_key": resume_key_for(model_id=model_id, prompt_uid=prompt_uid),
                "generated_text": str(generated_text),
                "first_generated_token_text": str(first_token_text),
                "parsed_choice": str(parser_choice) if parser_choice is not None else None,
                "parser_status": "resolved" if parser_choice is not None else "unresolved",
                "parser_signals": {
                    "decision": str(parser.get("decision") or ""),
                    "debug": parser.get("debug") or {},
                },
                "is_correct": bool(is_correct),
                "layerwise": layerwise,
            }
            rows_buffer.append(out_row)
            rows_since_checkpoint += 1
            if checkpoint_flush_required(rows_since_checkpoint=rows_since_checkpoint, checkpoint_every=checkpoint_every):
                _flush_checkpoint()

    start = time.monotonic()
    batch_sizes_used: List[int] = []
    thermal_action: Dict[str, Any] | None = None
    stopped_early = False
    try:
        backoff = execute_with_batch_backoff(
            items=pending_rows,
            batch_chain=batch_chain,
            run_batch=_run_batch,
        )
        batch_sizes_used = [int(x) for x in (backoff.get("batch_sizes_used") or [])]
        _flush_checkpoint()
    except ThermalCheckpointExit as exc:
        thermal_action = dict(exc.action)
        stopped_early = True
        _flush_checkpoint()
    finally:
        try:
            del model_obj
        except Exception:
            pass
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    elapsed = max(1e-9, float(time.monotonic() - start))
    if stopped_early:
        return {
            "pass": False,
            "stopped_early": True,
            "stop_reason": "thermal_checkpoint",
            "thermal_action": thermal_action,
            "rows_written": int(rows_written),
            "output_path": str(out_path),
            "rows_per_second": float(rows_written / elapsed),
            "batch_sizes_used": batch_sizes_used,
            "pending_rows": int(pending_rows_count),
            "resume_rows_total": int(resume_rows_total),
            "resume_rows_skipped": int(resume_rows_skipped),
            "checkpoint_flush_count": int(checkpoint_flush_count),
        }

    return {
        "pass": True,
        "stopped_early": False,
        "stop_reason": None,
        "thermal_action": None,
        "rows_written": int(rows_written),
        "output_path": str(out_path),
        "rows_per_second": float(rows_written / elapsed),
        "batch_sizes_used": batch_sizes_used,
        "pending_rows": int(pending_rows_count),
        "resume_rows_total": int(resume_rows_total),
        "resume_rows_skipped": int(resume_rows_skipped),
        "checkpoint_flush_count": int(checkpoint_flush_count),
    }
