from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sow.hashing import sha256_file
from sow.hashing import sha256_text
from sow.io_jsonl import iter_jsonl
from sow.judging.deterministic_parser import parse_choice
from sow.token_buckets.option_buckets import model_fs_id, piece_to_letter


def _domain_key(row: Dict[str, Any]) -> str:
    return str(row.get("coarse_domain") or "unknown")


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


def _parsed_choice(row: Dict[str, Any]) -> Optional[str]:
    p = row.get("parser")
    if isinstance(p, dict):
        v = p.get("parsed_choice")
        if v is None:
            return None
        return str(v)
    return None


def select_pilot_rows(
    *,
    baseline_manifest_path: Path,
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(baseline_manifest_path))
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(rows):
        raise ValueError(f"sample_size {sample_size} > baseline rows {len(rows)}")

    strata: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        strata.setdefault(_domain_key(r), []).append(r)

    keys = sorted(strata.keys())
    n = len(keys)
    if n == 0:
        raise ValueError("no strata found")

    base = sample_size // n
    rem = sample_size % n

    rng = random.Random(int(seed))
    keys_shuffled = list(keys)
    rng.shuffle(keys_shuffled)
    extra = set(keys_shuffled[:rem])

    picked: List[Dict[str, Any]] = []
    for k in keys:
        bucket = list(strata[k])
        bucket.sort(key=lambda r: str(r["prompt_uid"]))
        rng2 = random.Random(int(sha256_text(f"{seed}|{k}"), 16))
        rng2.shuffle(bucket)
        want = base + (1 if k in extra else 0)
        if want == 0:
            continue
        if len(bucket) < want:
            raise ValueError(f"domain {k} has only {len(bucket)} rows, need {want}")
        picked.extend(bucket[:want])

    # stable ordering
    picked.sort(key=lambda r: str(r["prompt_uid"]))
    if len(picked) != sample_size:
        raise RuntimeError(f"picked {len(picked)} rows, expected {sample_size}")
    return picked


def _load_token_buckets_for_model(*, run_dir: Path, model_id: str, model_revision: str) -> Dict[str, Any]:
    p = run_dir / "token_buckets" / f"{model_fs_id(model_id)}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing token bucket file: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("model_id") != model_id or obj.get("model_revision") != model_revision:
        raise ValueError("token bucket file does not match model id/revision")
    return {"path": p, "sha256": sha256_file(p), "obj": obj}


def _accumulate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    one_token = sum(1 for r in rows if r["one_token_compliance"])
    resolved = sum(1 for r in rows if _parsed_choice(r) is not None)
    unresolved = n - resolved
    resolved_correct = sum(1 for r in rows if (r["is_correct"] is True))
    resolved_total = sum(1 for r in rows if r["is_correct"] in (True, False))
    acc = (resolved_correct / resolved_total) if resolved_total else None
    return {
        "n": n,
        "one_token_compliance_rate": one_token / n,
        "parser_resolved_rate": resolved / n,
        "unresolved": unresolved,
        "accuracy_on_resolved": acc,
    }


def run_pilot_for_model(
    *,
    run_id: str,
    run_dir: Path,
    model: Dict[str, Any],
    generation: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    device_override: Optional[str] = None,
    thermal_hygiene: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Runs pilot inference for a single model over the provided sample rows.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_id = model["model_id"]
    revision = model["revision"]

    tb = _load_token_buckets_for_model(run_dir=run_dir, model_id=model_id, model_revision=revision)
    buckets = tb["obj"]["buckets"]
    bucket_union = set(buckets["A"]) | set(buckets["B"]) | set(buckets["C"]) | set(buckets["D"])

    device = device_override
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model_obj = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model_obj.eval()
    if device != "cpu":
        model_obj.to(device)
    t_load = time.perf_counter()

    # Cooperative thermal hygiene (no background thread): inference loop polls at an interval.
    from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig  # noqa: PLC0415

    th_cfg = ThermalHygieneConfig.from_cfg(thermal_hygiene)
    thermal_events_path = run_dir / "meta" / "thermal_events.jsonl"
    governor = ThermalGovernor(cfg=th_cfg, events_path=thermal_events_path) if th_cfg.enabled else None

    out_dir = run_dir / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = _next_available_path(out_dir / f"{model_fs_id(model_id)}_pilot_outputs.jsonl")
    out_report = _next_available_path(out_dir / f"{model_fs_id(model_id)}_pilot_report.json")

    outputs: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for r in sample_rows:
            if governor is not None:
                governor.maybe_cooldown(stage="pilot_inference", model_id=model_id, model_revision=revision)

            prompt_text = r["prompt_text"]
            inputs = tok(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask")
            if device != "cpu":
                input_ids = input_ids.to(device)
                if attn is not None:
                    attn = attn.to(device)

            gen_ids = model_obj.generate(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=False,
                max_new_tokens=int(generation.get("max_new_tokens", 24)),
                temperature=float(generation.get("temperature", 1.0)),
                top_p=float(generation.get("top_p", 1.0)),
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )

            input_len = int(input_ids.shape[1])
            new_ids = gen_ids[0, input_len:]
            first_id = int(new_ids[0].item()) if new_ids.numel() > 0 else None
            first_piece = (
                tok.decode([first_id], clean_up_tokenization_spaces=False, skip_special_tokens=False) if first_id is not None else None
            )
            gen_text = tok.decode(new_ids.tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=False)

            letter = piece_to_letter(first_piece or "")
            one_token_ok = bool(letter is not None)
            first_id_in_bucket_union = bool(first_id in bucket_union) if first_id is not None else False

            parsed = parse_choice(
                response_text=gen_text,
                first_token=first_piece,
                options=r["options"],
            )
            parsed_choice = parsed["parsed_choice"]
            is_correct = None
            if parsed_choice is not None:
                is_correct = bool(parsed_choice == r["correct_key"])

            outputs.append(
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_revision": revision,
                    "prompt_uid": r["prompt_uid"],
                    "prompt_id": r["prompt_id"],
                    "example_id": r["example_id"],
                    "wrapper_id": r["wrapper_id"],
                    "coarse_domain": r.get("coarse_domain") or "unknown",
                    "manifest_sha256": r["manifest_sha256"],
                    "generation": {
                        "max_new_tokens": int(generation.get("max_new_tokens", 24)),
                        "first_generated_token_id": first_id,
                        "first_generated_token_text": first_piece,
                        "generated_text": gen_text,
                    },
                    "one_token_compliance": one_token_ok,
                    "first_token_in_bucket_union": first_id_in_bucket_union,
                    "parser": {
                        "parsed_choice": parsed_choice,
                        "decision": parsed["decision"],
                    },
                    "correct_key": r["correct_key"],
                    "is_correct": is_correct,
                }
            )

    _write_jsonl_atomic_new(out_jsonl, outputs)
    t_done = time.perf_counter()

    # Aggregate metrics.
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for o in outputs:
        by_domain.setdefault(o["coarse_domain"], []).append(o)

    report = {
        "pass": True,  # gate decision is applied by the stage-level validator, not here
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "device": device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "thermal_hygiene": {
            "enabled": bool(th_cfg.enabled),
            "provider": th_cfg.provider,
            "cutoff_level": th_cfg.cutoff_level,
            "cooldown_seconds": int(th_cfg.cooldown_seconds),
            "check_interval_seconds": int(th_cfg.check_interval_seconds),
            "events_path": str(thermal_events_path),
        },
        "baseline_manifest_path": str(run_dir / "manifests" / "baseline_manifest.jsonl"),
        "baseline_manifest_sha256": sha256_file(run_dir / "manifests" / "baseline_manifest.jsonl"),
        "token_bucket_path": str(tb["path"]),
        "token_bucket_sha256": tb["sha256"],
        "sample_size": len(sample_rows),
        "sample_prompt_uids_sha256": sha256_text("\n".join([str(r["prompt_uid"]) for r in sample_rows]) + "\n"),
        "metrics_overall": _accumulate_metrics(outputs),
        "metrics_by_domain": {k: _accumulate_metrics(v) for k, v in sorted(by_domain.items(), key=lambda kv: kv[0])},
        "timing_seconds": {
            "total": float(t_done - t0),
            "load": float(t_load - t0),
            "inference": float(t_done - t_load),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "outputs_path": out_jsonl,
        "outputs_sha256": sha256_file(out_jsonl),
        "report_path": out_report,
        "report_sha256": sha256_file(out_report),
        "metrics_overall": report["metrics_overall"],
    }


def stage7_gate(
    *,
    overall_metrics: Dict[str, Any],
    min_one_token_compliance_rate: Optional[float],
    min_parser_resolved_rate: Optional[float],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if min_one_token_compliance_rate is None or min_parser_resolved_rate is None:
        reasons.append("pilot gate thresholds not configured")
        return False, reasons

    ok = True
    comp = overall_metrics.get("one_token_compliance_rate")
    res = overall_metrics.get("parser_resolved_rate")
    if comp is None or res is None:
        return False, ["pilot metrics missing"]
    if comp < float(min_one_token_compliance_rate):
        ok = False
        reasons.append(f"one_token_compliance_rate {comp:.4f} < min {float(min_one_token_compliance_rate):.4f}")
    if res < float(min_parser_resolved_rate):
        ok = False
        reasons.append(f"parser_resolved_rate {res:.4f} < min {float(min_parser_resolved_rate):.4f}")
    return ok, reasons
