from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.judging.deterministic_parser import parse_choice
from sow.token_buckets.option_buckets import model_fs_id, piece_to_letter
from sow.inference.model_adapter import resolve_model_components
from sow.inference.readout import BucketIndex, build_bucket_index, compute_candidate_readout, idx_to_letter, project_hidden_pca


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def _repair_trailing_partial_line(path: Path) -> None:
    """
    If a previous run crashed mid-write, the output file may end with a partial JSON line.
    Truncate back to the last newline so resume scans don't fail.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    if not data:
        return
    if data.endswith(b"\n"):
        return
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        # No newline at all -> treat as corrupt; refuse to proceed.
        raise ValueError(f"output file has no newline (corrupt / partial): {path}")
    path.write_bytes(data[: last_nl + 1])


def resume_key_for(*, model_id: str, prompt_uid: str) -> str:
    return sha256_text(f"{model_id}:{prompt_uid}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_token_buckets(*, run_dir: Path, model_id: str, model_revision: str) -> Tuple[BucketIndex, Dict[str, Any], Path]:
    p = run_dir / "token_buckets" / f"{model_fs_id(model_id)}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing token bucket file: {p}")
    obj = _load_json(p)
    if obj.get("model_id") != model_id or obj.get("model_revision") != model_revision:
        raise ValueError("token bucket file does not match model id/revision")
    buckets = obj.get("buckets") or {}
    bidx = build_bucket_index(buckets=buckets)
    return bidx, obj, p


def _load_pca_basis_from_sentinel(*, run_dir: Path, model_id: str) -> Dict[str, Any]:
    sent = run_dir / "sentinels" / f"pca_fit.{model_fs_id(model_id)}.done"
    if not sent.exists():
        raise FileNotFoundError(f"missing PCA basis sentinel: {sent}")
    s_obj = _load_json(sent)
    basis_path = Path(str(s_obj.get("basis_path") or ""))
    meta_path = Path(str(s_obj.get("meta_path") or ""))
    if not basis_path.exists():
        raise FileNotFoundError(f"missing PCA basis file referenced by sentinel: {basis_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing PCA basis meta referenced by sentinel: {meta_path}")
    if sha256_file(basis_path) != str(s_obj.get("basis_sha256") or ""):
        raise ValueError("PCA basis sha256 mismatch vs sentinel")
    if sha256_file(meta_path) != str(s_obj.get("meta_sha256") or ""):
        raise ValueError("PCA basis meta sha256 mismatch vs sentinel")
    meta = _load_json(meta_path)
    basis_hash = meta.get("basis_hash")
    if not isinstance(basis_hash, str) or not basis_hash:
        raise ValueError("PCA basis meta missing basis_hash")
    with np.load(basis_path) as z:
        mean = z["mean"].astype(np.float32, copy=False)
        components = z["components"].astype(np.float32, copy=False)
    return {
        "sentinel_path": str(sent),
        "basis_path": str(basis_path),
        "basis_sha256": sha256_file(basis_path),
        "meta_path": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "basis_hash": basis_hash,
        "mean": mean,
        "components": components,
    }


def _read_manifest_rows(*, manifest_path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(manifest_path))


def _select_smoke_rows_baseline(*, baseline_manifest: Path, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    from sow.pilot.pilot_inference import select_pilot_rows  # noqa: PLC0415

    return select_pilot_rows(baseline_manifest_path=baseline_manifest, sample_size=int(sample_size), seed=int(seed))


def _select_smoke_rows_robustness(*, robustness_manifest: Path, expected_wrapper_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministic 20-row robustness smoke: pick the lexicographically smallest example_id
    and take one row for each wrapper_id (ordered by expected wrapper list).
    """
    rows = _read_manifest_rows(manifest_path=robustness_manifest)
    by_ex: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_ex.setdefault(str(r["example_id"]), {})[str(r["wrapper_id"])] = r
    ex = sorted(by_ex.keys())[0] if by_ex else None
    if ex is None:
        raise ValueError("robustness manifest empty")
    picked = []
    mp = by_ex[ex]
    for wid in expected_wrapper_ids:
        if wid not in mp:
            raise ValueError(f"robustness smoke selection missing wrapper {wid} for example {ex}")
        picked.append(mp[wid])
    if len(picked) != len(expected_wrapper_ids):
        raise RuntimeError("unexpected smoke pick length")
    return picked


def _parse_status(decision: str) -> str:
    if decision.startswith("resolved_"):
        return "resolved"
    if decision == "unresolved_conflicting_signals":
        return "conflict"
    return "unresolved"


def _trajectory_metrics(
    *,
    top_candidate_idx: List[int],
    top2_margin_prob: List[float],
    thresholds: List[float],
) -> Dict[str, Any]:
    if len(top_candidate_idx) != len(top2_margin_prob):
        raise ValueError("trajectory arrays length mismatch")
    n = len(top_candidate_idx)
    if n == 0:
        return {"flip_count": 0, "commitment_layer_by_margin_threshold": {}}

    # Winner at final layer.
    final_winner = int(top_candidate_idx[-1])

    # Flip count across depth.
    flips = 0
    for i in range(1, n):
        if int(top_candidate_idx[i]) != int(top_candidate_idx[i - 1]):
            flips += 1

    # Commitment: earliest layer where final winner is stable for all later layers
    # and margin >= threshold for all later layers.
    commit: Dict[str, Any] = {}
    for t in thresholds:
        thr = float(t)
        layer = None
        for i in range(n):
            ok = True
            for j in range(i, n):
                if int(top_candidate_idx[j]) != final_winner or float(top2_margin_prob[j]) < thr:
                    ok = False
                    break
            if ok:
                layer = int(i)
                break
        # JSON keys must be strings; keep stable representation.
        key = str(thr)
        commit[key] = layer

    return {"flip_count": int(flips), "commitment_layer_by_margin_threshold": commit}


def _load_completed_resume_keys(path: Path) -> Dict[str, int]:
    """
    Return mapping resume_key -> line_no (1-based) for already-written rows.
    """
    seen: Dict[str, int] = {}
    if not path.exists():
        return seen
    for ln_no, row in enumerate(iter_jsonl(path), start=1):
        rk = row.get("resume_key")
        if isinstance(rk, str):
            seen[rk] = ln_no
    return seen


def _write_jsonl_append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg) or ("mps backend out of memory" in msg)


def _empty_device_cache(*, device: str) -> None:
    try:
        import torch  # noqa: PLC0415

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _pick_device(*, preferred: Optional[str] = None) -> str:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        torch = None
    if preferred:
        return preferred
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_longest_prompts_by_chars(*, rows: List[Dict[str, Any]], k: int) -> List[str]:
    if k <= 0:
        return []
    pairs = [(len(str(r.get("prompt_text") or "")), str(r.get("prompt_text") or "")) for r in rows]
    pairs.sort(key=lambda x: x[0])
    return [p[1] for p in pairs[-k:]]


def _auto_calibrate_batch_size(
    *,
    model_obj: Any,
    tok: Any,
    probe_rows: List[Dict[str, Any]],
    device: str,
    generation: Dict[str, Any],
) -> int:
    """
    Choose the largest batch size from a candidate set that does not OOM on:
      - a trace forward pass (output_hidden_states=True)
      - a short greedy generation (max_new_tokens per config)
    using the longest prompts in the target manifest.
    """
    import torch  # noqa: PLC0415

    # Conservative candidates to keep headroom on 48GB GPUs.
    candidates = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1] if device == "cuda" else [16, 8, 4, 2, 1]
    probe_prompts = _pick_longest_prompts_by_chars(rows=probe_rows, k=min(8, len(probe_rows)))
    if not probe_prompts:
        return 1

    # Generation settings are fixed by config (greedy).
    max_new = int(generation.get("max_new_tokens", 24))

    with torch.inference_mode():
        for b in candidates:
            batch_prompts = (probe_prompts * ((b + len(probe_prompts) - 1) // len(probe_prompts)))[:b]
            try:
                enc = tok(batch_prompts, return_tensors="pt", padding=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                if attn is None:
                    raise RuntimeError("tokenizer did not return attention_mask")
                if device != "cpu":
                    input_ids = input_ids.to(device)
                    attn = attn.to(device)

                _ = model_obj(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True, return_dict=True)
                _ = model_obj.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    do_sample=False,
                    max_new_tokens=max_new,
                    temperature=float(generation.get("temperature", 1.0)),
                    top_p=float(generation.get("top_p", 1.0)),
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
                return int(b)
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    _empty_device_cache(device=device)
                    continue
                raise
    return 1


@dataclass(frozen=True)
class InferencePaths:
    out_jsonl: Path
    run_meta_json: Path


def _paths_for_condition(*, run_dir: Path, model_id: str, condition: str) -> InferencePaths:
    if condition not in ("baseline", "robustness"):
        raise ValueError("condition must be baseline or robustness")
    out_dir = run_dir / "outputs" / model_fs_id(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{condition}_outputs.jsonl"
    run_meta = out_dir / "run_meta.json"
    return InferencePaths(out_jsonl=out_jsonl, run_meta_json=run_meta)


def validate_stage13_output(
    *,
    output_path: Path,
    manifest_rows: List[Dict[str, Any]],
    model_id: str,
    model_revision: str,
    basis_hash: str,
    expected_n_layers: Optional[int],
) -> Dict[str, Any]:
    """
    Deterministic validator for Stage 13 outputs.
    """
    required_top = [
        "run_id",
        "model_id",
        "model_revision",
        "prompt_uid",
        "example_id",
        "wrapper_id",
        "prompt_text_sha256",
        "prompt_length_tokens",
        "resume_key",
        "first_generated_token_id",
        "first_generated_token_text",
        "generated_text",
        "parsed_choice",
        "is_correct",
        "parser_status",
        "parser_signals",
        "flip_count",
        "commitment_layer_by_margin_threshold",
        "layerwise",
        "basis_hash",
    ]

    exp_uids = [str(r["prompt_uid"]) for r in manifest_rows]
    exp_set = set(exp_uids)

    seen_keys: set[str] = set()
    seen_uids: set[str] = set()
    n_rows = 0
    bad: List[str] = []
    n_layers_found = None

    for row in iter_jsonl(output_path):
        n_rows += 1
        for k in required_top:
            if k not in row:
                bad.append(f"missing key {k} on row {n_rows}")
                break
        if row.get("model_id") != model_id or row.get("model_revision") != model_revision:
            bad.append(f"model id/revision mismatch on row {n_rows}")
        if row.get("basis_hash") != basis_hash:
            bad.append(f"basis_hash mismatch on row {n_rows}")

        rk = row.get("resume_key")
        if not isinstance(rk, str):
            bad.append(f"resume_key not str on row {n_rows}")
        elif rk in seen_keys:
            bad.append(f"duplicate resume_key on row {n_rows}")
        else:
            seen_keys.add(rk)

        puid = row.get("prompt_uid")
        if not isinstance(puid, str):
            bad.append(f"prompt_uid not str on row {n_rows}")
        elif puid in seen_uids:
            bad.append(f"duplicate prompt_uid on row {n_rows}")
        else:
            seen_uids.add(puid)

        lw = row.get("layerwise")
        if not isinstance(lw, list) or not lw:
            bad.append(f"layerwise missing/empty on row {n_rows}")
        else:
            if n_layers_found is None:
                n_layers_found = len(lw)
            if len(lw) != n_layers_found:
                bad.append(f"layer count mismatch on row {n_rows}")
            for ent in lw:
                if not isinstance(ent, dict):
                    bad.append(f"layerwise entry not object on row {n_rows}")
                    break
                for k in ["layer_index", "candidate_logits", "candidate_probs", "candidate_entropy", "top_candidate", "top2_margin_prob", "projected_hidden_128"]:
                    if k not in ent:
                        bad.append(f"layerwise missing {k} on row {n_rows}")
                        break

    missing = sorted(exp_set - seen_uids)
    extra = sorted(seen_uids - exp_set)

    if expected_n_layers is not None and n_layers_found is not None and int(n_layers_found) != int(expected_n_layers):
        bad.append(f"n_layers mismatch: found {n_layers_found} expected {expected_n_layers}")

    ok = (not bad) and (not missing) and (not extra) and (n_rows == len(manifest_rows))
    return {
        "pass": bool(ok),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "expected_rows": len(manifest_rows),
        "found_rows": int(n_rows),
        "missing_prompt_uids": missing[:20],
        "extra_prompt_uids": extra[:20],
        "errors": bad[:20],
        "n_layers_found": n_layers_found,
    }


def run_stage13_inference_for_model(
    *,
    run_id: str,
    run_dir: Path,
    model: Dict[str, Any],
    generation: Dict[str, Any],
    manifest_path: Path,
    condition: str,
    batch_size: int | str,
    device_override: Optional[str],
    output_path_override: Optional[Path] = None,
    stop_after_rows: Optional[int] = None,
    limit_rows: Optional[int] = None,
    commitment_margin_thresholds: Optional[List[float]] = None,
    thermal_hygiene_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Stage 13: full inference with layerwise readouts + PCA projection, append-only JSONL with resume.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_id = str(model["model_id"])
    revision = str(model["revision"])
    device = _pick_device(preferred=device_override)

    if bool(generation.get("do_sample")) is not False:
        raise ValueError("Stage 13 requires greedy decoding (do_sample=false) for determinism")
    if int(generation.get("max_new_tokens", 24)) != 24:
        raise ValueError("Stage 13 requires max_new_tokens=24")

    thresholds = commitment_margin_thresholds or [0.05, 0.1, 0.2]

    # Load manifest rows (deterministic order).
    rows = _read_manifest_rows(manifest_path=manifest_path)
    if limit_rows is not None:
        rows = rows[: int(limit_rows)]

    # Paths + resume.
    paths = _paths_for_condition(run_dir=run_dir, model_id=model_id, condition=condition)
    out_jsonl = Path(output_path_override) if output_path_override is not None else paths.out_jsonl
    _repair_trailing_partial_line(out_jsonl)

    completed = _load_completed_resume_keys(out_jsonl)

    # Token buckets + PCA basis.
    bidx, tb_obj, tb_path = _load_token_buckets(run_dir=run_dir, model_id=model_id, model_revision=revision)
    pca = _load_pca_basis_from_sentinel(run_dir=run_dir, model_id=model_id)

    # Determinism knobs (CUDA).
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    # For decoder-only batched generation, left padding is the safest default.
    tok.padding_side = "left"
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

    comps = resolve_model_components(model_obj)

    # Calibrate batch size if requested.
    bs_requested = None
    if isinstance(batch_size, int):
        bs_requested = int(batch_size)
    else:
        s = str(batch_size).strip().lower()
        bs_requested = None if s == "auto" else int(s)
    if bs_requested is None:
        bs = _auto_calibrate_batch_size(model_obj=model_obj, tok=tok, probe_rows=rows, device=device, generation=generation)
    else:
        bs = int(bs_requested)
    if bs <= 0:
        raise ValueError("batch_size must be positive")

    # PCA basis tensors.
    mean_t = torch.tensor(pca["mean"], dtype=torch.float32, device=device)
    comps_t = torch.tensor(pca["components"], dtype=torch.float32, device=device)

    # Run meta (written once; safe to overwrite because it is derived, but keep append-only semantics by attempt suffix).
    run_meta = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "condition": condition,
        "device": device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "generation": {
            "do_sample": False,
            "temperature": float(generation.get("temperature", 1.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "max_new_tokens": int(generation.get("max_new_tokens", 24)),
        },
        "token_buckets_path": str(tb_path),
        "token_buckets_sha256": sha256_file(tb_path),
        "token_bucket_sizes": {k: len(tb_obj["buckets"][k]) for k in ["A", "B", "C", "D"]},
        "pca_basis": {
            "basis_hash": pca["basis_hash"],
            "basis_path": pca["basis_path"],
            "basis_sha256": pca["basis_sha256"],
            "meta_path": pca["meta_path"],
            "meta_sha256": pca["meta_sha256"],
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "commitment_margin_thresholds": [float(x) for x in thresholds],
        "batching": {"requested": str(batch_size), "resolved": int(bs)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = paths.run_meta_json
    if output_path_override is None:
        # Shared per-model meta, updated deterministically as conditions complete.
        if meta_path.exists():
            try:
                existing = _load_json(meta_path)
            except Exception:
                existing = None
            if isinstance(existing, dict):
                # Sanity: require base identity match.
                for k in ["run_id", "model_id", "model_revision"]:
                    if existing.get(k) != run_meta.get(k):
                        raise ValueError(f"run_meta.json mismatch for key {k}")
                conds = existing.get("conditions")
                if not isinstance(conds, dict):
                    conds = {}
                conds[str(condition)] = {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "output_path": str(out_jsonl),
                }
                existing["conditions"] = conds
                existing["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                tmp = meta_path.with_name(f".{meta_path.name}.tmp")
                tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
                tmp.replace(meta_path)
        else:
            run_meta2 = dict(run_meta)
            run_meta2["conditions"] = {
                str(condition): {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "output_path": str(out_jsonl),
                }
            }
            meta_path.write_text(json.dumps(run_meta2, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Inference loop.
    wrote = 0
    total = len(rows)
    bs_cur = bs
    bs_used: set[int] = set()

    def batch_iter() -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
        off = 0
        while off < total:
            batch = rows[off : off + int(bs_cur)]
            yield off, batch
            off += int(bs_cur)

    outputs_written = 0
    with torch.inference_mode():
        for off, batch in batch_iter():
            # Skip already completed by resume_key (append-only resume).
            batch2 = []
            for r in batch:
                rk = resume_key_for(model_id=model_id, prompt_uid=str(r["prompt_uid"]))
                if rk in completed:
                    continue
                batch2.append(r)
            if not batch2:
                continue

            prompts = [str(r["prompt_text"]) for r in batch2]
            enc = tok(prompts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"]
            attn = enc.get("attention_mask")
            if attn is None:
                raise RuntimeError("tokenizer did not return attention_mask")
            prompt_lens = attn.sum(dim=1).to(dtype=torch.long).tolist()
            if device != "cpu":
                input_ids = input_ids.to(device)
                attn = attn.to(device)

            try:
                # Trace forward pass (hidden states).
                out = model_obj(
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states
                if hs is None or len(hs) < 2:
                    raise RuntimeError("model did not return hidden_states")
                # Per spec: exclude embeddings (layer 0 = first transformer block output).
                hs_layers = hs[1:]
                n_layers = len(hs_layers)
                hidden_dim = int(hs_layers[0].shape[-1])

                # Left padding => decision position p is the last token for all sequences.
                hidden_last = torch.stack([h[:, -1, :] for h in hs_layers], dim=1)  # (batch, n_layers, hidden)

                readout = compute_candidate_readout(hidden_last=hidden_last, final_norm=comps.final_norm, lm_head=comps.lm_head, bucket_index=bidx)
                proj = project_hidden_pca(hidden_last=hidden_last, mean=mean_t, components=comps_t)

                # Generation (greedy, capped).
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
            except RuntimeError as exc:
                if _is_oom_error(exc) and int(bs_cur) > 1:
                    bs_cur = max(1, int(bs_cur) // 2)
                    _empty_device_cache(device=device)
                    continue
                raise

            bs_used.add(int(bs_cur))

            # Slice generated tokens after the (padded) input length.
            in_len = int(input_ids.shape[1])
            new_ids = gen_ids[:, in_len:]

            # Move tensors to CPU for JSON serialization.
            cand_logits = readout["candidate_logits"].detach().to("cpu").numpy()
            cand_probs = readout["candidate_probs"].detach().to("cpu").numpy()
            cand_ent = readout["candidate_entropy"].detach().to("cpu").numpy()
            top_idx = readout["top_candidate_idx"].detach().to("cpu").numpy()
            margin = readout["top2_margin_prob"].detach().to("cpu").numpy()
            proj_cpu = proj.detach().to("cpu").numpy()
            new_ids_cpu = new_ids.detach().to("cpu").numpy()

            out_rows: List[Dict[str, Any]] = []
            for i, r in enumerate(batch2):
                prompt_uid = str(r["prompt_uid"])
                rk = resume_key_for(model_id=model_id, prompt_uid=prompt_uid)

                ids_i = new_ids_cpu[i].tolist()
                first_id = int(ids_i[0]) if ids_i else None
                first_piece = (
                    tok.decode([first_id], clean_up_tokenization_spaces=False, skip_special_tokens=False) if first_id is not None else None
                )
                gen_text = tok.decode(ids_i, clean_up_tokenization_spaces=False, skip_special_tokens=False)

                parsed = parse_choice(
                    response_text=gen_text,
                    first_token=first_piece,
                    options=r["options"],
                )
                parsed_choice = parsed["parsed_choice"]
                is_correct = None
                if parsed_choice is not None:
                    is_correct = bool(str(parsed_choice) == str(r["correct_key"]))

                # Layerwise records.
                layerwise = []
                top_idx_list = []
                margin_list = []
                for li in range(int(n_layers)):
                    logits4 = cand_logits[i, li, :]
                    probs4 = cand_probs[i, li, :]
                    top_i = int(top_idx[i, li])
                    top_idx_list.append(top_i)
                    m_i = float(margin[i, li])
                    margin_list.append(m_i)
                    layerwise.append(
                        {
                            "layer_index": int(li),
                            "candidate_logits": {"A": float(logits4[0]), "B": float(logits4[1]), "C": float(logits4[2]), "D": float(logits4[3])},
                            "candidate_probs": {"A": float(probs4[0]), "B": float(probs4[1]), "C": float(probs4[2]), "D": float(probs4[3])},
                            "candidate_entropy": float(cand_ent[i, li]),
                            "top_candidate": idx_to_letter(top_i),
                            "top2_margin_prob": float(m_i),
                            "projected_hidden_128": [float(x) for x in proj_cpu[i, li, :].tolist()],
                        }
                    )

                traj = _trajectory_metrics(top_candidate_idx=top_idx_list, top2_margin_prob=margin_list, thresholds=thresholds)

                row_out = {
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_revision": revision,
                    "prompt_uid": prompt_uid,
                    "prompt_id": str(r["prompt_id"]),
                    "example_id": str(r["example_id"]),
                    "wrapper_id": str(r["wrapper_id"]),
                    "coarse_domain": str(r.get("coarse_domain") or "unknown"),
                    "manifest_sha256": str(r["manifest_sha256"]),
                    "prompt_text_sha256": sha256_text(str(r["prompt_text"])),
                    "prompt_length_tokens": int(prompt_lens[i]),
                    "resume_key": rk,
                    "generation_settings": run_meta["generation"],
                    "first_generated_token_id": first_id,
                    "first_generated_token_text": first_piece,
                    "generated_text": gen_text,
                    "first_token_is_option_letter": bool(piece_to_letter(first_piece or "") is not None),
                    "parsed_choice": parsed_choice,
                    "is_correct": is_correct,
                    "parser_status": _parse_status(str(parsed.get("decision") or "")),
                    "parser_signals": parsed,
                    "flip_count": int(traj["flip_count"]),
                    "commitment_layer_by_margin_threshold": traj["commitment_layer_by_margin_threshold"],
                    "layerwise": layerwise,
                    "basis_hash": str(pca["basis_hash"]),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                out_rows.append(row_out)
                completed[rk] = -1
                outputs_written += 1

                if stop_after_rows is not None and outputs_written >= int(stop_after_rows):
                    _write_jsonl_append(out_jsonl, out_rows)
                    return {
                        "pass": False,
                        "stopped_early": True,
                        "stop_after_rows": int(stop_after_rows),
                        "output_path": str(out_jsonl),
                        "output_sha256": sha256_file(out_jsonl),
                        "batching": {"resolved": int(bs), "batch_sizes_used": sorted(int(x) for x in bs_used)},
                    }

            _write_jsonl_append(out_jsonl, out_rows)

    t1 = time.perf_counter()
    v = validate_stage13_output(
        output_path=out_jsonl,
        manifest_rows=rows,
        model_id=model_id,
        model_revision=revision,
        basis_hash=str(pca["basis_hash"]),
        expected_n_layers=None,
    )
    v["timing_seconds_total"] = float(t1 - t0)
    v["model_id"] = model_id
    v["model_revision"] = revision
    v["condition"] = condition
    v["batching"] = {"requested": str(batch_size), "resolved": int(bs), "batch_sizes_used": sorted(int(x) for x in bs_used)}
    v["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    return {
        "pass": bool(v["pass"]),
        "stopped_early": False,
        "output_path": str(out_jsonl),
        "output_sha256": sha256_file(out_jsonl),
        "validator": v,
        "batching": v["batching"],
    }


def batch_consistency_gate(
    *,
    run_id: str,
    run_dir: Path,
    model: Dict[str, Any],
    generation: Dict[str, Any],
    manifest_path: Path,
    condition: str,
    rows: List[Dict[str, Any]],
    device_override: Optional[str],
    atol_logits: float,
) -> Dict[str, Any]:
    """
    Stage 13 gate: run same rows with batch_size=1 and batch_size=4 and compare
    first token ids and candidate logits (within atol).
    """
    # Use temp outputs under validation dir (never overwrite).
    v_dir = run_dir / "validation"
    v_dir.mkdir(parents=True, exist_ok=True)
    tmp_run = f"{run_id}__gate"
    model_id = str(model["model_id"])
    model_key = model_fs_id(model_id)

    # Materialize a tiny manifest file for deterministic ordering.
    tiny_manifest = _next_available_path(v_dir / f"stage13_gate_manifest.{condition}.{model_key}.jsonl")
    with tiny_manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    out_bs1 = _next_available_path(v_dir / f"stage13_gate_outputs.{condition}.{model_key}.bs1.jsonl")
    out_bs4 = _next_available_path(v_dir / f"stage13_gate_outputs.{condition}.{model_key}.bs4.jsonl")

    # Run bs=1.
    res1 = run_stage13_inference_for_model(
        run_id=tmp_run,
        run_dir=run_dir,
        model=model,
        generation=generation,
        manifest_path=tiny_manifest,
        condition=condition,
        batch_size=1,
        device_override=device_override,
        output_path_override=out_bs1,
        stop_after_rows=None,
        limit_rows=None,
        commitment_margin_thresholds=None,
        thermal_hygiene_cfg=None,
    )

    # Run bs=4.
    res4 = run_stage13_inference_for_model(
        run_id=tmp_run,
        run_dir=run_dir,
        model=model,
        generation=generation,
        manifest_path=tiny_manifest,
        condition=condition,
        batch_size=4,
        device_override=device_override,
        output_path_override=out_bs4,
        stop_after_rows=None,
        limit_rows=None,
        commitment_margin_thresholds=None,
        thermal_hygiene_cfg=None,
    )

    # Load both outputs keyed by prompt_uid.
    by_uid_1 = {str(r["prompt_uid"]): r for r in iter_jsonl(out_bs1)}
    by_uid_4 = {str(r["prompt_uid"]): r for r in iter_jsonl(out_bs4)}

    max_abs = 0.0
    mismatches: List[str] = []
    for uid in sorted(set(by_uid_1.keys()) | set(by_uid_4.keys())):
        a = by_uid_1.get(uid)
        b = by_uid_4.get(uid)
        if a is None or b is None:
            mismatches.append(f"missing uid in one output: {uid}")
            continue
        if a.get("first_generated_token_id") != b.get("first_generated_token_id"):
            mismatches.append(f"first token id mismatch uid={uid}")
        # Compare candidate logits per layer.
        la = a.get("layerwise") or []
        lb = b.get("layerwise") or []
        if len(la) != len(lb):
            mismatches.append(f"layer count mismatch uid={uid}")
            continue
        for i in range(len(la)):
            ca = la[i]["candidate_logits"]
            cb = lb[i]["candidate_logits"]
            for k in ["A", "B", "C", "D"]:
                da = float(ca[k])
                db = float(cb[k])
                max_abs = max(max_abs, abs(da - db))

    ok = (not mismatches) and (max_abs <= float(atol_logits))
    return {
        "pass": bool(ok),
        "model_id": model_id,
        "model_revision": str(model["revision"]),
        "condition": condition,
        "n_rows": len(rows),
        "atol_logits": float(atol_logits),
        "max_abs_diff_candidate_logits": float(max_abs),
        "mismatches": mismatches[:20],
        "artifacts": {"manifest": str(tiny_manifest), "bs1": str(out_bs1), "bs4": str(out_bs4)},
    }
