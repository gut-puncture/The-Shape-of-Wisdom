from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.token_buckets.option_buckets import model_fs_id
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig


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


def _write_json_atomic_new(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic_replace(path: Path, obj: Dict[str, Any]) -> None:
    """
    Atomic replace writer (used for checkpoints/progress files).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_npz_atomic_new(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    # Important: numpy.savez appends ".npz" when given a string/path that doesn't end
    # with ".npz". Use a file handle to guarantee the exact tmp path.
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def build_prompt_map(*, baseline_manifest: Path, robustness_manifest: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load baseline+robustness manifests into a prompt_uid -> row mapping.

    This is safe for our current sizes (~63k rows total).
    """
    m: Dict[str, Dict[str, Any]] = {}
    for r in iter_jsonl(baseline_manifest):
        m[str(r["prompt_uid"])] = r
    for r in iter_jsonl(robustness_manifest):
        m[str(r["prompt_uid"])] = r
    return m


def load_membership(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "membership" not in obj:
        raise ValueError(f"invalid membership file: {path}")
    return obj


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "mps backend out of memory" in msg


def _empty_device_cache(*, device: str) -> None:
    try:
        import torch  # noqa: PLC0415

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        # Best effort only.
        pass
    gc.collect()


def _parse_batch_size_arg(batch_size: int | str) -> Optional[int]:
    if isinstance(batch_size, int):
        return int(batch_size)
    s = str(batch_size).strip().lower()
    if s in ("auto", ""):
        return None
    return int(s)


def _extract_hidden_lastpos(
    *,
    model_obj: Any,
    input_ids: Any,
    attention_mask: Any,
    include_embedding: bool,
) -> Tuple[Any, int, int]:
    """
    Returns:
      - batch_hidden: torch.Tensor on CPU, shape (batch, n_layers, hidden_dim), dtype float16/float32
      - n_layers
      - hidden_dim
    """
    import torch  # noqa: PLC0415

    out = model_obj(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("model output_hidden_states returned None")
    if len(hs) < 2:
        raise RuntimeError(f"unexpected hidden_states length: {len(hs)}")

    # Per spec: layer 0 is first transformer block output (exclude embeddings).
    start = 0 if include_embedding else 1
    hs_layers = hs[start:]
    n_layers = len(hs_layers)
    hidden_dim = int(hs_layers[0].shape[-1])
    batch = int(hs_layers[0].shape[0])

    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    idx = (lengths - 1).to(hs_layers[0].device)
    ar = torch.arange(batch, device=hs_layers[0].device)

    # Gather last-position hidden per layer into a CPU tensor.
    out_cpu = torch.empty((batch, n_layers, hidden_dim), dtype=hs_layers[0].dtype, device="cpu")
    for li, layer_h in enumerate(hs_layers):
        # layer_h: (batch, seq, hidden)
        v = layer_h[ar, idx, :]  # (batch, hidden)
        out_cpu[:, li, :] = v.to("cpu")
    return out_cpu, n_layers, hidden_dim


def _pick_longest_prompts(*, prompts: List[str], tok: Any, k: int) -> List[str]:
    if k <= 0:
        return []
    if not prompts:
        return []
    k = min(int(k), len(prompts))
    # Fast-tokenizer path: return_length provides stable lengths without needing attention masks.
    try:
        enc = tok(prompts, add_special_tokens=False, return_length=True, padding=False, truncation=False)
        lengths = enc.get("length") or enc.get("lengths")
        if lengths is None:
            ids = enc.get("input_ids")
            if ids is None:
                raise RuntimeError("tokenizer did not return length(s) or input_ids")
            lengths = [len(x) for x in ids]
    except Exception:
        # Fallback: approximate by characters (still deterministic).
        lengths = [len(p) for p in prompts]
    idxs = np.argsort(np.asarray(lengths, dtype=np.int64))[-k:]
    return [prompts[int(i)] for i in idxs.tolist()]


def _auto_calibrate_batch_size(
    *,
    model_obj: Any,
    tok: Any,
    prompts: List[str],
    device: str,
    include_embedding: bool,
    governor: Optional[ThermalGovernor],
) -> int:
    """
    Choose the largest batch size from a small candidate set that does not OOM on the
    longest prompts in the membership list.
    """
    import torch  # noqa: PLC0415

    candidates = [32, 16, 8, 4, 2, 1] if device == "cuda" else [16, 8, 4, 2, 1]
    probe = _pick_longest_prompts(prompts=prompts, tok=tok, k=min(8, len(prompts)))
    if not probe:
        return 1

    with torch.inference_mode():
        for b in candidates:
            batch_prompts = (probe * ((b + len(probe) - 1) // len(probe)))[:b]
            try:
                if governor is not None:
                    governor.maybe_cooldown(stage="pca_sample_inference_calibration", model_id="(calibration)", model_revision="(calibration)")
                enc = tok(batch_prompts, return_tensors="pt", padding=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                if attn is None:
                    raise RuntimeError("tokenizer did not return attention_mask")
                if device != "cpu":
                    input_ids = input_ids.to(device)
                    attn = attn.to(device)
                _extract_hidden_lastpos(model_obj=model_obj, input_ids=input_ids, attention_mask=attn, include_embedding=include_embedding)
                return int(b)
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    _empty_device_cache(device=device)
                    continue
                raise
    return 1


@dataclass(frozen=True)
class _SampleHiddenCheckpoint:
    ckpt_dir: Path
    hidden_npy: Path
    ckpt_meta_json: Path
    progress_json: Path


def _checkpoint_paths(*, out_dir: Path, model_key: str, resume_key: str) -> _SampleHiddenCheckpoint:
    d = out_dir / "_ckpt" / f"{model_key}.{resume_key[:12]}"
    return _SampleHiddenCheckpoint(
        ckpt_dir=d,
        hidden_npy=d / "hidden.npy",
        ckpt_meta_json=d / "checkpoint_meta.json",
        progress_json=d / "progress.json",
    )


def run_pca_sample_inference_for_model(
    *,
    run_id: str,
    run_dir: Path,
    model: Dict[str, Any],
    generation: Dict[str, Any],
    baseline_manifest: Path,
    robustness_manifest: Path,
    membership_path: Path,
    device_override: Optional[str],
    batch_size: int | str,
    repro_check_k: int,
    repro_atol: float,
    thermal_hygiene_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Stage 11: forward-only hidden extraction over the frozen PCA membership prompts.

    We store only the last-position hidden state for each transformer layer.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if int(generation.get("max_new_tokens", 24)) != 24 or bool(generation.get("do_sample")) is not False:
        raise ValueError("PCA sample inference expects canonical greedy generation settings in run_config")
    if repro_check_k <= 0:
        raise ValueError("repro_check_k must be positive")
    if repro_atol <= 0:
        raise ValueError("repro_atol must be positive")

    model_id = str(model["model_id"])
    revision = str(model["revision"])

    device = device_override
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    mem = load_membership(membership_path)
    membership = mem["membership"]
    seed = int(mem.get("seed", 0))

    # Deterministic seeds for any stochastic ops (none expected under eval/no dropout).
    torch.manual_seed(int(seed))

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    prompt_uids = [str(m["prompt_uid"]) for m in membership]
    prompt_uids_sha = sha256_text("\n".join(prompt_uids) + "\n")

    prompt_map = build_prompt_map(baseline_manifest=baseline_manifest, robustness_manifest=robustness_manifest)
    prompts: List[str] = []
    for uid in prompt_uids:
        row = prompt_map.get(uid)
        if row is None:
            raise KeyError(f"membership prompt_uid not found in manifests: {uid}")
        prompts.append(str(row["prompt_text"]))

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    t_tok = time.perf_counter()

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
    t_model = time.perf_counter()

    # Cooperative thermal hygiene.
    th_cfg = ThermalHygieneConfig.from_cfg(thermal_hygiene_cfg)
    thermal_events_path = run_dir / "meta" / "thermal_events.jsonl"
    governor = ThermalGovernor(cfg=th_cfg, events_path=thermal_events_path) if th_cfg.enabled else None

    n = len(prompts)
    if n == 0:
        raise ValueError("membership contains zero prompts")

    out_dir = run_dir / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_bs = _parse_batch_size_arg(batch_size)
    if requested_bs is not None and requested_bs <= 0:
        requested_bs = None
    init_bs = int(requested_bs) if requested_bs is not None else _auto_calibrate_batch_size(
        model_obj=model_obj,
        tok=tok,
        prompts=prompts,
        device=device,
        include_embedding=False,
        governor=governor,
    )
    if init_bs <= 0:
        raise ValueError("resolved batch size must be positive")

    membership_sha = sha256_file(membership_path)
    resume_key = sha256_text(
        "\n".join(
            [
                "pca_sample_hidden",
                model_id,
                revision,
                str(device),
                str(torch_dtype).replace("torch.", ""),
                str(membership_sha),
                str(prompt_uids_sha),
                "include_embedding=false",
            ]
        )
        + "\n"
    )
    ckpt = _checkpoint_paths(out_dir=out_dir, model_key=model_fs_id(model_id), resume_key=resume_key)
    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Final output paths are chosen once and recorded in checkpoint metadata.
    out_npz: Path
    out_meta: Path
    resumed = False
    offset = 0
    n_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    hidden_mm: Optional[np.memmap] = None

    if ckpt.progress_json.exists():
        resumed = True
        if not ckpt.ckpt_meta_json.exists() or not ckpt.hidden_npy.exists():
            raise RuntimeError(f"checkpoint is incomplete (missing meta or hidden.npy): {ckpt.ckpt_dir}")
        ckpt_meta = json.loads(ckpt.ckpt_meta_json.read_text(encoding="utf-8"))
        if ckpt_meta.get("resume_key") != resume_key:
            raise RuntimeError(f"checkpoint resume_key mismatch; refusing to resume: {ckpt.ckpt_dir}")
        out_npz = Path(str(ckpt_meta["final_npz"]))
        out_meta = Path(str(ckpt_meta["final_meta"]))
        n_layers = int(ckpt_meta["n_layers"])
        hidden_dim = int(ckpt_meta["hidden_dim"])

        prog = json.loads(ckpt.progress_json.read_text(encoding="utf-8"))
        if prog.get("resume_key") != resume_key:
            raise RuntimeError(f"checkpoint progress resume_key mismatch; refusing to resume: {ckpt.ckpt_dir}")
        offset = int(prog.get("next_offset", 0))
        if offset < 0 or offset > n:
            raise RuntimeError(f"invalid checkpoint offset {offset} for total {n}")
        hidden_mm = np.load(str(ckpt.hidden_npy), mmap_mode="r+")
        if hidden_mm.shape != (n, int(n_layers), int(hidden_dim)):
            raise RuntimeError("checkpoint hidden.npy shape mismatch vs membership/meta")
    else:
        out_npz = _next_available_path(out_dir / f"{model_fs_id(model_id)}_sample_hidden.npz")
        out_meta = _next_available_path(out_dir / f"{model_fs_id(model_id)}_sample_hidden.meta.json")

    # If a prior attempt wrote final outputs but crashed before sentinel/report, make finalize idempotent.
    if out_npz.exists() and out_meta.exists():
        report = {
            "pass": True,
            "run_id": run_id,
            "model_id": model_id,
            "model_revision": revision,
            "sample_size": int(n),
            "n_layers": int(n_layers) if n_layers is not None else None,
            "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
            "hidden_sha256": sha256_file(out_npz),
            "meta_sha256": sha256_file(out_meta),
            "prompt_uids_sha256": prompt_uids_sha,
            "repro_check": {"k": 0, "atol": float(repro_atol), "max_abs_diff": 0.0, "pass": True, "skipped": True},
            "batch_invariance_check": {"skipped": True},
            "run_config_sha256": sha256_file(run_dir / "run_config.yaml"),
            "config_snapshot_sha256": sha256_file(run_dir / "meta" / "config_snapshot.yaml"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "report": report,
            "hidden_path": out_npz,
            "hidden_sha256": sha256_file(out_npz),
            "meta_path": out_meta,
            "meta_sha256": sha256_file(out_meta),
        }

    # Forward-only extraction with checkpointing (hidden.npy + progress.json).
    bs_cur = int(init_bs)
    bs_used = {int(bs_cur)}

    def run_batches(
        *,
        m_prompts: List[str],
        use_batch_size: int,
    ) -> np.ndarray:
        total = len(m_prompts)
        if total == 0:
            raise ValueError("run_batches: empty prompt list")
        out_arr = np.empty(
            (total, int(n_layers), int(hidden_dim)),
            dtype=np.float16 if torch_dtype == torch.float16 else np.float32,
        )
        with torch.inference_mode():
            off = 0
            while off < total:
                batch_prompts = m_prompts[off : off + int(use_batch_size)]
                enc = tok(batch_prompts, return_tensors="pt", padding=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                if attn is None:
                    raise RuntimeError("tokenizer did not return attention_mask")
                if device != "cpu":
                    input_ids = input_ids.to(device)
                    attn = attn.to(device)
                batch_hidden_t, bl, hd = _extract_hidden_lastpos(
                    model_obj=model_obj,
                    input_ids=input_ids,
                    attention_mask=attn,
                    include_embedding=False,
                )
                if int(bl) != int(n_layers) or int(hd) != int(hidden_dim):
                    raise RuntimeError("model hidden shape changed across batches")
                bsz = int(batch_hidden_t.shape[0])
                out_arr[off : off + bsz, :, :] = batch_hidden_t.numpy()
                off += bsz
        return out_arr

    t_forward_start = time.perf_counter()
    with torch.inference_mode():
        start_wall = time.perf_counter()
        last_print = start_wall

        while offset < n:
            if governor is not None:
                governor.maybe_cooldown(stage="pca_sample_inference", model_id=model_id, model_revision=revision)

            batch_prompts = prompts[offset : offset + int(bs_cur)]
            try:
                enc = tok(batch_prompts, return_tensors="pt", padding=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                if attn is None:
                    raise RuntimeError("tokenizer did not return attention_mask")
                if device != "cpu":
                    input_ids = input_ids.to(device)
                    attn = attn.to(device)
                batch_hidden_t, bl, hd = _extract_hidden_lastpos(
                    model_obj=model_obj,
                    input_ids=input_ids,
                    attention_mask=attn,
                    include_embedding=False,
                )
            except RuntimeError as exc:
                if _is_oom_error(exc) and int(bs_cur) > 1:
                    bs_cur = max(1, int(bs_cur) // 2)
                    bs_used.add(int(bs_cur))
                    _empty_device_cache(device=device)
                    continue
                raise

            if hidden_mm is None:
                n_layers = int(bl)
                hidden_dim = int(hd)
                # Initialize checkpoint memmap file.
                if ckpt.hidden_npy.exists():
                    raise FileExistsError(f"refusing to overwrite existing checkpoint hidden.npy: {ckpt.hidden_npy}")
                hidden_mm = np.lib.format.open_memmap(
                    str(ckpt.hidden_npy),
                    mode="w+",
                    dtype=np.float16 if torch_dtype == torch.float16 else np.float32,
                    shape=(n, int(n_layers), int(hidden_dim)),
                )
                ckpt_meta = {
                    "resume_key": resume_key,
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_revision": revision,
                    "device": device,
                    "torch_dtype": str(torch_dtype).replace("torch.", ""),
                    "membership_path": str(membership_path),
                    "membership_sha256": membership_sha,
                    "prompt_uids_sha256": prompt_uids_sha,
                    "prompt_uids_count": int(n),
                    "include_embedding": False,
                    "n_layers": int(n_layers),
                    "hidden_dim": int(hidden_dim),
                    "hidden_dtype": str(hidden_mm.dtype),
                    "checkpoint_dir": str(ckpt.ckpt_dir),
                    "hidden_npy": str(ckpt.hidden_npy),
                    "final_npz": str(out_npz),
                    "final_meta": str(out_meta),
                    "batching": {"requested": str(batch_size), "initial_resolved": int(init_bs)},
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_json_atomic_replace(ckpt.ckpt_meta_json, ckpt_meta)

            if n_layers is None or hidden_dim is None or hidden_mm is None:
                raise RuntimeError("internal error: checkpoint memmap not initialized")
            if int(bl) != int(n_layers) or int(hd) != int(hidden_dim):
                raise RuntimeError("model hidden shape changed across batches")

            bsz = int(batch_hidden_t.shape[0])
            hidden_mm[offset : offset + bsz, :, :] = batch_hidden_t.numpy()
            hidden_mm.flush()
            offset += bsz

            _write_json_atomic_replace(
                ckpt.progress_json,
                {
                    "resume_key": resume_key,
                    "next_offset": int(offset),
                    "total": int(n),
                    "batch_size_current": int(bs_cur),
                    "batch_sizes_used": sorted(int(x) for x in bs_used),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Lightweight progress print (~every 15s) for ETA visibility.
            now = time.perf_counter()
            if (now - last_print) >= 15.0 or offset >= n:
                elapsed = now - start_wall
                rate = (offset / elapsed) if elapsed > 0 else 0.0
                eta_s = ((n - offset) / rate) if rate > 0 else None
                eta_txt = f"{eta_s/60:.1f}m" if eta_s is not None else "?"
                print(f"[pca-sample-inference] {model_id}@{revision} {offset}/{n} bs={bs_cur} rate={rate:.2f} rows/s eta={eta_txt}")
                last_print = now

    t_done = time.perf_counter()

    if n_layers is None or hidden_dim is None or hidden_mm is None:
        raise RuntimeError("hidden extraction produced no data")
    if hidden_mm.shape != (n, int(n_layers), int(hidden_dim)):
        raise RuntimeError(f"unexpected hidden_mm shape: {hidden_mm.shape} expected {(n, int(n_layers), int(hidden_dim))}")

    # Reproducibility spot-check: recompute first K prompts and compare (same batch size).
    k = min(int(repro_check_k), n)
    hidden_k = run_batches(m_prompts=prompts[:k], use_batch_size=int(max(bs_used)))
    diff = float(np.max(np.abs(hidden_k.astype(np.float32) - np.asarray(hidden_mm[:k]).astype(np.float32))))
    repro_ok = bool(diff <= float(repro_atol))

    # Batch invariance spot-check (if we ever ran with batching > 1): compare bs=1 vs max(bs_used).
    bs_max = int(max(bs_used))
    if bs_max > 1:
        hidden_bs1 = run_batches(m_prompts=prompts[:k], use_batch_size=1)
        hidden_bsn = run_batches(m_prompts=prompts[:k], use_batch_size=bs_max)
        bdiff = float(np.max(np.abs(hidden_bs1.astype(np.float32) - hidden_bsn.astype(np.float32))))
        batch_invariance_ok = bool(bdiff <= float(repro_atol))
        batch_invariance = {"k": int(k), "atol": float(repro_atol), "max_abs_diff": float(bdiff), "batch_size": int(bs_max), "pass": bool(batch_invariance_ok)}
    else:
        batch_invariance_ok = True
        batch_invariance = {"skipped": True, "reason": "batch_size_max==1"}

    meta = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "device": device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "membership_path": str(membership_path),
        "membership_sha256": membership_sha,
        "prompt_uids": prompt_uids,
        "prompt_uids_sha256": prompt_uids_sha,
        "layer_indices": list(range(int(n_layers))),
        "hidden_dim": int(hidden_dim),
        "hidden_dtype": str(hidden_mm.dtype),
        "baseline_manifest_path": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "robustness_manifest_path": str(robustness_manifest),
        "robustness_manifest_sha256": sha256_file(robustness_manifest),
        "thermal_hygiene": {
            "enabled": bool(th_cfg.enabled),
            "provider": th_cfg.provider,
            "cutoff_level": th_cfg.cutoff_level,
            "cooldown_seconds": int(th_cfg.cooldown_seconds),
            "check_interval_seconds": int(th_cfg.check_interval_seconds),
            "events_path": str(thermal_events_path),
        },
        "checkpointing": {
            "enabled": True,
            "resume_key": resume_key,
            "checkpoint_dir": str(ckpt.ckpt_dir),
            "resumed": bool(resumed),
            "progress_path": str(ckpt.progress_json),
            "checkpoint_meta_path": str(ckpt.ckpt_meta_json),
            "hidden_npy_path": str(ckpt.hidden_npy),
        },
        "batching": {"requested": str(batch_size), "initial_resolved": int(init_bs), "batch_sizes_used": sorted(int(x) for x in bs_used)},
        "repro_check": {"k": int(k), "atol": float(repro_atol), "max_abs_diff": float(diff), "pass": bool(repro_ok)},
        "batch_invariance_check": batch_invariance,
        "timing_seconds": {
            "total": float(t_done - t0),
            "tokenizer_load": float(t_tok - t0),
            "model_load": float(t_model - t_tok),
            "forward_only": float(t_done - t_forward_start),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Finalize (idempotent-ish): if one exists, attempt to complete the other.
    if not out_npz.exists():
        _write_npz_atomic_new(out_npz, hidden=hidden_mm)
    if not out_meta.exists():
        _write_json_atomic_new(out_meta, meta)

    report = {
        "pass": bool(repro_ok and batch_invariance_ok),
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "sample_size": int(n),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "hidden_sha256": sha256_file(out_npz),
        "meta_sha256": sha256_file(out_meta),
        "prompt_uids_sha256": prompt_uids_sha,
        "repro_check": meta["repro_check"],
        "batch_invariance_check": meta["batch_invariance_check"],
        "run_config_sha256": sha256_file(run_dir / "run_config.yaml"),
        "config_snapshot_sha256": sha256_file(run_dir / "meta" / "config_snapshot.yaml"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # On success, remove checkpoint files to avoid doubling disk usage.
    if report["pass"]:
        try:
            for p in [ckpt.progress_json, ckpt.ckpt_meta_json, ckpt.hidden_npy]:
                if p.exists():
                    p.unlink()
            # Remove the directory if empty.
            if ckpt.ckpt_dir.exists() and not any(ckpt.ckpt_dir.iterdir()):
                ckpt.ckpt_dir.rmdir()
            # Keep parent _ckpt dir if it contains other checkpoints.
        except Exception:
            # Best effort cleanup only.
            pass

    return {
        "report": report,
        "hidden_path": out_npz,
        "hidden_sha256": sha256_file(out_npz),
        "meta_path": out_meta,
        "meta_sha256": sha256_file(out_meta),
    }
