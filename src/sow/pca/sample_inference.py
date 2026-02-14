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
    batch_size: int,
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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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

    # Deterministic seeds for any stochastic ops (none expected under eval/no dropout).
    torch.manual_seed(int(json.loads(membership_path.read_text(encoding="utf-8"))["seed"]))

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    mem = load_membership(membership_path)
    membership = mem["membership"]
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

    # Forward-only extraction.
    n = len(prompts)
    n_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    hidden_all: Optional[np.ndarray] = None

    def run_batches(*, take_first_k: Optional[int] = None) -> np.ndarray:
        nonlocal n_layers, hidden_dim
        m_prompts = prompts[:take_first_k] if take_first_k is not None else prompts
        total = len(m_prompts)
        out_arr: Optional[np.ndarray] = None

        with torch.inference_mode():
            offset = 0
            while offset < total:
                if governor is not None:
                    governor.maybe_cooldown(stage="pca_sample_inference", model_id=model_id, model_revision=revision)

                batch_prompts = m_prompts[offset : offset + batch_size]
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
                if n_layers is None:
                    n_layers = int(bl)
                    hidden_dim = int(hd)
                    out_arr = np.empty((total, n_layers, hidden_dim), dtype=np.float16 if torch_dtype == torch.float16 else np.float32)
                else:
                    if int(bl) != int(n_layers) or int(hd) != int(hidden_dim):
                        raise RuntimeError("model hidden shape changed across batches")
                if out_arr is None:
                    raise RuntimeError("internal error: out_arr not initialized")

                bsz = int(batch_hidden_t.shape[0])
                out_arr[offset : offset + bsz, :, :] = batch_hidden_t.numpy()
                offset += bsz
        if out_arr is None:
            raise RuntimeError("internal error: out_arr not initialized")
        return out_arr

    hidden_all = run_batches()
    t_done = time.perf_counter()

    if n_layers is None or hidden_dim is None:
        raise RuntimeError("hidden extraction produced no data")
    if hidden_all.shape != (n, n_layers, hidden_dim):
        raise RuntimeError(f"unexpected hidden_all shape: {hidden_all.shape} expected {(n, n_layers, hidden_dim)}")

    # Reproducibility spot-check: recompute first K prompts and compare.
    k = min(int(repro_check_k), n)
    hidden_k = run_batches(take_first_k=k)
    diff = np.max(np.abs(hidden_k.astype(np.float32) - hidden_all[:k].astype(np.float32)))
    repro_ok = bool(diff <= float(repro_atol))

    out_dir = run_dir / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = _next_available_path(out_dir / f"{model_fs_id(model_id)}_sample_hidden.npz")
    out_meta = _next_available_path(out_dir / f"{model_fs_id(model_id)}_sample_hidden.meta.json")

    meta = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "device": device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "membership_path": str(membership_path),
        "membership_sha256": sha256_file(membership_path),
        "prompt_uids": prompt_uids,
        "prompt_uids_sha256": prompt_uids_sha,
        "layer_indices": list(range(int(n_layers))),
        "hidden_dim": int(hidden_dim),
        "hidden_dtype": str(hidden_all.dtype),
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
        "repro_check": {"k": int(k), "atol": float(repro_atol), "max_abs_diff": float(diff), "pass": bool(repro_ok)},
        "timing_seconds": {
            "total": float(t_done - t0),
            "tokenizer_load": float(t_tok - t0),
            "model_load": float(t_model - t_tok),
            "forward_only": float(t_done - t_model),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _write_npz_atomic_new(out_npz, hidden=hidden_all)
    _write_json_atomic_new(out_meta, meta)

    report = {
        "pass": bool(repro_ok),
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
