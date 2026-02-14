from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sow.hashing import sha256_file, sha256_text
from sow.token_buckets.option_buckets import build_buckets_from_tokenizer, validate_bucket_obj


def _run_cmd(argv: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip()


def collect_environment(*, repo_root: Path) -> Dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        torch = None
    try:
        import transformers  # noqa: PLC0415
    except Exception:
        transformers = None

    env: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {
            "commit": _run_cmd(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "status_porcelain": _run_cmd(["git", "-C", str(repo_root), "status", "--porcelain=v1"]),
        },
        "libraries": {
            "torch": getattr(torch, "__version__", None) if torch else None,
            "transformers": getattr(transformers, "__version__", None) if transformers else None,
        },
        "accel": {},
    }

    if torch is not None:
        env["accel"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            try:
                env["accel"]["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                env["accel"]["cuda_device_name"] = None

    return env


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


def run_smoke_test(
    *,
    model_id: str,
    revision: str,
    generation: Dict[str, Any],
    seed: int,
    preferred_device: Optional[str] = None,
    thermal_governor: Any | None = None,
) -> Dict[str, Any]:
    """
    Stage 0 smoke: verify tokenizer + model load, forward pass w/ hidden states, bucket scoring, and a short greedy generate.
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = _pick_device(preferred=preferred_device)
    torch.manual_seed(int(seed))

    prompt = (
        "Question: What is 2+2?\n"
        "A) 3\n"
        "B) 4\n"
        "C) 5\n"
        "D) 22\n\n"
        "Return only the letter (A, B, C, or D).\n"
        "Answer: "
    )

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    t_tok = time.perf_counter()

    # float16 is required for feasibility on most accelerators; fall back to float32 if forced.
    torch_dtype = torch.float16
    if device == "cpu":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    t_model = time.perf_counter()

    # Token buckets (deterministic) + fail-fast on overlaps.
    bucket_obj = build_buckets_from_tokenizer(tok)
    validate_bucket_obj({"buckets": bucket_obj["buckets"], "overlaps": bucket_obj.get("overlaps", {})})

    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    if device != "cpu":
        input_ids = input_ids.to(device)
        if attn is not None:
            attn = attn.to(device)

    # Forward pass with hidden states
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
    t_fwd = time.perf_counter()

    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("model forward did not return hidden_states")
    n_layers_total = len(hs)
    hidden_dim = int(hs[0].shape[-1])

    # Candidate bucket scores at the decision position (next-token logits).
    logits = out.logits[0, -1, :]
    scores = {}
    for letter in ["A", "B", "C", "D"]:
        ids = bucket_obj["buckets"][letter]
        tids = torch.tensor(ids, device=logits.device, dtype=torch.long)
        scores[letter] = float(torch.logsumexp(logits.index_select(0, tids), dim=0).item())

    # Short greedy generation
    do_sample = bool(generation.get("do_sample"))
    if do_sample:
        raise ValueError("Smoke test requires greedy decoding (do_sample=false)")

    thermal_action = None
    if thermal_governor is not None:
        # Cooperative thermal hygiene (may sleep) before generation.
        thermal_action = thermal_governor.maybe_cooldown(
            stage="stage0_generate",
            model_id=model_id,
            model_revision=revision,
        )

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        do_sample=False,
        max_new_tokens=int(generation.get("max_new_tokens", 24)),
        temperature=float(generation.get("temperature", 1.0)),
        top_p=float(generation.get("top_p", 1.0)),
        use_cache=True,
    )
    t_gen = time.perf_counter()

    input_len = int(input_ids.shape[1])
    new_ids = gen_ids[0, input_len:]
    first_id = int(new_ids[0].item()) if new_ids.numel() > 0 else None
    first_piece = tok.decode([first_id], clean_up_tokenization_spaces=False, skip_special_tokens=False) if first_id is not None else None
    gen_text = tok.decode(new_ids.tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=False)

    report = {
        "pass": True,
        "model_id": model_id,
        "model_revision": revision,
        "device": device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "transformers_version": str(transformers.__version__),
        "torch_version": str(torch.__version__),
        "seed": int(seed),
        "prompt_text_sha256": sha256_text(prompt),
        "prompt_text": prompt,
        "input_len_tokens": input_len,
        "hidden_states": {"n_layers_total": n_layers_total, "hidden_dim": hidden_dim},
        "bucket_sizes": {k: len(bucket_obj["buckets"][k]) for k in ["A", "B", "C", "D"]},
        "bucket_scores_logsumexp": scores,
        "generation": {
            "max_new_tokens": int(generation.get("max_new_tokens", 24)),
            "first_generated_token_id": first_id,
            "first_generated_token_text": first_piece,
            "generated_text": gen_text,
        },
        "thermal_hygiene": {
            "enabled": bool(getattr(getattr(thermal_governor, "cfg", None), "enabled", False)) if thermal_governor else False,
            "provider": getattr(getattr(thermal_governor, "cfg", None), "provider", None) if thermal_governor else None,
            "cutoff_level": getattr(getattr(thermal_governor, "cfg", None), "cutoff_level", None) if thermal_governor else None,
            "cooldown_seconds": getattr(getattr(thermal_governor, "cfg", None), "cooldown_seconds", None) if thermal_governor else None,
            "check_interval_seconds": getattr(getattr(thermal_governor, "cfg", None), "check_interval_seconds", None) if thermal_governor else None,
            "action": thermal_action,
        },
        "timing_seconds": {
            "total": float(t_gen - t0),
            "tokenizer_load": float(t_tok - t0),
            "model_load": float(t_model - t_tok),
            "forward": float(t_fwd - t_model),
            "generate": float(t_gen - t_fwd),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report
