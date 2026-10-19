from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

import transformers

from model_registry import ModelSpec


@dataclass
class LoadedModelBundle:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    hidden_state_projection_mode: str


def _set_default_tokenizer_behavior(tokenizer: Any) -> None:
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def validate_requested_device(device: str) -> str:
    normalized = str(device).strip()
    if not normalized:
        raise ValueError("device must be a non-empty string")
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        raise ValueError(f"unable to validate device without torch: {exc}") from exc

    if normalized == "cpu":
        return normalized
    if normalized == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("requested device=mps but MPS is not available")
        return normalized
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("requested cuda device but CUDA is not available")
        if ":" in normalized:
            _, _, raw_index = normalized.partition(":")
            if raw_index:
                index = int(raw_index)
                if index < 0 or index >= torch.cuda.device_count():
                    raise ValueError(
                        f"requested cuda device index {index} but available count is {torch.cuda.device_count()}"
                    )
        return normalized
    try:
        torch.device(normalized)
    except Exception as exc:
        raise ValueError(f"unsupported device={normalized}: {exc}") from exc
    return normalized


def resolve_requested_dtype(*, device: str, dtype: str) -> tuple[Any | None, str]:
    normalized = str(dtype).strip().lower() or "auto"
    import torch  # noqa: PLC0415

    if normalized in {"", "auto"}:
        if str(device).startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bfloat16"
            return torch.float16, "float16"
        return torch.float32, "float32"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported dtype={dtype}")
    if resolved == torch.bfloat16 and str(device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise ValueError("requested dtype=bfloat16 but CUDA bf16 is not supported")
    if str(device) == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        raise ValueError(f"requested dtype={normalized} on cpu is unsupported for this pipeline")
    return resolved, normalized


def _device_memory_summary(device: str) -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        return {"torch_import_error": str(exc)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = 0
        if ":" in str(device):
            _, _, raw_index = str(device).partition(":")
            if raw_index:
                index = int(raw_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        props = torch.cuda.get_device_properties(index)
        return {
            "device_index": index,
            "device_name": props.name,
            "total_vram_gb": total_bytes / float(1024**3),
            "free_vram_gb": free_bytes / float(1024**3),
        }
    return {}


def load_model_bundle(spec: ModelSpec, *, device: str = "cpu", dtype: str = "auto") -> LoadedModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    resolved_device = validate_requested_device(device)
    torch_dtype, dtype_name = resolve_requested_dtype(device=resolved_device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.tokenizer_revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=spec.local_files_only,
    )
    _set_default_tokenizer_behavior(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=spec.local_files_only,
        dtype=torch_dtype,
    )
    model.eval()
    model.to(resolved_device)
    return LoadedModelBundle(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=dtype_name,
        hidden_state_projection_mode=spec.hidden_state_projection_mode,
    )


def capture_environment_manifest(*, device: str | None = None) -> dict[str, Any]:
    pip_freeze = subprocess.run(
        ["python3", "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    manifest: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "transformers_version": transformers.__version__,
        "pip_freeze": pip_freeze.stdout.splitlines(),
        "env_hf_home": os.environ.get("HF_HOME"),
    }
    try:
        import torch  # noqa: PLC0415

        manifest.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_device_names": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else [],
                "cuda_bf16_supported": bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()),
            }
        )
    except Exception as exc:
        manifest["torch_import_error"] = str(exc)
    summary_device = str(device) if device is not None else ("cuda:0" if manifest.get("cuda_available") else "cpu")
    manifest["device_memory"] = _device_memory_summary(summary_device)
    return manifest
