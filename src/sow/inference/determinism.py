from __future__ import annotations

from typing import Any, Dict


def configure_torch_determinism(*, device: str) -> Dict[str, Any]:
    """
    Best-effort determinism knobs for inference.

    Goals:
    - Make greedy decoding + layerwise readouts stable across batch sizes.
    - Avoid TF32 / autotuning differences.
    - Avoid non-deterministic SDPA kernels when possible.

    Notes:
    - We intentionally do NOT call torch.use_deterministic_algorithms(True) here because
      it can raise hard errors for some transformer kernels on some builds.
      Instead, we rely on our explicit batch-consistency gates to fail-fast.
    """
    import torch  # noqa: PLC0415

    applied: Dict[str, Any] = {"device": str(device)}

    if device == "cuda" and torch.cuda.is_available():
        # TF32 can introduce subtle numeric drift across kernel choices.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # cuDNN determinism knobs (mostly relevant for conv, but harmless here).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # SDPA kernel selection. Flash/mem-efficient can be nondeterministic across shapes.
        cuda_b = getattr(torch.backends, "cuda", None)
        if cuda_b is not None:
            fn = getattr(cuda_b, "enable_flash_sdp", None)
            if callable(fn):
                fn(False)
                applied["sdpa_flash_disabled"] = True
            fn = getattr(cuda_b, "enable_mem_efficient_sdp", None)
            if callable(fn):
                fn(False)
                applied["sdpa_mem_efficient_disabled"] = True
            fn = getattr(cuda_b, "enable_math_sdp", None)
            if callable(fn):
                fn(True)
                applied["sdpa_math_enabled"] = True

        applied["allow_tf32"] = False
        applied["cudnn_deterministic"] = True
        applied["cudnn_benchmark"] = False

    return applied

