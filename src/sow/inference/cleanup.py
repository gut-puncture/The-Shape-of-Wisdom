from __future__ import annotations

import gc
from typing import Any, Dict, Optional


def cleanup_torch(*, device: str) -> Dict[str, Any]:
    """
    Best-effort memory cleanup between heavyweight model runs.

    This is critical on single-GPU machines when running multiple models sequentially
    in a single Python process (pilot / PCA sample extraction / Stage 13 inference).
    """
    out: Dict[str, Any] = {"device": str(device)}

    # GC first to drop Python refs before touching the allocator.
    gc.collect()
    out["gc_collect"] = True

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        out["torch_import_error"] = {"type": type(exc).__name__, "msg": str(exc)}
        return out

    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            out["cuda_empty_cache"] = True
        except Exception as exc:  # pragma: no cover
            out["cuda_empty_cache_error"] = {"type": type(exc).__name__, "msg": str(exc)}
        try:
            torch.cuda.ipc_collect()
            out["cuda_ipc_collect"] = True
        except Exception as exc:  # pragma: no cover
            out["cuda_ipc_collect_error"] = {"type": type(exc).__name__, "msg": str(exc)}

    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
            out["mps_empty_cache"] = True
        except Exception as exc:  # pragma: no cover
            out["mps_empty_cache_error"] = {"type": type(exc).__name__, "msg": str(exc)}

    return out


def cleanup_model(*, device: str, model_obj: Optional[Any], tok: Optional[Any]) -> Dict[str, Any]:
    """
    Explicitly drop strong refs to model/tokenizer then run allocator cleanup.
    """
    try:
        if model_obj is not None:
            del model_obj
        if tok is not None:
            del tok
    except Exception:
        # Best-effort only.
        pass
    return cleanup_torch(device=device)

