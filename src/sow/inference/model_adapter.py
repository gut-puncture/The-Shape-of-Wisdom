from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelComponents:
    final_norm: Any
    lm_head: Any


def resolve_model_components(model_obj: Any) -> ModelComponents:
    """
    Resolve the two modules required by the Stage 13 logit-lens style readout:
      - final normalization module
      - LM head (vocab projection)

    We only support the three pinned HF model families for this project.
    """
    final_norm = None
    # Llama / Mistral / many decoder-only models
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "norm"):
        final_norm = model_obj.model.norm
    # Some older/alternate layouts
    if final_norm is None and hasattr(model_obj, "transformer") and hasattr(model_obj.transformer, "norm"):
        final_norm = model_obj.transformer.norm
    if final_norm is None and hasattr(model_obj, "model") and hasattr(model_obj.model, "final_layernorm"):
        final_norm = model_obj.model.final_layernorm
    if final_norm is None:
        raise ValueError("could not resolve final_norm module on model (expected .model.norm or .transformer.norm)")

    lm_head = None
    if hasattr(model_obj, "lm_head"):
        lm_head = model_obj.lm_head
    if lm_head is None and hasattr(model_obj, "model") and hasattr(model_obj.model, "lm_head"):
        lm_head = model_obj.model.lm_head
    if lm_head is None:
        raise ValueError("could not resolve lm_head on model (expected .lm_head)")

    # Sanity checks: weight must exist.
    w = getattr(lm_head, "weight", None)
    if w is None:
        raise ValueError("lm_head missing .weight")

    return ModelComponents(final_norm=final_norm, lm_head=lm_head)


def resolve_vocab_size(model_obj: Any) -> Optional[int]:
    cfg = getattr(model_obj, "config", None)
    if cfg is None:
        return None
    vs = getattr(cfg, "vocab_size", None)
    return int(vs) if isinstance(vs, int) else None

