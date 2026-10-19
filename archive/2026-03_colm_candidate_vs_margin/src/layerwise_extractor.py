from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from continuation_tokenizer import contextual_token_ids
from model_loader import LoadedModelBundle


PROJECTION_APPLY_FINAL_NORM = "apply_final_norm"
PROJECTION_APPLY_FINAL_NORM_EXCEPT_LAST = "apply_final_norm_except_last"
PROJECTION_SKIP_FINAL_NORM = "skip_final_norm"


def _find_final_norm(model: Any) -> Any | None:
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    for root_name, attr_name in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, attr_name):
            return getattr(root, attr_name)
    base = getattr(model, "base_model", None)
    if base is not None:
        nested = getattr(base, "model", None)
        if nested is not None and hasattr(nested, "norm"):
            return getattr(nested, "norm")
    return None


def _hidden_state_sequence(outputs: Any) -> list[torch.Tensor]:
    hidden_states = list(outputs.hidden_states or [])
    if len(hidden_states) >= 2:
        return hidden_states[1:]
    return hidden_states


def project_hidden_state_to_logits(
    hidden_state: torch.Tensor,
    *,
    output_embedding: Any,
    final_norm: Any | None = None,
    apply_final_norm: bool = True,
    model_config: Any | None = None,
) -> torch.Tensor:
    state = hidden_state
    if apply_final_norm and final_norm is not None:
        state = final_norm(state)
    logits = output_embedding(state)
    softcap = getattr(model_config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = logits / float(softcap)
        logits = torch.tanh(logits)
        logits = logits * float(softcap)
    return logits.float()


def should_apply_final_norm(
    projection_mode: str,
    *,
    is_last_hidden_state: bool,
) -> bool:
    if projection_mode == PROJECTION_APPLY_FINAL_NORM:
        return True
    if projection_mode == PROJECTION_SKIP_FINAL_NORM:
        return False
    if projection_mode == PROJECTION_APPLY_FINAL_NORM_EXCEPT_LAST:
        return not is_last_hidden_state
    raise ValueError(f"unsupported hidden_state_projection_mode={projection_mode}")


@dataclass
class LayerwiseStepResult:
    forced_token_id: int
    forced_token_logprobs_by_layer: list[float]
    direct_final_logprob: float


@dataclass
class LayerwiseOptionScore:
    token_ids: list[int]
    steps: list[LayerwiseStepResult]
    layer_sums: list[float]
    layer_means: list[float]


def assert_final_layer_equivalence(
    *,
    outputs: Any,
    output_embedding: Any,
    final_norm: Any | None,
    projection_mode: str,
    model_config: Any | None = None,
    atol: float = 1e-4,
) -> None:
    hidden_states = _hidden_state_sequence(outputs)
    if not hidden_states:
        raise ValueError("model outputs did not include usable hidden states")
    projected = project_hidden_state_to_logits(
        hidden_states[-1],
        output_embedding=output_embedding,
        final_norm=final_norm,
        apply_final_norm=should_apply_final_norm(projection_mode, is_last_hidden_state=True),
        model_config=model_config,
    )
    direct = outputs.logits.float()
    if not torch.allclose(projected, direct, atol=atol, rtol=0.0):
        delta = torch.max(torch.abs(projected - direct)).item()
        raise ValueError(f"final-layer equivalence failed; max_abs_delta={delta}")


def score_option_layerwise(bundle: LoadedModelBundle, prompt_text: str, option_surface: str) -> LayerwiseOptionScore:
    tokenizer = bundle.tokenizer

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    option_ids = contextual_token_ids(tokenizer, prompt_text=prompt_text, surface=option_surface)
    if not prompt_ids:
        raise ValueError("prompt tokenization produced no tokens")
    if not option_ids:
        raise ValueError("option tokenization produced no tokens")
    from scoring_engine import _score_token_ids_reference  # noqa: PLC0415

    return _score_token_ids_reference(bundle, prompt_token_ids=list(prompt_ids), option_ids=list(option_ids))
