from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelNuance:
    model_id: str
    model_card_url: str
    model_doc_url: str
    min_transformers_version: str
    requires_left_padding: bool
    force_eager_attention_for_tracing: bool
    mps_dtype: str
    max_new_tokens_default: int


_MODEL_NUANCES = {
    "Qwen/Qwen2.5-7B-Instruct": ModelNuance(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        model_doc_url="https://huggingface.co/docs/transformers/model_doc/qwen2",
        min_transformers_version="4.37.0",
        requires_left_padding=True,
        force_eager_attention_for_tracing=True,
        mps_dtype="float16",
        max_new_tokens_default=24,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelNuance(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_card_url="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        model_doc_url="https://huggingface.co/docs/transformers/model_doc/llama",
        min_transformers_version="4.43.0",
        requires_left_padding=True,
        force_eager_attention_for_tracing=True,
        mps_dtype="float16",
        max_new_tokens_default=24,
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelNuance(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_card_url="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
        model_doc_url="https://huggingface.co/docs/transformers/model_doc/mistral",
        min_transformers_version="4.42.0",
        requires_left_padding=True,
        force_eager_attention_for_tracing=True,
        mps_dtype="float16",
        max_new_tokens_default=24,
    ),
}


def get_model_nuance(model_id: str) -> ModelNuance:
    if model_id not in _MODEL_NUANCES:
        raise KeyError(f"unsupported model_id for v2: {model_id}")
    return _MODEL_NUANCES[model_id]


def apply_tokenizer_nuance(tokenizer: Any, *, model_id: str) -> None:
    nuance = get_model_nuance(model_id)
    if nuance.requires_left_padding:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def pick_torch_dtype(*, device: str):
    import torch  # noqa: PLC0415

    if str(device) == "cpu":
        return torch.float32
    return torch.float16
