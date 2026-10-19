from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TOKENIZATION_VERSION = "contextual_v1"


@dataclass(frozen=True)
class ContextualTokenization:
    prompt_text: str
    surface: str
    prompt_token_ids: list[int]
    full_token_ids: list[int]
    token_ids: list[int]
    context_preceded_by_whitespace: bool
    context_preceded_by_newline: bool


def tokenize_continuation_in_context(tokenizer: Any, *, prompt_text: str, surface: str) -> ContextualTokenization:
    prompt_token_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full_token_ids = list(tokenizer.encode(f"{prompt_text}{surface}", add_special_tokens=False))
    prefix_len = len(prompt_token_ids)
    if full_token_ids[:prefix_len] != prompt_token_ids:
        raise ValueError(
            "contextual tokenization mismatch: prompt prefix tokens do not match prompt+surface prefix"
        )
    token_ids = list(full_token_ids[prefix_len:])
    return ContextualTokenization(
        prompt_text=prompt_text,
        surface=surface,
        prompt_token_ids=prompt_token_ids,
        full_token_ids=full_token_ids,
        token_ids=token_ids,
        context_preceded_by_whitespace=bool(prompt_text) and prompt_text[-1].isspace(),
        context_preceded_by_newline=bool(prompt_text) and prompt_text[-1] == "\n",
    )


def contextual_token_ids(tokenizer: Any, *, prompt_text: str, surface: str) -> list[int]:
    return tokenize_continuation_in_context(tokenizer, prompt_text=prompt_text, surface=surface).token_ids


def tokenize_surfaces_in_context(
    tokenizer: Any,
    *,
    prompt_text: str,
    option_surfaces: dict[str, str],
) -> dict[str, ContextualTokenization]:
    return {
        option_id: tokenize_continuation_in_context(tokenizer, prompt_text=prompt_text, surface=surface)
        for option_id, surface in option_surfaces.items()
    }
