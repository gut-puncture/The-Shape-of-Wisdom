from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from dataset_loader import CanonicalExample, CanonicalOption

SYSTEM_PROMPT_HANDLING_SYSTEM_ROLE = "system_role"
SYSTEM_PROMPT_HANDLING_USER_PREFIX = "user_prefix"


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    prompt_hash: str
    example_id: str
    prompt_mode: str
    template_id: str
    readout_id: str
    permutation_id: str
    rendered_prompt: str
    answer_prefix: str
    prompt_token_count: int | None
    answer_start_token_index: int | None
    correct_option_after_permutation: str | None
    option_order: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def option_label(index: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]


def apply_permutation(example: CanonicalExample, permutation: list[int]) -> tuple[list[CanonicalOption], str | None]:
    if sorted(permutation) != list(range(len(example.options))):
        raise ValueError(f"invalid permutation: {permutation}")
    reordered = [example.options[idx] for idx in permutation]
    answer = None
    if example.answer_option_id is not None:
        original_index = next(
            (idx for idx, option in enumerate(example.options) if option.option_id == example.answer_option_id),
            None,
        )
        if original_index is not None:
            new_index = permutation.index(original_index)
            answer = option_label(new_index)
    rebuilt = [
        CanonicalOption(
            option_id=option_label(new_idx),
            ordinal=new_idx,
            raw_text=option.raw_text,
            canonical_text=option.canonical_text,
        )
        for new_idx, option in enumerate(reordered)
    ]
    return rebuilt, answer


def identity_permutation(option_count: int) -> list[int]:
    return list(range(option_count))


def _render_question_block(question: str, options: list[CanonicalOption]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    for option in options:
        lines.append(f"{option.option_id}. {option.raw_text}")
    return "\n".join(lines)


def build_chat_messages(
    *,
    user_text: str,
    system_prompt: str | None = None,
    system_prompt_handling: str = SYSTEM_PROMPT_HANDLING_SYSTEM_ROLE,
) -> list[dict[str, str]]:
    resolved_user_text = user_text
    messages: list[dict[str, str]] = []
    if system_prompt:
        if system_prompt_handling == SYSTEM_PROMPT_HANDLING_SYSTEM_ROLE:
            messages.append({"role": "system", "content": system_prompt})
        elif system_prompt_handling == SYSTEM_PROMPT_HANDLING_USER_PREFIX:
            resolved_user_text = f"System instruction: {system_prompt}\n\n{user_text}"
        else:
            raise ValueError(f"unsupported system_prompt_handling={system_prompt_handling}")
    messages.append({"role": "user", "content": resolved_user_text})
    return messages


def build_prompt(
    example: CanonicalExample,
    *,
    template_id: str,
    readout_id: str,
    prompt_mode: str,
    answer_prefix: str,
    permutation: list[int] | None = None,
    tokenizer: Any | None = None,
    system_prompt: str | None = None,
    system_prompt_handling: str = SYSTEM_PROMPT_HANDLING_SYSTEM_ROLE,
) -> RenderedPrompt:
    permutation = identity_permutation(len(example.options)) if permutation is None else permutation
    options, answer_after = apply_permutation(example, permutation)
    question_block = _render_question_block(example.question_raw, options)
    user_text = f"{question_block}\n\n{answer_prefix}"

    if prompt_mode == "chat":
        messages = build_chat_messages(
            user_text=user_text,
            system_prompt=system_prompt,
            system_prompt_handling=system_prompt_handling,
        )
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = "\n".join(
                [f"[{m['role'].upper()}]\n{m['content']}" for m in messages] + ["[ASSISTANT]"]
            )
    elif prompt_mode == "raw":
        rendered_prompt = user_text
    else:
        raise ValueError(f"unsupported prompt_mode={prompt_mode}")

    prompt_token_count = None
    answer_start_token_index = None
    if tokenizer is not None:
        prompt_token_count = len(tokenizer.encode(rendered_prompt, add_special_tokens=False))
        answer_start_token_index = prompt_token_count
    prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()

    permutation_id = "-".join(str(x) for x in permutation)
    return RenderedPrompt(
        prompt_id=f"{example.example_id}:{template_id}:{readout_id}:{permutation_id}",
        prompt_hash=prompt_hash,
        example_id=example.example_id,
        prompt_mode=prompt_mode,
        template_id=template_id,
        readout_id=readout_id,
        permutation_id=permutation_id,
        rendered_prompt=rendered_prompt,
        answer_prefix=answer_prefix,
        prompt_token_count=prompt_token_count,
        answer_start_token_index=answer_start_token_index,
        correct_option_after_permutation=answer_after,
        option_order=[option.option_id for option in options],
        metadata={
            "system_prompt": system_prompt,
            "system_prompt_handling": system_prompt_handling,
            "permutation": permutation,
        },
    )
