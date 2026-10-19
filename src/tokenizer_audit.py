from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from continuation_tokenizer import tokenize_continuation_in_context


@dataclass(frozen=True)
class TokenAuditRecord:
    model_id: str
    example_id: str
    readout_id: str
    prompt_id: str
    prompt_token_count: int
    answer_start_token_index: int
    option_id: str
    option_surface: str
    constructed_surface: str
    token_ids: list[int]
    token_strings: list[str]
    token_count: int
    context_preceded_by_whitespace: bool
    context_preceded_by_newline: bool
    suspicious_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_token_audit_records(
    *,
    tokenizer: Any,
    model_id: str,
    example_id: str,
    readout_id: str,
    prompt_id: str,
    prompt_text: str,
    answer_start_token_index: int,
    option_surfaces: dict[str, str],
    suspicious_length_threshold: int = 16,
) -> list[TokenAuditRecord]:
    prompt_token_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    prompt_token_count = len(prompt_token_ids)
    if prompt_token_count != int(answer_start_token_index):
        raise ValueError(
            f"answer_start_token_index mismatch: expected {prompt_token_count}, got {answer_start_token_index}"
        )
    token_sequences: dict[tuple[int, ...], list[str]] = {}
    records: list[TokenAuditRecord] = []
    for option_id, surface in option_surfaces.items():
        tokenization = tokenize_continuation_in_context(tokenizer, prompt_text=prompt_text, surface=surface)
        constructed_surface = f"{prompt_text}{surface}"
        full_token_ids = list(tokenization.full_token_ids)
        token_ids = list(tokenization.token_ids)
        token_strings = (
            list(tokenizer.convert_ids_to_tokens(token_ids))
            if hasattr(tokenizer, "convert_ids_to_tokens")
            else [str(token_id) for token_id in token_ids]
        )
        flags: list[str] = []
        if len(token_ids) == 0:
            flags.append("empty_tokenization")
        if len(token_ids) > suspicious_length_threshold:
            flags.append("suspiciously_long_option_tokenization")
        if surface.startswith(" "):
            flags.append("option_starts_with_space")
        context_preceded_by_whitespace = tokenization.context_preceded_by_whitespace
        context_preceded_by_newline = tokenization.context_preceded_by_newline
        token_sequences.setdefault(tuple(token_ids), []).append(option_id)
        records.append(
            TokenAuditRecord(
                model_id=model_id,
                example_id=example_id,
                readout_id=readout_id,
                prompt_id=prompt_id,
                prompt_token_count=prompt_token_count,
                answer_start_token_index=answer_start_token_index,
                option_id=option_id,
                option_surface=surface,
                constructed_surface=constructed_surface,
                token_ids=token_ids,
                token_strings=token_strings,
                token_count=len(token_ids),
                context_preceded_by_whitespace=context_preceded_by_whitespace,
                context_preceded_by_newline=context_preceded_by_newline,
                suspicious_flags=flags,
            )
        )
    collisions = {seq: ids for seq, ids in token_sequences.items() if len(ids) > 1}
    if collisions:
        augmented = []
        for record in records:
            flags = list(record.suspicious_flags)
            if tuple(record.token_ids) in collisions:
                flags.append("tokenization_collision")
            augmented.append(TokenAuditRecord(**{**record.to_dict(), "suspicious_flags": sorted(set(flags))}))
        return augmented
    return records
