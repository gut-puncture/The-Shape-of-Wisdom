from __future__ import annotations

import pytest

from continuation_tokenizer import contextual_token_ids, tokenize_continuation_in_context


class _ContextSensitiveTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text == "Answer: ":
            return [10]
        if text == "Answer: A":
            return [10, 20]
        if text == "A":
            return [99]
        if text == "Answer: alpha":
            return [10, 30, 31]
        if text == "alpha":
            return [98, 97]
        raise AssertionError(f"unexpected text={text!r}")

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(item) for item in ids]


def test_contextual_tokenization_uses_prompt_plus_surface() -> None:
    tokenizer = _ContextSensitiveTokenizer()
    tokenization = tokenize_continuation_in_context(tokenizer, prompt_text="Answer: ", surface="A")
    assert tokenization.prompt_token_ids == [10]
    assert tokenization.token_ids == [20]
    assert contextual_token_ids(tokenizer, prompt_text="Answer: ", surface="alpha") == [30, 31]


def test_contextual_tokenization_raises_on_prefix_mismatch() -> None:
    class _BrokenTokenizer(_ContextSensitiveTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text == "Answer: ":
                return [10]
            if text == "Answer: A":
                return [11, 20]
            return super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError):
        tokenize_continuation_in_context(_BrokenTokenizer(), prompt_text="Answer: ", surface="A")
