from __future__ import annotations

from tokenizer_audit import build_token_audit_records


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [chr(i) for i in ids]


def test_tokenizer_audit_detects_collisions() -> None:
    tokenizer = DummyTokenizer()
    records = build_token_audit_records(
        tokenizer=tokenizer,
        model_id="demo",
        example_id="ex1",
        readout_id="semantic_exact",
        prompt_id="prompt-1",
        prompt_text="Answer: ",
        answer_start_token_index=len("Answer: "),
        option_surfaces={"A": "x", "B": "x"},
    )
    collision_counts = sum("tokenization_collision" in record.suspicious_flags for record in records)
    assert collision_counts == 2


def test_tokenizer_audit_raises_on_contextual_prefix_mismatch() -> None:
    class _BrokenTokenizer(DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text == "Answer: ":
                return [1]
            if text == "Answer: x":
                return [2, 3]
            return super().encode(text, add_special_tokens=add_special_tokens)

    try:
        build_token_audit_records(
            tokenizer=_BrokenTokenizer(),
            model_id="demo",
            example_id="ex1",
            readout_id="semantic_exact",
            prompt_id="prompt-1",
            prompt_text="Answer: ",
            answer_start_token_index=1,
            option_surfaces={"A": "x"},
        )
        raise AssertionError("expected contextual tokenization mismatch")
    except ValueError:
        pass


def test_tokenizer_audit_uses_prompt_context_for_boundary_metadata() -> None:
    tokenizer = DummyTokenizer()
    records = build_token_audit_records(
        tokenizer=tokenizer,
        model_id="demo",
        example_id="ex1",
        readout_id="semantic_exact",
        prompt_id="prompt-2",
        prompt_text="Answer:\n",
        answer_start_token_index=len("Answer:\n"),
        option_surfaces={"A": "x"},
    )
    record = records[0]
    assert record.prompt_id == "prompt-2"
    assert record.prompt_token_count == len("Answer:\n")
    assert record.answer_start_token_index == len("Answer:\n")
    assert record.context_preceded_by_newline is True
    assert record.context_preceded_by_whitespace is True
    assert record.token_ids == [ord("x")]
