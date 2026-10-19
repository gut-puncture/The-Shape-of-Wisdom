from __future__ import annotations

from dataset_loader import CanonicalExample, CanonicalOption
from prompt_builder import apply_permutation, build_prompt


def _example() -> CanonicalExample:
    return CanonicalExample(
        example_id="ex1",
        dataset_name="demo",
        dataset_id="demo/id",
        dataset_revision="rev",
        split="validation",
        source_row_index=0,
        question_raw="Which one?",
        question_canonical="Which one?",
        options=[
            CanonicalOption(option_id="A", ordinal=0, raw_text="one", canonical_text="one"),
            CanonicalOption(option_id="B", ordinal=1, raw_text="two", canonical_text="two"),
            CanonicalOption(option_id="C", ordinal=2, raw_text="three", canonical_text="three"),
        ],
        answer_key_raw="B",
        answer_option_id="B",
        metadata={},
        anomaly_flags=[],
        exclusion_reason=None,
        raw_record={},
    )


def test_permutation_remaps_correct_answer() -> None:
    options, answer = apply_permutation(_example(), [2, 0, 1])
    assert [opt.raw_text for opt in options] == ["three", "one", "two"]
    assert answer == "C"


def test_prompt_build_is_deterministic() -> None:
    example = _example()
    prompt_a = build_prompt(
        example,
        template_id="plain_exam_v1",
        readout_id="semantic_exact",
        prompt_mode="raw",
        answer_prefix="Final answer: ",
    )
    prompt_b = build_prompt(
        example,
        template_id="plain_exam_v1",
        readout_id="semantic_exact",
        prompt_mode="raw",
        answer_prefix="Final answer: ",
    )
    assert prompt_a.rendered_prompt == prompt_b.rendered_prompt
    assert prompt_a.prompt_id == prompt_b.prompt_id


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]


class _ChatTokenizer(_DummyTokenizer):
    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        _ = tokenize, add_generation_prompt
        return "\n".join(f"{message['role']}::{message['content']}" for message in messages)


def test_prompt_records_answer_start_token_index() -> None:
    prompt = build_prompt(
        _example(),
        template_id="plain_exam_v1",
        readout_id="semantic_exact",
        prompt_mode="raw",
        answer_prefix="Final answer: ",
        tokenizer=_DummyTokenizer(),
    )
    assert prompt.prompt_token_count == len(prompt.rendered_prompt)
    assert prompt.answer_start_token_index == prompt.prompt_token_count


def test_chat_prompt_can_inline_system_prompt_into_user_message() -> None:
    prompt = build_prompt(
        _example(),
        template_id="chat_exam_v1",
        readout_id="semantic_exact",
        prompt_mode="chat",
        answer_prefix="Final answer: ",
        tokenizer=_ChatTokenizer(),
        system_prompt="Choose carefully.",
        system_prompt_handling="user_prefix",
    )
    assert "system::" not in prompt.rendered_prompt
    assert "user::System instruction: Choose carefully." in prompt.rendered_prompt
    assert prompt.metadata["system_prompt_handling"] == "user_prefix"
