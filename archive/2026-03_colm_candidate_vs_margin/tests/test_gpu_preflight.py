from __future__ import annotations

import pytest

from gpu_preflight import (
    OOMErrorSignal,
    calibration_forecast,
    determine_safe_batch_size,
    verify_prompt_template_compatibility,
)


def test_determine_safe_batch_size_retries_downward() -> None:
    attempts: list[int] = []

    def _probe(size: int) -> None:
        attempts.append(size)
        if size > 4:
            raise OOMErrorSignal("oom")

    safe = determine_safe_batch_size(candidate_batch_sizes=[16, 8, 4, 2, 1], probe_batch_size=_probe)
    assert safe == 4
    assert attempts[:3] == [16, 8, 4]


def test_determine_safe_batch_size_raises_when_everything_ooms() -> None:
    def _probe(_: int) -> None:
        raise OOMErrorSignal("oom")

    with pytest.raises(RuntimeError):
        determine_safe_batch_size(candidate_batch_sizes=[2, 1], probe_batch_size=_probe)


def test_calibration_forecast_reports_runtime_and_cost() -> None:
    report = calibration_forecast(
        total_prefix_states=1000,
        measured_prefix_states=100,
        elapsed_seconds=5.0,
        hourly_rate_usd=3.0,
    )
    assert report["estimated_runtime_seconds"] == 50.0
    assert report["estimated_cost_usd"] == pytest.approx((50.0 / 3600.0) * 3.0)


def test_verify_prompt_template_compatibility_surfaces_failures() -> None:
    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            _ = tokenize, add_generation_prompt
            if any(message["role"] == "system" for message in messages):
                raise RuntimeError("system role not supported")
            return "ok"

    result = verify_prompt_template_compatibility(
        tokenizer=_Tokenizer(),
        prompt_mode="chat",
        question_text="Which one?",
        options=["A. one", "B. two"],
        answer_prefix="Final answer: ",
        system_prompt="Choose one.",
        system_prompt_handling="system_role",
    )
    assert result["passed"] is False
    assert "system role not supported" in result["error"]


def test_verify_prompt_template_compatibility_supports_user_prefixed_system_prompt() -> None:
    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            _ = tokenize, add_generation_prompt
            assert all(message["role"] != "system" for message in messages)
            return messages[0]["content"]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            return [ord(ch) for ch in text]

    result = verify_prompt_template_compatibility(
        tokenizer=_Tokenizer(),
        prompt_mode="chat",
        question_text="Which one?",
        options=["A. one", "B. two"],
        answer_prefix="Final answer: ",
        system_prompt="Choose one.",
        system_prompt_handling="user_prefix",
    )
    assert result["passed"] is True
    assert result["rendered_prompt"].startswith("System instruction: Choose one.")
