from __future__ import annotations

from dataclasses import dataclass

from family_readiness import summarize_family_readiness, timed_out_readiness_result


@dataclass(frozen=True)
class _Spec:
    name: str
    family: str
    size: str
    model_id: str
    model_revision: str
    tokenizer_revision: str


def test_timed_out_readiness_result_marks_failure() -> None:
    spec = _Spec(
        name="model_x",
        family="family_x",
        size="1b",
        model_id="org/model_x",
        model_revision="rev1",
        tokenizer_revision="rev1",
    )
    result = timed_out_readiness_result(spec, timeout_seconds=60)
    assert result["overall_passed"] is False
    assert result["timed_out"] is True
    assert result["checks"]["model_load_and_equivalence"]["stage"] == "timeout"


def test_family_readiness_summary_requires_two_sizes_for_gate() -> None:
    rows = [
        {"model_family": "family_x", "model_name": "m1", "model_size": "1b", "overall_passed": True},
        {"model_family": "family_x", "model_name": "m2", "model_size": "2b", "overall_passed": True},
    ]
    summary = summarize_family_readiness(rows)
    assert summary[0]["at_least_two_sizes_passed"] is True
