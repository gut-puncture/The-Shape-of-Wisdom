from __future__ import annotations

from dataset_loader import canonicalize_arc_row, canonicalize_commonsenseqa_row, canonicalize_mmlu_row
from dataset_registry import DatasetSpec


def _spec(name: str, dataset_id: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        dataset_id=dataset_id,
        config_name=None,
        revision="rev",
        split="validation",
        enabled=True,
        smoke_examples=1,
        note="",
    )


def test_mmlu_accepts_integer_answer_key() -> None:
    row = {
        "question": "Q",
        "subject": "abstract_algebra",
        "choices": ["a", "b", "c", "d"],
        "answer": 2,
    }
    example = canonicalize_mmlu_row(row, spec=_spec("mmlu", "cais/mmlu"), source_row_index=0)
    assert example.answer_option_id == "C"
    assert example.exclusion_reason is None


def test_arc_variable_option_count_is_flagged_but_not_silently_dropped() -> None:
    row = {
        "id": "arc1",
        "question": "Q",
        "choices": {"text": ["x", "y", "z"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    example = canonicalize_arc_row(row, spec=_spec("arc", "allenai/ai2_arc"), source_row_index=0)
    assert "variable_option_count:3" in example.anomaly_flags
    assert example.answer_option_id == "B"


def test_commonsenseqa_preserves_five_options() -> None:
    row = {
        "id": "csqa1",
        "question": "Q",
        "question_concept": "door",
        "choices": {"text": ["a", "b", "c", "d", "e"], "label": ["A", "B", "C", "D", "E"]},
        "answerKey": "E",
    }
    example = canonicalize_commonsenseqa_row(row, spec=_spec("csqa", "tau/commonsense_qa"), source_row_index=0)
    assert len(example.options) == 5
    assert "variable_option_count:5" in example.anomaly_flags
    assert example.answer_option_id == "E"

