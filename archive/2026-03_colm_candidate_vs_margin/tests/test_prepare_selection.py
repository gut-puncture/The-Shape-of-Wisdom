from __future__ import annotations

from dataset_loader import CanonicalExample, CanonicalOption
from prepare_selection import build_selection_manifest, select_examples_for_dataset


def _example(*, dataset: str, source_row_index: int, option_count: int, question_length: int) -> CanonicalExample:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return CanonicalExample(
        example_id=f"{dataset}:validation:{source_row_index}",
        dataset_name=dataset,
        dataset_id=f"{dataset}_id",
        dataset_revision="rev",
        split="validation",
        source_row_index=source_row_index,
        question_raw=("q" * question_length),
        question_canonical=("q" * question_length),
        options=[
            CanonicalOption(
                option_id=labels[index],
                ordinal=index,
                raw_text=f"option-{index}",
                canonical_text=f"option-{index}",
            )
            for index in range(option_count)
        ],
        answer_key_raw="A",
        answer_option_id="A",
        metadata={},
        anomaly_flags=[],
        exclusion_reason=None,
        raw_record={},
    )


def test_representative_pilot_v1_covers_non_majority_option_counts_first() -> None:
    examples = [
        _example(dataset="arc", source_row_index=0, option_count=3, question_length=20),
        _example(dataset="arc", source_row_index=1, option_count=3, question_length=40),
        _example(dataset="arc", source_row_index=2, option_count=3, question_length=60),
        _example(dataset="arc", source_row_index=3, option_count=5, question_length=55),
    ]
    examples.extend(
        _example(dataset="arc", source_row_index=10 + index, option_count=4, question_length=10 + (index * 10))
        for index in range(8)
    )

    selected, records = select_examples_for_dataset(
        examples,
        max_examples=5,
        strategy="representative_pilot_v1",
    )

    assert len(selected) == 5
    assert [len(example.options) for example in selected].count(3) == 1
    assert [len(example.options) for example in selected].count(5) == 1
    assert [len(example.options) for example in selected].count(4) == 3
    assert [record.selection_reason for record in records if record.option_count == 3] == [
        "non_majority_option_count_median:3"
    ]
    assert [record.selection_reason for record in records if record.option_count == 5] == [
        "non_majority_option_count_median:5"
    ]


def test_representative_pilot_v1_uses_evenly_spaced_majority_quantiles() -> None:
    examples = [_example(dataset="mmlu", source_row_index=index, option_count=4, question_length=10 + index) for index in range(11)]
    selected, records = select_examples_for_dataset(
        examples,
        max_examples=5,
        strategy="representative_pilot_v1",
    )

    assert [example.source_row_index for example in selected] == [1, 3, 5, 7, 9]
    assert [record.selection_reason for record in records] == [
        "majority_option_count_quantile:4:1_of_5",
        "majority_option_count_quantile:4:2_of_5",
        "majority_option_count_quantile:4:3_of_5",
        "majority_option_count_quantile:4:4_of_5",
        "majority_option_count_quantile:4:5_of_5",
    ]


def test_build_selection_manifest_hash_is_deterministic() -> None:
    selected, records = select_examples_for_dataset(
        [_example(dataset="csqa", source_row_index=index, option_count=5, question_length=20 + index) for index in range(6)],
        max_examples=5,
        strategy="representative_pilot_v1",
    )
    assert len(selected) == 5

    manifest_a = build_selection_manifest(
        run_id="paper_pilot_a",
        panel_name="paper_pilot",
        selection_strategy="representative_pilot_v1",
        records=records,
    )
    manifest_b = build_selection_manifest(
        run_id="paper_pilot_b",
        panel_name="paper_pilot",
        selection_strategy="representative_pilot_v1",
        records=records,
    )

    assert manifest_a["selection_manifest_sha256"] == manifest_b["selection_manifest_sha256"]
    assert len(manifest_a["selections"]) == 5


def test_representative_panel_v1_alias_matches_pilot_strategy() -> None:
    examples = [_example(dataset="csqa", source_row_index=index, option_count=5, question_length=20 + index) for index in range(9)]
    selected_panel, records_panel = select_examples_for_dataset(
        examples,
        max_examples=5,
        strategy="representative_panel_v1",
    )
    selected_pilot, records_pilot = select_examples_for_dataset(
        examples,
        max_examples=5,
        strategy="representative_pilot_v1",
    )

    assert [example.example_id for example in selected_panel] == [example.example_id for example in selected_pilot]
    assert [record.selection_reason for record in records_panel] == [record.selection_reason for record in records_pilot]
