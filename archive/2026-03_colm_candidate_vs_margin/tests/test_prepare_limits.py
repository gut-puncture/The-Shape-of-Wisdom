from __future__ import annotations

from dataclasses import dataclass

from run_prepare import _max_examples_for_dataset


@dataclass(frozen=True)
class _DatasetSpecStub:
    name: str
    smoke_examples: int


def test_max_examples_for_dataset_uses_panel_default() -> None:
    panel_cfg = {"max_examples_per_dataset": 5}
    assert _max_examples_for_dataset(panel_cfg, dataset_name="csqa", smoke_examples=2, smoke=False) == 5


def test_max_examples_for_dataset_uses_dataset_specific_override() -> None:
    panel_cfg = {
        "max_examples_per_dataset": 5,
        "dataset_example_limits": {
            "mmlu_abstract_algebra": 11,
            "ai2_arc_challenge": 195,
        },
    }
    assert _max_examples_for_dataset(panel_cfg, dataset_name="mmlu_abstract_algebra", smoke_examples=2, smoke=False) == 11
    assert _max_examples_for_dataset(panel_cfg, dataset_name="ai2_arc_challenge", smoke_examples=2, smoke=False) == 195
    assert _max_examples_for_dataset(panel_cfg, dataset_name="commonsenseqa_validation", smoke_examples=2, smoke=False) == 5


def test_max_examples_for_dataset_respects_smoke_cap() -> None:
    panel_cfg = {
        "dataset_example_limits": {
            "ai2_arc_challenge": 195,
        },
    }
    assert _max_examples_for_dataset(panel_cfg, dataset_name="ai2_arc_challenge", smoke_examples=3, smoke=True) == 3
