from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from runtime_panels import get_panel_config, load_runtime_config, resolve_named_specs, resolve_panel_name


@dataclass(frozen=True)
class _Spec:
    name: str
    enabled: bool = True
    family_ready: bool = False


def test_resolve_panel_name_prefers_smoke_when_requested() -> None:
    runtime_cfg = {"default_panel": "paper", "smoke": {}, "paper": {}}
    assert resolve_panel_name(runtime_cfg, smoke=True, explicit_panel=None) == "smoke"


def test_resolve_panel_name_rejects_conflicting_override() -> None:
    runtime_cfg = {"default_panel": "paper", "smoke": {}, "paper": {}}
    with pytest.raises(ValueError):
        resolve_panel_name(runtime_cfg, smoke=True, explicit_panel="paper")


def test_resolve_named_specs_raises_on_unready_models() -> None:
    specs = [_Spec(name="ready_model", family_ready=True), _Spec(name="blocked_model", family_ready=False)]
    with pytest.raises(ValueError):
        resolve_named_specs(
            specs,
            ["ready_model", "blocked_model"],
            require_enabled=True,
            require_family_ready=True,
            label="model:paper",
        )


def test_get_panel_config_requires_known_panel() -> None:
    with pytest.raises(ValueError):
        get_panel_config({"smoke": {}}, "paper")


def test_paper_pilot_panel_matches_paper_panel_with_reduced_examples() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    runtime_cfg = load_runtime_config(workspace_root / "configs" / "runtime.yaml")
    paper_cfg = get_panel_config(runtime_cfg, "paper")
    pilot_cfg = get_panel_config(runtime_cfg, "paper_pilot")

    assert pilot_cfg["models"] == paper_cfg["models"]
    assert pilot_cfg["datasets"] == paper_cfg["datasets"]
    assert pilot_cfg["readouts"] == paper_cfg["readouts"]
    assert pilot_cfg["permutations"] == paper_cfg["permutations"]
    assert pilot_cfg["max_examples_per_dataset"] == 5
    assert pilot_cfg["selection_strategy"] == "representative_pilot_v1"


def test_paper_final_panel_has_expected_budget_shape() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    runtime_cfg = load_runtime_config(workspace_root / "configs" / "runtime.yaml")
    final_cfg = get_panel_config(runtime_cfg, "paper_final_6x1000")

    assert final_cfg["models"] == [
        "qwen2p5_1p5b_instruct",
        "qwen2p5_7b_instruct",
        "gemma2_2b_it",
        "gemma2_27b_it",
        "llama3p2_1b_instruct",
        "llama3p1_8b_instruct",
    ]
    assert final_cfg["readouts"] == ["semantic_exact", "templated_semantic", "letter_label"]
    assert final_cfg["permutations"] == ["identity", "reverse"]
    assert final_cfg["selection_strategy"] == "representative_panel_v1"
    assert final_cfg["dataset_example_limits"] == {
        "mmlu_abstract_algebra": 11,
        "ai2_arc_challenge": 195,
        "commonsenseqa_validation": 794,
    }
