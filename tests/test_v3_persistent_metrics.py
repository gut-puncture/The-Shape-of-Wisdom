from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "v3_persistent_contender_margin_fix"
    / "src"
    / "persistent_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("persistent_metrics_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_compute_example_metrics_persistent_depths_and_writes() -> None:
    bundle = {
        "model_name": "toy_model",
        "model_family": "toy_family",
        "model_size": "toy",
        "dataset_name": "toy_dataset",
        "example_id": "toy:0",
        "final_winner": "A",
        "layers": [
            {
                "layer_index": 0,
                "options": {
                    "A": {"normalized_probability": 0.10},
                    "B": {"normalized_probability": 0.50},
                    "C": {"normalized_probability": 0.40},
                },
            },
            {
                "layer_index": 1,
                "options": {
                    "A": {"normalized_probability": 0.20},
                    "B": {"normalized_probability": 0.45},
                    "C": {"normalized_probability": 0.35},
                },
            },
            {
                "layer_index": 2,
                "options": {
                    "A": {"normalized_probability": 0.30},
                    "B": {"normalized_probability": 0.40},
                    "C": {"normalized_probability": 0.30},
                },
            },
            {
                "layer_index": 3,
                "options": {
                    "A": {"normalized_probability": 0.45},
                    "B": {"normalized_probability": 0.35},
                    "C": {"normalized_probability": 0.20},
                },
            },
            {
                "layer_index": 4,
                "options": {
                    "A": {"normalized_probability": 0.55},
                    "B": {"normalized_probability": 0.25},
                    "C": {"normalized_probability": 0.20},
                },
            },
        ],
    }

    metrics = MODULE.compute_example_metrics(
        bundle=bundle,
        cache_key="cache-key",
        cache_relpath="outputs/caches/semantic_scores/xx/cache-key.json",
        canonical_option_ids=["A", "B", "C"],
    )

    assert metrics.final_winner == "A"
    assert metrics.d2_pos == 2
    assert metrics.d1_pos == 3
    assert metrics.d50_pos == 4
    assert metrics.d25_pos == 3
    assert metrics.d75_pos == 4
    assert metrics.delta_21_pos == 1
    assert metrics.delta_150_pos == 1
    assert metrics.delta_250_pos == 2
    assert metrics.norm_d2 == 0.5
    assert metrics.norm_d1 == 0.75
    assert metrics.norm_d50 == 1.0
    assert metrics.norm_delta_250 == 0.5
    assert metrics.write_share_after_d2 is not None
    assert metrics.write_share_after_d1 is not None
    assert metrics.write_share_last20 is not None
    assert metrics.write_center_of_mass is not None
    assert not metrics.final_layer_top_tie
    assert not metrics.final_winner_mismatch


def test_compute_example_metrics_logs_final_layer_tie_and_nonpositive_gap() -> None:
    bundle = {
        "model_name": "toy_model",
        "model_family": "toy_family",
        "model_size": "toy",
        "dataset_name": "toy_dataset",
        "example_id": "toy:1",
        "final_winner": "B",
        "layers": [
            {
                "layer_index": 0,
                "options": {
                    "A": {"normalized_probability": 0.40},
                    "B": {"normalized_probability": 0.40},
                    "C": {"normalized_probability": 0.20},
                },
            },
            {
                "layer_index": 1,
                "options": {
                    "A": {"normalized_probability": 0.45},
                    "B": {"normalized_probability": 0.45},
                    "C": {"normalized_probability": 0.10},
                },
            },
        ],
    }

    metrics = MODULE.compute_example_metrics(
        bundle=bundle,
        cache_key="cache-key-2",
        cache_relpath="outputs/caches/semantic_scores/yy/cache-key-2.json",
        canonical_option_ids=["A", "B", "C"],
    )

    assert metrics.final_winner == "A"
    assert metrics.final_layer_top_tie
    assert metrics.final_winner_mismatch
    assert metrics.g_final == 0.0
    assert not metrics.normalized_gap_valid
    assert metrics.d2_pos == 0
    assert metrics.d1_pos == 0
    assert metrics.d50_pos is None
