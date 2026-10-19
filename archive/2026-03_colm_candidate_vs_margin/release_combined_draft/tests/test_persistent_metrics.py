from __future__ import annotations

from shape_of_wisdom_release.persistent_metrics import compute_example_metrics


def test_persistent_depths_and_late_write_shares() -> None:
    bundle = {
        "model_name": "toy_model",
        "model_family": "toy_family",
        "model_size": "toy",
        "dataset_name": "toy_dataset",
        "example_id": "toy:0",
        "final_winner": "A",
        "layers": [
            {"layer_index": 0, "options": {"A": {"normalized_probability": 0.1}, "B": {"normalized_probability": 0.5}, "C": {"normalized_probability": 0.4}}},
            {"layer_index": 1, "options": {"A": {"normalized_probability": 0.2}, "B": {"normalized_probability": 0.45}, "C": {"normalized_probability": 0.35}}},
            {"layer_index": 2, "options": {"A": {"normalized_probability": 0.3}, "B": {"normalized_probability": 0.4}, "C": {"normalized_probability": 0.3}}},
            {"layer_index": 3, "options": {"A": {"normalized_probability": 0.45}, "B": {"normalized_probability": 0.35}, "C": {"normalized_probability": 0.2}}},
            {"layer_index": 4, "options": {"A": {"normalized_probability": 0.55}, "B": {"normalized_probability": 0.25}, "C": {"normalized_probability": 0.2}}},
        ],
    }
    metrics = compute_example_metrics(
        bundle=bundle,
        cache_key="cache-key",
        cache_relpath="outputs/caches/semantic_scores/xx/cache-key.json",
        canonical_option_ids=["A", "B", "C"],
    )
    assert metrics.d2_pos == 2
    assert metrics.d1_pos == 3
    assert metrics.d50_pos == 4
    assert metrics.delta_21_pos == 1
    assert metrics.delta_250_pos == 2
    assert metrics.norm_delta_250 == 0.5
    assert metrics.write_share_after_d2 is not None
    assert metrics.write_share_after_d1 is not None
    assert metrics.write_share_last20 is not None


def test_final_tie_is_logged_and_gap_metrics_are_excluded() -> None:
    bundle = {
        "model_name": "toy_model",
        "model_family": "toy_family",
        "model_size": "toy",
        "dataset_name": "toy_dataset",
        "example_id": "toy:1",
        "final_winner": "B",
        "layers": [
            {"layer_index": 0, "options": {"A": {"normalized_probability": 0.4}, "B": {"normalized_probability": 0.4}, "C": {"normalized_probability": 0.2}}},
            {"layer_index": 1, "options": {"A": {"normalized_probability": 0.45}, "B": {"normalized_probability": 0.45}, "C": {"normalized_probability": 0.1}}},
        ],
    }
    metrics = compute_example_metrics(
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
    assert metrics.d50_pos is None
