from __future__ import annotations

from metrics_candidate_vs_margin import candidate_entry_depth, compute_primary_metrics, margin_isolation_depth, stable_top1_depth


def _layers() -> list[dict]:
    return [
        {"layer_index": 0, "options": {"A": {"normalized_probability": 0.40}, "B": {"normalized_probability": 0.35}, "C": {"normalized_probability": 0.25}}},
        {"layer_index": 1, "options": {"A": {"normalized_probability": 0.45}, "B": {"normalized_probability": 0.30}, "C": {"normalized_probability": 0.25}}},
        {"layer_index": 2, "options": {"A": {"normalized_probability": 0.60}, "B": {"normalized_probability": 0.25}, "C": {"normalized_probability": 0.15}}},
    ]


def test_depth_metrics_on_toy_example() -> None:
    layers = _layers()
    assert candidate_entry_depth(layers, anchor_option="A", tau=0.8, tie_eps=1e-12) == 0
    assert stable_top1_depth(layers, anchor_option="A", tie_eps=1e-12) == 0
    assert margin_isolation_depth(layers, anchor_option="A", delta=0.2, tie_eps=1e-12) == 2


def test_compute_primary_metrics_includes_both_anchors() -> None:
    bundle = {"layers": _layers(), "final_winner": "A"}
    metrics = compute_primary_metrics(bundle, gold_answer="B", tau=0.8, delta=0.2, tie_eps=1e-12)
    assert "final_winner_anchor" in metrics
    assert "gold_answer_anchor" in metrics
