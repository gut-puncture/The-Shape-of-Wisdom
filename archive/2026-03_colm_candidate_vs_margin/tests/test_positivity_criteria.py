from __future__ import annotations

from positivity_criteria import evaluate_model_dataset_positivity, evaluate_rule, summarize_family_counts, summarize_model_counts


def test_primary_rule_positive_when_median_ci_stays_above_zero() -> None:
    rows = [{"group_id": f"g{i}", "delta_margin": float(i + 1)} for i in range(10)]
    result = evaluate_rule(
        rows,
        rule_cfg={
            "statistic": "median_delta_margin",
            "bootstrap_iterations": 200,
            "min_valid_examples": 10,
            "positive_if": {"ci_lo_gt": 0.0},
        },
    )
    assert result["status"] == "evaluated"
    assert result["positive"] is True
    assert result["observed_estimate"] > 0.0
    assert result["bootstrap"]["ci_lo"] > 0.0


def test_secondary_rule_insufficient_n_stays_noncommittal() -> None:
    rows = [{"group_id": f"g{i}", "delta_margin": 1.0} for i in range(4)]
    result = evaluate_rule(
        rows,
        rule_cfg={
            "statistic": "proportion_delta_margin_gt_zero",
            "bootstrap_iterations": 50,
            "min_valid_examples": 10,
            "threshold_gt": 0.6,
            "ci_lo_gt": 0.5,
        },
    )
    assert result["status"] == "insufficient_n"
    assert "positive" not in result


def test_model_dataset_positivity_summaries_count_statuses() -> None:
    rows = []
    for idx in range(10):
        rows.append(
            {
                "example_id": f"e{idx}",
                "dataset_name": "dataset_a",
                "model_name": "model_x",
                "model_family": "family_x",
                "model_size": "1b",
                "readout_id": "semantic_exact",
                "permutation_id": "identity",
                "metrics": {
                    "final_winner_anchor": {"delta_margin": 1.0},
                    "gold_answer_anchor": {"delta_margin": 0.5},
                },
            }
        )
    criteria_cfg = {
        "primary_rule": {
            "anchor": "final_winner_anchor",
            "readout_id": "semantic_exact",
            "permutation_id": "identity",
            "statistic": "median_delta_margin",
            "bootstrap_iterations": 100,
            "min_valid_examples": 10,
            "positive_if": {"ci_lo_gt": 0.0},
        },
        "secondary_robustness": {
            "anchor": "final_winner_anchor",
            "readout_id": "semantic_exact",
            "permutation_id": "identity",
            "statistic": "proportion_delta_margin_gt_zero",
            "bootstrap_iterations": 100,
            "min_valid_examples": 10,
            "threshold_gt": 0.6,
            "ci_lo_gt": 0.5,
            "report_only": True,
        },
        "gold_answer_sensitivity": {
            "enabled": True,
            "anchor": "gold_answer_anchor",
            "readout_id": "semantic_exact",
            "permutation_id": "identity",
            "statistic": "median_delta_margin",
            "bootstrap_iterations": 100,
            "min_valid_examples": 10,
            "positive_if": {"ci_lo_gt": 0.0},
            "report_only": True,
        },
    }
    evaluations = evaluate_model_dataset_positivity(rows, criteria_cfg=criteria_cfg)
    assert len(evaluations) == 3
    model_summary = summarize_model_counts(evaluations)
    family_summary = summarize_family_counts(evaluations)
    assert model_summary[0]["datasets_positive"] >= 1
    assert family_summary[0]["rows_positive"] >= 1
