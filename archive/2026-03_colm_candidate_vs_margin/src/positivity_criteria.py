from __future__ import annotations

from statistics import median
from typing import Any, Callable

from stats_bootstrap import grouped_bootstrap


def _median(values: list[float]) -> float:
    return float(median(values))


def _proportion_gt_zero(values: list[float]) -> float:
    positives = sum(1 for value in values if value > 0.0)
    return positives / len(values)


def _statistic_fn(statistic: str) -> Callable[[list[float]], float]:
    if statistic == "median_delta_margin":
        return _median
    if statistic == "proportion_delta_margin_gt_zero":
        return _proportion_gt_zero
    raise ValueError(f"unsupported positivity statistic={statistic}")


def evaluate_rule(rows: list[dict[str, Any]], *, rule_cfg: dict[str, Any], group_field: str = "group_id") -> dict[str, Any]:
    valid = [row for row in rows if row.get("delta_margin") is not None]
    statistic_name = str(rule_cfg["statistic"])
    iterations = int(rule_cfg.get("bootstrap_iterations", 1000))
    min_valid_examples = int(rule_cfg.get("min_valid_examples", 1))
    calc = _statistic_fn(statistic_name)
    result: dict[str, Any] = {
        "statistic": statistic_name,
        "n_total": len(rows),
        "n_valid": len(valid),
        "min_valid_examples": min_valid_examples,
        "status": "insufficient_n" if len(valid) < min_valid_examples else "evaluated",
    }
    if len(valid) < min_valid_examples:
        return result

    values = [float(row["delta_margin"]) for row in valid]
    observed = calc(values)
    bootstrap = grouped_bootstrap(
        valid,
        group_field=group_field,
        iterations=iterations,
        value_fn=lambda sampled: calc([float(item["delta_margin"]) for item in sampled]),
    )
    result["observed_estimate"] = observed
    result["bootstrap"] = bootstrap
    if statistic_name == "median_delta_margin":
        threshold = float(rule_cfg.get("positive_if", {}).get("ci_lo_gt", 0.0))
        result["decision_threshold"] = {"ci_lo_gt": threshold}
        result["positive"] = bool(observed > 0.0 and float(bootstrap["ci_lo"]) > threshold)
    elif statistic_name == "proportion_delta_margin_gt_zero":
        threshold = float(rule_cfg["threshold_gt"])
        ci_floor = float(rule_cfg["ci_lo_gt"])
        result["decision_threshold"] = {"observed_gt": threshold, "ci_lo_gt": ci_floor}
        result["positive"] = bool(observed > threshold and float(bootstrap["ci_lo"]) > ci_floor)
    else:
        raise ValueError(f"unsupported positivity statistic={statistic_name}")
    return result


def evaluate_model_dataset_positivity(rows: list[dict[str, Any]], *, criteria_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    primary_cfg = dict(criteria_cfg["primary_rule"])
    secondary_cfg = dict(criteria_cfg["secondary_robustness"])
    gold_cfg = dict(criteria_cfg.get("gold_answer_sensitivity", {}))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("model_name")), str(row.get("dataset_name")), str(row.get("model_family")))
        groups.setdefault(key, []).append(row)
    evaluations: list[dict[str, Any]] = []
    for (model_name, dataset_name, model_family), group_rows in sorted(groups.items()):
        base_row = group_rows[0]
        records = [
            ("primary_rule", primary_cfg),
            ("secondary_robustness", secondary_cfg),
        ]
        if gold_cfg.get("enabled", False):
            records.append(("gold_answer_sensitivity", gold_cfg))
        for label, rule_cfg in records:
            anchor = str(rule_cfg["anchor"])
            readout_id = str(rule_cfg["readout_id"])
            permutation_id = str(rule_cfg["permutation_id"])
            matching = []
            for row in group_rows:
                if row.get("readout_id") != readout_id or row.get("permutation_id") != permutation_id:
                    continue
                anchor_metrics = row.get("metrics", {}).get(anchor)
                if not isinstance(anchor_metrics, dict):
                    continue
                matching.append(
                    {
                        "group_id": f"{row.get('dataset_name')}::{row.get('example_id')}",
                        "example_id": row.get("example_id"),
                        "delta_margin": anchor_metrics.get("delta_margin"),
                    }
                )
            evaluation = {
                "criterion_label": label,
                "model_name": model_name,
                "model_family": model_family,
                "model_size": base_row.get("model_size"),
                "dataset_name": dataset_name,
                "anchor": anchor,
                "readout_id": readout_id,
                "permutation_id": permutation_id,
                "report_only": bool(rule_cfg.get("report_only", False)),
            }
            evaluation.update(evaluate_rule(matching, rule_cfg=rule_cfg))
            evaluations.append(evaluation)
    return evaluations


def summarize_model_counts(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in evaluations:
        key = (
            str(row.get("criterion_label")),
            str(row.get("anchor")),
            str(row.get("model_name")),
            str(row.get("model_family")),
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (criterion_label, anchor, model_name, model_family), rows in sorted(grouped.items()):
        summary.append(
            {
                "criterion_label": criterion_label,
                "anchor": anchor,
                "model_name": model_name,
                "model_family": model_family,
                "model_size": rows[0].get("model_size"),
                "datasets_evaluated": len(rows),
                "datasets_positive": sum(1 for row in rows if row.get("positive") is True),
                "datasets_negative": sum(1 for row in rows if row.get("positive") is False),
                "datasets_insufficient_n": sum(1 for row in rows if row.get("status") == "insufficient_n"),
                "positive_dataset_names": sorted(row["dataset_name"] for row in rows if row.get("positive") is True),
                "negative_dataset_names": sorted(row["dataset_name"] for row in rows if row.get("positive") is False),
                "insufficient_dataset_names": sorted(row["dataset_name"] for row in rows if row.get("status") == "insufficient_n"),
            }
        )
    return summary


def summarize_family_counts(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in evaluations:
        key = (
            str(row.get("criterion_label")),
            str(row.get("anchor")),
            str(row.get("model_family")),
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (criterion_label, anchor, model_family), rows in sorted(grouped.items()):
        model_names = sorted({str(row["model_name"]) for row in rows})
        summary.append(
            {
                "criterion_label": criterion_label,
                "anchor": anchor,
                "model_family": model_family,
                "models_evaluated": model_names,
                "model_count": len(model_names),
                "model_dataset_rows": len(rows),
                "rows_positive": sum(1 for row in rows if row.get("positive") is True),
                "rows_negative": sum(1 for row in rows if row.get("positive") is False),
                "rows_insufficient_n": sum(1 for row in rows if row.get("status") == "insufficient_n"),
            }
        )
    return summary
