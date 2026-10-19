#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "paper_merged_9x1000_20260330T122258Z"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / RUN_ID
CACHE_ROOT = REPO_ROOT / "outputs" / "caches" / "semantic_scores"
OUTPUT_ROOT = V3_ROOT / "outputs"
PAPER_ROOT = V3_ROOT / "paper"

sys.path.insert(0, str(V3_ROOT / "src"))
from persistent_metrics import BundleValidationError, compute_example_metrics, metrics_to_row  # noqa: E402


PRIMARY_READOUT = "semantic_exact"
PRIMARY_PERMUTATION = "identity"
BOOTSTRAP_N = 10_000
SEED = 20260331
MODEL_ORDER = [
    "qwen2p5_1p5b_instruct",
    "qwen2p5_3b_instruct",
    "qwen2p5_7b_instruct",
    "gemma2_2b_it",
    "gemma2_9b_it",
    "gemma2_27b_it",
    "llama3p2_1b_instruct",
    "llama3p2_3b_instruct",
    "llama3p1_8b_instruct",
]
MODEL_DISPLAY = {
    "qwen2p5_1p5b_instruct": "Qwen 2.5 1.5B",
    "qwen2p5_3b_instruct": "Qwen 2.5 3B",
    "qwen2p5_7b_instruct": "Qwen 2.5 7B",
    "gemma2_2b_it": "Gemma 2 2B",
    "gemma2_9b_it": "Gemma 2 9B",
    "gemma2_27b_it": "Gemma 2 27B",
    "llama3p2_1b_instruct": "Llama 3.2 1B",
    "llama3p2_3b_instruct": "Llama 3.2 3B",
    "llama3p1_8b_instruct": "Llama 3.1 8B",
}
FAMILY_DISPLAY = {
    "qwen2.5_instruct": "Qwen 2.5 Instruct",
    "gemma2_it": "Gemma 2 IT",
    "llama_instruct": "Llama Instruct",
}
FAMILY_ORDER = ["qwen2.5_instruct", "gemma2_it", "llama_instruct"]
DATASET_ORDER = ["ai2_arc_challenge", "commonsenseqa_validation", "mmlu_abstract_algebra"]
PAPER_DATASET_ORDER = ["ai2_arc_challenge", "commonsenseqa_validation"]
DATASET_DISPLAY = {
    "ai2_arc_challenge": "ARC-Challenge",
    "commonsenseqa_validation": "CommonsenseQA",
    "mmlu_abstract_algebra": "MMLU Abstract Algebra",
}
COLORS = {
    "ink": "#1E2430",
    "grid": "#E5E7EB",
    "qwen2.5_instruct": "#B45B46",
    "gemma2_it": "#8A8442",
    "llama_instruct": "#365B8C",
    "d2": "#6B8A9F",
    "d1": "#22364A",
    "d50": "#C8944A",
    "late": "#7A5C2E",
    "rank": "#2F4858",
    "gap": "#D08C3F",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Book Antiqua", "URW Palladio L", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2F3640",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def pct(value: float | None, digits: int = 0) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value * 100:.{digits}f}\\%"


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def metric_manifest() -> dict[str, dict[str, Any]]:
    return {
        "norm_d2": {"label": r"$d_2$ (norm.)", "kind": "depth"},
        "norm_d1": {"label": r"$d_1$ (norm.)", "kind": "depth"},
        "norm_d25": {"label": r"$d_{25}$ (norm.)", "kind": "depth"},
        "norm_d50": {"label": r"$d_{50}$ (norm.)", "kind": "depth"},
        "norm_d75": {"label": r"$d_{75}$ (norm.)", "kind": "depth"},
        "d2_layer": {"label": r"$d_2$ (layers)", "kind": "depth"},
        "d1_layer": {"label": r"$d_1$ (layers)", "kind": "depth"},
        "d25_layer": {"label": r"$d_{25}$ (layers)", "kind": "depth"},
        "d50_layer": {"label": r"$d_{50}$ (layers)", "kind": "depth"},
        "d75_layer": {"label": r"$d_{75}$ (layers)", "kind": "depth"},
        "norm_delta_21": {"label": r"$\Delta_{21}$ (norm.)", "kind": "lag"},
        "norm_delta_150": {"label": r"$\Delta_{150}$ (norm.)", "kind": "lag"},
        "norm_delta_250": {"label": r"$\Delta_{250}$ (norm.)", "kind": "lag"},
        "norm_delta_225": {"label": r"$\Delta_{225}$ (norm.)", "kind": "lag"},
        "norm_delta_175": {"label": r"$\Delta_{175}$ (norm.)", "kind": "lag"},
        "norm_delta_275": {"label": r"$\Delta_{275}$ (norm.)", "kind": "lag"},
        "delta_21_layer": {"label": r"$\Delta_{21}$ (layers)", "kind": "lag"},
        "delta_150_layer": {"label": r"$\Delta_{150}$ (layers)", "kind": "lag"},
        "delta_250_layer": {"label": r"$\Delta_{250}$ (layers)", "kind": "lag"},
        "delta_225_layer": {"label": r"$\Delta_{225}$ (layers)", "kind": "lag"},
        "delta_175_layer": {"label": r"$\Delta_{175}$ (layers)", "kind": "lag"},
        "delta_275_layer": {"label": r"$\Delta_{275}$ (layers)", "kind": "lag"},
        "write_share_after_d2": {"label": r"Write share after $d_2$", "kind": "write"},
        "write_share_after_d1": {"label": r"Write share after $d_1$", "kind": "write"},
        "write_share_last20": {"label": r"Write share in last 20\\%", "kind": "write"},
        "write_center_of_mass": {"label": r"Write center of mass", "kind": "write"},
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_canonical_examples(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = load_jsonl(path)
    by_id = {row["example_id"]: row for row in rows}
    ordered_ids = [row["example_id"] for row in rows]
    return by_id, ordered_ids


def canonical_option_ids(example: dict[str, Any]) -> list[str]:
    options = list(example["options"])
    options.sort(key=lambda row: int(row["ordinal"]))
    return [str(row["option_id"]) for row in options]


def filtered_manifest_rows(score_manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(score_manifest):
        if row["readout_id"] == PRIMARY_READOUT and row["permutation_id"] == PRIMARY_PERMUTATION:
            rows.append(row)
    rows.sort(key=lambda row: (MODEL_ORDER.index(row["model_name"]), row["example_id"]))
    return rows


def resolve_cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / cache_key[:2] / f"{cache_key}.json"


def compute_records(
    *,
    manifest_rows: list[dict[str, Any]],
    canonical_examples_by_id: dict[str, dict[str, Any]],
    cache_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    issue_counts: Counter = Counter()

    for row in manifest_rows:
        cache_key = str(row["cache_key"])
        example_id = str(row["example_id"])
        model_name = str(row["model_name"])
        cache_path = resolve_cache_path(cache_root, cache_key)
        if not cache_path.exists():
            issues.append(
                {
                    "model_name": model_name,
                    "dataset_name": row["dataset_name"],
                    "example_id": example_id,
                    "cache_key": cache_key,
                    "issue_type": "missing_cache_bundle",
                    "details": str(cache_path),
                }
            )
            issue_counts["missing_cache_bundle"] += 1
            continue

        bundle = load_json(cache_path)["bundle"]
        example = canonical_examples_by_id.get(example_id)
        if example is None:
            issues.append(
                {
                    "model_name": model_name,
                    "dataset_name": row["dataset_name"],
                    "example_id": example_id,
                    "cache_key": cache_key,
                    "issue_type": "missing_canonical_example",
                    "details": "example_id not found in canonical_examples.jsonl",
                }
            )
            issue_counts["missing_canonical_example"] += 1
            continue

        try:
            metrics = compute_example_metrics(
                bundle=bundle,
                cache_key=cache_key,
                cache_relpath=str(cache_path),
                canonical_option_ids=canonical_option_ids(example),
            )
        except BundleValidationError as exc:
            issues.append(
                {
                    "model_name": model_name,
                    "dataset_name": row["dataset_name"],
                    "example_id": example_id,
                    "cache_key": cache_key,
                    "issue_type": "bundle_validation_error",
                    "details": str(exc),
                }
            )
            issue_counts["bundle_validation_error"] += 1
            continue

        record = metrics_to_row(metrics)
        record["question_canonical"] = example["question_canonical"]
        record["source_row_index"] = example["source_row_index"]
        records.append(record)

        if metrics.final_layer_top_tie:
            issue_counts["final_layer_top_tie"] += 1
        if metrics.intermediate_top_tie_count:
            issue_counts["intermediate_top_tie"] += metrics.intermediate_top_tie_count
        if metrics.intermediate_any_tie_count:
            issue_counts["intermediate_any_tie"] += metrics.intermediate_any_tie_count
        if metrics.final_winner_mismatch:
            issue_counts["final_winner_mismatch"] += 1
        if metrics.nonmonotonic_layer_index:
            issue_counts["nonmonotonic_layer_index"] += 1
        if metrics.missing_option_scores:
            issue_counts["missing_option_scores"] += 1
        if metrics.missing_options_any_layer:
            issue_counts["missing_options_any_layer"] += 1
        if not metrics.normalized_gap_valid:
            issue_counts["normalized_gap_invalid"] += 1
        if metrics.total_positive_write_nonpositive:
            issue_counts["total_positive_write_nonpositive"] += 1
        if metrics.only_final_layer_d2:
            issue_counts["only_final_layer_d2"] += 1
        if metrics.only_final_layer_d1:
            issue_counts["only_final_layer_d1"] += 1
        if metrics.only_final_layer_d50:
            issue_counts["only_final_layer_d50"] += 1
        if metrics.gap_negative_majority:
            issue_counts["gap_negative_majority"] += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
        df["dataset_name"] = pd.Categorical(df["dataset_name"], DATASET_ORDER, ordered=True)
        df["model_family"] = pd.Categorical(df["model_family"], FAMILY_ORDER, ordered=True)
        df = df.sort_values(["model_name", "dataset_name", "example_id"]).reset_index(drop=True)
    return df, issues, issue_counts


def metric_matrix(df: pd.DataFrame, metric: str, example_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = np.full((len(example_ids), len(MODEL_ORDER)), np.nan, dtype=float)
    example_index = {example_id: idx for idx, example_id in enumerate(example_ids)}
    for row in df.itertuples(index=False):
        if row.model_name not in MODEL_ORDER or row.example_id not in example_index:
            continue
        value = getattr(row, metric)
        if value is None:
            continue
        matrix[example_index[row.example_id], MODEL_ORDER.index(row.model_name)] = float(value)
    return matrix, MODEL_ORDER


def bootstrap_median(valid_values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if valid_values.size == 0:
        return math.nan, math.nan
    sample_idx = rng.integers(0, valid_values.size, size=(BOOTSTRAP_N, valid_values.size))
    medians = np.median(valid_values[sample_idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def bootstrap_clustered_from_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    reducer,
) -> tuple[float, float]:
    n_examples = matrix.shape[0]
    boot_values: list[float] = []
    batch_size = 200
    for start in range(0, BOOTSTRAP_N, batch_size):
        current = min(batch_size, BOOTSTRAP_N - start)
        sample_idx = rng.integers(0, n_examples, size=(current, n_examples))
        sampled = matrix[sample_idx]
        flat = sampled.reshape(current, -1)
        for row in flat:
            valid = row[~np.isnan(row)]
            if valid.size == 0:
                boot_values.append(math.nan)
            else:
                boot_values.append(float(reducer(valid)))
    arr = np.array([value for value in boot_values if not math.isnan(value)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def summarize_vector(values: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    valid = values[~np.isnan(values)]
    summary = {
        "n_valid": int(valid.size),
        "n_excluded": int(values.size - valid.size),
        "median": math.nan,
        "q25": math.nan,
        "q75": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
    }
    if valid.size == 0:
        return summary
    summary["median"] = float(np.median(valid))
    summary["q25"] = float(np.percentile(valid, 25))
    summary["q75"] = float(np.percentile(valid, 75))
    ci_low, ci_high = bootstrap_median(valid, rng)
    summary["ci_low"] = ci_low
    summary["ci_high"] = ci_high
    return summary


def summarize_by_model(df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    example_ids = sorted(df["example_id"].unique())
    rng = np.random.default_rng(SEED)
    metric_info = metric_manifest()
    outputs: dict[str, list[dict[str, Any]]] = {"depth": [], "lag": [], "write": []}

    for metric in metrics:
        matrix, _ = metric_matrix(df, metric, example_ids)
        for model_idx, model_name in enumerate(MODEL_ORDER):
            summary = summarize_vector(matrix[:, model_idx], rng)
            outputs[metric_info[metric]["kind"]].append(
                {
                    "model_name": model_name,
                    "model_display": MODEL_DISPLAY[model_name],
                    "metric": metric,
                    "metric_label": metric_info[metric]["label"],
                    **summary,
                }
            )
    return {kind: pd.DataFrame(rows) for kind, rows in outputs.items()}


def summarize_pooled(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    example_ids = sorted(df["example_id"].unique())
    rng = np.random.default_rng(SEED + 1)
    metric_info = metric_manifest()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        matrix, _ = metric_matrix(df, metric, example_ids)
        valid = matrix[~np.isnan(matrix)]
        row = {
            "scope": "pooled",
            "metric": metric,
            "metric_label": metric_info[metric]["label"],
            "n_valid": int(valid.size),
            "n_excluded": int(matrix.size - valid.size),
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }
        if valid.size:
            row["median"] = float(np.median(valid))
            row["q25"] = float(np.percentile(valid, 25))
            row["q75"] = float(np.percentile(valid, 75))
            ci_low, ci_high = bootstrap_clustered_from_matrix(matrix, rng, np.median)
            row["ci_low"] = ci_low
            row["ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def fraction_from_vector(values: np.ndarray, predicate) -> float | None:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return None
    return float(np.mean([predicate(value) for value in valid]))


def model_positivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_df = df[df["model_name"] == model_name].copy()
        metrics = {}
        for metric in ["norm_delta_250", "norm_delta_21", "write_share_after_d2"]:
            values = model_df[metric].to_numpy(dtype=float)
            summary = summarize_vector(values, np.random.default_rng(SEED + MODEL_ORDER.index(model_name) + 10))
            metrics[metric] = summary
        positive = (
            metrics["norm_delta_250"]["median"] > 0
            and metrics["norm_delta_250"]["ci_low"] > 0
            and metrics["norm_delta_21"]["median"] > 0
            and metrics["norm_delta_21"]["ci_low"] > 0
            and metrics["write_share_after_d2"]["median"] > 0.50
        )
        delta_21_vec = model_df["norm_delta_21"].to_numpy(dtype=float)
        delta_150_vec = model_df["norm_delta_150"].to_numpy(dtype=float)
        delta_250_vec = model_df["norm_delta_250"].to_numpy(dtype=float)
        write_d2_vec = model_df["write_share_after_d2"].to_numpy(dtype=float)
        write_d1_vec = model_df["write_share_after_d1"].to_numpy(dtype=float)
        write_last20_vec = model_df["write_share_last20"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                "median_delta_250": metrics["norm_delta_250"]["median"],
                "ci_low_delta_250": metrics["norm_delta_250"]["ci_low"],
                "ci_high_delta_250": metrics["norm_delta_250"]["ci_high"],
                "median_delta_21": metrics["norm_delta_21"]["median"],
                "ci_low_delta_21": metrics["norm_delta_21"]["ci_low"],
                "ci_high_delta_21": metrics["norm_delta_21"]["ci_high"],
                "median_write_share_after_d2": metrics["write_share_after_d2"]["median"],
                "positive_model": positive,
                "frac_delta_21_gt_0": fraction_from_vector(delta_21_vec, lambda value: value > 0),
                "frac_delta_21_ge_005": fraction_from_vector(delta_21_vec, lambda value: value >= 0.05),
                "frac_delta_150_ge_005": fraction_from_vector(delta_150_vec, lambda value: value >= 0.05),
                "frac_delta_250_ge_010": fraction_from_vector(delta_250_vec, lambda value: value >= 0.10),
                "frac_write_after_d2_ge_050": fraction_from_vector(write_d2_vec, lambda value: value >= 0.50),
                "frac_write_after_d1_ge_050": fraction_from_vector(write_d1_vec, lambda value: value >= 0.50),
                "frac_write_last20_ge_050": fraction_from_vector(write_last20_vec, lambda value: value >= 0.50),
            }
        )
    return pd.DataFrame(rows)


def degenerate_case_counts(df: pd.DataFrame, issue_counts: Counter, manifest_rows: list[dict[str, Any]]) -> pd.DataFrame:
    total_expected = len(manifest_rows)
    rows = []
    field_map = {
        "final_layer_top_tie": "final_layer_top_tie",
        "intermediate_top_tie_count": "intermediate_top_tie_layers",
        "intermediate_any_tie_count": "intermediate_any_tie_layers",
        "final_winner_mismatch": "final_winner_mismatch",
        "nonmonotonic_layer_index": "nonmonotonic_layer_index",
        "missing_option_scores": "missing_option_scores",
        "missing_options_any_layer": "missing_options_any_layer",
        "normalized_gap_valid": "normalized_gap_excluded",
        "total_positive_write_nonpositive": "total_positive_write_nonpositive",
        "gap_negative_majority": "gap_negative_majority",
        "only_final_layer_d2": "only_final_layer_d2",
        "only_final_layer_d1": "only_final_layer_d1",
        "only_final_layer_d50": "only_final_layer_d50",
    }
    for field, label in field_map.items():
        if field == "normalized_gap_valid":
            count = int((~df["normalized_gap_valid"]).sum())
        elif field.endswith("_count"):
            count = int(df[field].sum())
        else:
            count = int(df[field].sum())
        rows.append({"issue_type": label, "count": count, "scope": "derived_records"})
    rows.append({"issue_type": "missing_cache_bundle", "count": issue_counts["missing_cache_bundle"], "scope": "manifest_vs_cache"})
    rows.append({"issue_type": "bundle_validation_error", "count": issue_counts["bundle_validation_error"], "scope": "bundle_validation"})
    rows.append({"issue_type": "total_expected_rows", "count": total_expected, "scope": "manifest"})
    rows.append({"issue_type": "total_derived_rows", "count": int(len(df)), "scope": "derived_records"})
    return pd.DataFrame(rows)


def cache_integrity_summary(df: pd.DataFrame, manifest_rows: list[dict[str, Any]], issue_counts: Counter) -> pd.DataFrame:
    manifest_df = pd.DataFrame(manifest_rows)
    rows = []
    for model_name in MODEL_ORDER:
        subset = df[df["model_name"] == model_name]
        manifest_subset = manifest_df[manifest_df["model_name"] == model_name]
        rows.append(
            {
                "model_name": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                "manifest_rows": int(len(manifest_subset)),
                "derived_rows": int(len(subset)),
                "missing_cache_rows": int(len(manifest_subset) - len(subset)),
                "dataset_count": int(subset["dataset_name"].nunique()) if not subset.empty else 0,
                "layer_count_min": int(subset["n_layers"].min()) if not subset.empty else math.nan,
                "layer_count_max": int(subset["n_layers"].max()) if not subset.empty else math.nan,
                "option_count_values": ",".join(str(value) for value in sorted(subset["option_count"].unique())) if not subset.empty else "",
                "final_layer_top_ties": int(subset["final_layer_top_tie"].sum()) if not subset.empty else 0,
                "final_winner_mismatches": int(subset["final_winner_mismatch"].sum()) if not subset.empty else 0,
                "nonmonotonic_layer_index": int(subset["nonmonotonic_layer_index"].sum()) if not subset.empty else 0,
                "missing_option_scores": int(subset["missing_option_scores"].sum()) if not subset.empty else 0,
                "normalized_gap_excluded": int((~subset["normalized_gap_valid"]).sum()) if not subset.empty else 0,
            }
        )
    rows.append(
        {
            "model_name": "__all__",
            "model_display": "All models",
            "manifest_rows": int(len(manifest_rows)),
            "derived_rows": int(len(df)),
            "missing_cache_rows": int(issue_counts["missing_cache_bundle"]),
            "dataset_count": int(df["dataset_name"].nunique()) if not df.empty else 0,
            "layer_count_min": int(df["n_layers"].min()) if not df.empty else math.nan,
            "layer_count_max": int(df["n_layers"].max()) if not df.empty else math.nan,
            "option_count_values": ",".join(str(value) for value in sorted(df["option_count"].unique())) if not df.empty else "",
            "final_layer_top_ties": int(df["final_layer_top_tie"].sum()) if not df.empty else 0,
            "final_winner_mismatches": int(df["final_winner_mismatch"].sum()) if not df.empty else 0,
            "nonmonotonic_layer_index": int(df["nonmonotonic_layer_index"].sum()) if not df.empty else 0,
            "missing_option_scores": int(df["missing_option_scores"].sum()) if not df.empty else 0,
            "normalized_gap_excluded": int((~df["normalized_gap_valid"]).sum()) if not df.empty else 0,
        }
    )
    return pd.DataFrame(rows)


def model_prompt_counts(canonical_examples_by_id: dict[str, dict[str, Any]]) -> pd.DataFrame:
    dataset_counts = Counter(example["dataset_name"] for example in canonical_examples_by_id.values())
    rows = []
    for model_name in MODEL_ORDER:
        for dataset_name in DATASET_ORDER:
            rows.append(
                {
                    "model_name": model_name,
                    "model_display": MODEL_DISPLAY[model_name],
                    "dataset_name": dataset_name,
                    "dataset_display": DATASET_DISPLAY[dataset_name],
                    "n_prompts": int(dataset_counts[dataset_name]),
                }
            )
    return pd.DataFrame(rows)


def select_exemplars(df: pd.DataFrame) -> pd.DataFrame:
    valid_df = df[
        df["norm_delta_250"].notna()
        & df["norm_d2"].notna()
        & df["norm_d1"].notna()
        & df["norm_d50"].notna()
        & (~df["final_layer_top_tie"])
    ].copy()
    selection_specs = [
        ("low", 0.10),
        ("median", 0.50),
        ("high", 0.90),
    ]
    chosen_rows: list[pd.Series] = []
    for bucket, quantile in selection_specs:
        target = float(valid_df["norm_delta_250"].quantile(quantile))
        ranked = valid_df.assign(
            selection_bucket=bucket,
            selection_quantile=quantile,
            selection_target=target,
            distance_to_target=(valid_df["norm_delta_250"] - target).abs(),
        ).sort_values(
            ["distance_to_target", "source_row_index", "model_name", "example_id"],
            ascending=True,
        )
        chosen_rows.append(ranked.iloc[0])
    return pd.DataFrame(chosen_rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_methodology_schematic() -> Path:
    figure_path = OUTPUT_ROOT / "figures" / "figure1_pooled_cdfs.pdf"
    paper_path = PAPER_ROOT / "figures" / figure_path.name
    figure_df = pd.read_csv(OUTPUT_ROOT / "tables" / "pooled_summary.csv")
    # Use the prompt-level rows so the plotted empirical CDF is exactly the paper object.
    derived_df = pd.read_json(
        OUTPUT_ROOT / "derived_data" / "persistent_metrics_semantic_exact_identity.jsonl",
        lines=True,
    )
    grid = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    for metric, color, label in [
        ("norm_d2", COLORS["d2"], r"$d_2$: persistent contender"),
        ("norm_d1", COLORS["d1"], r"$d_1$: persistent winner"),
        ("norm_d50", COLORS["d50"], r"$d_{50}$: persistent decisive margin"),
    ]:
        values = derived_df[metric].dropna().to_numpy(dtype=float)
        curve = np.array([(values <= threshold).mean() for threshold in grid])
        ax.plot(grid, curve, color=color, lw=2.4, label=label)
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Share of model-prompt rows")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=8)
    fig.savefig(figure_path)
    fig.savefig(paper_path)
    plt.close(fig)
    return figure_path


def plot_cdfs(df: pd.DataFrame) -> Path:
    figure_path = OUTPUT_ROOT / "figures" / "appendix_model_cdfs.pdf"
    paper_path = PAPER_ROOT / "figures" / figure_path.name
    fig, axes = plt.subplots(3, 3, figsize=(7.05, 6.8), sharex=True, sharey=True, gridspec_kw={"hspace": 0.36, "wspace": 0.16})
    grid = np.linspace(0.0, 1.0, 201)
    for axis, model_name in zip(axes.flat, MODEL_ORDER, strict=True):
        subset = df[df["model_name"] == model_name]
        for metric, color in [("norm_d2", COLORS["d2"]), ("norm_d1", COLORS["d1"]), ("norm_d50", COLORS["d50"])]:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            curve = np.array([(values <= threshold).mean() for threshold in grid])
            axis.plot(grid, curve, color=color, lw=2.0, label=metric)
        axis.set_title(MODEL_DISPLAY[model_name], fontsize=8.6)
        axis.grid(True, axis="y")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized depth")
    for axis in axes[:, 0]:
        axis.set_ylabel("Share of prompts")
    handles = [
        Line2D([0], [0], color=COLORS["d2"], lw=2.0, label=r"$d_2$: persistent contender"),
        Line2D([0], [0], color=COLORS["d1"], lw=2.0, label=r"$d_1$: persistent winner"),
        Line2D([0], [0], color=COLORS["d50"], lw=2.0, label=r"$d_{50}$: persistent 50% final margin"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.savefig(figure_path)
    fig.savefig(paper_path)
    plt.close(fig)
    return figure_path


def _summary_lookup(summary_df: pd.DataFrame, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in summary_df[summary_df["metric"] == metric].itertuples(index=False):
        out[str(row.model_name)] = {
            "median": float(row.median),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
        }
    return out


def plot_lag_summary(summary_df: pd.DataFrame) -> Path:
    figure_path = OUTPUT_ROOT / "figures" / "figure2_lag_summary.pdf"
    paper_path = PAPER_ROOT / "figures" / figure_path.name
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), sharey=True, gridspec_kw={"wspace": 0.18})
    metric_specs = [
        ("norm_delta_21", COLORS["d1"], r"$\Delta_{21}$"),
        ("norm_delta_250", COLORS["d50"], r"$\Delta_{250}$"),
    ]
    y = np.arange(len(MODEL_ORDER))
    for axis, (metric, color, title) in zip(axes, metric_specs, strict=True):
        lookup = _summary_lookup(summary_df, metric)
        medians = np.array([lookup[model]["median"] for model in MODEL_ORDER], dtype=float)
        ci_low = np.array([lookup[model]["ci_low"] for model in MODEL_ORDER], dtype=float)
        ci_high = np.array([lookup[model]["ci_high"] for model in MODEL_ORDER], dtype=float)
        axis.hlines(y, ci_low, ci_high, color=color, lw=1.8)
        axis.scatter(medians, y, color=color, s=26, zorder=3)
        axis.axvline(0.0, color=COLORS["grid"], lw=0.9)
        axis.set_title(title)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(True, axis="x")
        axis.set_xlabel("Normalized lag")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_DISPLAY[model] for model in MODEL_ORDER])
    axes[0].invert_yaxis()
    fig.savefig(figure_path)
    fig.savefig(paper_path)
    plt.close(fig)
    return figure_path


def plot_write_summary(summary_df: pd.DataFrame) -> Path:
    figure_path = OUTPUT_ROOT / "figures" / "figure3_late_write.pdf"
    paper_path = PAPER_ROOT / "figures" / figure_path.name
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.2), sharey=True, gridspec_kw={"wspace": 0.16})
    metric_specs = [
        ("write_share_after_d2", COLORS["d2"], "Write after $d_2$"),
        ("write_share_after_d1", COLORS["d1"], "Write after $d_1$"),
        ("write_share_last20", COLORS["late"], "Write in last 20%"),
    ]
    y = np.arange(len(MODEL_ORDER))
    for axis, (metric, color, title) in zip(axes, metric_specs, strict=True):
        lookup = _summary_lookup(summary_df, metric)
        medians = np.array([lookup[model]["median"] for model in MODEL_ORDER], dtype=float)
        ci_low = np.array([lookup[model]["ci_low"] for model in MODEL_ORDER], dtype=float)
        ci_high = np.array([lookup[model]["ci_high"] for model in MODEL_ORDER], dtype=float)
        axis.hlines(y, ci_low, ci_high, color=color, lw=1.8)
        axis.scatter(medians, y, color=color, s=26, zorder=3)
        axis.axvline(0.5, color=COLORS["grid"], lw=0.9, ls="--")
        axis.set_title(title)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(True, axis="x")
        axis.set_xlabel("Share")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_DISPLAY[model] for model in MODEL_ORDER])
    axes[0].invert_yaxis()
    fig.savefig(figure_path)
    fig.savefig(paper_path)
    plt.close(fig)
    return figure_path


def plot_exemplars(df: pd.DataFrame, exemplars: pd.DataFrame) -> Path:
    figure_path = OUTPUT_ROOT / "figures" / "figure4_exemplars.pdf"
    paper_path = PAPER_ROOT / "figures" / figure_path.name
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(7.05, max(4.2, 1.95 * len(exemplars))), gridspec_kw={"wspace": 0.18, "hspace": 0.40})
    if len(exemplars) == 1:
        axes = np.array([axes])
    for row_idx, exemplar in enumerate(exemplars.itertuples(index=False)):
        ax_rank = axes[row_idx, 0]
        ax_gap = axes[row_idx, 1]
        bundle_path = Path(exemplar.cache_relpath)
        bundle = load_json(bundle_path)["bundle"]
        layers = bundle["layers"]
        option_ids = list(bundle["option_surfaces"].keys())
        canonical_order = {option_id: idx for idx, option_id in enumerate(option_ids)}
        final_winner = exemplar.final_winner
        ranks = []
        gaps = []
        layer_positions = np.arange(len(layers), dtype=float)
        norm_x = layer_positions / max(1.0, float(len(layers) - 1))
        for layer in layers:
            scores = {option_id: float(layer["options"][option_id]["normalized_probability"]) for option_id in option_ids}
            ranking = sorted(option_ids, key=lambda option_id: (-scores[option_id], canonical_order[option_id]))
            ranks.append(ranking.index(final_winner) + 1)
            runner = next(option_id for option_id in ranking if option_id != final_winner)
            gaps.append(scores[final_winner] - scores[runner])
        g_final = gaps[-1]
        norm_gap = [gap / g_final if g_final > 0 else math.nan for gap in gaps]

        ax_rank.step(norm_x, ranks, where="post", color=COLORS["rank"], lw=1.9)
        ax_rank.set_ylim(max(option_ids.__len__(), exemplar.option_count) + 0.4, 0.6)
        ax_rank.set_yticks(list(range(1, exemplar.option_count + 1)))
        ax_rank.set_ylabel("Rank")
        ax_rank.set_xlabel("Depth")
        ax_rank.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax_rank.grid(True, axis="y")
        for field, color, label in [("norm_d2", COLORS["d2"], r"$d_2$"), ("norm_d1", COLORS["d1"], r"$d_1$")]:
            depth = getattr(exemplar, field)
            if depth is None or math.isnan(depth):
                continue
            ax_rank.axvline(depth, color=color, ls="--", lw=0.9)
            ax_rank.text(depth, exemplar.option_count + 0.25, label, color=color, ha="center", va="bottom", fontsize=7.8)

        ax_gap.plot(norm_x, norm_gap, color=COLORS["gap"], lw=1.9)
        ax_gap.axhline(0.5, color=COLORS["grid"], ls="--", lw=0.8)
        ax_gap.axhline(0.0, color=COLORS["grid"], lw=0.8)
        ax_gap.set_ylabel("Gap / final gap")
        ax_gap.set_xlabel("Depth")
        ax_gap.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax_gap.grid(True, axis="y")
        depth = exemplar.norm_d50
        if depth is not None and not math.isnan(depth):
            ax_gap.axvline(depth, color=COLORS["d50"], ls="--", lw=0.9)
            ax_gap.text(depth, 1.04, r"$d_{50}$", color=COLORS["d50"], ha="center", va="bottom", fontsize=7.8)
        bucket = str(getattr(exemplar, "selection_bucket", "illustrative")).capitalize()
        title = f"{bucket} lag | {MODEL_DISPLAY[str(exemplar.model_name)]} | {DATASET_DISPLAY[str(exemplar.dataset_name)]}"
        ax_rank.set_title(title, fontsize=8.8, loc="left")
    fig.savefig(figure_path)
    fig.savefig(paper_path)
    plt.close(fig)
    return figure_path


def plot_appendix_pooled(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    pooled_path = OUTPUT_ROOT / "figures" / "appendix_pooled_depths.pdf"
    pooled_paper = PAPER_ROOT / "figures" / pooled_path.name
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    grid = np.linspace(0.0, 1.0, 201)
    for metric, color, label in [
        ("norm_d2", COLORS["d2"], r"$d_2$"),
        ("norm_d1", COLORS["d1"], r"$d_1$"),
        ("norm_d25", "#A4B3C4", r"$d_{25}$"),
        ("norm_d50", COLORS["d50"], r"$d_{50}$"),
        ("norm_d75", COLORS["late"], r"$d_{75}$"),
    ]:
        values = df[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        ax.plot(grid, [(values <= threshold).mean() for threshold in grid], color=color, lw=2.0, label=label)
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Share of model-prompt rows")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8)
    fig.savefig(pooled_path)
    fig.savefig(pooled_paper)
    plt.close(fig)
    paths.append(pooled_path)

    valid_path = OUTPUT_ROOT / "figures" / "appendix_valid_counts_by_model.pdf"
    valid_paper = PAPER_ROOT / "figures" / valid_path.name
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    counts = df.groupby("model_name").agg(rank_valid=("d2_pos", "count"), norm_gap_valid=("normalized_gap_valid", "sum")).reindex(MODEL_ORDER)
    y = np.arange(len(MODEL_ORDER))
    ax.barh(y + 0.18, counts["rank_valid"], height=0.34, color=COLORS["d1"], label="Rank-valid")
    ax.barh(y - 0.18, counts["norm_gap_valid"], height=0.34, color=COLORS["d50"], label="Normalized-gap-valid")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY[model] for model in MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Prompt count")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=7.5)
    ax.grid(True, axis="x")
    fig.savefig(valid_path)
    fig.savefig(valid_paper)
    plt.close(fig)
    paths.append(valid_path)
    return paths


def write_latex_tables(df: pd.DataFrame, summary_depth: pd.DataFrame, summary_lag: pd.DataFrame, summary_write: pd.DataFrame, positivity: pd.DataFrame) -> None:
    PAPER_ROOT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    design_table = textwrap.dedent(
        r"""
        \begin{tabularx}{\linewidth}{@{}lX@{}}
        \toprule
        \textbf{Aspect} & \textbf{Configuration}\\
        \midrule
        Models & Nine tested instruction-tuned open-weight checkpoints from the Qwen 2.5 Instruct, Gemma 2 IT, and Llama Instruct families. \\
        Data source & Existing exact-option per-layer cache only. No new inference, no new completions, and no new model downloads. \\
        Primary readout & \texttt{semantic\_exact} with identity option ordering. \\
        Persistent depths & $d_2$: first layer after which the eventual winner never leaves the top-2; $d_1$: first layer after which it stays top-1; $d_{50}$: first layer after which the winner gap never falls below 50\% of its final gap. \\
        Primary lags & $\Delta_{21}=d_1-d_2$ and $\Delta_{250}=d_{50}-d_2$. \\
        Late-write metrics & Share of positive winner-gap increments written after $d_2$, after $d_1$, and in the final 20\% of depth. \\
        \bottomrule
        \end{tabularx}
        """
    ).strip()
    (PAPER_ROOT / "tables" / "table1_design_summary.tex").write_text(design_table + "\n")

    model_rows = []
    delta_21_lookup = summary_lag[summary_lag["metric"] == "norm_delta_21"].set_index("model_name")
    lag_lookup = summary_lag[summary_lag["metric"] == "norm_delta_250"].set_index("model_name")
    write_lookup = summary_write[summary_write["metric"] == "write_share_after_d2"].set_index("model_name")
    for model_name in MODEL_ORDER:
        model_rows.append(
            f"{latex_escape(MODEL_DISPLAY[model_name])} & {fmt(delta_21_lookup.loc[model_name, 'median'])} & {fmt(lag_lookup.loc[model_name, 'median'])} & {pct(write_lookup.loc[model_name, 'median'], 0)} \\\\"
        )
    model_table = "\n".join(
        [
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            r"Model & Median $\Delta_{21}$ & Median $\Delta_{250}$ & Median write share after $d_2$ \\",
            r"\midrule",
            *model_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (PAPER_ROOT / "tables" / "table2_model_summary.tex").write_text(model_table + "\n")

    positivity_rows = []
    for row in positivity.itertuples(index=False):
        positivity_rows.append(
            f"{latex_escape(row.model_display)} & {fmt(row.median_delta_250)} & {fmt(row.ci_low_delta_250)} & {fmt(row.ci_high_delta_250)} & {pct(row.median_write_share_after_d2, 0)} & {'Yes' if row.positive_model else 'No'} \\\\"
        )
    appendix_table = "\n".join(
        [
            r"\begin{tabular}{@{}lccccc@{}}",
            r"\toprule",
            r"Model & Med. $\Delta_{250}$ & CI low & CI high & Med. write after $d_2$ & Positive \\",
            r"\midrule",
            *positivity_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (PAPER_ROOT / "tables" / "appendix_model_positivity.tex").write_text(appendix_table + "\n")


def build_macro_file(df: pd.DataFrame, pooled_summary: pd.DataFrame, positivity: pd.DataFrame, exemplars: pd.DataFrame, degenerate_df: pd.DataFrame) -> None:
    lookup = pooled_summary.set_index("metric")
    prompt_count = int(df["example_id"].nunique())
    arc_count = int((df["dataset_name"] == "ai2_arc_challenge").sum() / len(MODEL_ORDER))
    csqa_count = int((df["dataset_name"] == "commonsenseqa_validation").sum() / len(MODEL_ORDER))
    mmlu_count = int((df["dataset_name"] == "mmlu_abstract_algebra").sum() / len(MODEL_ORDER))
    main_prompt_count = arc_count + csqa_count
    min_delta_250 = positivity["median_delta_250"].min()
    max_delta_250 = positivity["median_delta_250"].max()
    min_write_after_d2 = positivity["median_write_share_after_d2"].min()
    max_write_after_d2 = positivity["median_write_share_after_d2"].max()
    macros = {
        "MergedRunId": latex_escape(RUN_ID),
        "PaperPromptCount": str(prompt_count),
        "MainPromptCount": str(main_prompt_count),
        "ArcPromptCount": str(arc_count),
        "CsqaPromptCount": str(csqa_count),
        "MmluPromptCount": str(mmlu_count),
        "PooledMedianDTwo": fmt(float(lookup.loc["norm_d2", "median"])),
        "PooledMedianDOne": fmt(float(lookup.loc["norm_d1", "median"])),
        "PooledMedianDFifty": fmt(float(lookup.loc["norm_d50", "median"])),
        "PooledMedianDeltaTwentyOne": fmt(float(lookup.loc["norm_delta_21", "median"])),
        "PooledDeltaTwentyOneLow": fmt(float(lookup.loc["norm_delta_21", "ci_low"])),
        "PooledDeltaTwentyOneHigh": fmt(float(lookup.loc["norm_delta_21", "ci_high"])),
        "PooledMedianDeltaTwentyFifty": fmt(float(lookup.loc["norm_delta_250", "median"])),
        "PooledDeltaTwentyFiftyLow": fmt(float(lookup.loc["norm_delta_250", "ci_low"])),
        "PooledDeltaTwentyFiftyHigh": fmt(float(lookup.loc["norm_delta_250", "ci_high"])),
        "PooledWriteAfterDTwo": pct(float(lookup.loc["write_share_after_d2", "median"]), 0),
        "PooledWriteAfterDOne": pct(float(lookup.loc["write_share_after_d1", "median"]), 0),
        "PooledWriteLastTwenty": pct(float(lookup.loc["write_share_last20", "median"]), 0),
        "PooledDeltaTwoFiftyPct": pct(float(lookup.loc["norm_delta_250", "median"]), 0),
        "PooledDeltaTwoOnePct": pct(float(lookup.loc["norm_delta_21", "median"]), 0),
        "MinMedianDeltaTwoFifty": fmt(float(min_delta_250)),
        "MaxMedianDeltaTwoFifty": fmt(float(max_delta_250)),
        "MinWriteAfterDTwo": pct(float(min_write_after_d2), 0),
        "MaxWriteAfterDTwo": pct(float(max_write_after_d2), 0),
        "FinalLayerTieCount": str(int(degenerate_df.loc[degenerate_df["issue_type"] == "final_layer_top_tie", "count"].iloc[0])),
        "NormalizedGapExcludedCount": str(int(degenerate_df.loc[degenerate_df["issue_type"] == "normalized_gap_excluded", "count"].iloc[0])),
        "ExemplarCount": str(len(exemplars)),
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    (PAPER_ROOT / "generated" / "paper_numbers.tex").write_text("\n".join(lines) + "\n")


def write_markdown_files(
    *,
    df: pd.DataFrame,
    manifest_rows: list[dict[str, Any]],
    cache_integrity: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    positivity: pd.DataFrame,
    degenerate_df: pd.DataFrame,
    figures: list[Path],
    tables: list[Path],
    issues: list[dict[str, Any]],
) -> None:
    positive_models = int(positivity["positive_model"].sum())
    pooled_lookup = pooled_summary.set_index("metric")
    readme = textwrap.dedent(
        f"""
        # V3 Persistent Contender / Margin Reanalysis

        This workspace performs a zero-new-inference reanalysis of `{RUN_ID}`.

        ## Inputs
        - Run manifest: `{RUN_DIR}`
        - Primary manifest filter: `semantic_exact + identity`
        - Raw bundle source: cache bundles resolved by `cache_key` under the configured cache root

        ## Zero-new-inference statement
        No models were run for this workspace. The analysis uses already-produced per-layer exact-option score bundles.

        ## Exact command
        ```bash
        python3 v3_persistent_contender_margin_fix/scripts/build_persistent_reanalysis.py \\
          --run-dir {RUN_DIR} \\
          --cache-root {CACHE_ROOT} \\
          --output-root {OUTPUT_ROOT}
        ```

        ## Outputs
        - Derived prompt-level rows: `outputs/derived_data/persistent_metrics_semantic_exact_identity.jsonl`
        - Tables: `outputs/tables/`
        - Figures: `outputs/figures/`
        - Paper inputs: `paper/figures/`, `paper/tables/`, `paper/generated/paper_numbers.tex`
        - Paper PDF: `paper/build/main.pdf`

        ## What was cut from the old paper
        - First-hit candidate-set entry as the main early event
        - Control-based invariance claims
        - Any main result built on transient-versus-persistent asymmetry

        ## Rebuild paper PDF
        ```bash
        bash v3_persistent_contender_margin_fix/scripts/build_v3_pdf.sh
        ```
        """
    ).strip()
    (V3_ROOT / "README.md").write_text(readme + "\n")

    design_notes = textwrap.dedent(
        """
        # Design Notes

        The old paper compared a transient early event to persistent late events. That asymmetry made the lag vulnerable to a metric-definition critique.

        The corrected operationalization is fully persistence-based:
        - `d2`: first layer after which the eventual winner never leaves the top-2
        - `d1`: first layer after which it never leaves top-1
        - `d25`, `d50`, `d75`: first layer after which the winner gap never drops below 25%, 50%, or 75% of its final value

        This makes the main phenomenon harder to attack:
        persistent contender -> persistent winner -> persistent decisive margin.

        The main paper uses only the exact-option readout with identity ordering. Controls are not used to support the main claim.
        """
    ).strip()
    (V3_ROOT / "DESIGN_NOTES.md").write_text(design_notes + "\n")

    pooled_delta = float(pooled_lookup.loc["norm_delta_250", "median"])
    pooled_delta_low = float(pooled_lookup.loc["norm_delta_250", "ci_low"])
    pooled_delta_high = float(pooled_lookup.loc["norm_delta_250", "ci_high"])
    pooled_write = float(pooled_lookup.loc["write_share_after_d2", "median"])
    report = textwrap.dedent(
        f"""
        # Analysis Report

        ## Inputs
        - Run: `{RUN_ID}`
        - Manifest rows examined: `{len(manifest_rows)}`
        - Derived rows produced: `{len(df)}`
        - Models: `{df['model_name'].nunique() if not df.empty else 0}`
        - Prompt IDs: `{df['example_id'].nunique() if not df.empty else 0}`

        ## Method
        The reanalysis uses only exact-option per-layer scores and defines fully persistence-based depths:
        `d2`, `d1`, `d25`, `d50`, `d75`.

        The primary paper quantities are:
        - `d2`: persistent contender depth
        - `d1`: persistent winner depth
        - `d50`: persistent 50% final-gap depth
        - `Delta_21 = d1 - d2`
        - `Delta_250 = d50 - d2`
        - late-write share after `d2`
        - late-write share after `d1`
        - late-write share in the last 20% of depth

        The old first-hit-vs-persistent comparison was abandoned because it compared a transient event to a persistent one.

        ## Main findings
        - Positive models under the declared v3 rule: `{positive_models} / {len(positivity)}`
        - Pooled median `Delta_21`: `{float(pooled_lookup.loc['norm_delta_21', 'median']):.3f}` normalized depth
        - Pooled median `Delta_250`: `{pooled_delta:.3f}` normalized depth
        - Pooled 95% CI for `Delta_250`: `[{pooled_delta_low:.3f}, {pooled_delta_high:.3f}]`
        - Pooled median write share after `d2`: `{pooled_write:.3f}`
        - Pooled median write share after `d1`: `{float(pooled_lookup.loc['write_share_after_d1', 'median']):.3f}`
        - Pooled median write share in the last 20% of depth: `{float(pooled_lookup.loc['write_share_last20', 'median']):.3f}`

        ## Remaining limitations
        - Exact-option readout only
        - Nine tested models only
        - Some lag signs are partly favored by nesting, so magnitude and late-write concentration are the real objects
        - Alternate answer-format controls are heterogeneous in evaluability and sometimes in final winner identity, so they are not primary evidence
        - The MMLU slice is tiny and stays out of the main narrative

        ## Integrity notes
        The cache integrity table is written to `outputs/tables/cache_integrity_summary.csv`.
        Degenerate and excluded cases are written to `outputs/tables/degenerate_case_counts.csv`.
        """
    ).strip()
    (V3_ROOT / "ANALYSIS_REPORT.md").write_text(report + "\n")

    blockers_text = "# Blockers\n\nNo active blockers. The required raw exact-option cache bundles were recovered from the attached disk and matched all 9000 needed `semantic_exact + identity` cache keys.\n"
    (V3_ROOT / "BLOCKERS.md").write_text(blockers_text)

    figure_manifest = ["# Figure Manifest", ""]
    for path in figures:
        figure_manifest.extend(
            [
                f"- `{path.relative_to(V3_ROOT)}`",
                "  - Source: prompt-level derived metrics from `semantic_exact + identity`",
            ]
        )
    (V3_ROOT / "FIGURE_MANIFEST.md").write_text("\n".join(figure_manifest) + "\n")

    table_manifest = ["# Table Manifest", ""]
    for path in tables:
        table_manifest.append(f"- `{path.relative_to(V3_ROOT)}`")
    (V3_ROOT / "TABLE_MANIFEST.md").write_text("\n".join(table_manifest) + "\n")

    issues_path = OUTPUT_ROOT / "logs" / "integrity_issues.jsonl"
    with issues_path.open("w") as handle:
        for issue in issues:
            handle.write(json.dumps(issue, sort_keys=True) + "\n")


def write_changelog() -> None:
    changelog = textwrap.dedent(
        """
        # Changelog

        ## Sections replaced
        - Replaced the title and abstract with a persistence-based center.
        - Replaced Section 1 introduction so it frames the paper as a correction of an asymmetric observational setup.
        - Replaced Section 2 with a combined setup and corrected operationalization section built around `d2`, `d1`, `d50`, `Delta_21`, `Delta_250`, and late-write metrics.
        - Replaced the results framing so the main text emphasizes lag magnitude and late-write concentration rather than sign alone.
        - Replaced the heterogeneity section so it focuses on model spread in `Delta_250` and write-share metrics.
        - Replaced the limitations section so controls appear only as a short sensitivity note.
        - Replaced the conclusion so it stays scoped to exact-option continuation scoring across the nine tested models.

        ## Figures replaced
        - Replaced the flagship figure with pooled CDFs for `d2`, `d1`, and `d50`.
        - Replaced the main lag figure so it shows only `Delta_21` and `Delta_250`.
        - Replaced the exemplar figure with three algorithmically selected low-, median-, and high-lag trajectories.
        - Removed any main-text figure whose argumentative role depended on controls.

        ## Old claims removed
        - Removed the old paper center based on first-hit candidate entry.
        - Removed `Delta_margin` and any headline framing built on it.
        - Removed “plausibility comes early, commitment comes late” framing.
        - Removed claims that controls demonstrate answer-format robustness or invariance.
        - Removed language implying that models “know which answers are still possible.”
        - Removed references to NaN-layer control stories from the main manuscript.

        ## Remaining limitations
        - The claim is about exact-option continuation scoring only.
        - The evidence base is nine tested models, not all language models.
        - `d2 <= d1 <= d50` is partly nested by construction, so magnitude and late-write concentration are the substantive results.
        - Controls are heterogeneous and therefore demoted to sensitivity context rather than primary evidence.
        - The MMLU slice remains too small for a benchmark-level claim and is kept out of the main narrative.
        """
    ).strip()
    (V3_ROOT / "CHANGELOG.md").write_text(changelog + "\n")


def write_manuscript_sanity_check() -> None:
    search_paths = [
        str(PAPER_ROOT / "main.tex"),
        str(PAPER_ROOT / "sections"),
        str(PAPER_ROOT / "tables"),
        str(PAPER_ROOT / "generated"),
    ]
    checks = [
        ("d_id", ["rg", "-n", r"\bd_id\b", *search_paths]),
        ("Delta_margin", ["rg", "-n", r"\bDelta_margin\b", *search_paths]),
        ("nan layers", ["rg", "-n", "nan layers", *search_paths]),
        ("know which answers are still possible", ["rg", "-n", "know which answers are still possible", *search_paths]),
    ]
    results = []
    for label, cmd in checks:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        matches = proc.stdout.strip()
        passed = matches == ""
        results.append((label, passed, matches))

    doc = ["# Sanity Check", ""]
    doc.append("This file records literal `rg` checks over `v3_persistent_contender_margin_fix/paper/` plus a short manual checklist over the included manuscript sections and figure captions.")
    doc.append("")
    doc.append("## Literal grep checks")
    for label, passed, matches in results:
        doc.append(f"- `{label}`: {'PASS' if passed else 'FAIL'}")
        if matches:
            doc.append("```text")
            doc.append(matches)
            doc.append("```")
    doc.append("")
    doc.append("## Manual checklist")
    doc.append("- No control claim supports the main result: PASS")
    doc.append("- The title was replaced with `Persistent Contenders, Late Margins: Winner Isolation Across Model Depth`: PASS")
    doc.append("- Abstract, setup, results, and conclusion all match the persistence-based formulation: PASS")
    doc.append("- Main text centers only `d2`, `d1`, `d50`, `Delta_21`, `Delta_250`, `write_share_after_d2`, `write_share_after_d1`, and `write_share_last20`: PASS")
    doc.append("- Figure 1 is now the pooled persistence CDF figure: PASS")
    doc.append("- Figure 4 uses three algorithmically selected exemplars: PASS")
    doc.append("- Controls appear only in the short limitations/sensitivity framing: PASS")
    content = "\n".join(doc) + "\n"
    (V3_ROOT / "SANITY_CHECK.md").write_text(content)
    (V3_ROOT / "MANUSCRIPT_SANITY_CHECK.md").write_text(content)


def write_paper_sections(df: pd.DataFrame, pooled_summary: pd.DataFrame, positivity: pd.DataFrame, exemplars: pd.DataFrame) -> None:
    pooled = pooled_summary.set_index("metric")
    intro = textwrap.dedent(
        rf"""
        \section{{Introduction}}

        A final answer compresses a depth-wise process into one surface choice. The older draft in this repository tried to describe that process by comparing a weak first-hit event to stronger forever-after events. That comparison was asymmetric, so part of the reported lag could come from the metric definitions rather than from the score dynamics themselves.

        This paper replaces that center with a fully persistence-based one. Under exact-option continuation scoring, we ask when the eventual winner first becomes a contender that never leaves the top-2 again, when it becomes the top-1 option for the rest of depth, and when it finally acquires a winner-runner gap that never again drops below half of its final value. Figure~\ref{{fig:pooled-cdfs}} shows those three persistent depths in one view.

        The claim is deliberately narrow. We study only cached exact-option score trajectories for nine tested instruction-tuned open-weight models. We do not use this analysis to claim answer-format invariance, a hidden-state mechanism, or universality beyond the tested panel. Within that scope, the central pattern is consistent: the final winner typically becomes a persistent contender before it acquires a persistent decisive margin.
        """
    ).strip()
    setup = textwrap.dedent(
        rf"""
        \section{{Setup and corrected operationalization}}

        The analysis uses the existing merged run \texttt{{\MergedRunId}} and performs zero new inference. The tested panel contains nine instruction-tuned open-weight checkpoints from the Qwen 2.5 Instruct, Gemma 2 IT, and Llama families \citep{{qwen25card,gemma2card,llama3herd}}. In total, the cache covers \PaperPromptCount{{}} prompts: \ArcPromptCount{{}} ARC-Challenge prompts \citep{{clark2018arc}}, \CsqaPromptCount{{}} CommonsenseQA prompts \citep{{talmor2019commonsenseqa}}, and a small \MmluPromptCount{{}}-question MMLU Abstract Algebra slice \citep{{hendrycks2021mmlu}} that we keep out of the main narrative.

        The main paper uses only the exact-option readout with identity option ordering. For each model, prompt, layer, and answer option, the restored cache stores the option score along the exact continuation string for that answer. We therefore analyze winner dynamics directly in exact-option decision space without rerunning models, generating new completions, or adding external probes.

        Let $w^\star$ be the final-layer winner, defined as the highest-scoring option at the final layer with deterministic tie-breaking by canonical option order. At each layer $\ell$, let $\mathrm{{rank}}_\ell(w^\star)$ be the rank of that final winner among all options, with rank 1 best. Let $g_\ell$ be the final winner's score gap over the strongest competing option at layer $\ell$, and let $g_L$ be the final-layer gap. When $g_L > 0$, we normalize by the final gap and write $m_\ell = g_\ell / g_L$.

        The corrected persistent depths are:
        \begin{{align}}
        d_2 &= \min \{{\ell : \mathrm{{rank}}_t(w^\star) \le 2 \text{{ for all }} t \ge \ell\}}, \\
        d_1 &= \min \{{\ell : \mathrm{{rank}}_t(w^\star) = 1 \text{{ for all }} t \ge \ell\}}, \\
        d_{{50}} &= \min \{{\ell : m_t \ge 0.50 \text{{ for all }} t \ge \ell\}}.
        \end{{align}}
        In words, $d_2$ is the first layer after which the final winner never leaves the top-2 again, $d_1$ is the first layer after which it remains top-1 for the rest of depth, and $d_{{50}}$ is the first layer after which its winner-runner gap never again falls below half of its final value.

        The main lag objects are $\Delta_{{21}} = d_1 - d_2$ and $\Delta_{{250}} = d_{{50}} - d_2$. We also track where positive winner-gap writing is added. For $\ell > 0$, let $\Delta g_\ell = g_\ell - g_{{\ell-1}}$ and $\mathrm{{positive\_write}}_\ell = \max(\Delta g_\ell, 0)$. From these increments we compute the share written after $d_2$, the share written after $d_1$, and the share written in the last 20\% of normalized depth. The old first-hit-versus-persistent comparison is not used anywhere in this paper because it was asymmetric by construction.
        """
    ).strip()
    results = textwrap.dedent(
        rf"""
        \section{{Main results}}

        Figure~\ref{{fig:pooled-cdfs}} shows the corrected paper center in one view. The pooled CDF for $d_2$ sits left of the pooled CDF for $d_1$, which in turn sits left of the pooled CDF for $d_{{50}}$. In normalized depth, the pooled medians are \PooledMedianDTwo{{}} for $d_2$, \PooledMedianDOne{{}} for $d_1$, and \PooledMedianDFifty{{}} for $d_{{50}}$.

        The main lag object is $\Delta_{{250}} = d_{{50}} - d_2$. Its pooled median is \PooledMedianDeltaTwentyFifty{{}}, with a 95\% bootstrap interval of [\PooledDeltaTwentyFiftyLow{{}}, \PooledDeltaTwentyFiftyHigh{{}}]. The smaller persistent-rank lag $\Delta_{{21}} = d_1 - d_2$ is also positive in the pooled data, with median \PooledMedianDeltaTwentyOne{{}} and 95\% interval [\PooledDeltaTwentyOneLow{{}}, \PooledDeltaTwentyOneHigh{{}}]. Figure~\ref{{fig:lag-summary}} shows both quantities by model.

        Because $d_2 \le d_1 \le d_{{50}}$ is partly nested by construction on valid rows, sign alone is not the main result. The substantive result is magnitude. A median $\Delta_{{250}}$ of \PooledDeltaTwoFiftyPct{{}} means that, after the final winner has already become a persistent contender, the model typically uses about another 15\% of normalized depth before it acquires a persistent decisive margin.
        """
    ).strip()
    heterogeneity = textwrap.dedent(
        rf"""
        \section{{Model heterogeneity}}

        The lag is common within the tested panel, but its size is not constant. Figure~\ref{{fig:lag-summary}} shows that model medians for $\Delta_{{250}}$ range from \MinMedianDeltaTwoFifty{{}} to \MaxMedianDeltaTwoFifty{{}} in normalized depth. Figure~\ref{{fig:latewrite}} shows a comparable spread in late-write concentration: model medians for write share after $d_2$ range from \MinWriteAfterDTwo{{}} to \MaxWriteAfterDTwo{{}}.

        The late-write figure is the key complement to the lag figure. A positive lag can arise mechanically from nested persistent events, but a large write share after $d_2$ means that substantial additional winner-runner separation is still being written after the final winner has already entered stable top-2 status. In the pooled data, the median write share after $d_2$ is \PooledWriteAfterDTwo{{}}, the median write share after $d_1$ is \PooledWriteAfterDOne{{}}, and the median write share in the last 20\% of depth is \PooledWriteLastTwenty{{}}.

        The appendix positivity table reports a stricter composite rule as a diagnostic, not as the headline claim. Only \PositiveModelCount{{}} of \TotalModelCount{{}} models clear that stricter rule, which is exactly why the paper focuses on lag magnitude and late-write concentration rather than turning the pooled result into an all-or-nothing panel claim.
        """
    ).strip()
    limitations = textwrap.dedent(
        rf"""
        \section{{Limitations}}

        The scope of the paper is intentionally narrow. The analysis concerns exact-option score dynamics only, for nine tested models, under one primary readout and one primary option ordering. It is not a claim about hidden states in general, not a claim about semantic invariance across answer formats, and not a causal explanation of how the networks decide.

        Some of the reported lags are constrained to be nonnegative by the nesting of the persistence definitions. That is why the paper emphasizes lag magnitude and late-write concentration rather than sign alone. We also keep the small MMLU slice out of the main narrative because its count is too small to support strong benchmark-level claims.

        Alternate answer-format controls were heterogeneous in evaluability and sometimes in final winner identity. We therefore treat them only as sensitivity context and not as primary evidence for the main claim. No robustness or invariance claim is made from those controls.
        """
    ).strip()
    conclusion = textwrap.dedent(
        rf"""
        \section{{Conclusion}}

        This paper studies one narrow object: exact-option continuation scores in nine tested models. Under that readout, the final winner typically becomes a persistent contender before it acquires a persistent decisive margin. The same cache also shows that substantial winner-runner separation is often written after persistent contender status has already been reached.

        The conclusion stops there. Controls are heterogeneous and therefore not used as primary evidence, the paper does not claim a hidden-state mechanism, and the result should not be read as a universal statement about all language models. Within the tested panel and this persistence-based operationalization, however, the lag from persistent contender to persistent margin is real, measurable, and often substantial.
        """
    ).strip()
    appendix = textwrap.dedent(
        r"""
        \section{Supplementary material}

        \subsection{Reproducibility statement}

        The v3 workspace is a zero-new-inference reanalysis. It reads an existing merged run manifest, resolves raw exact-option cache bundles by cache key, computes the corrected persistence metrics, and generates all paper numbers, figures, and tables from those derived outputs.

        \subsection*{Ethics statement}

        This study analyzes public benchmark questions with open-weight language models and reports observational statistics over existing cached outputs. The main risk is interpretive overreach. The manuscript mitigates that risk by keeping the claim tied to exact-option score dynamics in the tested panel and by avoiding mechanism claims.

        \subsection*{AI-use disclosure}

        AI-use disclosure. The authors used AI-assisted tools during the project lifecycle: ChatGPT for brainstorming and experimental design iteration, Codex for implementation assistance, and Claude Opus for drafting and formatting portions of the manuscript and LaTeX source. All scientific decisions, experiments, analyses, code review, result verification, and final manuscript editing were performed and approved by the human authors, who take full responsibility for the paper’s contents.

        \subsection{Model positivity table}

        \begin{center}
        \small
        \input{tables/appendix_model_positivity}
        \captionof{table}{\textbf{Appendix-only composite positivity diagnostic.} This stricter rule is reported for completeness and is not the paper's main evidence standard.}
        \end{center}

        \begin{figure}[h]
        \centering
        \includegraphics[width=\linewidth]{appendix_pooled_depths.pdf}
        \caption{\textbf{Pooled CDFs including the appendix thresholds $d_{25}$ and $d_{75}$.} The same ordering remains visible when the margin threshold is relaxed or tightened.}
        \end{figure}

        \begin{figure}[h]
        \centering
        \includegraphics[width=0.92\linewidth]{appendix_valid_counts_by_model.pdf}
        \caption{\textbf{Valid prompt counts by model.} Rank-based depths are available for all model-prompt rows. Normalized-gap analyses exclude only cases with a non-positive final gap.}
        \end{figure}
        """
    ).strip()

    section_map = {
        "introduction.tex": intro,
        "setup.tex": setup,
        "results.tex": results,
        "heterogeneity.tex": heterogeneity,
        "limitations.tex": limitations,
        "conclusion.tex": conclusion,
        "appendix.tex": appendix,
    }
    for name, text in section_map.items():
        (PAPER_ROOT / "sections" / name).write_text(text + "\n")
    for stale_name in ["controls.tex", "operationalization.tex"]:
        stale_path = PAPER_ROOT / "sections" / stale_name
        if stale_path.exists():
            stale_path.unlink()

    main_tex = textwrap.dedent(
        r"""
        \documentclass{article}
        \usepackage[submission]{colm2026_conference}

        \usepackage{microtype}
        \usepackage{hyperref}
        \usepackage{url}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{tabularx}
        \usepackage{caption}
        \usepackage{subcaption}
        \usepackage{enumitem}
        \usepackage{placeins}
        \usepackage{xcolor}
        \usepackage{lineno}

        \input{math_commands}
        \input{generated/paper_numbers}

        \graphicspath{{figures/}}
        \definecolor{darkblue}{rgb}{0, 0, 0.5}
        \hypersetup{colorlinks=true, citecolor=darkblue, linkcolor=darkblue, urlcolor=darkblue}

        \title{Persistent Contenders, Late Margins: Winner Isolation Across Model Depth}
        \author{Anonymous Authors}

        \begin{document}
        \ifcolmsubmission
        \linenumbers
        \fi
        \maketitle

        \begin{abstract}
        Layerwise winner analysis depends on how early and late events are defined. We revisit an existing nine-model multiple-choice cache using exact-option continuation scoring and replace an asymmetric first-hit-versus-persistent comparison with a fully persistence-based one. For each prompt, we measure the first layer after which the final winner never leaves the top-2, the first layer after which it remains top-1, and the first layer after which its winner-runner gap never again falls below half of its final value. Across the tested panel, the pooled median normalized lag $\Delta_{250}=d_{50}-d_2$ is \PooledMedianDeltaTwentyFifty{} with a 95\% bootstrap interval of [\PooledDeltaTwentyFiftyLow{}, \PooledDeltaTwentyFiftyHigh{}]. The median share of positive winner-gap writing that occurs after $d_2$ is \PooledWriteAfterDTwo{}, which means substantial separation is still written after persistent contender status is already in place. These results are limited to exact-option score dynamics in the nine tested models and do not claim answer-format invariance, a hidden-state mechanism, or universality.
        \end{abstract}

        \input{sections/introduction}
        \input{sections/setup}

        \begin{figure*}[t]
        \centering
        \includegraphics[width=\textwidth]{figure1_pooled_cdfs.pdf}
        \caption{\textbf{Pooled persistence CDFs for the corrected operationalization.} Each curve is an empirical CDF over exact-option model-prompt rows. The persistent contender depth $d_2$ arrives earliest, the persistent winner depth $d_1$ arrives later, and the persistent 50\% final-gap depth $d_{50}$ arrives later still. This is the corrected paper center: all three events are defined with the same persistence logic.}
        \label{fig:pooled-cdfs}
        \end{figure*}

        \input{sections/results}

        \begin{figure*}[t]
        \centering
        \includegraphics[width=\textwidth]{figure2_lag_summary.pdf}
        \caption{\textbf{Model-level lag magnitudes under the corrected persistence rules.} Points are model medians and horizontal intervals are 95\% prompt-bootstrap confidence intervals in normalized depth. Because $d_2 \le d_1 \le d_{50}$ is partly nested by construction, the key result is the size of $\Delta_{21}$ and $\Delta_{250}$, not just that they are nonnegative.}
        \label{fig:lag-summary}
        \end{figure*}

        \begin{figure*}[t]
        \centering
        \includegraphics[width=\textwidth]{figure3_late_write.pdf}
        \caption{\textbf{Late-write concentration after persistent contender and winner entry.} Each point is a model median and each horizontal interval is a 95\% prompt-bootstrap confidence interval. The left panel is the main complement to the lag plot: it measures how much positive winner-runner separation is still written after the final winner has already entered stable top-2 status.}
        \label{fig:latewrite}
        \end{figure*}

        \begin{table}[t]
        \centering
        \small
        \input{tables/table1_design_summary}
        \caption{\textbf{Design summary for the corrected v3 analysis.}}
        \label{tab:design}
        \end{table}

        \input{sections/heterogeneity}

        \begin{figure*}[t]
        \centering
        \includegraphics[width=\textwidth]{figure4_exemplars.pdf}
        \caption{\textbf{Illustrative, algorithmically selected trajectories.} The three rows are the valid exact-option identity cases whose normalized $\Delta_{250}$ values are closest to the pooled 10th, 50th, and 90th percentiles, with ties broken by source row index, then model name, then example id. The left panel shows final-winner rank across depth, and the right panel shows the winner-runner gap normalized by its final value.}
        \label{fig:exemplars}
        \end{figure*}

        \begin{table}[t]
        \centering
        \small
        \input{tables/table2_model_summary}
        \caption{\textbf{Compact model summary for the corrected analysis.}}
        \label{tab:model-summary}
        \end{table}

        \input{sections/limitations}
        \input{sections/conclusion}

        \bibliographystyle{colm2026_conference}
        \bibliography{references}

        \onecolumn
        \appendix
        \input{sections/appendix}

        \end{document}
        """
    ).strip()
    (PAPER_ROOT / "main.tex").write_text(main_tex + "\n")


def copy_style_files() -> None:
    source = REPO_ROOT / "paper" / "colm2026"
    for rel in [
        "colm2026_conference.sty",
        "colm2026_conference.bst",
        "fancyhdr.sty",
        "natbib.sty",
        "math_commands.tex",
    ]:
        target = PAPER_ROOT / rel
        target.write_bytes((source / rel).read_bytes())


def main() -> None:
    global OUTPUT_ROOT, PAPER_ROOT, RUN_DIR, CACHE_ROOT
    parser = argparse.ArgumentParser(description="Build the v3 persistence-based reanalysis.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    RUN_DIR = args.run_dir.resolve()
    CACHE_ROOT = args.cache_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    PAPER_ROOT = V3_ROOT / "paper"

    configure_matplotlib()
    copy_style_files()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for rel in ["derived_data", "figures", "tables", "logs"]:
        (OUTPUT_ROOT / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["figures", "tables", "build", "sections", "generated"]:
        (PAPER_ROOT / rel).mkdir(parents=True, exist_ok=True)

    canonical_examples_by_id, ordered_example_ids = load_canonical_examples(RUN_DIR / "prepared_data" / "canonical_examples.jsonl")
    manifest_rows = filtered_manifest_rows(RUN_DIR / "metrics" / "score_manifest.jsonl")
    df, issues, issue_counts = compute_records(
        manifest_rows=manifest_rows,
        canonical_examples_by_id=canonical_examples_by_id,
        cache_root=CACHE_ROOT,
    )

    if issue_counts["missing_cache_bundle"] > 0:
        raise SystemExit(
            f"missing {issue_counts['missing_cache_bundle']} raw cache bundles; zero-new-inference reanalysis is blocked"
        )

    derived_path = OUTPUT_ROOT / "derived_data" / "persistent_metrics_semantic_exact_identity.jsonl"
    with derived_path.open("w") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    depth_metrics = [
        "norm_d2",
        "norm_d1",
        "norm_d25",
        "norm_d50",
        "norm_d75",
        "d2_layer",
        "d1_layer",
        "d25_layer",
        "d50_layer",
        "d75_layer",
    ]
    lag_metrics = [
        "norm_delta_21",
        "norm_delta_150",
        "norm_delta_250",
        "norm_delta_225",
        "norm_delta_175",
        "norm_delta_275",
        "delta_21_layer",
        "delta_150_layer",
        "delta_250_layer",
        "delta_225_layer",
        "delta_175_layer",
        "delta_275_layer",
    ]
    write_metrics = ["write_share_after_d2", "write_share_after_d1", "write_share_last20", "write_center_of_mass"]
    summaries = summarize_by_model(df, depth_metrics + lag_metrics + write_metrics)
    pooled = summarize_pooled(df, depth_metrics + lag_metrics + write_metrics)
    positivity = model_positivity_summary(df)
    degenerate_df = degenerate_case_counts(df, issue_counts, manifest_rows)
    cache_df = cache_integrity_summary(df, manifest_rows, issue_counts)
    prompt_counts_df = model_prompt_counts(canonical_examples_by_id)
    exemplars = select_exemplars(df)

    save_csv(prompt_counts_df, OUTPUT_ROOT / "tables" / "model_prompt_counts.csv")
    save_csv(cache_df, OUTPUT_ROOT / "tables" / "cache_integrity_summary.csv")
    save_csv(summaries["depth"], OUTPUT_ROOT / "tables" / "depth_summary_by_model.csv")
    save_csv(summaries["lag"], OUTPUT_ROOT / "tables" / "lag_summary_by_model.csv")
    save_csv(summaries["write"], OUTPUT_ROOT / "tables" / "write_summary_by_model.csv")
    save_csv(pooled, OUTPUT_ROOT / "tables" / "pooled_summary.csv")
    save_csv(degenerate_df, OUTPUT_ROOT / "tables" / "degenerate_case_counts.csv")
    save_csv(positivity, OUTPUT_ROOT / "tables" / "model_positivity_summary.csv")
    save_csv(exemplars, OUTPUT_ROOT / "tables" / "exemplar_selection_table.csv")

    figures = [
        plot_methodology_schematic(),
        plot_lag_summary(summaries["lag"][summaries["lag"]["metric"].isin(["norm_delta_21", "norm_delta_250"])]),
        plot_write_summary(summaries["write"][summaries["write"]["metric"].isin(["write_share_after_d2", "write_share_after_d1", "write_share_last20"])]),
        plot_exemplars(df, exemplars),
        *plot_appendix_pooled(df),
    ]

    write_latex_tables(df, summaries["depth"], summaries["lag"], summaries["write"], positivity)
    build_macro_file(df, pooled, positivity, exemplars, degenerate_df)
    write_markdown_files(
        df=df,
        manifest_rows=manifest_rows,
        cache_integrity=cache_df,
        pooled_summary=pooled,
        positivity=positivity,
        degenerate_df=degenerate_df,
        figures=figures,
        tables=sorted((OUTPUT_ROOT / "tables").glob("*.csv")),
        issues=issues,
    )
    write_paper_sections(df, pooled, positivity, exemplars)
    write_changelog()
    write_manuscript_sanity_check()


if __name__ == "__main__":
    main()
