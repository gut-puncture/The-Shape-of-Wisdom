from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RELEASE_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RELEASE_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(RELEASE_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DERIVED_PATH = RELEASE_ROOT / "data" / "derived" / "trajectory_rows.jsonl"
MECHANISTIC_PATH = RELEASE_ROOT / "data" / "derived" / "mechanistic_rows.jsonl"
EXEMPLARS_PATH = RELEASE_ROOT / "data" / "derived" / "exemplar_trajectories.json"
PAPER_ROOT = RELEASE_ROOT / "paper"
QA_ROOT = RELEASE_ROOT / "qa"
TABLE_ROOT = RELEASE_ROOT / "data" / "tables"
MANIFEST_ROOT = RELEASE_ROOT / "data" / "manifests"

BOOTSTRAP_N = 10_000
SEED = 20260331
PRIMARY_DATASETS = ["ai2_arc_challenge", "commonsenseqa_validation"]
APPENDIX_DATASETS: list[str] = []
DATASET_DISPLAY = {
    "ai2_arc_challenge": "ARC-Challenge",
    "commonsenseqa_validation": "CommonsenseQA",
}
MODEL_ORDER = [
    "qwen2p5_1p5b_instruct",
    "qwen2p5_3b_instruct",
    "qwen2p5_7b_instruct",
    "gemma2_2b_it",
    "gemma2_9b_it",
    "gemma2_27b_it",
    "llama3p2_1b_instruct",
    "llama3p2_3b_instruct",
    "llama3p2_3b_instruct",
    "llama3p1_8b_instruct",
]
MODEL_ORDER = list(dict.fromkeys(MODEL_ORDER))
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
COLORS = {
    "paper": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#111111",
    "muted": "#666666",
    "grid": "#E6E6E6",
    "d2": "#2166AC",
    "d1": "#4D4D4D",
    "d50": "#B2182B",
    "lag21": "#2166AC",
    "lag250": "#B2182B",
    "write": "#B2182B",
    "rank": "#2166AC",
    "gap": "#E68613",
    "green": "#2A7F3E",
    "orange": "#E68613",
    "purple": "#7B3294",
}
FORBIDDEN_TEXT_PATTERNS = [
    r"MMLU Abstract Algebra",
    r"mmlu_abstract_algebra",
    r"Abstract Algebra",
    r"Delta_margin",
    r"d_id",
    r"first-hit",
    r"plausible answer set",
    r"know which answers",
    r"Easily Explained",
    r"TODO",
    r"FIXME",
    r"\bnan\b",
    r"old draft",
    r"COLM",
    r"corrected v3",
]


@dataclass(frozen=True)
class Summary:
    metric: str
    n_valid: int
    n_excluded: int
    median: float | None
    q25: float | None
    q75: float | None
    ci_low: float | None
    ci_high: float | None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "axes.grid": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(load_jsonl(DERIVED_PATH))
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df["dataset_name"] = pd.Categorical(
        df["dataset_name"],
        PRIMARY_DATASETS,
        ordered=True,
    )
    return df.sort_values(["model_name", "dataset_name", "example_id"]).reset_index(drop=True)


def load_mechanistic_dataframe() -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(MECHANISTIC_PATH))


def primary_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["dataset_name"].astype(str).isin(PRIMARY_DATASETS)].copy()


def finite_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    idx = rng.integers(0, values.size, size=(BOOTSTRAP_N, values.size))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def metric_matrix(df: pd.DataFrame, metric: str, example_ids: list[str]) -> np.ndarray:
    matrix = np.full((len(example_ids), len(MODEL_ORDER)), np.nan, dtype=float)
    example_index = {example_id: idx for idx, example_id in enumerate(example_ids)}
    for row in df.itertuples(index=False):
        example_id = str(row.example_id)
        model_name = str(row.model_name)
        if example_id not in example_index or model_name not in MODEL_ORDER:
            continue
        value = getattr(row, metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        matrix[example_index[example_id], MODEL_ORDER.index(model_name)] = float(value)
    return matrix


def bootstrap_clustered_median(matrix: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if matrix.size == 0 or np.isfinite(matrix).sum() == 0:
        return None, None
    values: list[float] = []
    n_examples = matrix.shape[0]
    for _ in range(BOOTSTRAP_N):
        sample = rng.integers(0, n_examples, size=n_examples)
        boot_values = matrix[sample, :].reshape(-1)
        boot_values = boot_values[np.isfinite(boot_values)]
        if boot_values.size:
            values.append(float(np.median(boot_values)))
    if not values:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize(series: pd.Series, *, rng: np.random.Generator) -> Summary:
    values = finite_values(series)
    n_total = int(series.shape[0])
    if values.size == 0:
        return Summary(series.name, 0, n_total, None, None, None, None, None)
    ci_low, ci_high = bootstrap_median(values, rng)
    return Summary(
        str(series.name),
        int(values.size),
        int(n_total - values.size),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        ci_low,
        ci_high,
    )


def summarize_clustered(df: pd.DataFrame, metric: str, *, rng: np.random.Generator) -> Summary:
    values = finite_values(df[metric])
    if values.size == 0:
        return Summary(metric, 0, int(df.shape[0]), None, None, None, None, None)
    example_ids = sorted(df["example_id"].astype(str).unique())
    matrix = metric_matrix(df, metric, example_ids)
    ci_low, ci_high = bootstrap_clustered_median(matrix, rng)
    return Summary(
        metric,
        int(values.size),
        int(df.shape[0] - values.size),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        ci_low,
        ci_high,
    )


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value * 100:.{digits}f}\\%"


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


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n")


def generate_summaries(df: pd.DataFrame, mech_df: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    p_df = primary_df(df)
    pooled_metrics = [
        "norm_d2",
        "norm_d1",
        "norm_d25",
        "norm_d50",
        "norm_d75",
        "norm_delta_21",
        "norm_delta_250",
        "write_share_after_d2",
        "write_share_after_d1",
        "write_share_last20",
        "write_center_of_mass",
    ]
    pooled = {metric: summarize_clustered(p_df, metric, rng=rng) for metric in pooled_metrics}
    model_summaries: dict[str, dict[str, Summary]] = {}
    for model in MODEL_ORDER:
        m_df = p_df[p_df["model_name"].astype(str) == model]
        model_summaries[model] = {
            metric: summarize(m_df[metric], rng=rng)
            for metric in ["norm_delta_21", "norm_delta_250", "write_share_after_d2", "write_share_after_d1", "write_share_last20"]
        }
    dataset_counts = (
        p_df.groupby(["dataset_name"], observed=True)["example_id"].nunique().reset_index(name="n_prompts")
    )
    regime_counts = (
        p_df.groupby(["trajectory_regime"], observed=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    drift_rows = mech_df[mech_df["analysis"] == "drift_decomposition"].copy()
    span_rows = mech_df[mech_df["analysis"] == "span_effect_by_model"].copy()
    substitution_rows = mech_df[
        (mech_df["analysis"] == "substitution_sensitivity")
        & (mech_df["setting_group"] == "pairing_mode")
        & (mech_df["setting_name"] == "all_pairs_within_model")
    ].copy()
    legacy_type_rows = mech_df[mech_df["analysis"] == "legacy_trajectory_type_count"].copy()
    return {
        "paper_df": p_df,
        "mechanistic_df": mech_df,
        "pooled": pooled,
        "model_summaries": model_summaries,
        "dataset_counts": dataset_counts,
        "regime_counts": regime_counts,
        "drift_rows": drift_rows,
        "span_rows": span_rows,
        "substitution_rows": substitution_rows,
        "legacy_type_rows": legacy_type_rows,
    }


def build_tables(df: pd.DataFrame, summaries: dict[str, Any]) -> None:
    p_df: pd.DataFrame = summaries["paper_df"]
    mech_df: pd.DataFrame = summaries["mechanistic_df"]
    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_df = p_df[p_df["model_name"].astype(str) == model]
        s_delta = summaries["model_summaries"][model]["norm_delta_250"]
        s_write = summaries["model_summaries"][model]["write_share_after_d2"]
        model_rows.append(
            {
                "model_name": model,
                "model_display": MODEL_DISPLAY[model],
                "family": FAMILY_DISPLAY[str(model_df["model_family"].iloc[0])],
                "n_rows": int(model_df.shape[0]),
                "valid_delta_250": s_delta.n_valid,
                "median_delta_250": s_delta.median,
                "delta_250_ci_low": s_delta.ci_low,
                "delta_250_ci_high": s_delta.ci_high,
                "median_write_after_d2": s_write.median,
                "write_after_d2_ci_low": s_write.ci_low,
                "write_after_d2_ci_high": s_write.ci_high,
            }
        )
    write_csv(
        TABLE_ROOT / "model_summary.csv",
        model_rows,
        [
            "model_name",
            "model_display",
            "family",
            "n_rows",
            "valid_delta_250",
            "median_delta_250",
            "delta_250_ci_low",
            "delta_250_ci_high",
            "median_write_after_d2",
            "write_after_d2_ci_low",
            "write_after_d2_ci_high",
        ],
    )

    regime_rows = [
        {
            "trajectory_regime": str(row.trajectory_regime),
            "n_rows": int(row.n_rows),
            "share": int(row.n_rows) / int(p_df.shape[0]),
        }
        for row in summaries["regime_counts"].itertuples(index=False)
    ]
    write_csv(TABLE_ROOT / "regime_summary.csv", regime_rows, ["trajectory_regime", "n_rows", "share"])

    threshold_rows: list[dict[str, Any]] = []
    for metric in ["norm_d25", "norm_d50", "norm_d75", "norm_delta_21", "norm_delta_250"]:
        s = summaries["pooled"][metric]
        threshold_rows.append(
            {
                "metric": metric,
                "n_valid": s.n_valid,
                "n_excluded": s.n_excluded,
                "median": s.median,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
            }
        )
    write_csv(TABLE_ROOT / "threshold_sensitivity.csv", threshold_rows, list(threshold_rows[0].keys()))

    qc_rows = qc_summary_rows(df)
    write_csv(TABLE_ROOT / "qc_summary.csv", qc_rows, ["item", "count", "scope"])

    counts_rows = [
        {
            "dataset": DATASET_DISPLAY[str(row.dataset_name)],
            "n_prompts": int(row.n_prompts),
            "role": "broad trajectory panel",
        }
        for row in summaries["dataset_counts"].itertuples(index=False)
    ]
    write_csv(TABLE_ROOT / "dataset_counts.csv", counts_rows, ["dataset", "n_prompts", "role"])

    drift_rows = summaries["drift_rows"]
    span_rows = summaries["span_rows"]
    sub_rows = summaries["substitution_rows"]
    mech_summary_rows = [
        {"item": "Instrumented models", "value": int(drift_rows["model_id"].nunique())},
        {
            "item": "Component-depth summary rows",
            "value": int(mech_df[mech_df["analysis"] == "component_depth_bin"].shape[0]),
        },
        {"item": "Minimum held-out drift R-squared", "value": float(drift_rows["test_r2"].min())},
        {"item": "Mean evidence-minus-distractor gap", "value": float(span_rows["gap_mean"].mean())},
        {
            "item": "All-pairs MLP substitution mean shift",
            "value": float(sub_rows.loc[sub_rows["component"] == "mlp", "mean_delta_shift"].iloc[0]),
        },
        {
            "item": "All-pairs attention substitution mean shift",
            "value": float(sub_rows.loc[sub_rows["component"] == "attention", "mean_delta_shift"].iloc[0]),
        },
    ]
    write_csv(TABLE_ROOT / "mechanistic_summary.csv", mech_summary_rows, ["item", "value"])

    write_latex_tables(model_rows, threshold_rows, qc_rows, counts_rows, regime_rows, mech_summary_rows)


def write_latex_tables(
    model_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    counts_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    mech_summary_rows: list[dict[str, Any]],
) -> None:
    design = r"""
\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Aspect} & \textbf{Release setting}\\
\midrule
Broad trajectory panel & Nine instruction-tuned open-weight checkpoints from Qwen 2.5 Instruct, Gemma 2 IT, and Llama Instruct on ARC-Challenge and CommonsenseQA. \\
Instrumented panel & Three 7--8B checkpoints on a 3,000-prompt MMLU cache with attention/MLP traces, span counterfactuals, and cached counterfactual accounting. \\
Primary readout & Exact-option continuation scores after the prompt suffix \texttt{Final answer:}. \\
Primary quantities & Leader, runner-up, entropy, soft margin, persistent contender depth $d_2$, persistent winner depth $d_1$, persistent 50\% final-gap depth $d_{50}$, late write, and trajectory regimes. \\
Main claim type & Cached readout-space dynamics plus mechanistic accounting of margin drift; no full circuit-discovery claim. \\
\bottomrule
\end{tabularx}
"""
    write_latex_table(PAPER_ROOT / "tables" / "table1_design_summary.tex", design)

    lines = [
        r"\begin{tabularx}{\linewidth}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Valid} & \textbf{Median $\Delta_{250}$} & \textbf{95\% CI} & \textbf{Write after $d_2$} \\",
        r"\midrule",
    ]
    for row in model_rows:
        lines.append(
            f"{latex_escape(row['model_display'])} & "
            f"{row['valid_delta_250']} & "
            f"{fmt(row['median_delta_250'], 2)} & "
            f"[{fmt(row['delta_250_ci_low'], 2)}, {fmt(row['delta_250_ci_high'], 2)}] & "
            f"{pct(row['median_write_after_d2'], 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    write_latex_table(PAPER_ROOT / "tables" / "table2_model_summary.tex", "\n".join(lines))

    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"\textbf{Trajectory regime} & \textbf{Rows} & \textbf{Share} \\",
        r"\midrule",
    ]
    for row in regime_rows:
        lines.append(f"{latex_escape(row['trajectory_regime'])} & {row['n_rows']} & {pct(row['share'], 0)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(PAPER_ROOT / "tables" / "table3_regime_summary.tex", "\n".join(lines))

    lines = [
        r"\begin{tabularx}{\linewidth}{@{}lX@{}}",
        r"\toprule",
        r"\textbf{Mechanistic artifact} & \textbf{Value} \\",
        r"\midrule",
    ]
    for row in mech_summary_rows:
        value = row["value"]
        rendered = fmt(value, 2) if isinstance(value, float) else str(value)
        lines.append(f"{latex_escape(row['item'])} & {rendered} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    write_latex_table(PAPER_ROOT / "tables" / "table4_mechanistic_summary.tex", "\n".join(lines))

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Valid} & \textbf{Median} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    label = {
        "norm_d25": r"$d_{25}$",
        "norm_d50": r"$d_{50}$",
        "norm_d75": r"$d_{75}$",
        "norm_delta_21": r"$\Delta_{21}$",
        "norm_delta_250": r"$\Delta_{250}$",
    }
    for row in threshold_rows:
        lines.append(
            f"{label[row['metric']]} & {row['n_valid']} & {fmt(row['median'], 2)} & "
            f"[{fmt(row['ci_low'], 2)}, {fmt(row['ci_high'], 2)}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(PAPER_ROOT / "tables" / "appendix_threshold_summary.tex", "\n".join(lines))

    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"\textbf{Dataset} & \textbf{Prompts} & \textbf{Role} \\", r"\midrule"]
    for row in counts_rows:
        lines.append(f"{latex_escape(row['dataset'])} & {row['n_prompts']} & {latex_escape(row['role'])} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(PAPER_ROOT / "tables" / "appendix_dataset_counts.tex", "\n".join(lines))

    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"\textbf{QC item} & \textbf{Count} \\", r"\midrule"]
    for row in qc_rows:
        lines.append(f"{latex_escape(row['item'])} & {row['count']} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(PAPER_ROOT / "tables" / "appendix_qc_summary.tex", "\n".join(lines))


def qc_summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    def count_bool(column: str) -> int:
        return int(df[column].fillna(False).astype(bool).sum())

    return [
        {"item": "Derived exact-option rows", "count": int(df.shape[0]), "scope": "all"},
        {"item": "Primary ARC/CommonsenseQA rows", "count": int(primary_df(df).shape[0]), "scope": "primary"},
        {"item": "Final-layer top ties", "count": count_bool("final_layer_top_tie"), "scope": "all"},
        {"item": "Rows excluded from normalized-gap metrics", "count": int(df["norm_d50"].isna().sum()), "scope": "all"},
        {"item": "Missing option scores", "count": count_bool("missing_option_scores"), "scope": "all"},
        {"item": "Missing options in any layer", "count": count_bool("missing_options_any_layer"), "scope": "all"},
        {"item": "Final winner mismatches", "count": count_bool("final_winner_mismatch"), "scope": "all"},
        {"item": "Nonmonotonic layer index rows", "count": count_bool("nonmonotonic_layer_index"), "scope": "all"},
        {"item": "Rows with only-final-layer d2", "count": count_bool("only_final_layer_d2"), "scope": "all"},
        {"item": "Rows with only-final-layer d50", "count": count_bool("only_final_layer_d50"), "scope": "all"},
    ]


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def build_figures(df: pd.DataFrame, summaries: dict[str, Any]) -> None:
    configure_matplotlib()
    p_df: pd.DataFrame = summaries["paper_df"]
    mech_df: pd.DataFrame = summaries["mechanistic_df"]
    fig_dir = PAPER_ROOT / "figures"

    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for metric, label, color in [
        ("norm_d2", r"$d_2$: persistent contender", COLORS["d2"]),
        ("norm_d1", r"$d_1$: persistent winner", COLORS["d1"]),
        ("norm_d50", r"$d_{50}$: persistent decisive margin", COLORS["d50"]),
    ]:
        x, y = ecdf(finite_values(p_df[metric]))
        ax.plot(x, y, lw=1.8, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Rows at or before depth")
    ax.set_title("Persistent events by depth", loc="left", color=COLORS["ink"], fontsize=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        frameon=False,
        fontsize=7.5,
        handlelength=2.0,
    )
    savefig(fig_dir / "figure1_pooled_cdfs.pdf")

    lag_rows = []
    for model in MODEL_ORDER:
        for metric, label in [("norm_delta_21", r"$\Delta_{21}$"), ("norm_delta_250", r"$\Delta_{250}$")]:
            s = summaries["model_summaries"][model][metric]
            lag_rows.append((model, label, s))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    for ax, metric, title, color in [
        (axes[0], "norm_delta_21", r"Persistent winner lag $\Delta_{21}$", COLORS["lag21"]),
        (axes[1], "norm_delta_250", r"Persistent margin lag $\Delta_{250}$", COLORS["lag250"]),
    ]:
        y = np.arange(len(MODEL_ORDER))
        med = [summaries["model_summaries"][m][metric].median for m in MODEL_ORDER]
        low = [summaries["model_summaries"][m][metric].ci_low for m in MODEL_ORDER]
        high = [summaries["model_summaries"][m][metric].ci_high for m in MODEL_ORDER]
        left = [m - l if m is not None and l is not None else 0 for m, l in zip(med, low, strict=True)]
        right = [h - m if m is not None and h is not None else 0 for m, h in zip(med, high, strict=True)]
        ax.errorbar(
            med,
            y,
            xerr=[left, right],
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=3,
            lw=1.1,
            ms=4.0,
        )
        ax.set_title(title, loc="left", color=COLORS["ink"], fontsize=10)
        ax.set_xlim(0, 0.26)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Normalized depth")
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=7)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=1.0)
    savefig(fig_dir / "figure2_lag_summary.pdf")
    build_staged_summary_figure(p_df, summaries, fig_dir / "figure1_staged_summary.pdf")

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.35), sharey=True)
    write_metrics = [
        ("write_share_after_d2", r"After $d_2$"),
        ("write_share_after_d1", r"After $d_1$"),
        ("write_share_last20", "Last 20% depth"),
    ]
    for ax, (metric, title) in zip(axes, write_metrics, strict=True):
        y = np.arange(len(MODEL_ORDER))
        med = [summaries["model_summaries"][m][metric].median for m in MODEL_ORDER]
        low = [summaries["model_summaries"][m][metric].ci_low for m in MODEL_ORDER]
        high = [summaries["model_summaries"][m][metric].ci_high for m in MODEL_ORDER]
        left = [m - l if m is not None and l is not None else 0 for m, l in zip(med, low, strict=True)]
        right = [h - m if m is not None and h is not None else 0 for m, h in zip(med, high, strict=True)]
        ax.errorbar(
            med,
            y,
            xerr=[left, right],
            fmt="o",
            color=COLORS["write"],
            ecolor=COLORS["write"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=3,
            lw=1.1,
            ms=4.2,
        )
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title, loc="left", color=COLORS["ink"], fontsize=9)
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=7)
    axes[0].invert_yaxis()
    fig.supxlabel("Share of positive winner-gap growth", y=0.01, fontsize=9)
    fig.tight_layout()
    savefig(fig_dir / "figure3_late_write.pdf")

    build_regime_figure(p_df, fig_dir / "figure4_regime_summary.pdf")
    build_mechanistic_figure(mech_df, fig_dir / "figure5_mechanistic_accounting.pdf")
    build_exemplar_figure(fig_dir / "figure4_exemplars.pdf")
    build_appendix_depth_figure(p_df, fig_dir / "appendix_threshold_cdfs.pdf")


def build_staged_summary_figure(df: pd.DataFrame, summaries: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.65),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )
    ax_cdf, ax_lag21, ax_lag250 = axes

    for metric, label, color in [
        ("norm_d2", r"$d_2$", COLORS["d2"]),
        ("norm_d1", r"$d_1$", COLORS["d1"]),
        ("norm_d50", r"$d_{50}$", COLORS["d50"]),
    ]:
        x, y = ecdf(finite_values(df[metric]))
        ax_cdf.plot(x, y, lw=1.7, color=color, label=label)
    ax_cdf.set_xlim(0, 1)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_cdf.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_cdf.set_xlabel("Normalized depth")
    ax_cdf.set_ylabel("Rows at or before depth")
    ax_cdf.set_title("A. Persistent events", loc="left", fontsize=9)
    ax_cdf.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        columnspacing=0.8,
    )

    for ax, metric, title, color in [
        (ax_lag21, "norm_delta_21", r"B. $\Delta_{21}$", COLORS["lag21"]),
        (ax_lag250, "norm_delta_250", r"C. $\Delta_{250}$", COLORS["lag250"]),
    ]:
        y = np.arange(len(MODEL_ORDER))
        med = [summaries["model_summaries"][m][metric].median for m in MODEL_ORDER]
        low = [summaries["model_summaries"][m][metric].ci_low for m in MODEL_ORDER]
        high = [summaries["model_summaries"][m][metric].ci_high for m in MODEL_ORDER]
        left = [m - l if m is not None and l is not None else 0 for m, l in zip(med, low, strict=True)]
        right = [h - m if m is not None and h is not None else 0 for m, h in zip(med, high, strict=True)]
        ax.errorbar(
            med,
            y,
            xerr=[left, right],
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=2.6,
            lw=1.0,
            ms=3.4,
        )
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(0, 0.26)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Normalized depth")
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=6.4)
    ax_lag21.invert_yaxis()
    ax_lag250.tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=0.7)
    savefig(path)


def build_regime_figure(df: pd.DataFrame, path: Path) -> None:
    counts = (
        df.groupby(["trajectory_regime"], observed=True)
        .size()
        .sort_values(ascending=True)
    )
    regime_colors = {
        "stable-correct": COLORS["green"],
        "stable-wrong": COLORS["d50"],
        "late-crystallizing": COLORS["orange"],
        "switch-heavy": COLORS["purple"],
        "unstable": COLORS["muted"],
    }
    colors = [regime_colors.get(str(idx), COLORS["muted"]) for idx in counts.index]
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    ax.barh(range(len(counts)), counts.values / counts.values.sum(), color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(idx) for idx in counts.index])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of broad trajectory rows")
    ax.set_title("Trajectory regimes in the broad exact-option panel", loc="left", color=COLORS["ink"])
    for pos, value in enumerate(counts.values / counts.values.sum()):
        ax.text(value + 0.01, pos, f"{value:.0%}", va="center", color=COLORS["ink"], fontsize=9)
    ax.set_xlim(0, max(counts.values / counts.values.sum()) * 1.18)
    savefig(path)


def build_mechanistic_figure(mech_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(3.35, 7.2))
    ax_r2, ax_trace, ax_span, ax_sub = axes

    drift = mech_df[mech_df["analysis"] == "drift_decomposition"].copy()
    model_labels = [str(x).split("/")[-1].replace("-Instruct", "") for x in drift["model_id"]]
    ax_r2.bar(model_labels, drift["test_r2"].astype(float), color=COLORS["d2"], edgecolor="black", linewidth=0.4)
    ax_r2.axhline(0.70, color=COLORS["muted"], ls="--", lw=1)
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("Held-out $R^2$")
    ax_r2.set_title("Drift decomposition passes held-out gates", loc="left", fontsize=10)
    ax_r2.tick_params(axis="x", labelrotation=18, labelsize=7)

    traces = mech_df[
        (mech_df["analysis"] == "component_depth_bin")
        & (mech_df["trajectory_type"].isin(["stable_correct", "stable_wrong"]))
        & (mech_df["component"].isin(["attention", "mlp"]))
    ].copy()
    for (trajectory_type, component), group in traces.groupby(["trajectory_type", "component"]):
        style = "-" if component == "attention" else "--"
        color = COLORS["d2"] if trajectory_type == "stable_correct" else COLORS["d50"]
        label = f"{trajectory_type.replace('_', '-')} {component}"
        grouped = group.groupby("depth_mid", as_index=False)["mean_value"].mean()
        ax_trace.plot(grouped["depth_mid"], grouped["mean_value"], style, color=color, lw=1.3, marker="o", ms=2.8, label=label)
    ax_trace.axhline(0, color=COLORS["muted"], lw=0.8)
    ax_trace.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_trace.set_xlabel("Normalized depth")
    ax_trace.set_ylabel("Mean contribution")
    ax_trace.set_title("Traces: blue correct, red wrong; solid attention, dashed MLP", loc="left", fontsize=8.2)

    span = mech_df[mech_df["analysis"] == "span_deletion_summary"].copy()
    ax_span.bar(
        span["span_label"].astype(str),
        span["mean_effect_delta"].astype(float),
        color=[COLORS["d2"], COLORS["d50"], COLORS["muted"]][: len(span)],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_span.axhline(0, color=COLORS["muted"], lw=0.8)
    ax_span.set_ylabel("Mean effect on margin")
    ax_span.set_title("Span deletion separates evidence and distractors", loc="left", fontsize=10)

    sub = mech_df[
        (mech_df["analysis"] == "substitution_sensitivity")
        & (mech_df["setting_group"] == "pairing_mode")
        & (mech_df["setting_name"].isin(["all_pairs_within_model", "legacy_first_per_model"]))
    ].copy()
    setting_labels = ["all-pairs", "legacy first"]
    setting_names = ["all_pairs_within_model", "legacy_first_per_model"]
    x = np.arange(len(setting_names), dtype=float)
    width = 0.34
    for offset, component, color in [(-width / 2, "attention", COLORS["d2"]), (width / 2, "mlp", COLORS["d50"])]:
        values = []
        for setting in setting_names:
            match = sub[(sub["setting_name"] == setting) & (sub["component"] == component)]
            values.append(float(match["mean_delta_shift"].iloc[0]))
        ax_sub.bar(x + offset, values, width=width, color=color, edgecolor="black", linewidth=0.4, label=component)
    ax_sub.axhline(0, color=COLORS["muted"], lw=0.8)
    ax_sub.set_xticks(x)
    ax_sub.set_xticklabels(setting_labels, fontsize=8)
    ax_sub.set_ylabel("Mean simulated shift")
    ax_sub.set_title("Counterfactual accounting is useful but protocol-sensitive", loc="left", fontsize=10)
    ax_sub.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout(h_pad=1.0)
    savefig(path)


def build_exemplar_figure(path: Path) -> None:
    rows = load_json(EXEMPLARS_PATH)
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.4, 6.2), sharex=True)
    if len(rows) == 1:
        axes = np.array([axes])
    for idx, row in enumerate(rows):
        depths = np.array(row["norm_depths"], dtype=float)
        ranks = np.array(row["ranks"], dtype=float)
        gap = np.array(row["norm_gap"], dtype=float)
        label = f"{row['selection_bucket'].title()} lag: {MODEL_DISPLAY.get(row['model_name'], row['model_name'])}"
        ax_rank, ax_gap = axes[idx]
        ax_rank.plot(depths, ranks, color=COLORS["rank"], lw=1.8)
        ax_rank.invert_yaxis()
        ax_rank.set_ylabel("Winner rank")
        ax_rank.set_title(label, loc="left", fontsize=9, color=COLORS["ink"])
        ax_rank.set_ylim(max(ranks) + 0.25, 0.75)
        ax_gap.plot(depths, gap, color=COLORS["gap"], lw=1.8)
        ax_gap.axhline(0.5, color=COLORS["muted"], ls="--", lw=0.9)
        ax_gap.axhline(1.0, color=COLORS["muted"], ls=":", lw=0.9)
        ax_gap.set_ylabel("Gap / final gap")
        lo = float(np.nanpercentile(gap, 2))
        hi = float(np.nanpercentile(gap, 98))
        ax_gap.set_ylim(min(-0.2, lo - 0.2), max(1.2, hi + 0.2))
    for ax in axes[-1]:
        ax.set_xlabel("Normalized depth")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    savefig(path)


def build_appendix_depth_figure(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for metric, label, color in [
        ("norm_d25", r"$d_{25}$", COLORS["green"]),
        ("norm_d50", r"$d_{50}$", COLORS["d50"]),
        ("norm_d75", r"$d_{75}$", COLORS["orange"]),
    ]:
        x, y = ecdf(finite_values(df[metric]))
        ax.plot(x, y, lw=1.7, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Rows at or before depth")
    ax.set_title("Persistent margin thresholds", loc="left", color=COLORS["ink"], fontsize=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        frameon=False,
        fontsize=7.5,
        handlelength=2.0,
    )
    savefig(path)


def build_macros(df: pd.DataFrame, summaries: dict[str, Any]) -> None:
    p_df: pd.DataFrame = summaries["paper_df"]
    mech_df: pd.DataFrame = summaries["mechanistic_df"]
    pooled = summaries["pooled"]
    model_delta = [summaries["model_summaries"][m]["norm_delta_250"].median for m in MODEL_ORDER]
    write_after = [summaries["model_summaries"][m]["write_share_after_d2"].median for m in MODEL_ORDER]
    dataset_counts = {
        str(row.dataset_name): int(row.n_prompts)
        for row in summaries["dataset_counts"].itertuples(index=False)
    }
    drift_rows = summaries["drift_rows"]
    span_rows = summaries["span_rows"]
    sub_rows = summaries["substitution_rows"]
    regime_counts = {str(row.trajectory_regime): int(row.n_rows) for row in summaries["regime_counts"].itertuples(index=False)}
    all_pairs_attn = float(sub_rows.loc[sub_rows["component"] == "attention", "mean_delta_shift"].iloc[0])
    all_pairs_mlp = float(sub_rows.loc[sub_rows["component"] == "mlp", "mean_delta_shift"].iloc[0])

    macros = [
        "% Auto-generated by scripts/rebuild_all.py. Do not edit by hand.",
        macro("ModelCount", str(len(MODEL_ORDER))),
        macro("PaperPromptCount", str(sum(dataset_counts.get(d, 0) for d in PRIMARY_DATASETS))),
        macro("ArcPromptCount", str(dataset_counts.get("ai2_arc_challenge", 0))),
        macro("CsqaPromptCount", str(dataset_counts.get("commonsenseqa_validation", 0))),
        macro("DerivedRowCount", f"{df.shape[0]:,}"),
        macro("PrimaryRowCount", f"{p_df.shape[0]:,}"),
        macro("MechanisticRowCount", f"{mech_df.shape[0]:,}"),
        macro("InstrumentedModelCount", str(int(drift_rows["model_id"].nunique()))),
        macro("InstrumentedPromptCount", "3,000"),
        macro("MechanisticMinHeldoutRtwo", fmt(float(drift_rows["test_r2"].min()), 2)),
        macro("MechanisticMaxHeldoutRtwo", fmt(float(drift_rows["test_r2"].max()), 2)),
        macro("SpanEvidenceDistractorGap", fmt(float(span_rows["gap_mean"].mean()), 2)),
        macro("SpanMinPValue", fmt(float(span_rows["p_value"].min()), 4)),
        macro("SubstitutionAttentionMean", fmt(all_pairs_attn, 2)),
        macro("SubstitutionMLPMean", fmt(all_pairs_mlp, 2)),
        macro("LateCrystallizingShare", pct(regime_counts.get("late-crystallizing", 0) / max(1, p_df.shape[0]), 0)),
        macro("SwitchHeavyShare", pct(regime_counts.get("switch-heavy", 0) / max(1, p_df.shape[0]), 0)),
        macro("PooledMedianDTwo", fmt(pooled["norm_d2"].median, 2)),
        macro("PooledMedianDOne", fmt(pooled["norm_d1"].median, 2)),
        macro("PooledMedianDFifty", fmt(pooled["norm_d50"].median, 2)),
        macro("PooledMedianDeltaTwentyOne", fmt(pooled["norm_delta_21"].median, 2)),
        macro("PooledDeltaTwentyOneLow", fmt(pooled["norm_delta_21"].ci_low, 2)),
        macro("PooledDeltaTwentyOneHigh", fmt(pooled["norm_delta_21"].ci_high, 2)),
        macro("PooledMedianDeltaTwentyFifty", fmt(pooled["norm_delta_250"].median, 2)),
        macro("PooledDeltaTwentyFiftyLow", fmt(pooled["norm_delta_250"].ci_low, 2)),
        macro("PooledDeltaTwentyFiftyHigh", fmt(pooled["norm_delta_250"].ci_high, 2)),
        macro("PooledDeltaTwoFiftyPct", pct(pooled["norm_delta_250"].median, 0)),
        macro("PooledWriteAfterDTwo", pct(pooled["write_share_after_d2"].median, 0)),
        macro("PooledWriteAfterDOne", pct(pooled["write_share_after_d1"].median, 0)),
        macro("PooledWriteLastTwenty", pct(pooled["write_share_last20"].median, 0)),
        macro("MinMedianDeltaTwoFifty", fmt(min(v for v in model_delta if v is not None), 2)),
        macro("MaxMedianDeltaTwoFifty", fmt(max(v for v in model_delta if v is not None), 2)),
        macro("MinWriteAfterDTwo", pct(min(v for v in write_after if v is not None), 0)),
        macro("MaxWriteAfterDTwo", pct(max(v for v in write_after if v is not None), 0)),
        macro("NormalizedGapExcludedCount", str(int(df["norm_d50"].isna().sum()))),
        macro("FinalLayerTieCount", str(int(df["final_layer_top_tie"].fillna(False).astype(bool).sum()))),
    ]
    path = PAPER_ROOT / "generated" / "paper_numbers.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(macros) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> None:
    paths: list[Path] = []
    for root in ["data", "paper", "src", "scripts", "tests", "docs", "qa", "arxiv_source"]:
        base = RELEASE_ROOT / root
        if not base.exists():
            continue
        paths.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    rows = []
    for path in sorted(paths):
        rel = path.relative_to(RELEASE_ROOT).as_posix()
        if rel.startswith("paper/build/") and rel != "paper/build/main.pdf":
            continue
        if "/build/" in f"/{rel}" and rel != "paper/build/main.pdf":
            continue
        if rel.startswith("paper/."):
            continue
        rows.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(MANIFEST_ROOT / "release_manifest.json", {"files": rows})


def scan_text() -> list[str]:
    failures: list[str] = []
    paper_files = list((PAPER_ROOT).rglob("*.tex")) + [PAPER_ROOT / "references.bib"]
    for path in paper_files:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                failures.append(f"{path.relative_to(RELEASE_ROOT)} contains forbidden pattern: {pattern}")
    macro_text = (PAPER_ROOT / "generated" / "paper_numbers.tex").read_text()
    if re.search(r"\bnan\b", macro_text, flags=re.IGNORECASE):
        failures.append("generated/paper_numbers.tex contains nan")
    return failures


def run_tectonic(*, offline: bool = False) -> None:
    build_dir = PAPER_ROOT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    args = ["tectonic", "--keep-logs", "--keep-intermediates", "--outdir", str(build_dir), "main.tex"]
    if offline:
        args.insert(1, "--only-cached")
    subprocess.run(args, cwd=PAPER_ROOT, check=True)


def render_pages() -> None:
    page_dir = QA_ROOT / "page_pngs"
    page_dir.mkdir(parents=True, exist_ok=True)
    for path in page_dir.glob("page-*.png"):
        path.unlink()
    pdf = PAPER_ROOT / "build" / "main.pdf"
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", str(pdf), str(page_dir / "page")], check=True)


def sync_arxiv_source() -> None:
    target = RELEASE_ROOT / "arxiv_source"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for rel in ["main.tex", "references.bib"]:
        shutil.copy2(PAPER_ROOT / rel, target / rel)
    for dirname in ["sections", "tables", "figures", "generated"]:
        src = PAPER_ROOT / dirname
        if src.exists():
            shutil.copytree(src, target / dirname)
    bbl = PAPER_ROOT / "build" / "main.bbl"
    if bbl.exists():
        shutil.copy2(bbl, target / "main.bbl")


def inspect_latex_log() -> list[str]:
    failures: list[str] = []
    log_path = PAPER_ROOT / "build" / "main.log"
    if not log_path.exists():
        return ["paper/build/main.log is missing"]
    text = log_path.read_text(errors="replace")
    for pattern in ["Overfull", "Undefined control sequence", "Citation .* undefined", "Reference .* undefined", "LaTeX Error"]:
        if re.search(pattern, text):
            failures.append(f"LaTeX log contains: {pattern}")
    return failures


def write_render_report(text_failures: list[str], log_failures: list[str]) -> None:
    page_dir = QA_ROOT / "page_pngs"
    pngs = sorted(page_dir.glob("page-*.png"))
    pdf = PAPER_ROOT / "build" / "main.pdf"
    lines = [
        "# Render QA Report",
        "",
        f"- PDF: `{pdf.relative_to(RELEASE_ROOT)}`",
        f"- Rendered pages: {len(pngs)}",
        f"- Text QA failures: {len(text_failures)}",
        f"- LaTeX log failures: {len(log_failures)}",
        "",
        "## Text QA",
    ]
    lines.extend([f"- FAIL: {failure}" for failure in text_failures] or ["- PASS"])
    lines.append("")
    lines.append("## LaTeX Log QA")
    lines.extend([f"- FAIL: {failure}" for failure in log_failures] or ["- PASS"])
    lines.append("")
    lines.append("## Visual QA")
    lines.append("- Rendered page PNGs are stored in `qa/page_pngs/` for manual inspection.")
    lines.append("- Manual inspection is required before release acceptance.")
    (QA_ROOT / "render_report.md").write_text("\n".join(lines) + "\n")


def rebuild_all(*, build_pdf: bool = True, offline: bool = False) -> None:
    for path in [TABLE_ROOT, PAPER_ROOT / "figures", PAPER_ROOT / "tables", PAPER_ROOT / "generated", QA_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    mech_df = load_mechanistic_dataframe()
    summaries = generate_summaries(df, mech_df)
    build_tables(df, summaries)
    build_figures(df, summaries)
    build_macros(df, summaries)
    text_failures = scan_text()
    if text_failures:
        write_render_report(text_failures, [])
        raise SystemExit("Text QA failed:\n" + "\n".join(text_failures))
    if build_pdf:
        run_tectonic(offline=offline)
        render_pages()
        log_failures = inspect_latex_log()
        write_render_report([], log_failures)
        if log_failures:
            raise SystemExit("LaTeX QA failed:\n" + "\n".join(log_failures))
    sync_arxiv_source()
    build_manifest()
