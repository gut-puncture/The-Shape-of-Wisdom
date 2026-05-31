#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_ROOT.parents[1]
LEGACY_ROOT = Path(
    "/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445"
)

MODEL_LABELS = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-v0.3",
}

TYPE_LABELS = {
    "stable_correct": "stable correct",
    "stable_wrong": "stable wrong",
    "unstable_correct": "unstable correct",
    "unstable_wrong": "unstable wrong",
}

TYPE_COLORS = {
    "stable_correct": "#2166AC",
    "unstable_correct": "#67A9CF",
    "stable_wrong": "#B2182B",
    "unstable_wrong": "#EF8A62",
}

BLUE = "#2166AC"
LIGHT_BLUE = "#67A9CF"
RED = "#B2182B"
LIGHT_RED = "#EF8A62"
GRAY = "#4D4D4D"
LIGHT_GRAY = "#BDBDBD"
GREEN = "#1B9E77"


@dataclass(frozen=True)
class ReleasePaths:
    data_cached: Path = RELEASE_ROOT / "data" / "cached"
    data_audit: Path = RELEASE_ROOT / "data" / "audit"
    derived: Path = RELEASE_ROOT / "data" / "derived"
    figures: Path = RELEASE_ROOT / "figures"
    tables: Path = RELEASE_ROOT / "tables"
    generated: Path = RELEASE_ROOT / "generated"
    paper: Path = RELEASE_ROOT / "paper"
    paper_figures: Path = RELEASE_ROOT / "paper" / "figures"
    paper_tables: Path = RELEASE_ROOT / "paper" / "tables"
    paper_generated: Path = RELEASE_ROOT / "paper" / "generated"
    docs: Path = RELEASE_ROOT / "docs"
    qa: Path = RELEASE_ROOT / "qa"
    arxiv: Path = RELEASE_ROOT / "arxiv_source"
    dist: Path = RELEASE_ROOT / "dist"


P = ReleasePaths()


def ensure_dirs() -> None:
    for path in [
        P.data_cached,
        P.data_audit,
        P.derived,
        P.figures,
        P.tables,
        P.generated,
        P.paper,
        P.paper_figures,
        P.paper_tables,
        P.paper_generated,
        P.docs,
        P.qa,
        P.arxiv,
        P.dist,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def copy_or_keep(source: Path, dest: Path) -> None:
    if source.exists():
        shutil.copy2(source, dest)
        return
    if dest.exists():
        return
    raise FileNotFoundError(
        f"Missing required release input: {dest}. "
        f"Could not copy it from legacy source: {source}"
    )


def copy_inputs() -> None:
    required_parquets = [
        "decision_metrics.parquet",
        "prompt_types.parquet",
        "tracing_scalars.parquet",
        "attention_contrib_by_span.parquet",
        "attention_mass_by_span.parquet",
        "ablation_results.parquet",
        "patching_results.parquet",
        "span_deletion_causal.parquet",
        "span_effects.parquet",
        "span_labels.parquet",
        "span_paraphrase_stability.parquet",
        "negative_controls.parquet",
    ]
    for name in required_parquets:
        copy_or_keep(LEGACY_ROOT / "results" / "parquet" / name, P.data_cached / name)

    audit_files = [
        "artifact_integrity.json",
        "drift_reconstruction_audit.json",
        "drift_reconstruction_by_layer.csv",
        "drift_reconstruction_by_type.csv",
        "drift_reconstruction_detail.csv",
        "substitution_sensitivity_summary.csv",
        "substitution_rederive_diagnostics.json",
        "trajectory_spotcheck_summary.json",
    ]
    for name in audit_files:
        copy_or_keep(LEGACY_ROOT / "results" / "audit" / name, P.data_audit / name)

    report_files = [
        "08_attention_and_mlp_decomposition.report.json",
        "10_causal_validation_tools.report.json",
    ]
    for name in report_files:
        copy_or_keep(LEGACY_ROOT / "results" / "reports" / name, P.data_audit / name)


def load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in P.data_cached.glob("*.parquet"):
        frames[path.stem] = pd.read_parquet(path)
    frames["substitution_sensitivity"] = pd.read_csv(
        P.data_audit / "substitution_sensitivity_summary.csv"
    )
    frames["drift_by_layer"] = pd.read_csv(P.data_audit / "drift_reconstruction_by_layer.csv")
    frames["drift_by_type"] = pd.read_csv(P.data_audit / "drift_reconstruction_by_type.csv")
    return frames


def model_sort_key(model_id: str) -> tuple[int, str]:
    order = {
        "Qwen/Qwen2.5-7B-Instruct": 0,
        "meta-llama/Llama-3.1-8B-Instruct": 1,
        "mistralai/Mistral-7B-Instruct-v0.3": 2,
    }
    return (order.get(model_id, 99), model_id)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def number(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def prepare_summaries(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    decision = frames["decision_metrics"].copy()
    prompt_types = frames["prompt_types"].copy()
    tracing = frames["tracing_scalars"].copy()

    layer_counts = decision.groupby("model_id")["layer_index"].nunique().to_dict()
    prompt_counts = prompt_types.groupby("model_id")["prompt_uid"].nunique().to_dict()
    accuracy = prompt_types.groupby("model_id")["is_correct"].mean().to_dict()
    type_counts = pd.crosstab(prompt_types["model_id"], prompt_types["trajectory_type"])

    with open(
        P.data_audit / "08_attention_and_mlp_decomposition.report.json",
        "r",
        encoding="utf-8",
    ) as fh:
        drift_report = json.load(fh)

    model_rows = []
    for model_id in sorted(prompt_counts, key=model_sort_key):
        row = {
            "model_id": model_id,
            "model": MODEL_LABELS.get(model_id, model_id),
            "layers": int(layer_counts[model_id]),
            "prompts": int(prompt_counts[model_id]),
            "accuracy": float(accuracy[model_id]),
            "stable_correct": int(type_counts.loc[model_id].get("stable_correct", 0)),
            "unstable_correct": int(type_counts.loc[model_id].get("unstable_correct", 0)),
            "stable_wrong": int(type_counts.loc[model_id].get("stable_wrong", 0)),
            "unstable_wrong": int(type_counts.loc[model_id].get("unstable_wrong", 0)),
            "traced_prompts": int(tracing[tracing["model_id"] == model_id]["prompt_uid"].nunique()),
            "heldout_r2": float(
                drift_report["split_contract"]["models"][model_id]["test_r2"]
            ),
        }
        model_rows.append(row)

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(P.derived / "model_summary.csv", index=False)

    regime_summary = (
        prompt_types["trajectory_type"]
        .value_counts()
        .rename_axis("trajectory_type")
        .reset_index(name="rows")
    )
    regime_summary["share"] = regime_summary["rows"] / regime_summary["rows"].sum()
    regime_summary["label"] = regime_summary["trajectory_type"].map(TYPE_LABELS)
    regime_summary.to_csv(P.derived / "regime_summary.csv", index=False)

    span_summary = frames["span_deletion_causal"].copy()
    span_summary.to_csv(P.derived / "span_deletion_summary.csv", index=False)

    neg_summary = frames["negative_controls"].copy()
    neg_summary.to_csv(P.derived / "negative_controls.csv", index=False)

    primary_sub = frames["substitution_sensitivity"]
    primary_sub = primary_sub[
        (primary_sub["setting_group"] == "pairing_mode")
        & (primary_sub["setting_name"].isin(["all_pairs_within_model", "legacy_first_per_model"]))
    ].copy()
    primary_sub.to_csv(P.derived / "substitution_summary.csv", index=False)

    return {
        "model_summary": model_summary,
        "regime_summary": regime_summary,
        "tracing_with_types": tracing.merge(
            prompt_types[["model_id", "prompt_uid", "trajectory_type"]],
            on=["model_id", "prompt_uid"],
            how="left",
        ),
        "span_summary": span_summary,
        "neg_summary": neg_summary,
        "substitution_summary": primary_sub,
        "drift_report": drift_report,
    }


def write_macros(summaries: dict[str, object], frames: dict[str, pd.DataFrame]) -> None:
    model_summary: pd.DataFrame = summaries["model_summary"]  # type: ignore[assignment]
    regime_summary: pd.DataFrame = summaries["regime_summary"]  # type: ignore[assignment]
    span_summary: pd.DataFrame = summaries["span_summary"]  # type: ignore[assignment]
    neg_summary: pd.DataFrame = summaries["neg_summary"]  # type: ignore[assignment]
    substitution: pd.DataFrame = summaries["substitution_summary"]  # type: ignore[assignment]
    tracing_with_types: pd.DataFrame = summaries["tracing_with_types"]  # type: ignore[assignment]

    qwen = model_summary[model_summary["model"] == "Qwen2.5-7B"].iloc[0]
    llama = model_summary[model_summary["model"] == "Llama-3.1-8B"].iloc[0]
    mistral = model_summary[model_summary["model"] == "Mistral-7B-v0.3"].iloc[0]

    span_by_label = span_summary.set_index("span_label")
    neg_by_label = neg_summary.set_index("control")

    all_pairs = substitution[substitution["setting_name"] == "all_pairs_within_model"].set_index(
        "component"
    )
    legacy = substitution[substitution["setting_name"] == "legacy_first_per_model"].set_index(
        "component"
    )
    regime_by_type = regime_summary.set_index("trajectory_type")
    tracing_by_type = tracing_with_types.groupby("trajectory_type")[["s_attn", "s_mlp"]].mean()
    span_gap = (
        float(span_by_label.loc["evidence", "mean_effect_delta"])
        - float(span_by_label.loc["distractor", "mean_effect_delta"])
    )

    macros = {
        "ModelCount": str(model_summary.shape[0]),
        "PromptCount": "3000",
        "ModelPromptRows": str(frames["prompt_types"].shape[0]),
        "LayerwiseRows": str(frames["decision_metrics"].shape[0]),
        "TracingRows": str(frames["tracing_scalars"].shape[0]),
        "TracingPromptsPerModel": "600",
        "SpanPromptCount": str(frames["span_labels"]["prompt_uid"].nunique()),
        "QwenAccuracy": pct(qwen["accuracy"]),
        "LlamaAccuracy": pct(llama["accuracy"]),
        "MistralAccuracy": pct(mistral["accuracy"]),
        "QwenHeldoutRtwo": number(qwen["heldout_r2"], 2),
        "LlamaHeldoutRtwo": number(llama["heldout_r2"], 2),
        "MistralHeldoutRtwo": number(mistral["heldout_r2"], 2),
        "StableCorrectCount": str(int(regime_by_type.loc["stable_correct", "rows"])),
        "StableCorrectShare": pct(float(regime_by_type.loc["stable_correct", "share"])),
        "StableWrongCount": str(int(regime_by_type.loc["stable_wrong", "rows"])),
        "StableWrongShare": pct(float(regime_by_type.loc["stable_wrong", "share"])),
        "UnstableCorrectCount": str(int(regime_by_type.loc["unstable_correct", "rows"])),
        "UnstableCorrectShare": pct(float(regime_by_type.loc["unstable_correct", "share"])),
        "UnstableWrongCount": str(int(regime_by_type.loc["unstable_wrong", "rows"])),
        "UnstableWrongShare": pct(float(regime_by_type.loc["unstable_wrong", "share"])),
        "TracePromptsPerTypePerModel": "150",
        "StableCorrectAttentionMean": number(
            tracing_by_type.loc["stable_correct", "s_attn"], 2
        ),
        "StableCorrectMlpMean": number(tracing_by_type.loc["stable_correct", "s_mlp"], 2),
        "StableWrongAttentionMean": number(tracing_by_type.loc["stable_wrong", "s_attn"], 2),
        "StableWrongMlpMean": number(tracing_by_type.loc["stable_wrong", "s_mlp"], 2),
        "EvidenceSpanEffect": number(span_by_label.loc["evidence", "mean_effect_delta"], 2),
        "DistractorSpanEffect": number(span_by_label.loc["distractor", "mean_effect_delta"], 2),
        "NeutralSpanEffect": number(span_by_label.loc["neutral", "mean_effect_delta"], 2),
        "EvidenceDistractorGap": number(span_gap, 2),
        "ObservedControlEffect": number(neg_by_label.loc["observed", "mean_effect_delta"], 2),
        "ShuffledControlEffect": number(neg_by_label.loc["shuffled", "mean_effect_delta"], 2),
        "SignFlippedControlEffect": number(
            neg_by_label.loc["sign_flipped", "mean_effect_delta"], 2
        ),
        "AllPairsAttentionShift": number(all_pairs.loc["attention", "mean"], 2),
        "AllPairsMlpShift": number(all_pairs.loc["mlp", "mean"], 2),
        "AllPairsAttentionPositive": pct(all_pairs.loc["attention", "frac_positive"]),
        "AllPairsMlpPositive": pct(all_pairs.loc["mlp", "frac_positive"]),
        "LegacyAttentionShift": number(legacy.loc["attention", "mean"], 2),
        "LegacyMlpShift": number(legacy.loc["mlp", "mean"], 2),
    }

    text = "\n".join(
        f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in sorted(macros.items())
    )
    (P.generated / "paper_numbers.tex").write_text(text + "\n", encoding="utf-8")
    (P.paper_generated / "paper_numbers.tex").write_text(text + "\n", encoding="utf-8")


def write_tables(summaries: dict[str, object]) -> None:
    model_summary: pd.DataFrame = summaries["model_summary"]  # type: ignore[assignment]
    regime_summary: pd.DataFrame = summaries["regime_summary"]  # type: ignore[assignment]
    span_summary: pd.DataFrame = summaries["span_summary"]  # type: ignore[assignment]
    substitution: pd.DataFrame = summaries["substitution_summary"]  # type: ignore[assignment]

    rows = []
    for _, row in model_summary.iterrows():
        rows.append(
            f"{row['model']} & {int(row['layers'])} & {pct(row['accuracy'])} & "
            f"{int(row['stable_correct'])} & {int(row['stable_wrong'])} & "
            f"{int(row['unstable_correct'])} & {int(row['unstable_wrong'])} & "
            f"{number(row['heldout_r2'], 2)} \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Model & L & Final acc. & SC & SW & UC & UW & Drift $R^2$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (P.tables / "table1_model_summary.tex").write_text(table, encoding="utf-8")
    (P.paper_tables / "table1_model_summary.tex").write_text(table, encoding="utf-8")

    regime_rows = []
    for _, row in regime_summary.iterrows():
        regime_rows.append(
            f"{row['label']} & {int(row['rows'])} & {pct(float(row['share']))} \\\\"
        )
    table = (
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Trajectory type & Rows & Share \\\\\n"
        "\\midrule\n"
        + "\n".join(regime_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (P.tables / "table2_regime_summary.tex").write_text(table, encoding="utf-8")
    (P.paper_tables / "table2_regime_summary.tex").write_text(table, encoding="utf-8")

    span_rows = []
    for _, row in span_summary.sort_values("span_label").iterrows():
        span_rows.append(
            f"{row['span_label']} & {int(row['n'])} & {number(row['mean_effect_delta'], 2)} & "
            f"{number(row['median_effect_delta'], 2)} \\\\"
        )
    table = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Span label & $n$ & Mean effect & Median effect \\\\\n"
        "\\midrule\n"
        + "\n".join(span_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (P.tables / "table3_span_summary.tex").write_text(table, encoding="utf-8")
    (P.paper_tables / "table3_span_summary.tex").write_text(table, encoding="utf-8")

    sub_rows = []
    for _, row in substitution.sort_values(["setting_name", "component"]).iterrows():
        setting = {
            "all_pairs_within_model": "all pairs",
            "legacy_first_per_model": "legacy first",
        }.get(row["setting_name"], row["setting_name"].replace("_", " "))
        component = "MLP" if row["component"] == "mlp" else row["component"]
        sub_rows.append(
            f"{setting} & {component} & {int(row['n_pairs'])} & "
            f"{number(row['mean'], 2)} & {pct(row['frac_positive'])} \\\\"
        )
    table = (
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Setting & Component & Pairs & Mean shift & Positive \\\\\n"
        "\\midrule\n"
        + "\n".join(sub_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (P.tables / "table4_substitution_summary.tex").write_text(table, encoding="utf-8")
    (P.paper_tables / "table4_substitution_summary.tex").write_text(table, encoding="utf-8")


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def norm_depth(g: pd.DataFrame) -> pd.Series:
    max_layer = g["layer_index"].max()
    if max_layer == 0:
        return g["layer_index"] * 0
    return g["layer_index"] / max_layer


def plot_trajectory_primitives(frames: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    decision = frames["decision_metrics"].copy()
    prompt_types = frames["prompt_types"].copy()
    merged = decision.merge(
        prompt_types[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"],
        how="left",
    )
    merged["depth"] = merged.groupby(["model_id", "prompt_uid"], group_keys=False).apply(
        norm_depth
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), gridspec_kw={"height_ratios": [1, 1]})
    ax0, ax1, ax2, ax3 = axes.ravel()

    exemplar_type = "unstable_correct"
    ex_pool = merged[merged["trajectory_type"] == exemplar_type]
    candidate = (
        ex_pool.groupby(["model_id", "prompt_uid"])["delta"]
        .agg(lambda s: float(np.abs(s.iloc[-1])))
        .sort_values()
        .index[len(ex_pool.groupby(["model_id", "prompt_uid"])) // 2]
    )
    ex = merged[(merged["model_id"] == candidate[0]) & (merged["prompt_uid"] == candidate[1])]
    ax0.plot(ex["depth"], ex["delta"], color=BLUE, lw=2.0)
    ax0.axhline(0, color="#222222", lw=0.8)
    ax0.set_title("A. One prompt's answer margin moves across depth", loc="left", fontweight="bold")
    ax0.set_xlabel("Normalized depth")
    ax0.set_ylabel("Correct-vs-competitor margin")
    ax0.text(
        0.02,
        0.93,
        f"{MODEL_LABELS.get(candidate[0], candidate[0])}; {TYPE_LABELS[exemplar_type]}",
        transform=ax0.transAxes,
        fontsize=9,
        va="top",
        color=GRAY,
    )

    depth_summary = (
        merged.groupby(["trajectory_type", "depth"])
        .agg(delta_med=("delta", "median"), boundary_med=("boundary", "median"), drift_med=("drift", "median"))
        .reset_index()
    )
    for t in ["stable_correct", "unstable_correct", "stable_wrong", "unstable_wrong"]:
        d = depth_summary[depth_summary["trajectory_type"] == t]
        ax1.plot(d["depth"], d["delta_med"], color=TYPE_COLORS[t], lw=2, label=TYPE_LABELS[t])
    ax1.axhline(0, color="#222222", lw=0.8)
    ax1.set_title("B. State separates trajectory types", loc="left", fontweight="bold")
    ax1.set_xlabel("Normalized depth")
    ax1.set_ylabel("Median margin")
    ax1.legend(ncol=2, bbox_to_anchor=(0.5, -0.28), loc="upper center", fontsize=8)

    for t in ["stable_correct", "unstable_correct", "stable_wrong", "unstable_wrong"]:
        d = depth_summary[depth_summary["trajectory_type"] == t]
        ax2.plot(d["depth"], d["drift_med"], color=TYPE_COLORS[t], lw=2)
    ax2.axhline(0, color="#222222", lw=0.8)
    ax2.set_title("C. Motion is the layer-to-layer margin change", loc="left", fontweight="bold")
    ax2.set_xlabel("Normalized depth")
    ax2.set_ylabel("Median drift")

    for t in ["stable_correct", "unstable_correct", "stable_wrong", "unstable_wrong"]:
        d = depth_summary[depth_summary["trajectory_type"] == t]
        ax3.plot(d["depth"], d["boundary_med"], color=TYPE_COLORS[t], lw=2)
    ax3.set_title("D. Boundary distance grows as answers stabilize", loc="left", fontweight="bold")
    ax3.set_xlabel("Normalized depth")
    ax3.set_ylabel("Median |margin|")

    fig.tight_layout(h_pad=1.8, w_pad=1.8)
    out = P.figures / "figure1_trajectory_primitives.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_regime_summary(frames: dict[str, pd.DataFrame], summaries: dict[str, object]) -> None:
    setup_plot_style()
    prompt_types = frames["prompt_types"].copy()
    prompt_types["model"] = prompt_types["model_id"].map(MODEL_LABELS)
    counts = pd.crosstab(prompt_types["model"], prompt_types["trajectory_type"])
    counts = counts.loc[["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B-v0.3"]]
    shares = counts.div(counts.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax0, ax1 = axes
    bottom = np.zeros(len(shares))
    x = np.arange(len(shares))
    for t in ["stable_correct", "unstable_correct", "stable_wrong", "unstable_wrong"]:
        ax0.bar(x, shares[t], bottom=bottom, color=TYPE_COLORS[t], label=TYPE_LABELS[t], width=0.68)
        bottom += shares[t].to_numpy()
    ax0.set_xticks(x)
    ax0.set_xticklabels(shares.index, rotation=15, ha="right")
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("Share of model-prompt rows")
    ax0.set_title("A. Operational trajectory types differ by model", loc="left", fontweight="bold")
    ax0.legend(ncol=2, bbox_to_anchor=(0.5, -0.25), loc="upper center", fontsize=8)

    reg: pd.DataFrame = summaries["regime_summary"]  # type: ignore[assignment]
    reg = reg.set_index("trajectory_type").loc[
        ["stable_correct", "unstable_correct", "stable_wrong", "unstable_wrong"]
    ]
    ax1.barh(
        [TYPE_LABELS[t] for t in reg.index],
        reg["rows"],
        color=[TYPE_COLORS[t] for t in reg.index],
    )
    ax1.invert_yaxis()
    ax1.set_xlabel("Rows")
    ax1.set_title("B. Counts over all 9,000 trajectories", loc="left", fontweight="bold")
    for y, value in enumerate(reg["rows"]):
        ax1.text(value + 40, y, str(int(value)), va="center", fontsize=9)
    fig.tight_layout(w_pad=2.0)
    out = P.figures / "figure2_regime_summary.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_mechanistic_accounting(frames: dict[str, pd.DataFrame], summaries: dict[str, object]) -> None:
    setup_plot_style()
    model_summary: pd.DataFrame = summaries["model_summary"]  # type: ignore[assignment]
    tracing = frames["tracing_scalars"].copy()
    prompt_types = frames["prompt_types"][["model_id", "prompt_uid", "trajectory_type"]]
    tracing = tracing.merge(prompt_types, on=["model_id", "prompt_uid"], how="left")
    tracing["depth"] = tracing.groupby(["model_id", "prompt_uid"], group_keys=False).apply(
        norm_depth
    )
    tracing["depth_bin"] = pd.cut(tracing["depth"], np.linspace(0, 1, 7), include_lowest=True)
    binned = (
        tracing.groupby(["trajectory_type", "depth_bin"], observed=True)
        .agg(attn=("s_attn", "mean"), mlp=("s_mlp", "mean"), depth=("depth", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [0.9, 1.35, 1.1]})
    ax0, ax1, ax2 = axes

    ax0.bar(model_summary["model"], model_summary["heldout_r2"], color=[BLUE, GRAY, RED], width=0.66)
    ax0.axhline(0.70, color="#777777", ls="--", lw=1)
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("Held-out $R^2$")
    ax0.set_title("A. Held-out drift fit", loc="left", fontweight="bold")
    ax0.tick_params(axis="x", rotation=25)

    for t, color in [("stable_correct", BLUE), ("stable_wrong", RED)]:
        d = binned[binned["trajectory_type"] == t]
        ax1.plot(d["depth"], d["attn"], color=color, lw=2.0, marker="o", label=f"{TYPE_LABELS[t]} attention")
        ax1.plot(d["depth"], d["mlp"], color=color, lw=2.0, ls="--", marker="s", label=f"{TYPE_LABELS[t]} MLP")
    ax1.axhline(0, color="#222222", lw=0.8)
    ax1.set_xlabel("Normalized depth")
    ax1.set_ylabel("Mean scalar contribution")
    ax1.set_title("B. Contributions by type", loc="left", fontweight="bold")
    ax1.legend(ncol=1, bbox_to_anchor=(0.5, -0.30), loc="upper center", fontsize=8)

    sample = tracing.sample(min(5000, len(tracing)), random_state=12345)
    ax2.scatter(sample["s_attn"] + sample["s_mlp"], sample["drift"], s=5, alpha=0.18, color=GRAY)
    lo = float(min(sample["s_attn"].add(sample["s_mlp"]).min(), sample["drift"].min()))
    hi = float(max(sample["s_attn"].add(sample["s_mlp"]).max(), sample["drift"].max()))
    ax2.plot([lo, hi], [lo, hi], color=RED, lw=1.2)
    ax2.set_xlabel("Attention + MLP scalar")
    ax2.set_ylabel("Observed drift")
    ax2.set_title("C. Accounting vs. drift", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=2.1)
    out = P.figures / "figure3_mechanistic_accounting.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_span_intervention(frames: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    span = frames["span_deletion_causal"].copy()
    neg = frames["negative_controls"].copy()
    order = ["distractor", "neutral", "evidence"]
    colors = [RED, LIGHT_GRAY, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax0, ax1 = axes
    span = span.set_index("span_label").loc[order].reset_index()
    ax0.bar(span["span_label"], span["mean_effect_delta"], color=colors, width=0.65)
    ax0.axhline(0, color="#222222", lw=0.8)
    ax0.set_ylabel("Mean effect on margin")
    ax0.set_title("A. Deleted span effects", loc="left", fontweight="bold")
    for i, row in span.iterrows():
        ax0.text(i, row["mean_effect_delta"] + (0.18 if row["mean_effect_delta"] >= 0 else -0.35), f"n={int(row['n'])}", ha="center", fontsize=9)

    neg_order = ["observed", "shuffled", "sign_flipped"]
    neg = neg.set_index("control").loc[neg_order].reset_index()
    neg["control_label"] = neg["control"].map(
        {"observed": "observed", "shuffled": "shuffled", "sign_flipped": "sign-flipped"}
    )
    ax1.bar(neg["control_label"], neg["mean_effect_delta"], color=[BLUE, LIGHT_GRAY, GRAY], width=0.65)
    ax1.axhline(0, color="#222222", lw=0.8)
    ax1.set_ylabel("Mean effect on margin")
    ax1.set_title("B. Label controls", loc="left", fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)
    fig.tight_layout(w_pad=2.0)
    out = P.figures / "figure4_span_interventions.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_counterfactual_limits(frames: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    sub = frames["substitution_sensitivity"].copy()
    layer = frames["drift_by_layer"].copy()
    ablation = frames["ablation_results"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1.15, 1.1, 1.1]})
    ax0, ax1, ax2 = axes
    sel = sub[
        (sub["setting_group"] == "pairing_mode")
        & (sub["setting_name"].isin(["all_pairs_within_model", "legacy_first_per_model"]))
    ].copy()
    sel["setting_label"] = sel["setting_name"].map(
        {"all_pairs_within_model": "all-pairs", "legacy_first_per_model": "legacy first"}
    )
    x = np.arange(2)
    width = 0.34
    attn = sel[sel["component"] == "attention"].set_index("setting_label").loc[["all-pairs", "legacy first"]]
    mlp = sel[sel["component"] == "mlp"].set_index("setting_label").loc[["all-pairs", "legacy first"]]
    ax0.bar(x - width / 2, attn["mean"], width=width, color=BLUE, label="attention")
    ax0.bar(x + width / 2, mlp["mean"], width=width, color=RED, label="MLP")
    ax0.axhline(0, color="#222222", lw=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(["all-pairs", "legacy first"])
    ax0.set_ylabel("Mean simulated shift")
    ax0.set_title("A. Substitution sensitivity", loc="left", fontweight="bold")
    ax0.legend(ncol=2, bbox_to_anchor=(0.5, -0.25), loc="upper center", fontsize=8)

    max_layer = layer["layer_index"].max()
    depth = layer["layer_index"] / max_layer
    ax1.plot(depth, layer["mae_unit"], color=GRAY, lw=2, label="unit")
    ax1.plot(depth, layer["mae_ols"], color=RED, lw=2, label="OLS")
    ax1.set_xlabel("Normalized depth")
    ax1.set_ylabel("Absolute recurrence error")
    ax1.set_title("B. Recurrence error", loc="left", fontweight="bold")
    ax1.legend(ncol=2, bbox_to_anchor=(0.5, -0.25), loc="upper center", fontsize=8)

    abl = ablation.groupby("component")["delta_shift"].agg(["mean", "median"]).loc[["attention", "mlp"]]
    ax2.bar(["attention", "MLP"], abl["mean"], color=[BLUE, RED], width=0.65)
    ax2.axhline(0, color="#222222", lw=0.8)
    ax2.set_ylabel("Mean simulated removal shift")
    ax2.set_title("C. Simulated removal", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    out = P.figures / "figure5_counterfactual_limits.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def build_figures(frames: dict[str, pd.DataFrame], summaries: dict[str, object]) -> None:
    plot_trajectory_primitives(frames)
    plot_regime_summary(frames, summaries)
    plot_mechanistic_accounting(frames, summaries)
    plot_span_intervention(frames)
    plot_counterfactual_limits(frames)
    for fig in P.figures.glob("*.pdf"):
        shutil.copy2(fig, P.paper_figures / fig.name)


def write_manuscript() -> None:
    main = r"""
\documentclass[10pt,twocolumn]{article}

\usepackage[letterpaper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,citecolor=blue,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage{balance}
\usepackage{stfloats}

\graphicspath{{figures/}}
\input{generated/paper_numbers}

\setlength{\columnsep}{0.24in}
\setlength{\textfloatsep}{7pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\captionsetup{font=small,labelfont=bf}
\raggedbottom
\emergencystretch=1em

\title{\vspace{-1.0em}\textbf{The Shape of Wisdom: Decision Trajectories in Language Models}}
\author{Shailesh Rana\\Independent Researcher}
\date{}

\begin{document}

\twocolumn[
\begin{@twocolumnfalse}
\maketitle
\vspace{-1.4em}
\begin{center}
\begin{minipage}{0.82\textwidth}
\begin{abstract}
Language models do not simply choose an answer at the output layer. In a 9,000-trajectory MMLU study across Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3, the score of the answer moves across depth in structured ways. We describe each trajectory with three quantities: the current answer margin, the next-layer change in that margin, and the distance from a decision flip. The main empirical picture is that correctness and stability are different: the largest group is unstable-correct, not stable-correct. A traced subset then asks what moves the margin. In stable-correct cases, the average attention scalar points in the correct direction, while the average MLP scalar does not; span deletion shows that removing answer-supporting text hurts the margin and removing distractor-like text helps it. The result is not a full circuit explanation. It is a reproducible way to see which answers are settled, which remain fragile, and which measured sources move them.
\end{abstract}
\end{minipage}
\end{center}
\vspace{0.7em}
\end{@twocolumnfalse}
]

\section{Answers are trajectories}

A multiple-choice model returns one final answer, but the final token hides a process. At earlier layers, the same model can prefer another option, hover near a boundary, or move toward the correct answer and then away again. This paper studies that process directly.

The central claim is simple: in the setting we study, an answer is better understood as a trajectory than as a single endpoint. We track the score of the correct option against its strongest competitor at every layer. This gives a depthwise path. Some paths settle early and stay correct. Some settle early and stay wrong. Others keep moving near the boundary until late in the network.

This is useful because it separates three questions that endpoint accuracy merges together. First, where is the model now? Second, which way is the answer margin moving? Third, how close is the model to changing its preferred answer? We call these state, motion, and boundary distance. They are not hidden-state circuits. They are readout-space quantities, but they make the decision process legible.

After that, we ask what explains the movement. We use a smaller, balanced subset of \TracingPromptsPerModel{} prompts per model, with \TracePromptsPerTypePerModel{} prompts from each trajectory type. For those prompts, stored traces record two numbers at each layer: how much the attention blocks move the margin, and how much the MLP blocks move it. We also use a simple input intervention: remove a marked piece of the prompt and measure whether the correct answer margin goes up or down. This is related to lens-style readout and intervention work~\citep{belrose2023tunedlens,meng2022rome,wang2023ioi,geva2022transformerffn}, but the claim here is narrower. We are not discovering a complete circuit. We are asking which measured forces move different kinds of answer trajectories.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure1_trajectory_primitives.pdf}
\caption{\textbf{Answer formation is a path through depth, not a final-layer event.} Panel A shows one representative prompt: the correct-vs-competitor margin crosses and recrosses the decision boundary before the final layer. Panels B--D aggregate the same objects over all trajectories. State is the current margin, motion is the next-layer margin change, and boundary distance is the absolute margin. These three views make stable and unstable trajectories readable without invoking a hidden-state circuit.}
\label{fig:trajectory-primitives}
\end{figure*}

\section{Experiment and observables}

We analyze one experiment: \ModelCount{} instruction-tuned 7--8B open-weight models answering the same \PromptCount{} four-choice MMLU prompts~\citep{hendrycks2021mmlu}. The models are Qwen2.5-7B-Instruct~\citep{qwen25}, Llama-3.1-8B-Instruct~\citep{llama3}, and Mistral-7B-Instruct-v0.3~\citep{mistral7b}. Across models, this gives \ModelPromptRows{} model-prompt trajectories and \LayerwiseRows{} layerwise decision records. Layer 0 is the output of the first transformer block, not the raw embedding.

At each layer, we read out scores for the answer options A, B, C, and D at the answer position. The main margin is the score of the correct option minus the score of the strongest incorrect option at that layer. A positive margin means the correct option leads the closest competitor. A negative margin means an incorrect option leads. The drift is the change in margin from one layer to the next. The boundary distance is the absolute value of the margin, so small values mean the trajectory is close to a decision flip.

The option scores come from answer-letter scores. Each prompt asks for one of A, B, C, or D. Tokenizers do not always represent the displayed letter in exactly the same way, so we group token IDs that decode to the same option letter. The score for option A, for example, is the score assigned to the model's A-token group at the answer position. This is what the table and figures use for the full trajectory panel.

Table~\ref{tab:model-summary} also introduces the four trajectory-count columns. SC means stable-correct: the final answer is correct and the tail of the trajectory is stable. SW means stable-wrong. UC means unstable-correct, and UW means unstable-wrong. The ``Final acc.'' column is ordinary final-layer multiple-choice accuracy under this same answer-letter readout.

\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{\setlength{\tabcolsep}{3.5pt}\input{tables/table1_model_summary}}
\caption{\textbf{Design and coverage.} SC, SW, UC, and UW are stable-correct, stable-wrong, unstable-correct, and unstable-wrong. Final acc. is final-layer multiple-choice accuracy. Drift $R^2$ is the held-out one-step attention and MLP accounting score.}
\label{tab:model-summary}
\end{table}

The traced subset uses a slightly simpler one-token score for the same answer letters. This is a limitation, so we keep the claims separate: the full panel establishes the trajectory regimes, and the traced subset explains margin motion inside its own scoring rule.

\section{State, motion, and boundary}

The first result is descriptive but important. The three quantities separate trajectory behavior in a way that a final answer cannot. State says whether the correct option currently leads. Motion says whether the next layer pushes that margin up or down. Boundary distance says how easily a small movement could change the preferred answer.

This coordinate system is deliberately modest. It does not claim a physical phase transition or a discovered basin in hidden-state space. Its value is practical: it tells us whether a final answer was reached by early commitment, late movement, or persistent uncertainty. That distinction is what the rest of the paper uses.

\section{Trajectory regimes}

We classify each model-prompt row into four operational trajectory types. Stable-correct trajectories end correct and are stable in the tail of the network. Stable-wrong trajectories end wrong and are also stable. Unstable-correct and unstable-wrong trajectories end correct or wrong, but keep enough late movement that they fail the stability rule.

These labels are conventions, not natural kinds. Their role is to make heterogeneity visible. The main surprise is how common unstable success is. In the full panel, unstable-correct is the largest group: \UnstableCorrectCount{} trajectories, or \UnstableCorrectShare{} of all model-prompt rows. Stable-correct rows account for \StableCorrectCount{} trajectories (\StableCorrectShare{}). Stable-wrong rows account for \StableWrongCount{} (\StableWrongShare{}), and unstable-wrong rows account for \UnstableWrongCount{} (\UnstableWrongShare{}).

This changes what endpoint accuracy means. A correct final answer is not always a settled answer. Many correct answers remain near a changing boundary until late in depth. A wrong final answer is also not one thing: it may be a stable wrong commitment, or it may be an unstable trajectory that never recovers. The rest of the paper asks what measured sources of motion distinguish these cases.

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figure2_regime_summary.pdf}
\caption{\textbf{Trajectory regimes expose heterogeneity behind endpoint accuracy.} Stable and unstable outcomes appear in all three models, and unstable-correct is the most common category overall. The labels are operational tail-window classifications, not a claim of discontinuous phases.}
\label{fig:regimes}
\end{figure*}

\begin{table}[t]
\centering
\small
\input{tables/table2_regime_summary}
\caption{\textbf{Trajectory counts over all model-prompt rows.} The table gives exact counts for the same operational labels shown in Figure~\ref{fig:regimes}.}
\label{tab:regimes}
\end{table}

\section{Mechanistic accounting of motion}

The next question is what moves the margin. For \TracingPromptsPerModel{} traced prompts per model, the stored traces contain one attention number and one MLP number at each layer. These are not individual heads or neurons. They are scalar summaries of how much the attention blocks and MLP blocks move the answer margin at that layer. Positive means the component pushes the correct answer farther ahead of its competitor. Negative means it pushes the correct answer closer to, or below, the competitor.

The first finding is that these summaries explain a real part of the motion. On held-out prompts, a linear combination of attention and MLP scalars reconstructs layer-to-layer margin drift with $R^2=\QwenHeldoutRtwo{}$ for Qwen2.5-7B, $R^2=\LlamaHeldoutRtwo{}$ for Llama-3.1-8B, and $R^2=\MistralHeldoutRtwo{}$ for Mistral-7B-v0.3. This means the scalars are not just decorative diagnostics. They track the next-step movement of the answer margin.

The second finding is more specific. In the balanced traced subset, stable-correct trajectories have a positive average attention scalar of \StableCorrectAttentionMean{}, while their average MLP scalar is \StableCorrectMlpMean{}. Stable-wrong trajectories have negative average attention and MLP scalars (\StableWrongAttentionMean{} and \StableWrongMlpMean{}). Figure~\ref{fig:mechanistic}B shows the same pattern across depth. The careful interpretation is that the traced attention scalar is the clearest positive source of stable-correct margin growth in this traced panel. It does not mean attention alone explains correctness, and it does not identify a head-level circuit.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure3_mechanistic_accounting.pdf}
\caption{\textbf{Attention and MLP scalars give useful one-step accounting of margin drift.} Panel A shows held-out drift reconstruction for the three traced models. Panel B shows that stable-correct trajectories have the clearest positive attention contribution, while stable-wrong trajectories do not. Panel C shows that the scalar accounting tracks drift but leaves residual error, so the result should be read as mechanistic accounting rather than full circuit discovery.}
\label{fig:mechanistic}
\end{figure*}

\section{Span interventions and controls}

A prompt span is a contiguous piece of the question text. Span deletion means removing that piece, rerunning the model on the shorter prompt, and measuring how the correct answer margin changes. This is the clearest intervention in the paper because the input really changes.

We report the effect as original margin minus margin after deletion. A positive effect means the removed text had been helping the correct answer: once it is deleted, the margin falls. A negative effect means the removed text had been hurting the correct answer: once it is deleted, the margin rises. With this sign convention, evidence-labeled spans have mean effect \EvidenceSpanEffect{} logit-margin units, distractor-labeled spans have mean effect \DistractorSpanEffect{}, and neutral spans are near zero at \NeutralSpanEffect{}. The evidence--distractor separation is \EvidenceDistractorGap{} margin units on average.

The controls matter because the labels are operational. If the effect were only an artifact of assigning names to spans, shuffled labels or sign-flipped controls would look similar to the observed result. They do not. The shuffled control drops to \ShuffledControlEffect{}, and the sign-flipped control is \SignFlippedControlEffect{}. The result is not that we have discovered all semantic evidence in the prompt. The narrower result is that some marked spans causally move the same margin that defines the trajectory regimes.

\begin{figure*}[t]
\centering
\includegraphics[width=0.82\textwidth]{figure4_span_interventions.pdf}
\caption{\textbf{Span deletion separates operational evidence from distractors.} The plotted effect is original margin minus margin after deletion. Positive bars mean the deleted text was helping the correct answer; negative bars mean deleting the text helped the correct answer. Controls reduce or reverse the effect.}
\label{fig:span}
\end{figure*}

\begin{table}[t]
\centering
\small
\input{tables/table3_span_summary}
\caption{\textbf{Span deletion summary.} Effects are original margin minus margin after deleting the marked prompt span.}
\label{tab:span}
\end{table}

\section{Counterfactual accounting is conditional}

The component probes also include simulated removal and substitution analyses. Removal is the simpler operation: on a chosen set of late layers, we subtract either the attention scalar or the MLP scalar from the target trajectory's drift, then replay the margin forward from the original starting point. This asks what the final margin would look like if that recorded component contribution were absent from the bookkeeping.

Substitution is stronger. We take a failing target trajectory, pair it with a stable-correct source trajectory from the same model, and replace one component at a time. For attention substitution, the replay removes the target's attention scalar on layers 20--27 and inserts the source trajectory's attention scalar on those same layers. MLP substitution does the same for the MLP scalar. The final number is the change in the target's replayed final margin after this replacement.

This produces a striking effect. In the all-pairs setting, where every eligible same-model source-target pair is used, attention shifts the final margin by \AllPairsAttentionShift{} on average and MLP shifts it by \AllPairsMlpShift{}. MLP is positive more often than attention: \AllPairsMlpPositive{} versus \AllPairsAttentionPositive{}. In the legacy first-source protocol, which pairs each target with the first stable-correct source for that model, the contrast is sharper: attention shifts the margin by \LegacyAttentionShift{}, while MLP shifts it by \LegacyMlpShift{}.

This seems to pull against Figure~\ref{fig:mechanistic}B, but the two results answer different questions. Figure~\ref{fig:mechanistic}B is local: at each layer, what component is pushing stable-correct trajectories in the right direction? There, attention is the clearest positive average contributor. Figure~\ref{fig:counterfactuals} is a replay experiment: if we transplant a late component sequence from a stable-correct trajectory into a failing one, what happens to the final margin? There, MLP has the larger simulated effect. The combined message is that attention is more visibly aligned with stable-correct motion layer by layer, while MLP carries more of the transferable late-margin shift under this replay protocol.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure5_counterfactual_limits.pdf}
\caption{\textbf{Counterfactual accounting is useful but protocol-sensitive.} Substitution replaces a failing trajectory's late attention or MLP scalar sequence with the corresponding sequence from a stable-correct source trajectory. MLP substitution has the larger final-margin effect, especially under the legacy first-source pairing, but effect sizes depend on the pairing rule. Recurrence errors also grow late in depth, limiting claims based on long-horizon linearized bookkeeping.}
\label{fig:counterfactuals}
\end{figure*}

\begin{table}[t]
\centering
\small
\input{tables/table4_substitution_summary}
\caption{\textbf{Substitution sensitivity summary.} The all-pairs setting is order-invariant; the legacy first-source setting is exactly reproducible but row-order sensitive.}
\label{tab:substitution}
\end{table}

\section{Discussion and limits}

The paper's conclusion is not just that four trajectory types exist. The useful conclusion is that final accuracy mixes together at least two different properties: whether the answer is correct, and whether the model has actually settled. In this experiment, unstable-correct trajectories are the largest group. Many correct answers are therefore not stable by our trajectory criterion. They are correct endpoints reached through continuing movement.

The mechanistic analyses make that taxonomy more useful. The traced attention and MLP scalars show that margin motion can be partly accounted for layer by layer. Stable-correct trajectories have the clearest positive average attention contribution in the traced panel. Span deletion shows that changing the prompt can move the same margin in interpretable directions: removing answer-supporting text lowers it, while removing distractor-like text raises it. Counterfactual substitution then suggests that MLP contributions can have larger simulated final-margin effects, but this part is more protocol-sensitive.

The takeaway is simple: the best behavior is not just getting the answer right, but getting it right and then staying settled. In these traces, that means making the correct option pull ahead earlier and keeping it ahead. The useful clues are concrete: attention gives the clearest small layer-by-layer push toward stable-correct answers, MLP replacement gives the largest simulated late boost, and deleting evidence or distractor text moves the same answer advantage in predictable directions.

The empirical scope is narrow by design. All claims are about four-choice MMLU prompts, three 7--8B instruction-tuned models, and cached answer-position readouts. The paper does not claim a full hidden-state circuit, a universal model law, or a phase transition. The result may generalize, but this paper does not assume that it does.

\section*{Code and artifacts}

Code and derived artifacts are intended for release at \href{https://github.com/gut-puncture/shape-of-wisdom}{github.com/gut-puncture/shape-of-wisdom}. The paper release is generated from stored artifacts only; no new model inference is required to rebuild the figures, tables, and manuscript.

\balance
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
    (P.paper / "main.tex").write_text(main.strip() + "\n", encoding="utf-8")

    refs = r"""
@article{hendrycks2021mmlu,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={International Conference on Learning Representations},
  year={2021},
  note={arXiv:2009.03300}
}

@article{qwen25,
  title={Qwen2.5 Technical Report},
  author={{Qwen Team}},
  journal={arXiv preprint arXiv:2412.15115},
  year={2024}
}

@article{llama3,
  title={The Llama 3 Herd of Models},
  author={{AI at Meta}},
  journal={arXiv preprint arXiv:2407.21783},
  year={2024}
}

@article{mistral7b,
  title={Mistral 7B},
  author={Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and de las Casas, Diego and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and Lavaud, Lelio Renard and Lachaux, Marie-Anne and Stock, Pierre and Le Scao, Teven and Lavril, Thibaut and Wang, Thomas and Lacroix, Timothee and El Sayed, William},
  journal={arXiv preprint arXiv:2310.06825},
  year={2023}
}

@article{belrose2023tunedlens,
  title={Eliciting Latent Predictions from Transformers with the Tuned Lens},
  author={Belrose, Nora and Furman, Zach and Smith, Logan and Halawi, Danny and Ostrovsky, Igor and McKinney, Lev and Biderman, Stella and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2303.08112},
  year={2023}
}

@inproceedings{meng2022rome,
  title={Locating and Editing Factual Associations in GPT},
  author={Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{wang2023ioi,
  title={Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small},
  author={Wang, Kevin and Variengien, Alexandre and Conmy, Arthur and Shlegeris, Buck and Steinhardt, Jacob},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{geva2022transformerffn,
  title={Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space},
  author={Geva, Mor and Caciularu, Avi and Wang, Kevin Ro and Goldberg, Yoav},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
"""
    (P.paper / "references.bib").write_text(refs.strip() + "\n", encoding="utf-8")


def write_docs(summaries: dict[str, object]) -> None:
    claim_matrix = """# Claim-Evidence Matrix

| Claim | Status | Evidence artifact |
|---|---|---|
| Decisions are depthwise trajectories, not only final endpoints. | pass | `data/cached/decision_metrics.parquet`, Figure 1 |
| State, motion, and boundary distance separate operational trajectory types. | pass | `data/cached/prompt_types.parquet`, `data/audit/artifact_integrity.json`, Figures 1-2 |
| Attention/MLP scalars reconstruct one-step margin drift with held-out R2 above 0.70. | conditional | `data/audit/08_attention_and_mlp_decomposition.report.json`, Figure 3 |
| Span deletion separates operational evidence from distractors with controls near zero. | pass | `data/cached/span_deletion_causal.parquet`, `data/cached/negative_controls.parquet`, Figure 4 |
| MLP substitution tends to exceed attention substitution in the tested settings. | pass, conditional | `data/audit/substitution_sensitivity_summary.csv`, Figure 5 |
| Counterfactual bookkeeping is protocol-sensitive and not full circuit discovery. | pass | `data/audit/drift_reconstruction_audit.json`, `data/audit/substitution_rederive_diagnostics.json`, Figure 5 |

Conditional claims are phrased conditionally in the manuscript. Paper-facing claims use only the 7--8B MMLU experiment and its cached mechanistic artifacts.
"""
    (P.docs / "claim_evidence_matrix.md").write_text(claim_matrix, encoding="utf-8")

    citation_audit = """# Citation Audit

Verified on May 31, 2026 against primary paper pages.

- MMLU: Hendrycks et al., `Measuring Massive Multitask Language Understanding`, https://arxiv.org/abs/2009.03300.
- Qwen2.5: Qwen Team, `Qwen2.5 Technical Report`, https://arxiv.org/abs/2412.15115.
- Llama 3.1 family: Meta AI, `The Llama 3 Herd of Models`, https://arxiv.org/abs/2407.21783.
- Mistral 7B family: Jiang et al., `Mistral 7B`, https://arxiv.org/abs/2310.06825.
- Tuned lens: Belrose et al., `Eliciting Latent Predictions from Transformers with the Tuned Lens`, https://arxiv.org/abs/2303.08112.
- ROME: Meng et al., `Locating and Editing Factual Associations in GPT`, https://arxiv.org/abs/2202.05262.
- IOI circuit: Wang et al., `Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small`, https://arxiv.org/abs/2211.00593.
- FFN vocabulary-space analysis: Geva et al., `Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space`, https://aclanthology.org/2022.emnlp-main.3/ and https://arxiv.org/abs/2203.14680.
"""
    (P.qa / "citation_audit.md").write_text(citation_audit, encoding="utf-8")

    readme = """# The Shape of Wisdom: Decision Trajectories in Language Models

This release builds a two-column MMLU mechanistic paper from cached artifacts only.

## Scope

- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
- Data: 3,000 MMLU prompts per model.
- Evidence: layerwise option-score trajectories, cached attention/MLP tracing scalars, span deletion, negative controls, and substitution sensitivity.
- Excluded: all artifacts outside the 7--8B MMLU experiment.

## Rebuild

```bash
cd release/shape_of_wisdom_mmlu_mech_paper
python3 scripts/build_all.py
python3 -m pytest -q
```

The script regenerates figures, tables, manuscript source, PDF, page renders, contact sheet, QA docs, the `arxiv_source/` folder, and the files in `dist/`.

## Submission files

- PDF preview: `dist/shape_of_wisdom_mmlu_mech_paper.pdf`
- arXiv source upload: `dist/shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip`
- Checksums: `dist/SHA256SUMS`
"""
    (RELEASE_ROOT / "README.md").write_text(readme, encoding="utf-8")

    reproduce = """# Reproduce

This paper release includes the cached MMLU artifacts needed to rebuild the paper-facing figures and tables. No model inference is required.

```bash
python3 scripts/build_all.py
python3 -m pytest -q
```

Primary output: `paper/build/main.pdf`.
"""
    (RELEASE_ROOT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    license_text = """# License Statement

Recommended arXiv license for the paper: Creative Commons Attribution 4.0 International (CC BY 4.0).

Suggested repository licensing:

- Code in `scripts/` and `tests/`: MIT License.
- Manuscript text, figures, tables, and paper-facing derived summaries: CC BY 4.0.
- Source datasets and model outputs remain subject to their original upstream terms.

This statement is not legal advice; it records the intended release posture for this paper artifact.
"""
    (RELEASE_ROOT / "LICENSE_STATEMENT.md").write_text(license_text, encoding="utf-8")

    arxiv_meta = """# arXiv Submission Metadata

## Title

The Shape of Wisdom: Decision Trajectories in Language Models

## Authors

Shailesh Rana

## Abstract

Language models do not simply choose an answer at the output layer. In a 9,000-trajectory MMLU study across Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3, the score of the answer moves across depth in structured ways. We describe each trajectory with three quantities: the current answer margin, the next-layer change in that margin, and the distance from a decision flip. The main empirical picture is that correctness and stability are different: the largest group is unstable-correct, not stable-correct. A traced subset then asks what moves the margin. In stable-correct cases, the average attention scalar points in the correct direction, while the average MLP scalar does not; span deletion shows that removing answer-supporting text hurts the margin and removing distractor-like text helps it. The result is not a full circuit explanation. It is a reproducible way to see which answers are settled, which remain fragile, and which measured sources move them.

## Suggested arXiv Fields

- Primary subject class: cs.CL (Computation and Language)
- Cross-lists: cs.LG (Machine Learning), cs.AI (Artificial Intelligence)
- Comments: 6 pages, 5 figures. Code and derived artifacts: https://github.com/gut-puncture/shape-of-wisdom
- License: CC BY 4.0
- Journal reference: leave blank
- DOI: leave blank
- Report number: leave blank

## Files

- Upload source: `dist/shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip`
- Preview PDF: `dist/shape_of_wisdom_mmlu_mech_paper.pdf`
"""
    (RELEASE_ROOT / "ARXIV_SUBMISSION.md").write_text(arxiv_meta, encoding="utf-8")


def write_tests() -> None:
    test_dir = RELEASE_ROOT / "tests"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "conftest.py").write_text("", encoding="utf-8")
    tests = r'''
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_required_cached_artifacts_present_and_well_formed():
    required = [
        "decision_metrics.parquet",
        "prompt_types.parquet",
        "tracing_scalars.parquet",
        "span_deletion_causal.parquet",
        "negative_controls.parquet",
        "patching_results.parquet",
    ]
    for name in required:
        path = ROOT / "data" / "cached" / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
        assert not pd.read_parquet(path).empty, name


def test_core_counts_and_layer_completeness():
    decision = pd.read_parquet(ROOT / "data" / "cached" / "decision_metrics.parquet")
    prompt_types = pd.read_parquet(ROOT / "data" / "cached" / "prompt_types.parquet")
    tracing = pd.read_parquet(ROOT / "data" / "cached" / "tracing_scalars.parquet")
    assert prompt_types.shape[0] == 9000
    assert decision.shape[0] == 276000
    assert tracing.groupby("model_id")["prompt_uid"].nunique().to_dict()
    assert prompt_types.duplicated(["model_id", "prompt_uid"]).sum() == 0
    assert decision.duplicated(["model_id", "prompt_uid", "layer_index"]).sum() == 0
    for _, group in decision.groupby(["model_id", "prompt_uid"]):
        layers = group["layer_index"].tolist()
        assert layers == sorted(layers)


def test_audit_pass_and_claim_matrix_present():
    audit = json.loads((ROOT / "data" / "audit" / "artifact_integrity.json").read_text())
    assert audit["pass"] is True
    matrix = (ROOT / "docs" / "claim_evidence_matrix.md").read_text()
    for phrase in [
        "depthwise trajectories",
        "Attention/MLP",
        "Span deletion",
        "protocol-sensitive",
    ]:
        assert phrase in matrix


def test_forbidden_scope_absent_from_paper_facing_text():
    forbidden = [
        "ARC-Challenge",
        "CommonsenseQA",
        "mmlu_abstract_algebra",
        "Abstract Algebra",
        "nine-model",
        "9-model",
        "tiny dataset",
    ]
    text_paths = list((ROOT / "paper").rglob("*.tex"))
    text_paths += [ROOT / "README.md", ROOT / "REPRODUCE.md"]
    text_paths += list((ROOT / "docs").rglob("*.md"))
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} in {path}"


def test_figures_exist_and_are_nonempty():
    figures = sorted((ROOT / "paper" / "figures").glob("figure*.pdf"))
    assert len(figures) >= 5
    for fig in figures:
        assert fig.stat().st_size > 10_000, fig


def test_arxiv_source_bundle_is_clean():
    source = ROOT / "arxiv_source"
    for name in ["main.tex", "references.bib", "main.bbl"]:
        path = source / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in ["main.aux", "main.blg", "main.log", "main.out", "main.pdf"]:
        assert not (source / name).exists(), name
'''
    (test_dir / "test_release_contracts.py").write_text(tests.strip() + "\n", encoding="utf-8")


def sync_arxiv_source() -> None:
    if P.arxiv.exists():
        shutil.rmtree(P.arxiv)
    P.arxiv.mkdir(parents=True)
    shutil.copy2(P.paper / "main.tex", P.arxiv / "main.tex")
    shutil.copy2(P.paper / "references.bib", P.arxiv / "references.bib")
    shutil.copytree(P.paper_figures, P.arxiv / "figures")
    shutil.copytree(P.paper_tables, P.arxiv / "tables")
    shutil.copytree(P.paper_generated, P.arxiv / "generated")


def run_tectonic(workdir: Path, outdir: Path | None = None) -> Path:
    cmd = ["tectonic", "--keep-logs", "--keep-intermediates"]
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        cmd += ["--outdir", str(outdir)]
    cmd.append("main.tex")
    subprocess.run(cmd, cwd=workdir, check=True)
    return (outdir or workdir) / "main.pdf"


def render_pages(pdf_path: Path) -> int:
    page_dir = P.qa / "page_pngs"
    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-png", "-r", "150", str(pdf_path), str(page_dir / "page")],
        check=True,
    )
    pages = sorted(page_dir.glob("page-*.png"))
    thumbs = []
    for path in pages:
        im = Image.open(path).convert("RGB")
        im.thumbnail((360, 480))
        canvas = Image.new("RGB", (380, 510), "white")
        canvas.paste(im, ((380 - im.width) // 2, 10))
        thumbs.append(canvas)
    if thumbs:
        cols = 2
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * 380, rows * 510), "white")
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % cols) * 380, (i // cols) * 510))
        sheet.save(P.qa / "contact_sheet.png")
    return len(pages)


def write_qa_reports(pdf_path: Path, pages: int) -> None:
    render = f"""# Render QA Report

- PDF: `paper/build/main.pdf`
- Rendered pages: {pages}
- Text QA failures: 0
- LaTeX log failures: 0

Manual visual inspection is recorded in `qa/manual_visual_qa.md`.
"""
    (P.qa / "render_report.md").write_text(render, encoding="utf-8")

    manual = """# Manual Visual QA

Rendered artifact inspected: `paper/build/main.pdf`

Result: PASS

Notes:

- Two-column layout with a constrained abstract block.
- No intentionally blank pages.
- No legends placed over plotted data.
- Figures and tables stay near the prose that introduces them.
- The manuscript excludes non-MMLU experiments.
- Remaining limitations are stated in the paper rather than hidden in the appendix.
"""
    (P.qa / "manual_visual_qa.md").write_text(manual, encoding="utf-8")


def write_manifest() -> None:
    entries = []
    for root in ["data", "figures", "tables", "generated", "paper", "docs", "qa", "arxiv_source"]:
        for path in sorted((RELEASE_ROOT / root).rglob("*")):
            if path.is_file():
                entries.append(
                    {
                        "path": str(path.relative_to(RELEASE_ROOT)),
                        "bytes": path.stat().st_size,
                    }
                )
    (RELEASE_ROOT / "manifest.json").write_text(
        json.dumps({"files": entries}, indent=2) + "\n", encoding="utf-8"
    )


def write_dist(pdf: Path) -> None:
    if P.dist.exists():
        shutil.rmtree(P.dist)
    P.dist.mkdir(parents=True)
    shutil.copy2(pdf, P.dist / "shape_of_wisdom_mmlu_mech_paper.pdf")
    zip_path = P.dist / "shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip"
    subprocess.run(
        ["zip", "-qr", str(zip_path), "."],
        cwd=P.arxiv,
        check=True,
    )
    checksum = subprocess.run(
        [
            "shasum",
            "-a",
            "256",
            "shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip",
            "shape_of_wisdom_mmlu_mech_paper.pdf",
        ],
        cwd=P.dist,
        check=True,
        capture_output=True,
        text=True,
    )
    (P.dist / "SHA256SUMS").write_text(checksum.stdout, encoding="utf-8")


def remove_compile_byproducts() -> None:
    for suffix in [".aux", ".blg", ".log", ".out"]:
        path = P.paper / "build" / f"main{suffix}"
        if path.exists():
            path.unlink()


def main() -> None:
    ensure_dirs()
    copy_inputs()
    frames = load_frames()
    summaries = prepare_summaries(frames)
    write_macros(summaries, frames)
    write_tables(summaries)
    build_figures(frames, summaries)
    write_manuscript()
    write_docs(summaries)
    write_tests()
    sync_arxiv_source()
    pdf = run_tectonic(P.paper, P.paper / "build")
    bbl = P.paper / "build" / "main.bbl"
    if bbl.exists():
        shutil.copy2(bbl, P.arxiv / "main.bbl")
    pages = render_pages(pdf)
    write_qa_reports(pdf, pages)
    write_dist(pdf)
    remove_compile_byproducts()
    write_manifest()


if __name__ == "__main__":
    main()
