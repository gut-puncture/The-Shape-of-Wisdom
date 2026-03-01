"""Centralised parquet loading with column validation for paper figures."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Model layer counts (pinned)
# ---------------------------------------------------------------------------

MODEL_LAYERS: Dict[str, int] = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
}

OPTION_COLS = ["option_A", "option_B", "option_C", "option_D"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_cols(df: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def _add_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised depth column l/(L-1) ∈ [0, 1]."""
    L = df["model_id"].map(MODEL_LAYERS)
    df = df.copy()
    df["depth"] = df["layer_index"] / (L - 1).clip(lower=1)
    return df


def _remap_option_to_relative(
    span_label: str, correct_key: str, competitor_key: str,
) -> str:
    """Map absolute span labels (option_A) to relative (option_correct)."""
    if span_label not in OPTION_COLS:
        return span_label
    letter = span_label.replace("option_", "")
    if letter == correct_key:
        return "option_correct"
    elif letter == competitor_key:
        return "option_competitor"
    return "option_other"


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_decision_metrics(parquet_dir: Path) -> pd.DataFrame:
    """Load decision_metrics + prompt_types, add depth column.

    Expected columns from decision_metrics.parquet:
        model_id, prompt_uid, layer_index, delta, drift, boundary,
        competitor, correct_key, is_correct

    Expected columns from prompt_types.parquet:
        model_id, prompt_uid, trajectory_type
    """
    dm = pd.read_parquet(parquet_dir / "decision_metrics.parquet")
    _assert_cols(dm, {"model_id", "prompt_uid", "layer_index", "delta", "drift",
                       "boundary", "competitor", "correct_key"}, "decision_metrics")

    pt = pd.read_parquet(parquet_dir / "prompt_types.parquet")
    _assert_cols(pt, {"model_id", "prompt_uid", "trajectory_type"}, "prompt_types")

    merged = dm.merge(
        pt[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"], how="left",
    )
    return _add_depth(merged)


def load_prompt_types(parquet_dir: Path) -> pd.DataFrame:
    pt = pd.read_parquet(parquet_dir / "prompt_types.parquet")
    _assert_cols(pt, {"model_id", "prompt_uid", "trajectory_type"}, "prompt_types")
    return pt


def load_tracing_scalars(parquet_dir: Path) -> pd.DataFrame:
    """Load tracing_scalars + prompt_types, add depth column.

    Expected columns from tracing_scalars.parquet:
        model_id, prompt_uid, layer_index, s_attn, s_mlp, delta, drift
    """
    ts = pd.read_parquet(parquet_dir / "tracing_scalars.parquet")
    _assert_cols(ts, {"model_id", "prompt_uid", "layer_index", "s_attn", "s_mlp",
                       "delta", "drift"}, "tracing_scalars")

    pt = pd.read_parquet(parquet_dir / "prompt_types.parquet")
    merged = ts.merge(
        pt[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"], how="left",
    )
    return _add_depth(merged)


def _get_final_layer_keys(dm: pd.DataFrame) -> pd.DataFrame:
    """Get correct_key and final-layer competitor for each (model, prompt)."""
    final = (
        dm.sort_values("layer_index")
        .groupby(["model_id", "prompt_uid"])
        .last()
        .reset_index()[["model_id", "prompt_uid", "correct_key", "competitor"]]
    )
    return final.rename(columns={"competitor": "competitor_key"})


def load_attention_data(
    parquet_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load attention mass and contribution with relative option remapping.

    Returns (mass_df, contrib_df) both with span_rel, depth, trajectory_type.
    """
    mass = pd.read_parquet(parquet_dir / "attention_mass_by_span.parquet")
    _assert_cols(mass, {"model_id", "prompt_uid", "layer_index", "span_label",
                         "attention_mass"}, "attention_mass_by_span")

    contrib = pd.read_parquet(parquet_dir / "attention_contrib_by_span.parquet")
    _assert_cols(contrib, {"model_id", "prompt_uid", "layer_index", "span_label",
                            "attention_contribution"}, "attention_contrib_by_span")

    # Get correct/competitor keys from decision_metrics
    dm = pd.read_parquet(parquet_dir / "decision_metrics.parquet")
    keys = _get_final_layer_keys(dm)

    pt = pd.read_parquet(parquet_dir / "prompt_types.parquet")
    traj = pt[["model_id", "prompt_uid", "trajectory_type"]]

    for df_name, df in [("mass", mass), ("contrib", contrib)]:
        pass  # validation already done above

    # Merge keys and remap
    mass_m = mass.merge(keys, on=["model_id", "prompt_uid"], how="inner")
    contrib_m = contrib.merge(keys, on=["model_id", "prompt_uid"], how="inner")

    mass_m["span_rel"] = mass_m.apply(
        lambda r: _remap_option_to_relative(r["span_label"], r["correct_key"], r["competitor_key"]),
        axis=1,
    )
    contrib_m["span_rel"] = contrib_m.apply(
        lambda r: _remap_option_to_relative(r["span_label"], r["correct_key"], r["competitor_key"]),
        axis=1,
    )

    # Add depth and trajectory type
    mass_m = _add_depth(mass_m).merge(traj, on=["model_id", "prompt_uid"], how="left")
    contrib_m = _add_depth(contrib_m).merge(traj, on=["model_id", "prompt_uid"], how="left")

    return mass_m, contrib_m


def load_causal_data(
    parquet_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ablation, patching, span_labels, negative_controls.

    Returns (ablation, patching, span_labels, negative_controls).
    """
    abl = pd.read_parquet(parquet_dir / "ablation_results.parquet")
    _assert_cols(abl, {"component", "delta_shift"}, "ablation_results")

    pat = pd.read_parquet(parquet_dir / "patching_results.parquet")
    _assert_cols(pat, {"component", "delta_shift"}, "patching_results")

    sl = pd.read_parquet(parquet_dir / "span_labels.parquet")
    _assert_cols(sl, {"span_label", "effect_delta"}, "span_labels")

    nc = pd.read_parquet(parquet_dir / "negative_controls.parquet")
    _assert_cols(nc, {"control", "mean_effect_delta"}, "negative_controls")

    return abl, pat, sl, nc


def load_config(config_path: Path) -> dict:
    """Load experiment config for threshold values."""
    with open(config_path) as f:
        return yaml.safe_load(f)
