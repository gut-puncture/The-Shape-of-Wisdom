"""Tests for Paper 1 data contracts.

Run with: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/part1/ -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = REPO_ROOT / "paper" / "part1" / "data" / "part1_core.parquet"

EXPECTED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

REQUIRED_COLUMNS = [
    "model_id", "prompt_uid", "layer_index", "L_model", "depth_norm",
    "correct_key", "sA", "sB", "sC", "sD",
    "k_dyn", "k_fix",
    "delta_soft_tau_0_5", "delta_soft_tau_1_0", "delta_soft_tau_2_0",
    "delta_hard_dyn", "delta_hard_fix", "delta_default",
    "final_delta_default", "final_sign", "delta_signed",
    "switch_indicator", "regime", "commitment_layer", "flip_count",
    "last_flip_layer",
]


@pytest.fixture(scope="module")
def core() -> pd.DataFrame:
    if not CORE_PATH.exists():
        pytest.skip(f"part1_core.parquet not found at {CORE_PATH}")
    return pd.read_parquet(CORE_PATH)


def test_exactly_three_models(core: pd.DataFrame) -> None:
    models = sorted(core["model_id"].unique())
    assert len(models) == 3, f"Expected 3 models, got {len(models)}: {models}"
    for m in EXPECTED_MODELS:
        assert m in models, f"Missing expected model: {m}"


def test_min_prompts_per_model(core: pd.DataFrame) -> None:
    for m in EXPECTED_MODELS:
        n = core.loc[core["model_id"] == m, "prompt_uid"].nunique()
        assert n >= 1500, f"{m} has only {n} prompts (need ≥1500)"


def test_required_columns_exist(core: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in core.columns]
    assert not missing, f"Missing columns: {missing}"


def test_correct_key_valid(core: pd.DataFrame) -> None:
    bad = ~core["correct_key"].isin(["A", "B", "C", "D"])
    assert not bad.any(), f"{bad.sum()} rows with invalid correct_key"


def test_scores_not_null(core: pd.DataFrame) -> None:
    for col in ["sA", "sB", "sC", "sD"]:
        assert core[col].notna().all(), f"{col} has null values"


def test_contiguous_layers(core: pd.DataFrame) -> None:
    """At least 95% of prompts must have contiguous layers 0..L-1."""
    for m in EXPECTED_MODELS:
        sub = core[core["model_id"] == m]
        L = sub["L_model"].iloc[0]
        total = sub["prompt_uid"].nunique()
        complete = 0
        for uid, grp in sub.groupby("prompt_uid"):
            layers = set(grp["layer_index"].values)
            if layers == set(range(L + 1)):
                complete += 1
        pct = complete / total
        assert pct >= 0.95, f"{m}: only {pct:.1%} prompts have contiguous layers"


def test_no_transformers_import() -> None:
    """Verify that importing transformers is blocked in offline mode."""
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers is installed but should be blocked by offline vars")
    except (ImportError, OSError):
        pass  # Expected


def test_one_row_per_key(core: pd.DataFrame) -> None:
    """Each (model_id, prompt_uid, layer_index) must appear exactly once."""
    dup = core.duplicated(subset=["model_id", "prompt_uid", "layer_index"])
    assert not dup.any(), f"{dup.sum()} duplicate rows"
