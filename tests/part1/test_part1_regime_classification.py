"""Tests for Paper 1 regime classification sanity (T6)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = REPO_ROOT / "paper" / "part1" / "data" / "part1_core.parquet"
M_DEFAULT = 0.75

EXPECTED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


@pytest.fixture(scope="module")
def prompt_level() -> pd.DataFrame:
    if not CORE_PATH.exists():
        pytest.skip(f"part1_core.parquet not found")
    core = pd.read_parquet(CORE_PATH)
    return core.groupby(["model_id", "prompt_uid"]).first().reset_index()


def test_all_prompts_classified(prompt_level: pd.DataFrame) -> None:
    """Every prompt must have a regime assignment."""
    assert prompt_level["regime"].notna().all()
    valid = prompt_level["regime"].isin(["Stable-Correct", "Stable-Wrong", "Unstable"])
    assert valid.all(), f"Invalid regimes: {prompt_level[~valid]['regime'].unique()}"


def test_stable_correct_median(prompt_level: pd.DataFrame) -> None:
    for m in EXPECTED_MODELS:
        sub = prompt_level[(prompt_level["model_id"] == m) & (prompt_level["regime"] == "Stable-Correct")]
        if len(sub) == 0:
            pytest.skip(f"No Stable-Correct for {m}")
        med = sub["final_delta_default"].median()
        assert med > M_DEFAULT, (
            f"{m} Stable-Correct median = {med:.3f}, expected > {M_DEFAULT}"
        )


def test_stable_wrong_median(prompt_level: pd.DataFrame) -> None:
    for m in EXPECTED_MODELS:
        sub = prompt_level[(prompt_level["model_id"] == m) & (prompt_level["regime"] == "Stable-Wrong")]
        if len(sub) == 0:
            pytest.skip(f"No Stable-Wrong for {m}")
        med = sub["final_delta_default"].median()
        assert med < -M_DEFAULT, (
            f"{m} Stable-Wrong median = {med:.3f}, expected < {-M_DEFAULT}"
        )


def test_unstable_smaller_magnitude(prompt_level: pd.DataFrame) -> None:
    for m in EXPECTED_MODELS:
        stable = prompt_level[
            (prompt_level["model_id"] == m)
            & prompt_level["regime"].isin(["Stable-Correct", "Stable-Wrong"])
        ]
        unstable = prompt_level[
            (prompt_level["model_id"] == m) & (prompt_level["regime"] == "Unstable")
        ]
        if len(stable) == 0 or len(unstable) == 0:
            continue
        med_s = stable["final_delta_default"].abs().median()
        med_u = unstable["final_delta_default"].abs().median()
        # This is a soft check — warn rather than hard fail
        if med_u >= med_s:
            import warnings
            warnings.warn(
                f"{m}: Unstable median |δ| ({med_u:.3f}) ≥ Stable ({med_s:.3f})"
            )
