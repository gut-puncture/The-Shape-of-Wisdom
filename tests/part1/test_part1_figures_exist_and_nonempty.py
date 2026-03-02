"""Tests that all Paper 1 figures exist and are non-empty."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "part1" / "figures"

EXPECTED_FIGURES = [
    "fig1_examples.pdf",
    "fig2_delta_distribution.pdf",
    "fig3_decision_space_trajectories.pdf",
    "fig4_flow_field.pdf",
    "fig5_commitment_and_flips.pdf",
    "fig6_robustness.pdf",
]

MIN_SIZE = 15_000


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_exists(name: str) -> None:
    path = FIG_DIR / name
    assert path.exists(), f"Figure not found: {path}"


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_nonempty(name: str) -> None:
    path = FIG_DIR / name
    if not path.exists():
        pytest.skip(f"Figure not found: {path}")
    size = path.stat().st_size
    assert size >= MIN_SIZE, f"{name} is {size} bytes, min {MIN_SIZE}"


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_is_pdf(name: str) -> None:
    path = FIG_DIR / name
    if not path.exists():
        pytest.skip(f"Figure not found: {path}")
    header = path.read_bytes()[:4]
    assert header == b"%PDF", f"{name} does not start with %PDF header"
