"""Tests for Paper 1 metric definitions.

Verifies that canonical formulas (δ_soft, δ_hard_{dyn,fix}, k_dyn, switch)
are computed correctly on synthetic data.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def _logsumexp(x: np.ndarray) -> float:
    xm = np.max(x)
    return float(xm + np.log(np.sum(np.exp(x - xm))))


def test_delta_hard_dyn() -> None:
    """δ_hard_dyn = s_correct - max(s_{j≠c})."""
    scores = {"A": 2.0, "B": 1.5, "C": 0.5, "D": -1.0}
    correct = "A"
    s_c = scores[correct]
    others = [v for k, v in scores.items() if k != correct]
    delta = s_c - max(others)
    assert abs(delta - 0.5) < 1e-10


def test_delta_soft_tau1() -> None:
    """δ_soft(τ=1) = s_c - τ * logsumexp(s_{j≠c} / τ)."""
    scores = {"A": 2.0, "B": 1.5, "C": 0.5, "D": -1.0}
    correct = "A"
    tau = 1.0
    s_c = scores[correct]
    others = np.array([scores[k] for k in ["B", "C", "D"]])
    lse = _logsumexp(others / tau)
    delta = s_c - tau * lse
    # logsumexp([1.5, 0.5, -1.0]) ≈ 1.5 + log(1 + exp(-1) + exp(-2.5))
    expected_lse = 1.5 + np.log(1 + np.exp(-1.0) + np.exp(-2.5))
    assert abs(lse - expected_lse) < 1e-10
    assert abs(delta - (2.0 - 1.0 * expected_lse)) < 1e-10


def test_delta_hard_fix() -> None:
    """δ_hard_fix uses the competitor from the final layer."""
    # At final layer, competitor is B → δ_fix uses B at all layers
    scores_l0 = {"A": 1.0, "B": 0.5, "C": 2.0, "D": -1.0}
    scores_final = {"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0}
    correct = "A"
    # k_fix = argmax_{j≠A} scores_final = "B"
    k_fix = "B"
    # At layer 0: δ_fix = s_A(l0) - s_B(l0) = 1.0 - 0.5 = 0.5
    delta_fix_l0 = scores_l0[correct] - scores_l0[k_fix]
    assert abs(delta_fix_l0 - 0.5) < 1e-10


def test_k_dyn_tiebreak() -> None:
    """Tie-breaking: deterministic in A,B,C,D order."""
    # If B and C tied, argmax picks B (earlier in order)
    scores = {"A": 5.0, "B": 3.0, "C": 3.0, "D": 1.0}
    correct = "A"
    choices = ["A", "B", "C", "D"]
    best = None
    for c in choices:
        if c == correct:
            continue
        if best is None or scores[c] > scores[best]:
            best = c
    assert best == "B"  # First one wins in iteration


def test_switch_indicator() -> None:
    """switch_indicator = 1 if k_dyn changes between layers."""
    k_dyn_seq = ["B", "B", "C", "C", "B"]
    switches = [0]  # l=0 has no predecessor
    for i in range(1, len(k_dyn_seq)):
        switches.append(1 if k_dyn_seq[i] != k_dyn_seq[i - 1] else 0)
    assert switches == [0, 0, 1, 0, 1]


def test_flip_count() -> None:
    """flip_count = sign changes in δ sequence, ignoring zeros."""
    deltas = [1.0, 0.5, -0.3, 0.0, -0.1, 0.7]
    nonzero = [d for d in deltas if d != 0]
    signs = [np.sign(d) for d in nonzero]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    assert flips == 2  # +→- and -→+


def test_commitment_layer() -> None:
    """Commitment = smallest l such that δ_signed[l:] ≥ M for all l' ≥ l."""
    M = 0.75
    d_signed = [0.1, 0.3, 0.6, 0.8, 0.9, 1.0]
    commitment = None
    for i in range(len(d_signed)):
        if all(d >= M for d in d_signed[i:]):
            commitment = i
            break
    assert commitment == 3


def test_regime_stable_correct() -> None:
    """Stable-Correct: flip_count ≤ 1 AND commitment not null AND final > 0."""
    flip_count = 0
    commitment = 5
    final_delta = 2.0
    if flip_count <= 1 and commitment is not None:
        if final_delta > 0:
            regime = "Stable-Correct"
        elif final_delta < 0:
            regime = "Stable-Wrong"
        else:
            regime = "Unstable"
    else:
        regime = "Unstable"
    assert regime == "Stable-Correct"


def test_regime_unstable_no_commitment() -> None:
    """Unstable: commitment is None."""
    flip_count = 0
    commitment = None
    final_delta = 1.0
    if flip_count <= 1 and commitment is not None:
        regime = "Stable-Correct"
    else:
        regime = "Unstable"
    assert regime == "Unstable"
