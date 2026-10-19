from __future__ import annotations

from dataset_loader import CanonicalOption
from execution_config import (
    PermutationSpec,
    ReadoutSpec,
    build_option_surfaces,
    permutation_for_spec,
)


def test_build_option_surfaces_uses_readout_template() -> None:
    options = [
        CanonicalOption(option_id="A", ordinal=0, raw_text="alpha", canonical_text="alpha"),
        CanonicalOption(option_id="B", ordinal=1, raw_text="beta", canonical_text="beta"),
    ]
    spec = ReadoutSpec(
        id="templated_semantic",
        kind="templated_semantic",
        normalization="mean_logprob",
        primary=False,
        template="Answer => {option_text}",
    )
    surfaces = build_option_surfaces(spec, options)
    assert surfaces == {"A": "Answer => alpha", "B": "Answer => beta"}


def test_build_option_surfaces_uses_letter_template() -> None:
    options = [
        CanonicalOption(option_id="A", ordinal=0, raw_text="alpha", canonical_text="alpha"),
    ]
    spec = ReadoutSpec(
        id="letter_label",
        kind="letter_label",
        normalization="mean_logprob",
        primary=False,
        template="({label})",
    )
    surfaces = build_option_surfaces(spec, options)
    assert surfaces == {"A": "(A)"}


def test_permutation_for_spec_comes_from_configured_strategy() -> None:
    assert permutation_for_spec(PermutationSpec(name="identity", strategy="identity"), option_count=4) == [0, 1, 2, 3]
    assert permutation_for_spec(PermutationSpec(name="reverse", strategy="reverse"), option_count=4) == [3, 2, 1, 0]
