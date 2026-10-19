from __future__ import annotations

import math
from typing import Any

from model_loader import LoadedModelBundle
from scoring_engine import score_readout_group


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def score_semantic_options(
    bundle: LoadedModelBundle,
    *,
    prompt_text: str,
    option_surfaces: dict[str, str],
    normalization: str = "mean_logprob",
    scorer_engine: str = "reference_exact",
    max_batch_size: int = 1,
) -> dict[str, Any]:
    _ = _softmax
    return score_readout_group(
        bundle,
        prompt_text=prompt_text,
        readout_option_surfaces={"semantic_exact": option_surfaces},
        normalization_by_readout={"semantic_exact": normalization},
        scorer_engine=scorer_engine,
        max_batch_size=max_batch_size,
    )["semantic_exact"]
