from __future__ import annotations

from typing import Any

from dataset_loader import CanonicalExample
from model_loader import LoadedModelBundle
from option_text_scorer import score_semantic_options


def _option_map(example: CanonicalExample) -> dict[str, str]:
    return {option.option_id: option.raw_text for option in example.options}


def score_templated_readout(
    bundle: LoadedModelBundle,
    *,
    prompt_text: str,
    example: CanonicalExample,
    template: str = "The answer is {option_text}",
) -> dict[str, Any]:
    option_surfaces = {
        option_id: template.format(option_text=text)
        for option_id, text in _option_map(example).items()
    }
    return score_semantic_options(bundle, prompt_text=prompt_text, option_surfaces=option_surfaces)


def score_letter_readout(
    bundle: LoadedModelBundle,
    *,
    prompt_text: str,
    example: CanonicalExample,
    template: str = "{label}",
) -> dict[str, Any]:
    option_surfaces = {
        option.option_id: template.format(label=option.option_id)
        for option in example.options
    }
    return score_semantic_options(bundle, prompt_text=prompt_text, option_surfaces=option_surfaces)
