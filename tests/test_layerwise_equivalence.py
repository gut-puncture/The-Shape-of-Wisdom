from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from layerwise_extractor import (
    PROJECTION_APPLY_FINAL_NORM,
    PROJECTION_APPLY_FINAL_NORM_EXCEPT_LAST,
    assert_final_layer_equivalence,
)


class DummyOutputs:
    def __init__(self, hidden_states, logits) -> None:
        self.hidden_states = hidden_states
        self.logits = logits


def test_final_layer_equivalence_passes_on_matching_projection() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]])
    embedding = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        embedding.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    logits = embedding(hidden)
    outputs = DummyOutputs(hidden_states=[torch.zeros_like(hidden), hidden], logits=logits)
    assert_final_layer_equivalence(
        outputs=outputs,
        output_embedding=embedding,
        final_norm=None,
        projection_mode=PROJECTION_APPLY_FINAL_NORM,
        atol=1e-6,
    )


def test_final_layer_equivalence_fails_on_mismatch() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]])
    embedding = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        embedding.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    logits = embedding(hidden) + 0.5
    outputs = DummyOutputs(hidden_states=[torch.zeros_like(hidden), hidden], logits=logits)
    with pytest.raises(ValueError):
        assert_final_layer_equivalence(
            outputs=outputs,
            output_embedding=embedding,
            final_norm=None,
            projection_mode=PROJECTION_APPLY_FINAL_NORM,
            atol=1e-6,
        )


def test_final_layer_equivalence_supports_skip_norm_on_last_hidden_state() -> None:
    class OffsetNorm(torch.nn.Module):
        def forward(self, value):
            return value + 10.0

    hidden = torch.tensor([[[1.0, 2.0]]])
    embedding = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        embedding.weight[:] = torch.eye(2)
    logits = embedding(hidden)
    outputs = DummyOutputs(hidden_states=[torch.zeros_like(hidden), hidden], logits=logits)
    assert_final_layer_equivalence(
        outputs=outputs,
        output_embedding=embedding,
        final_norm=OffsetNorm(),
        projection_mode=PROJECTION_APPLY_FINAL_NORM_EXCEPT_LAST,
        atol=1e-6,
    )


def test_final_layer_equivalence_applies_final_logit_softcap() -> None:
    hidden = torch.tensor([[[3.0, -1.0]]])
    embedding = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        embedding.weight[:] = torch.eye(2)
    raw_logits = embedding(hidden)
    softcap = 2.0
    logits = torch.tanh(raw_logits / softcap) * softcap
    outputs = DummyOutputs(hidden_states=[torch.zeros_like(hidden), hidden], logits=logits)
    assert_final_layer_equivalence(
        outputs=outputs,
        output_embedding=embedding,
        final_norm=None,
        projection_mode=PROJECTION_APPLY_FINAL_NORM,
        model_config=SimpleNamespace(final_logit_softcapping=softcap),
        atol=1e-6,
    )
