from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from continuation_tokenizer import contextual_token_ids
from layerwise_extractor import PROJECTION_APPLY_FINAL_NORM
from scoring_engine import (
    count_unique_prefix_states,
    find_equivalent_optimized_batch_size,
    label_fast_path_eligible,
    score_readout_group,
    scorer_feature_hash,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) + 1 for ch in text]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [chr(value - 1) for value in ids]


class _ToyModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 512, hidden_size: int = 16, layer_count: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(hidden_size, hidden_size) for _ in range(layer_count))
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        _ = use_cache
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        hidden = self.embed(input_ids) * mask
        hidden = torch.cumsum(hidden, dim=1)
        hidden_states = [hidden]
        value = hidden
        for layer in self.layers:
            value = torch.tanh(layer(value))
            hidden_states.append(value)
        logits = self.lm_head(value)
        if return_dict:
            return SimpleNamespace(logits=logits, hidden_states=hidden_states if output_hidden_states else None)
        return logits, hidden_states


class _BatchSensitiveModel(_ToyModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
        )
        if input_ids.shape[0] > 1:
            adjusted_hidden_states = [state.clone() for state in outputs.hidden_states]
            adjusted_hidden_states[-1][:, :, 0] = adjusted_hidden_states[-1][:, :, 0] + 0.25
            outputs.hidden_states = adjusted_hidden_states
            outputs.logits = self.lm_head(adjusted_hidden_states[-1])
        if return_dict:
            return outputs
        return outputs.logits, outputs.hidden_states


@dataclass
class _Bundle:
    tokenizer: _Tokenizer
    model: _ToyModel
    device: str
    hidden_state_projection_mode: str


def _bundle() -> _Bundle:
    return _Bundle(
        tokenizer=_Tokenizer(),
        model=_ToyModel(),
        device="cpu",
        hidden_state_projection_mode=PROJECTION_APPLY_FINAL_NORM,
    )


def _batch_sensitive_bundle() -> _Bundle:
    return _Bundle(
        tokenizer=_Tokenizer(),
        model=_BatchSensitiveModel(),
        device="cpu",
        hidden_state_projection_mode=PROJECTION_APPLY_FINAL_NORM,
    )


def _readout_surfaces() -> dict[str, dict[str, str]]:
    return {
        "semantic_exact": {"A": "alpha", "B": "alps", "C": "beta"},
        "templated_semantic": {
            "A": "The answer is alpha",
            "B": "The answer is alps",
            "C": "The answer is beta",
        },
        "letter_label": {"A": "A", "B": "B", "C": "C"},
    }


def _assert_scientific_equivalence(reference: dict, candidate: dict) -> None:
    assert reference["final_winner"] == candidate["final_winner"]
    assert reference["option_surfaces"] == candidate["option_surfaces"]
    assert reference["per_option_token_ids"] == candidate["per_option_token_ids"]
    assert len(reference["layers"]) == len(candidate["layers"])
    for ref_layer, cand_layer in zip(reference["layers"], candidate["layers"]):
        assert ref_layer["layer_index"] == cand_layer["layer_index"]
        assert ref_layer["options"].keys() == cand_layer["options"].keys()
        for option_id in ref_layer["options"]:
            ref_stats = ref_layer["options"][option_id]
            cand_stats = cand_layer["options"][option_id]
            assert ref_stats["token_count"] == cand_stats["token_count"]
            assert ref_stats["sum_logprob"] == pytest.approx(cand_stats["sum_logprob"], abs=1e-5)
            assert ref_stats["mean_logprob"] == pytest.approx(cand_stats["mean_logprob"], abs=1e-6)
            assert ref_stats["normalized_probability"] == pytest.approx(cand_stats["normalized_probability"], abs=1e-6)


def test_label_fast_path_detects_single_token_labels() -> None:
    tokenizer = _Tokenizer()
    surfaces = {"A": "A", "B": "B", "C": "C"}
    assert label_fast_path_eligible(tokenizer, prompt_text="Answer: ", option_surfaces=surfaces) is True


def test_label_fast_path_rejects_multi_token_labels() -> None:
    tokenizer = _Tokenizer()
    surfaces = {"A": "AA", "B": "B", "C": "C"}
    assert label_fast_path_eligible(tokenizer, prompt_text="Answer: ", option_surfaces=surfaces) is False


def test_count_unique_prefix_states_shares_common_prefixes() -> None:
    tokenizer = _Tokenizer()
    sequences = {
        "A": contextual_token_ids(tokenizer, prompt_text="Answer: ", surface="alpha"),
        "B": contextual_token_ids(tokenizer, prompt_text="Answer: ", surface="alps"),
        "C": contextual_token_ids(tokenizer, prompt_text="Answer: ", surface="beta"),
    }
    assert count_unique_prefix_states(sequences) < sum(len(value) for value in sequences.values())


def test_optimized_group_scoring_matches_reference() -> None:
    bundle = _bundle()
    surfaces = _readout_surfaces()
    normalizations = {readout_id: "mean_logprob" for readout_id in surfaces}
    reference = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="reference_exact",
    )
    optimized = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=3,
    )
    for readout_id in reference:
        _assert_scientific_equivalence(reference[readout_id], optimized[readout_id])
    assert optimized["letter_label"]["execution_mode"] == "optimized_exact_root_only"
    assert optimized["semantic_exact"]["execution_mode"] == "optimized_exact_prefix_dag"


def test_optimized_group_scoring_is_batch_partition_invariant() -> None:
    bundle = _bundle()
    surfaces = _readout_surfaces()
    normalizations = {readout_id: "mean_logprob" for readout_id in surfaces}
    batched_small = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=1,
    )
    batched_large = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=8,
    )
    for readout_id in batched_small:
        _assert_scientific_equivalence(batched_small[readout_id], batched_large[readout_id])


def test_disabling_letter_label_fast_path_preserves_semantic_readouts() -> None:
    bundle = _bundle()
    surfaces = _readout_surfaces()
    normalizations = {readout_id: "mean_logprob" for readout_id in surfaces}
    with_fast_path = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=8,
    )
    without_fast_path = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=8,
        disable_letter_label_fast_path=True,
    )
    _assert_scientific_equivalence(with_fast_path["semantic_exact"], without_fast_path["semantic_exact"])
    _assert_scientific_equivalence(with_fast_path["templated_semantic"], without_fast_path["templated_semantic"])
    _assert_scientific_equivalence(with_fast_path["letter_label"], without_fast_path["letter_label"])
    assert with_fast_path["letter_label"]["execution_mode"] == "optimized_exact_root_only"
    assert without_fast_path["letter_label"]["execution_mode"] == "optimized_exact_prefix_dag"


def test_scorer_feature_hash_changes_when_letter_fast_path_is_disabled() -> None:
    tokenizer = _Tokenizer()
    surfaces = {"A": "A", "B": "B", "C": "C"}
    enabled_hash = scorer_feature_hash(
        tokenizer=tokenizer,
        prompt_text="Answer: ",
        readout_id="letter_label",
        option_surfaces=surfaces,
        scorer_engine="optimized_exact",
    )
    disabled_hash = scorer_feature_hash(
        tokenizer=tokenizer,
        prompt_text="Answer: ",
        readout_id="letter_label",
        option_surfaces=surfaces,
        scorer_engine="optimized_exact",
        disable_letter_label_fast_path=True,
    )
    assert enabled_hash != disabled_hash


def test_find_equivalent_optimized_batch_size_falls_back_to_batch_one() -> None:
    bundle = _batch_sensitive_bundle()
    surfaces = _readout_surfaces()
    normalizations = {readout_id: "mean_logprob" for readout_id in surfaces}
    reference, optimized, equivalence_result = find_equivalent_optimized_batch_size(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        candidate_batch_sizes=[8, 4, 2, 1],
        sum_logprob_abs_tol=1.0e-5,
        mean_logprob_abs_tol=1.0e-6,
        probability_abs_tol=1.0e-6,
    )
    assert equivalence_result.batch_size == 1
    assert equivalence_result.attempts[0]["attempted_batch_size"] == 8
    assert equivalence_result.attempts[-1]["attempted_batch_size"] == 2
    assert equivalence_result.drift_by_readout["letter_label"]["final_winner_matches"] is True
    for readout_id in reference:
        _assert_scientific_equivalence(reference[readout_id], optimized[readout_id])


def test_multi_token_letter_labels_fall_back_to_prefix_dag() -> None:
    bundle = _bundle()
    surfaces = _readout_surfaces()
    surfaces["letter_label"] = {"A": "AA", "B": "B", "C": "C"}
    normalizations = {readout_id: "mean_logprob" for readout_id in surfaces}
    optimized = score_readout_group(
        bundle,
        prompt_text="Answer: ",
        readout_option_surfaces=surfaces,
        normalization_by_readout=normalizations,
        scorer_engine="optimized_exact",
        max_batch_size=4,
    )
    assert optimized["letter_label"]["execution_mode"] == "optimized_exact_prefix_dag"
