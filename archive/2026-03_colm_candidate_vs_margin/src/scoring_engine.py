from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch

from continuation_tokenizer import contextual_token_ids, tokenize_surfaces_in_context
from layerwise_extractor import (
    LayerwiseOptionScore,
    LayerwiseStepResult,
    _find_final_norm,
    _hidden_state_sequence,
    project_hidden_state_to_logits,
    should_apply_final_norm,
)


SCORER_FEATURE_VERSION = "optimized_exact_v2"
EXECUTION_MODE_REFERENCE = "reference_exact"
EXECUTION_MODE_ROOT_ONLY = "optimized_exact_root_only"
EXECUTION_MODE_PREFIX_DAG = "optimized_exact_prefix_dag"

# Target-GPU bounded-drift policy for batched optimized scoring.
# The hard gate remains: no final-winner flips.
# These tolerances are used only for reference-vs-optimized calibration, not for
# the reference scorer itself.
BOUNDED_DRIFT_SUM_LOGPROB_ABS_TOL = 0.31
BOUNDED_DRIFT_MEAN_LOGPROB_ABS_TOL = 0.16
BOUNDED_DRIFT_PROBABILITY_ABS_TOL = 0.025


@dataclass(frozen=True)
class PromptRootOutputs:
    prompt_hash: str
    prompt_token_count: int
    answer_start_token_index: int
    per_layer_logprobs: list[torch.Tensor]
    direct_final_logprobs: torch.Tensor


@dataclass(frozen=True)
class EquivalenceResult:
    batch_size: int
    disable_letter_label_fast_path: bool
    attempts: list[dict[str, Any]]
    drift_by_readout: dict[str, dict[str, Any]]
    execution_modes: dict[str, str]
    feature_flags_by_readout: dict[str, dict[str, Any]]
    profile: dict[str, Any] | None = None


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def label_fast_path_eligible(tokenizer: Any, *, prompt_text: str, option_surfaces: dict[str, str]) -> bool:
    try:
        return all(
            len(contextual_token_ids(tokenizer, prompt_text=prompt_text, surface=surface)) == 1
            for surface in option_surfaces.values()
        )
    except ValueError:
        return False


def scorer_feature_flags(
    *,
    tokenizer: Any,
    prompt_text: str,
    readout_id: str,
    option_surfaces: dict[str, str],
    scorer_engine: str,
    disable_letter_label_fast_path: bool = False,
) -> dict[str, Any]:
    if scorer_engine == "reference_exact":
        return {
            "feature_version": SCORER_FEATURE_VERSION,
            "grouped_prefix_scoring": False,
            "letter_label_fast_path_eligible": False,
            "letter_label_fast_path_enabled": False,
            "execution_mode": EXECUTION_MODE_REFERENCE,
        }
    if scorer_engine != "optimized_exact":
        raise ValueError(f"unsupported scorer_engine={scorer_engine}")
    eligible = bool(
        readout_id == "letter_label"
        and label_fast_path_eligible(
            tokenizer,
            prompt_text=prompt_text,
            option_surfaces=option_surfaces,
        )
    )
    enabled = bool(eligible and not disable_letter_label_fast_path)
    return {
        "feature_version": SCORER_FEATURE_VERSION,
        "grouped_prefix_scoring": True,
        "letter_label_fast_path_eligible": eligible,
        "letter_label_fast_path_enabled": enabled,
        "execution_mode": EXECUTION_MODE_ROOT_ONLY if enabled else EXECUTION_MODE_PREFIX_DAG,
    }


def scorer_feature_hash(
    *,
    tokenizer: Any,
    prompt_text: str,
    readout_id: str,
    option_surfaces: dict[str, str],
    scorer_engine: str,
    disable_letter_label_fast_path: bool = False,
) -> str:
    return _stable_hash(
        scorer_feature_flags(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            readout_id=readout_id,
            option_surfaces=option_surfaces,
            scorer_engine=scorer_engine,
            disable_letter_label_fast_path=disable_letter_label_fast_path,
        )
    )


def measure_scientific_drift(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["option_surfaces"] != candidate["option_surfaces"]:
        raise ValueError("option_surfaces mismatch")
    if reference["per_option_token_ids"] != candidate["per_option_token_ids"]:
        raise ValueError("per_option_token_ids mismatch")
    if len(reference["layers"]) != len(candidate["layers"]):
        raise ValueError("layer_count mismatch")
    if reference["layers"][-1]["options"].keys() != candidate["layers"][-1]["options"].keys():
        raise ValueError("layer option key mismatch")

    sum_deltas: list[float] = []
    mean_deltas: list[float] = []
    probability_deltas: list[float] = []
    for ref_layer, cand_layer in zip(reference["layers"], candidate["layers"]):
        if ref_layer["layer_index"] != cand_layer["layer_index"]:
            raise ValueError("layer_index mismatch")
        if ref_layer["options"].keys() != cand_layer["options"].keys():
            raise ValueError("layer option key mismatch")
        for option_id in ref_layer["options"]:
            ref_stats = ref_layer["options"][option_id]
            cand_stats = cand_layer["options"][option_id]
            if ref_stats["token_count"] != cand_stats["token_count"]:
                raise ValueError(f"token_count mismatch for option_id={option_id}")
            sum_deltas.append(abs(float(ref_stats["sum_logprob"]) - float(cand_stats["sum_logprob"])))
            mean_deltas.append(abs(float(ref_stats["mean_logprob"]) - float(cand_stats["mean_logprob"])))
            probability_deltas.append(
                abs(float(ref_stats["normalized_probability"]) - float(cand_stats["normalized_probability"]))
            )
    return {
        "reference_final_winner": reference["final_winner"],
        "candidate_final_winner": candidate["final_winner"],
        "final_winner_matches": reference["final_winner"] == candidate["final_winner"],
        "layer_count": len(reference["layers"]),
        "option_count": len(reference["layers"][-1]["options"]),
        "max_sum_logprob_abs_delta": max(sum_deltas, default=0.0),
        "mean_sum_logprob_abs_delta": (sum(sum_deltas) / len(sum_deltas)) if sum_deltas else 0.0,
        "max_mean_logprob_abs_delta": max(mean_deltas, default=0.0),
        "mean_mean_logprob_abs_delta": (sum(mean_deltas) / len(mean_deltas)) if mean_deltas else 0.0,
        "max_probability_abs_delta": max(probability_deltas, default=0.0),
        "mean_probability_abs_delta": (sum(probability_deltas) / len(probability_deltas))
        if probability_deltas
        else 0.0,
    }


def assert_scientific_equivalence(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    sum_logprob_abs_tol: float = 1.0e-5,
    mean_logprob_abs_tol: float = 1.0e-6,
    probability_abs_tol: float = 1.0e-6,
) -> dict[str, Any]:
    drift = measure_scientific_drift(reference, candidate)
    if not drift["final_winner_matches"]:
        raise ValueError(
            f"final_winner mismatch: reference={reference['final_winner']} candidate={candidate['final_winner']}"
        )
    if drift["max_sum_logprob_abs_delta"] > sum_logprob_abs_tol:
        raise ValueError(f"sum_logprob mismatch; drift_summary={json.dumps(drift, sort_keys=True)}")
    if drift["max_mean_logprob_abs_delta"] > mean_logprob_abs_tol:
        raise ValueError(f"mean_logprob mismatch; drift_summary={json.dumps(drift, sort_keys=True)}")
    if drift["max_probability_abs_delta"] > probability_abs_tol:
        raise ValueError(f"normalized_probability mismatch; drift_summary={json.dumps(drift, sort_keys=True)}")
    return drift


def _is_oom_like_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or exc.__class__.__name__ == "OutOfMemoryError"


def find_equivalent_optimized_batch_size(
    bundle: Any,
    *,
    prompt_text: str,
    readout_option_surfaces: dict[str, dict[str, str]],
    normalization_by_readout: dict[str, str],
    candidate_batch_sizes: list[int],
    force_disable_letter_label_fast_path: bool | None = None,
    sum_logprob_abs_tol: float = BOUNDED_DRIFT_SUM_LOGPROB_ABS_TOL,
    mean_logprob_abs_tol: float = BOUNDED_DRIFT_MEAN_LOGPROB_ABS_TOL,
    probability_abs_tol: float = BOUNDED_DRIFT_PROBABILITY_ABS_TOL,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], EquivalenceResult]:
    reference = score_readout_group(
        bundle,
        prompt_text=prompt_text,
        readout_option_surfaces=readout_option_surfaces,
        normalization_by_readout=normalization_by_readout,
        scorer_engine="reference_exact",
        max_batch_size=1,
    )
    tokenizer = bundle.tokenizer
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    letter_label_eligible = bool(
        "letter_label" in readout_option_surfaces
        and scorer_feature_flags(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            readout_id="letter_label",
            option_surfaces=readout_option_surfaces["letter_label"],
            scorer_engine="optimized_exact",
        )["letter_label_fast_path_eligible"]
    )
    if force_disable_letter_label_fast_path is None:
        disable_options = [False]
        if letter_label_eligible:
            disable_options.append(True)
    else:
        disable_options = [bool(force_disable_letter_label_fast_path)]
    for disable_letter_label_fast_path in disable_options:
        for raw_batch_size in candidate_batch_sizes:
            batch_size = max(1, int(raw_batch_size))
            try:
                optimized = score_readout_group(
                    bundle,
                    prompt_text=prompt_text,
                    readout_option_surfaces=readout_option_surfaces,
                    normalization_by_readout=normalization_by_readout,
                    scorer_engine="optimized_exact",
                    max_batch_size=batch_size,
                    disable_letter_label_fast_path=disable_letter_label_fast_path,
                )
                drift_by_readout = {
                    readout_id: assert_scientific_equivalence(
                        reference[readout_id],
                        optimized[readout_id],
                        sum_logprob_abs_tol=sum_logprob_abs_tol,
                        mean_logprob_abs_tol=mean_logprob_abs_tol,
                        probability_abs_tol=probability_abs_tol,
                    )
                    for readout_id in reference
                }
                profile = None
                if letter_label_eligible and "letter_label" in optimized:
                    profile = profile_letter_label_fast_path(
                        bundle,
                        prompt_text=prompt_text,
                        readout_option_surfaces=readout_option_surfaces,
                        normalization_by_readout=normalization_by_readout,
                        max_batch_size=batch_size,
                    )
                return reference, optimized, EquivalenceResult(
                    batch_size=batch_size,
                    disable_letter_label_fast_path=disable_letter_label_fast_path,
                    attempts=attempts,
                    drift_by_readout=drift_by_readout,
                    execution_modes={
                        readout_id: str(optimized_result["execution_mode"])
                        for readout_id, optimized_result in optimized.items()
                    },
                    feature_flags_by_readout={
                        readout_id: dict(optimized_result["scorer_feature_flags"])
                        for readout_id, optimized_result in optimized.items()
                    },
                    profile=profile,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                attempts.append(
                    {
                        "attempted_batch_size": batch_size,
                        "disable_letter_label_fast_path": disable_letter_label_fast_path,
                        "reason": "oom" if _is_oom_like_error(exc) else "equivalence",
                        "message": str(exc),
                    }
                )
                if _is_oom_like_error(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    if last_error is not None:
        raise RuntimeError(f"no equivalence-safe batch size found: {last_error}") from last_error
    raise RuntimeError("no equivalence-safe batch size candidates available")


def count_unique_prefix_states(token_sequences: dict[str, list[int]]) -> int:
    states = set()
    for token_ids in token_sequences.values():
        for prefix_length in range(len(token_ids)):
            states.add(tuple(token_ids[:prefix_length]))
    return len(states)


def _aggregate_readout(
    *,
    normalization: str,
    option_surfaces: dict[str, str],
    option_scores: dict[str, LayerwiseOptionScore],
    scorer_engine: str,
    scorer_feature_flags: dict[str, Any],
) -> dict[str, Any]:
    layer_count = len(next(iter(option_scores.values())).layer_sums)
    layers = []
    for layer_idx in range(layer_count):
        layer_records = {}
        values = []
        ordered_option_ids = sorted(option_scores)
        for option_id in ordered_option_ids:
            option_score = option_scores[option_id]
            aggregate = option_score.layer_means[layer_idx] if normalization == "mean_logprob" else option_score.layer_sums[layer_idx]
            layer_records[option_id] = {
                "sum_logprob": option_score.layer_sums[layer_idx],
                "mean_logprob": option_score.layer_means[layer_idx],
                "token_count": len(option_score.token_ids),
            }
            values.append(aggregate)
        probabilities = _softmax(values)
        for option_id, probability in zip(ordered_option_ids, probabilities):
            layer_records[option_id]["normalized_probability"] = probability
        layers.append({"layer_index": layer_idx, "options": layer_records})
    final_winner = max(
        layers[-1]["options"].items(),
        key=lambda item: item[1]["normalized_probability"],
    )[0]
    return {
        "normalization": normalization,
        "final_winner": final_winner,
        "option_surfaces": option_surfaces,
        "layers": layers,
        "per_option_token_ids": {option_id: score.token_ids for option_id, score in option_scores.items()},
        "scorer_engine": scorer_engine,
        "execution_mode": str(scorer_feature_flags.get("execution_mode", EXECUTION_MODE_REFERENCE)),
        "scorer_feature_flags": dict(scorer_feature_flags),
    }


def _score_token_ids_reference(bundle: Any, *, prompt_token_ids: list[int], option_ids: list[int]) -> LayerwiseOptionScore:
    if not prompt_token_ids:
        raise ValueError("prompt tokenization produced no tokens")
    if not option_ids:
        raise ValueError("option tokenization produced no tokens")
    model = bundle.model
    device = bundle.device
    sequence = list(prompt_token_ids)
    output_embedding = model.get_output_embeddings()
    final_norm = _find_final_norm(model)
    steps: list[LayerwiseStepResult] = []
    with torch.no_grad():
        for forced_id in option_ids:
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            logits = outputs.logits[:, -1, :].float()
            hidden_states = _hidden_state_sequence(outputs)
            forced_logprobs = []
            for idx, hidden in enumerate(hidden_states):
                layer_logits = project_hidden_state_to_logits(
                    hidden[:, -1, :],
                    output_embedding=output_embedding,
                    final_norm=final_norm,
                    apply_final_norm=should_apply_final_norm(
                        bundle.hidden_state_projection_mode,
                        is_last_hidden_state=idx == len(hidden_states) - 1,
                    ),
                    model_config=getattr(model, "config", None),
                )
                layer_logprobs = torch.log_softmax(layer_logits, dim=-1)
                forced_logprobs.append(float(layer_logprobs[0, forced_id].item()))
            direct_final = torch.log_softmax(logits, dim=-1)
            steps.append(
                LayerwiseStepResult(
                    forced_token_id=int(forced_id),
                    forced_token_logprobs_by_layer=forced_logprobs,
                    direct_final_logprob=float(direct_final[0, forced_id].item()),
                )
            )
            sequence.append(int(forced_id))
    layer_count = len(steps[0].forced_token_logprobs_by_layer)
    sums = [0.0 for _ in range(layer_count)]
    for step in steps:
        for idx, value in enumerate(step.forced_token_logprobs_by_layer):
            sums[idx] += float(value)
    means = [value / float(len(steps)) for value in sums]
    return LayerwiseOptionScore(token_ids=list(option_ids), steps=steps, layer_sums=sums, layer_means=means)


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[idx : idx + size] for idx in range(0, len(items), max(1, size))]


def _compute_prompt_root_outputs(
    bundle: Any,
    *,
    prompt_text: str,
    prompt_token_ids: list[int],
) -> PromptRootOutputs:
    outputs = _batched_prefix_outputs(bundle, prompt_token_ids=prompt_token_ids, prefixes=[tuple()], max_batch_size=1)
    prefix_outputs = outputs[tuple()]
    return PromptRootOutputs(
        prompt_hash=_stable_hash({"prompt_text": prompt_text}),
        prompt_token_count=len(prompt_token_ids),
        answer_start_token_index=len(prompt_token_ids),
        per_layer_logprobs=list(prefix_outputs["per_layer_logprobs"]),
        direct_final_logprobs=prefix_outputs["direct_final_logprobs"],
    )


def _batched_prefix_outputs(
    bundle: Any,
    *,
    prompt_token_ids: list[int],
    prefixes: list[tuple[int, ...]],
    max_batch_size: int,
) -> dict[tuple[int, ...], dict[str, Any]]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    output_embedding = model.get_output_embeddings()
    final_norm = _find_final_norm(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    results: dict[tuple[int, ...], dict[str, Any]] = {}
    with torch.no_grad():
        for batch_prefixes in _chunked(prefixes, max_batch_size):
            sequences = [prompt_token_ids + list(prefix) for prefix in batch_prefixes]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
            for row_idx, sequence in enumerate(sequences):
                input_ids[row_idx, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long, device=device)
                attention_mask[row_idx, -len(sequence) :] = 1
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=device)
            hidden_states = _hidden_state_sequence(outputs)
            per_layer_logprobs = []
            for layer_idx, hidden in enumerate(hidden_states):
                last_hidden = hidden[batch_indices, last_positions, :]
                layer_logits = project_hidden_state_to_logits(
                    last_hidden,
                    output_embedding=output_embedding,
                    final_norm=final_norm,
                    apply_final_norm=should_apply_final_norm(
                        bundle.hidden_state_projection_mode,
                        is_last_hidden_state=layer_idx == len(hidden_states) - 1,
                    ),
                    model_config=getattr(model, "config", None),
                )
                per_layer_logprobs.append(torch.log_softmax(layer_logits, dim=-1))
            direct_final_logprobs = torch.log_softmax(outputs.logits[batch_indices, last_positions, :].float(), dim=-1)
            for row_idx, prefix in enumerate(batch_prefixes):
                results[prefix] = {
                    "per_layer_logprobs": [tensor[row_idx].detach().cpu() for tensor in per_layer_logprobs],
                    "direct_final_logprobs": direct_final_logprobs[row_idx].detach().cpu(),
                }
    return results


def _score_token_sequences_optimized(
    bundle: Any,
    *,
    prompt_token_ids: list[int],
    token_sequences: dict[tuple[str, str], list[int]],
    max_batch_size: int,
    prompt_root_outputs: PromptRootOutputs | None = None,
) -> dict[tuple[str, str], LayerwiseOptionScore]:
    if not token_sequences:
        return {}
    layer_count = None
    item_steps: dict[tuple[str, str], list[LayerwiseStepResult]] = {key: [] for key in token_sequences}
    prefix_children: dict[tuple[int, ...], set[int]] = defaultdict(set)
    child_to_items: dict[tuple[tuple[int, ...], int], list[tuple[str, str]]] = defaultdict(list)
    queue: deque[tuple[int, ...]] = deque([tuple()])
    queued = {tuple()}
    for key, token_ids in token_sequences.items():
        for prefix_length, forced_id in enumerate(token_ids):
            prefix = tuple(token_ids[:prefix_length])
            prefix_children[prefix].add(int(forced_id))
            child_to_items[(prefix, int(forced_id))].append(key)
    while queue:
        frontier = list(queue)
        queue.clear()
        queued.clear()
        outputs_by_prefix = {}
        missing_prefixes = [
            prefix
            for prefix in frontier
            if not (prompt_root_outputs is not None and prefix == tuple())
        ]
        if missing_prefixes:
            outputs_by_prefix.update(
                _batched_prefix_outputs(
                    bundle,
                    prompt_token_ids=prompt_token_ids,
                    prefixes=missing_prefixes,
                    max_batch_size=max_batch_size,
                )
            )
        if prompt_root_outputs is not None and tuple() in frontier:
            outputs_by_prefix[tuple()] = {
                "per_layer_logprobs": list(prompt_root_outputs.per_layer_logprobs),
                "direct_final_logprobs": prompt_root_outputs.direct_final_logprobs,
            }
        for prefix in frontier:
            prefix_outputs = outputs_by_prefix[prefix]
            layer_logprobs = prefix_outputs["per_layer_logprobs"]
            direct_final = prefix_outputs["direct_final_logprobs"]
            if layer_count is None:
                layer_count = len(layer_logprobs)
            for forced_id in sorted(prefix_children.get(prefix, set())):
                step = LayerwiseStepResult(
                    forced_token_id=int(forced_id),
                    forced_token_logprobs_by_layer=[float(values[int(forced_id)].item()) for values in layer_logprobs],
                    direct_final_logprob=float(direct_final[int(forced_id)].item()),
                )
                next_prefix = tuple([*prefix, int(forced_id)])
                for item_key in child_to_items[(prefix, int(forced_id))]:
                    item_steps[item_key].append(step)
                    if len(item_steps[item_key]) < len(token_sequences[item_key]) and next_prefix not in queued:
                        queue.append(next_prefix)
                        queued.add(next_prefix)
    if layer_count is None:
        raise ValueError("optimized scorer did not produce any layers")
    results: dict[tuple[str, str], LayerwiseOptionScore] = {}
    for key, steps in item_steps.items():
        sums = [0.0 for _ in range(layer_count)]
        for step in steps:
            for idx, value in enumerate(step.forced_token_logprobs_by_layer):
                sums[idx] += float(value)
        means = [value / float(len(steps)) for value in sums]
        results[key] = LayerwiseOptionScore(
            token_ids=list(token_sequences[key]),
            steps=steps,
            layer_sums=sums,
            layer_means=means,
        )
    return results


def _score_letter_label_from_prompt_root(
    prompt_root_outputs: PromptRootOutputs,
    *,
    tokenizations: dict[str, Any],
) -> dict[str, LayerwiseOptionScore]:
    option_scores: dict[str, LayerwiseOptionScore] = {}
    for option_id, tokenization in tokenizations.items():
        if len(tokenization.token_ids) != 1:
            raise ValueError(f"letter_label fast path requires single-token labels, got option_id={option_id}")
        forced_id = int(tokenization.token_ids[0])
        step = LayerwiseStepResult(
            forced_token_id=forced_id,
            forced_token_logprobs_by_layer=[
                float(layer_logprobs[forced_id].item()) for layer_logprobs in prompt_root_outputs.per_layer_logprobs
            ],
            direct_final_logprob=float(prompt_root_outputs.direct_final_logprobs[forced_id].item()),
        )
        sums = [float(value) for value in step.forced_token_logprobs_by_layer]
        option_scores[option_id] = LayerwiseOptionScore(
            token_ids=list(tokenization.token_ids),
            steps=[step],
            layer_sums=sums,
            layer_means=list(sums),
        )
    return option_scores


def profile_letter_label_fast_path(
    bundle: Any,
    *,
    prompt_text: str,
    readout_option_surfaces: dict[str, dict[str, str]],
    normalization_by_readout: dict[str, str],
    max_batch_size: int,
) -> dict[str, Any]:
    if "letter_label" not in readout_option_surfaces:
        return {
            "eligible": False,
            "used": False,
            "baseline_elapsed_seconds": None,
            "fast_path_elapsed_seconds": None,
            "speedup_pct": None,
            "meets_threshold": False,
            "threshold_pct": 5.0,
        }
    eligible = scorer_feature_flags(
        tokenizer=bundle.tokenizer,
        prompt_text=prompt_text,
        readout_id="letter_label",
        option_surfaces=readout_option_surfaces["letter_label"],
        scorer_engine="optimized_exact",
    )["letter_label_fast_path_eligible"]
    threshold_pct = 5.0
    if not eligible:
        return {
            "eligible": False,
            "used": False,
            "baseline_elapsed_seconds": None,
            "fast_path_elapsed_seconds": None,
            "speedup_pct": None,
            "meets_threshold": False,
            "threshold_pct": threshold_pct,
        }
    start = time.perf_counter()
    baseline = score_readout_group(
        bundle,
        prompt_text=prompt_text,
        readout_option_surfaces=readout_option_surfaces,
        normalization_by_readout=normalization_by_readout,
        scorer_engine="optimized_exact",
        max_batch_size=max_batch_size,
        disable_letter_label_fast_path=True,
    )
    baseline_elapsed = max(time.perf_counter() - start, 1.0e-9)
    start = time.perf_counter()
    fast_path = score_readout_group(
        bundle,
        prompt_text=prompt_text,
        readout_option_surfaces=readout_option_surfaces,
        normalization_by_readout=normalization_by_readout,
        scorer_engine="optimized_exact",
        max_batch_size=max_batch_size,
        disable_letter_label_fast_path=False,
    )
    fast_path_elapsed = max(time.perf_counter() - start, 1.0e-9)
    drift_summary = assert_scientific_equivalence(baseline["letter_label"], fast_path["letter_label"])
    speedup_pct = ((baseline_elapsed - fast_path_elapsed) / baseline_elapsed) * 100.0
    return {
        "eligible": True,
        "used": bool(fast_path["letter_label"]["scorer_feature_flags"]["letter_label_fast_path_enabled"]),
        "baseline_elapsed_seconds": baseline_elapsed,
        "fast_path_elapsed_seconds": fast_path_elapsed,
        "speedup_pct": speedup_pct,
        "meets_threshold": speedup_pct >= threshold_pct,
        "threshold_pct": threshold_pct,
        "drift_summary": drift_summary,
    }


def score_readout_group(
    bundle: Any,
    *,
    prompt_text: str,
    readout_option_surfaces: dict[str, dict[str, str]],
    normalization_by_readout: dict[str, str],
    scorer_engine: str = "reference_exact",
    max_batch_size: int = 1,
    disable_letter_label_fast_path: bool = False,
) -> dict[str, dict[str, Any]]:
    tokenizer = bundle.tokenizer
    prompt_token_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    feature_flags_by_readout = {
        readout_id: scorer_feature_flags(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            readout_id=readout_id,
            option_surfaces=option_surfaces,
            scorer_engine=scorer_engine,
            disable_letter_label_fast_path=disable_letter_label_fast_path,
        )
        for readout_id, option_surfaces in readout_option_surfaces.items()
    }
    contextual_by_readout = {
        readout_id: tokenize_surfaces_in_context(tokenizer, prompt_text=prompt_text, option_surfaces=option_surfaces)
        for readout_id, option_surfaces in readout_option_surfaces.items()
    }
    if scorer_engine == "reference_exact":
        results = {}
        for readout_id, tokenizations in contextual_by_readout.items():
            option_scores = {
                option_id: _score_token_ids_reference(
                    bundle,
                    prompt_token_ids=prompt_token_ids,
                    option_ids=tokenization.token_ids,
                )
                for option_id, tokenization in tokenizations.items()
            }
            results[readout_id] = _aggregate_readout(
                normalization=normalization_by_readout[readout_id],
                option_surfaces=readout_option_surfaces[readout_id],
                option_scores=option_scores,
                scorer_engine=scorer_engine,
                scorer_feature_flags=feature_flags_by_readout[readout_id],
            )
        return results
    if scorer_engine != "optimized_exact":
        raise ValueError(f"unsupported scorer_engine={scorer_engine}")
    timings = {
        "prompt_root_evaluation_seconds": 0.0,
        "letter_label_scoring_seconds": 0.0,
        "continuation_scoring_seconds": 0.0,
        "total_seconds": 0.0,
    }
    total_start = time.perf_counter()
    root_start = time.perf_counter()
    prompt_root_outputs = _compute_prompt_root_outputs(
        bundle,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
    )
    timings["prompt_root_evaluation_seconds"] = time.perf_counter() - root_start
    grouped_sequences = {}
    root_only_option_scores: dict[str, dict[str, LayerwiseOptionScore]] = {}
    letter_start = time.perf_counter()
    for readout_id, tokenizations in contextual_by_readout.items():
        execution_mode = feature_flags_by_readout[readout_id]["execution_mode"]
        if execution_mode == EXECUTION_MODE_ROOT_ONLY:
            root_only_option_scores[readout_id] = _score_letter_label_from_prompt_root(
                prompt_root_outputs,
                tokenizations=tokenizations,
            )
            continue
        for option_id, tokenization in tokenizations.items():
            grouped_sequences[(readout_id, option_id)] = tokenization.token_ids
    timings["letter_label_scoring_seconds"] = time.perf_counter() - letter_start
    continuation_start = time.perf_counter()
    option_scores_all = _score_token_sequences_optimized(
        bundle,
        prompt_token_ids=prompt_token_ids,
        token_sequences=grouped_sequences,
        max_batch_size=max_batch_size,
        prompt_root_outputs=prompt_root_outputs,
    )
    timings["continuation_scoring_seconds"] = time.perf_counter() - continuation_start
    timings["total_seconds"] = time.perf_counter() - total_start
    results = {}
    for readout_id, tokenizations in contextual_by_readout.items():
        option_scores = root_only_option_scores.get(readout_id)
        if option_scores is None:
            option_scores = {
                option_id: option_scores_all[(readout_id, option_id)]
                for option_id in tokenizations
            }
        aggregated = _aggregate_readout(
            normalization=normalization_by_readout[readout_id],
            option_surfaces=readout_option_surfaces[readout_id],
            option_scores=option_scores,
            scorer_engine=scorer_engine,
            scorer_feature_flags=feature_flags_by_readout[readout_id],
        )
        aggregated["timing_seconds"] = dict(timings)
        results[readout_id] = aggregated
    return results
