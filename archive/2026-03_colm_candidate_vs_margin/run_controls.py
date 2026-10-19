from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_loader import CanonicalExample, CanonicalOption
from execution_config import (
    build_option_surfaces,
    load_controls_config,
    load_prompt_templates,
    load_readout_specs,
    permutation_for_spec,
    resolve_permutation_specs,
    resolve_prompt_template_for_mode,
    resolve_readout_specs,
)
from io_utils import read_jsonl, write_json_atomic, write_jsonl_atomic
from layerwise_extractor import _find_final_norm, assert_final_layer_equivalence
from model_loader import load_model_bundle, validate_requested_device
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from prompt_builder import apply_permutation, build_prompt
from run_manifest import load_run_manifest, resolve_manifest_path, update_run_manifest, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_named_specs, resolve_panel_name
from score_cache import ScoreCache, ScoreCacheKey
from scoring_engine import (
    find_equivalent_optimized_batch_size,
    score_readout_group,
    scorer_feature_hash,
)
from tokenizer_audit import build_token_audit_records


def _require_inference_allowed(env_var_name: str) -> None:
    if os.environ.get(env_var_name) != "1":
        raise SystemExit(f"inference blocked: expected {env_var_name}=1")


def _load_examples(path: Path) -> list[CanonicalExample]:
    examples: list[CanonicalExample] = []
    for row in read_jsonl(path):
        row["options"] = [CanonicalOption(**option) for option in row["options"]]
        examples.append(CanonicalExample(**row))
    return examples


def _validate_model_readout(bundle: Any) -> None:
    tokenizer = bundle.tokenizer
    model = bundle.model
    prompt_ids = tokenizer.encode("Sanity check.\nFinal answer:", add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("sanity prompt tokenized to zero length")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    assert_final_layer_equivalence(
        outputs=outputs,
        output_embedding=model.get_output_embeddings(),
        final_norm=_find_final_norm(model),
        projection_mode=bundle.hidden_state_projection_mode,
        model_config=getattr(model, "config", None),
    )


def _unit_key(model_name: str, example_id: str, permutation_name: str, readout_id: str) -> str:
    return f"{model_name}:{example_id}:{permutation_name}:{readout_id}"


def _existing_rows_by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit_key = row.get("unit_key")
        if unit_key:
            out[str(unit_key)] = row
    return out


def _existing_audits_by_unit(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unit_key = row.get("unit_key")
        if unit_key:
            out[str(unit_key)].append(row)
    return dict(out)


def _path_for_manifest_row(paths: ExperimentPaths, path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(paths.workspace_root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def _score_cache_key(
    *,
    spec: Any,
    example: CanonicalExample,
    rendered_prompt: Any,
    prompt_template: Any,
    readout_spec: Any,
    permutation_spec: Any,
    scorer_engine: str,
    scorer_feature_hash_value: str,
    dtype: str,
) -> ScoreCacheKey:
    return ScoreCacheKey(
        model_id=spec.model_id,
        tokenizer_revision=spec.tokenizer_revision,
        model_revision=spec.model_revision,
        dataset_revision=example.dataset_revision,
        prompt_template_hash=prompt_template.config_hash(),
        readout_id=readout_spec.id,
        permutation_id=permutation_spec.name,
        scoring_config_hash=readout_spec.normalization,
        unit_id=example.example_id,
        prompt_render_hash=rendered_prompt.prompt_hash,
        prompt_config_hash=prompt_template.config_hash(),
        readout_config_hash=readout_spec.config_hash(),
        permutation_config_hash=permutation_spec.config_hash(),
        scorer_engine=scorer_engine,
        scorer_feature_hash=scorer_feature_hash_value,
        dtype=dtype,
    )


def _failure_row(
    *,
    run_id: str,
    unit_key: str,
    reason_code: str,
    message: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": "run_controls",
        "unit_key": unit_key,
        "severity": "error",
        "reason_code": reason_code,
        "message": message,
        "metadata": metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--scorer-engine", default="optimized_exact")
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--resume-failed-only", action="store_true")
    return parser.parse_args()


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or exc.__class__.__name__ == "OutOfMemoryError"


def _candidate_batch_sizes(device: str, requested_max_batch_size: int) -> list[int]:
    if requested_max_batch_size > 0:
        sizes = [int(requested_max_batch_size)]
        current = int(requested_max_batch_size)
        while current > 1:
            current = max(1, current // 2)
            if current not in sizes:
                sizes.append(current)
        return sizes
    if str(device).startswith("cuda"):
        return [32, 16, 8, 4, 2, 1]
    return [1]


def _candidate_batch_sizes_from_safe_value(safe_batch_size: int | None, fallback: list[int]) -> list[int]:
    if safe_batch_size is None:
        return list(fallback)
    sizes = [int(safe_batch_size)]
    current = int(safe_batch_size)
    while current > 1:
        current = max(1, current // 2)
        if current not in sizes:
            sizes.append(current)
    return sizes


def _score_group_with_backoff(
    bundle: Any,
    *,
    prompt_text: str,
    readout_option_surfaces: dict[str, dict[str, str]],
    normalization_by_readout: dict[str, str],
    scorer_engine: str,
    candidate_batch_sizes: list[int],
    disable_letter_label_fast_path: bool = False,
) -> tuple[dict[str, dict[str, Any]], int, list[dict[str, Any]]]:
    import torch  # noqa: PLC0415

    oom_events: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for batch_size in candidate_batch_sizes:
        try:
            results = score_readout_group(
                bundle,
                prompt_text=prompt_text,
                readout_option_surfaces=readout_option_surfaces,
                normalization_by_readout=normalization_by_readout,
                scorer_engine=scorer_engine,
                max_batch_size=max(1, int(batch_size)),
                disable_letter_label_fast_path=disable_letter_label_fast_path,
            )
            return results, max(1, int(batch_size)), oom_events
        except Exception as exc:  # noqa: BLE001
            if scorer_engine == "optimized_exact" and _is_oom_error(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_events.append({"attempted_batch_size": int(batch_size), "message": str(exc)})
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise RuntimeError(f"no safe batch size found for optimized scorer: {last_error}") from last_error
    raise RuntimeError("no safe batch size candidates available")


def _rendered_group_prompts(
    *,
    example: CanonicalExample,
    readout_specs: list[Any],
    prompt_template: Any,
    spec: Any,
    permutation: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    rendered_by_readout = {
        readout_spec.id: build_prompt(
            example,
            template_id=prompt_template.template_id,
            readout_id=readout_spec.id,
            prompt_mode=spec.prompt_mode,
            answer_prefix=prompt_template.answer_prefix,
            permutation=permutation,
            tokenizer=tokenizer,
            system_prompt=prompt_template.system_prompt,
            system_prompt_handling=spec.system_prompt_handling,
        )
        for readout_spec in readout_specs
    }
    base = next(iter(rendered_by_readout.values()))
    for rendered in rendered_by_readout.values():
        if rendered.rendered_prompt != base.rendered_prompt:
            raise ValueError("grouped readouts produced inconsistent rendered prompts")
        if rendered.prompt_hash != base.prompt_hash:
            raise ValueError("grouped readouts produced inconsistent prompt hashes")
        if rendered.answer_start_token_index != base.answer_start_token_index:
            raise ValueError("grouped readouts produced inconsistent answer boundaries")
    return rendered_by_readout


def main() -> None:
    args = _parse_args()

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    _require_inference_allowed(str(runtime_cfg.get("inference_env_var", "SCWI_ALLOW_INFERENCE")))

    explicit_panel = None
    if args.smoke or args.panel is not None:
        explicit_panel = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    manifest = load_run_manifest(paths, run_id=args.run_id, panel_name=explicit_panel)
    validate_run_manifest_config_hashes(paths, manifest)
    run_id = str(manifest["run_id"])
    panel_name = str(manifest["panel_name"])
    run_paths = paths.run_paths(run_id)
    run_paths.ensure_dirs()
    device = validate_requested_device(str(args.device or runtime_cfg.get("default_device", "cpu")))

    prepared_examples_path = resolve_manifest_path(paths, manifest, "prepared_examples")
    score_manifest_path = resolve_manifest_path(paths, manifest, "score_manifest")
    token_audits_path = resolve_manifest_path(paths, manifest, "token_audits")
    failure_log_path = resolve_manifest_path(paths, manifest, "run_controls_failures")
    smoke_manifest_path = resolve_manifest_path(paths, manifest, "smoke_manifest")

    all_examples = _load_examples(prepared_examples_path)
    selected_dataset_names = set(str(name) for name in manifest.get("selections", {}).get("datasets", []))
    examples = [example for example in all_examples if not selected_dataset_names or example.dataset_name in selected_dataset_names]
    if args.max_examples > 0:
        examples = examples[: int(args.max_examples)]

    prompt_templates = load_prompt_templates(paths.configs_dir / "prompts.yaml")
    readout_specs = resolve_readout_specs(
        list(manifest.get("selections", {}).get("readouts", [])),
        load_readout_specs(paths.configs_dir / "readouts.yaml"),
    )
    controls_cfg = load_controls_config(paths.configs_dir / "controls.yaml")
    permutation_specs = resolve_permutation_specs(
        list(manifest.get("selections", {}).get("permutations", [])),
        controls_cfg,
    )

    all_specs = load_model_specs(paths.configs_dir / "models.yaml")
    resolved_specs = resolve_named_specs(
        all_specs,
        manifest.get("selections", {}).get("models", []),
        require_enabled=True,
        require_family_ready=True,
        label=f"model:{panel_name}",
    )
    if args.model_name:
        resolved_specs = [spec for spec in resolved_specs if spec.name == args.model_name]
        if not resolved_specs:
            raise SystemExit(f"unknown or disallowed model-name={args.model_name}")

    existing_score_rows = read_jsonl(score_manifest_path)
    existing_score_by_unit = _existing_rows_by_unit(existing_score_rows)
    existing_audits_by_unit = _existing_audits_by_unit(read_jsonl(token_audits_path))
    existing_failure_rows = read_jsonl(failure_log_path)
    existing_failure_by_unit = {
        str(row["unit_key"]): row
        for row in existing_failure_rows
        if row.get("unit_key") and str(row.get("unit_key")) not in existing_score_by_unit
    }
    unresolved_failure_keys = set(existing_failure_by_unit)

    cache = ScoreCache(paths)
    new_score_by_unit: dict[str, dict[str, Any]] = {}
    new_audits_by_unit: dict[str, list[dict[str, Any]]] = {}
    new_failure_by_unit: dict[str, dict[str, Any]] = {}
    attempted_unit_keys: set[str] = set()
    smoke_summary = {
        "run_id": run_id,
        "panel_name": panel_name,
        "device": device,
        "dtype": str(args.dtype),
        "scorer_engine": str(args.scorer_engine),
        "models_attempted": [],
        "examples_scored": 0,
        "cache_hits": 0,
        "computed_units": 0,
        "failed_units": 0,
        "readouts": [],
        "permutations": [],
        "batch_plan": {},
        "equivalence_checks": {},
        "oom_backoff_events": {},
        "letter_label_fast_path": {},
    }

    for spec in resolved_specs:
        prompt_template = resolve_prompt_template_for_mode(prompt_templates, prompt_mode=spec.prompt_mode)
        work_groups: list[tuple[CanonicalExample, Any, list[tuple[Any, str]]]] = []
        for example in examples:
            for permutation_spec in permutation_specs:
                missing_readouts: list[tuple[Any, str]] = []
                for readout_spec in readout_specs:
                    unit_key = _unit_key(spec.name, example.example_id, permutation_spec.name, readout_spec.id)
                    if unit_key in existing_score_by_unit:
                        continue
                    if args.resume_failed_only and unit_key not in unresolved_failure_keys:
                        continue
                    missing_readouts.append((readout_spec, unit_key))
                if missing_readouts:
                    work_groups.append((example, permutation_spec, missing_readouts))
        if not work_groups:
            continue

        try:
            bundle = load_model_bundle(spec, device=device, dtype=str(args.dtype))
            _validate_model_readout(bundle)
        except Exception as exc:
            for example, permutation_spec, missing_readouts in work_groups:
                for readout_spec, unit_key in missing_readouts:
                    attempted_unit_keys.add(unit_key)
                    new_failure_by_unit[unit_key] = _failure_row(
                        run_id=run_id,
                        unit_key=unit_key,
                        reason_code="final_layer_equivalence_failed",
                        message=str(exc),
                        metadata={
                            "model_name": spec.name,
                            "example_id": example.example_id,
                            "readout_id": readout_spec.id,
                            "permutation_id": permutation_spec.name,
                            "device": device,
                            "dtype": str(args.dtype),
                        },
                    )
            continue

        smoke_summary["models_attempted"].append(spec.name)
        calibrated_batch_size: int | None = None
        disable_letter_label_fast_path = False
        candidate_batch_sizes = _candidate_batch_sizes(device, int(args.max_batch_size or 0))
        equivalence_checked_per_permutation: set[str] = set()

        for example, permutation_spec, missing_readouts in work_groups:
            for _, unit_key in missing_readouts:
                attempted_unit_keys.add(unit_key)
            permutation = permutation_for_spec(permutation_spec, option_count=len(example.options))
            rendered_by_readout = _rendered_group_prompts(
                example=example,
                readout_specs=[readout_spec for readout_spec, _ in missing_readouts],
                prompt_template=prompt_template,
                spec=spec,
                permutation=permutation,
                tokenizer=bundle.tokenizer,
            )
            rendered = next(iter(rendered_by_readout.values()))
            permuted_options, _ = apply_permutation(example, permutation)
            readout_option_surfaces = {
                readout_spec.id: build_option_surfaces(readout_spec, permuted_options)
                for readout_spec, _ in missing_readouts
            }
            normalization_by_readout = {
                readout_spec.id: readout_spec.normalization
                for readout_spec, _ in missing_readouts
            }
            try:
                if str(args.scorer_engine) == "optimized_exact" and permutation_spec.name not in equivalence_checked_per_permutation:
                    reference_results, optimized_results, equivalence_result = find_equivalent_optimized_batch_size(
                        bundle,
                        prompt_text=rendered.rendered_prompt,
                        readout_option_surfaces=readout_option_surfaces,
                        normalization_by_readout=normalization_by_readout,
                        candidate_batch_sizes=_candidate_batch_sizes_from_safe_value(
                            calibrated_batch_size,
                            candidate_batch_sizes,
                        ),
                        force_disable_letter_label_fast_path=(disable_letter_label_fast_path or None),
                    )
                    equivalence_checked_per_permutation.add(permutation_spec.name)
                    disable_letter_label_fast_path = (
                        disable_letter_label_fast_path or equivalence_result.disable_letter_label_fast_path
                    )
                    smoke_summary["equivalence_checks"][f"{spec.name}:{permutation_spec.name}"] = {
                        "passed": True,
                        "checked_readouts": sorted(reference_results.keys()),
                        "equivalence_safe_batch_size": equivalence_result.batch_size,
                        "drift_by_readout": equivalence_result.drift_by_readout,
                        "execution_modes": equivalence_result.execution_modes,
                        "disable_letter_label_fast_path": disable_letter_label_fast_path,
                    }
                    if equivalence_result.attempts:
                        smoke_summary["oom_backoff_events"][f"{spec.name}:{permutation_spec.name}:canary"] = (
                            equivalence_result.attempts
                        )
                    calibrated_batch_size = (
                        equivalence_result.batch_size
                        if calibrated_batch_size is None
                        else min(int(calibrated_batch_size), int(equivalence_result.batch_size))
                    )
                    smoke_summary["batch_plan"][spec.name] = {"max_batch_size": calibrated_batch_size}
                    smoke_summary["letter_label_fast_path"][spec.name] = {
                        "eligible": bool(
                            equivalence_result.feature_flags_by_readout.get("letter_label", {}).get(
                                "letter_label_fast_path_eligible",
                                False,
                            )
                        ),
                        "used": bool(
                            equivalence_result.feature_flags_by_readout.get("letter_label", {}).get(
                                "letter_label_fast_path_enabled",
                                False,
                            )
                        ),
                        "disabled": bool(disable_letter_label_fast_path),
                        "execution_mode": equivalence_result.execution_modes.get("letter_label"),
                        "profile": equivalence_result.profile,
                    }

                cache_keys = {
                    readout_spec.id: _score_cache_key(
                        spec=spec,
                        example=example,
                        rendered_prompt=rendered_by_readout[readout_spec.id],
                        prompt_template=prompt_template,
                        readout_spec=readout_spec,
                        permutation_spec=permutation_spec,
                        scorer_engine=str(args.scorer_engine),
                        scorer_feature_hash_value=scorer_feature_hash(
                            tokenizer=bundle.tokenizer,
                            prompt_text=rendered.rendered_prompt,
                            readout_id=readout_spec.id,
                            option_surfaces=readout_option_surfaces[readout_spec.id],
                            scorer_engine=str(args.scorer_engine),
                            disable_letter_label_fast_path=disable_letter_label_fast_path,
                        ),
                        dtype=bundle.dtype,
                    )
                    for readout_spec, _ in missing_readouts
                }

                cache_misses = [
                    (readout_spec, unit_key)
                    for readout_spec, unit_key in missing_readouts
                    if not cache.exists(cache_keys[readout_spec.id])
                ]
                grouped_results: dict[str, dict[str, Any]] = {}
                if cache_misses:
                    grouped_results, used_batch_size, oom_events = _score_group_with_backoff(
                        bundle,
                        prompt_text=rendered.rendered_prompt,
                        readout_option_surfaces={
                            readout_spec.id: readout_option_surfaces[readout_spec.id]
                            for readout_spec, _ in cache_misses
                        },
                        normalization_by_readout={
                            readout_spec.id: normalization_by_readout[readout_spec.id]
                            for readout_spec, _ in cache_misses
                        },
                        scorer_engine=str(args.scorer_engine),
                        candidate_batch_sizes=_candidate_batch_sizes_from_safe_value(
                            calibrated_batch_size,
                            candidate_batch_sizes,
                        ),
                        disable_letter_label_fast_path=disable_letter_label_fast_path,
                    )
                    calibrated_batch_size = used_batch_size
                    smoke_summary["batch_plan"][spec.name] = {"max_batch_size": calibrated_batch_size}
                    if oom_events:
                        smoke_summary["oom_backoff_events"][
                            f"{spec.name}:{example.example_id}:{permutation_spec.name}"
                        ] = oom_events
            except Exception as exc:
                for readout_spec, unit_key in missing_readouts:
                    new_failure_by_unit[unit_key] = _failure_row(
                        run_id=run_id,
                        unit_key=unit_key,
                        reason_code="readout_scoring_failed",
                        message=str(exc),
                        metadata={
                            "model_name": spec.name,
                            "example_id": example.example_id,
                            "readout_id": readout_spec.id,
                            "permutation_id": permutation_spec.name,
                            "device": device,
                            "dtype": bundle.dtype,
                            "scorer_engine": str(args.scorer_engine),
                        },
                    )
                    smoke_summary["failed_units"] += 1
                continue

            for readout_spec, unit_key in missing_readouts:
                cache_key = cache_keys[readout_spec.id]
                rendered_for_readout = rendered_by_readout[readout_spec.id]
                if cache.exists(cache_key):
                    score_bundle = cache.load_bundle(cache_key)
                    source = "cache_hit"
                    smoke_summary["cache_hits"] += 1
                else:
                    score_bundle = {
                        "run_id": run_id,
                        "example_id": example.example_id,
                        "dataset_name": example.dataset_name,
                        "dataset_revision": example.dataset_revision,
                        "model_name": spec.name,
                        "model_family": spec.family,
                        "model_size": spec.size,
                        "model_id": spec.model_id,
                        "panel_name": panel_name,
                        "readout_id": readout_spec.id,
                        "readout_kind": readout_spec.kind,
                        "readout_config_hash": readout_spec.config_hash(),
                        "permutation_id": permutation_spec.name,
                        "permutation": permutation,
                        "permutation_config_hash": permutation_spec.config_hash(),
                        "prompt_mode": spec.prompt_mode,
                        "template_id": prompt_template.template_id,
                        "prompt_hash": rendered_for_readout.prompt_hash,
                        "prompt_config_hash": prompt_template.config_hash(),
                        "prompt_id": rendered_for_readout.prompt_id,
                        "rendered_prompt": rendered_for_readout.rendered_prompt,
                        "answer_prefix": prompt_template.answer_prefix,
                        "prompt_token_count": rendered_for_readout.prompt_token_count,
                        "answer_start_token_index": rendered_for_readout.answer_start_token_index,
                        "gold_answer": rendered_for_readout.correct_option_after_permutation,
                        "dtype": bundle.dtype,
                        "execution_mode": grouped_results[readout_spec.id]["execution_mode"],
                        "timing_seconds": grouped_results[readout_spec.id]["timing_seconds"],
                        "tokenization_version": "contextual_v1",
                        **grouped_results[readout_spec.id],
                    }
                    cache.save_bundle(cache_key, score_bundle)
                    source = "computed"
                    smoke_summary["computed_units"] += 1

                audits = build_token_audit_records(
                    tokenizer=bundle.tokenizer,
                    model_id=spec.model_id,
                    example_id=example.example_id,
                    readout_id=readout_spec.id,
                    prompt_id=rendered_for_readout.prompt_id,
                    prompt_text=rendered_for_readout.rendered_prompt,
                    answer_start_token_index=int(rendered_for_readout.answer_start_token_index or 0),
                    option_surfaces=readout_option_surfaces[readout_spec.id],
                )

                cache_path = cache.cache_path_for(cache_key)
                new_score_by_unit[unit_key] = {
                    "run_id": run_id,
                    "unit_key": unit_key,
                    "cache_key": cache_key.digest(),
                    "cache_path": _path_for_manifest_row(paths, cache_path),
                    "example_id": example.example_id,
                    "dataset_name": example.dataset_name,
                    "model_name": spec.name,
                    "model_family": spec.family,
                    "model_size": spec.size,
                    "readout_id": readout_spec.id,
                    "permutation_id": permutation_spec.name,
                    "gold_answer": rendered_for_readout.correct_option_after_permutation,
                    "final_winner": score_bundle["final_winner"],
                    "prompt_hash": rendered_for_readout.prompt_hash,
                    "prompt_config_hash": prompt_template.config_hash(),
                    "readout_config_hash": readout_spec.config_hash(),
                    "permutation_config_hash": permutation_spec.config_hash(),
                    "scorer_engine": score_bundle["scorer_engine"],
                    "execution_mode": score_bundle.get("execution_mode"),
                    "letter_label_fast_path_eligible": score_bundle.get("scorer_feature_flags", {}).get(
                        "letter_label_fast_path_eligible"
                    ),
                    "letter_label_fast_path_used": score_bundle.get("scorer_feature_flags", {}).get(
                        "letter_label_fast_path_enabled"
                    ),
                    "scorer_feature_hash": cache_key.scorer_feature_hash,
                    "dtype": bundle.dtype,
                    "source": source,
                }
                new_audits_by_unit[unit_key] = [
                    {
                        **audit.to_dict(),
                        "run_id": run_id,
                        "unit_key": unit_key,
                        "cache_key": cache_key.digest(),
                        "model_name": spec.name,
                        "dataset_name": example.dataset_name,
                        "permutation_id": permutation_spec.name,
                        "scorer_engine": score_bundle["scorer_engine"],
                    }
                    for audit in audits
                ]
                smoke_summary["examples_scored"] += 1
                if readout_spec.id not in smoke_summary["readouts"]:
                    smoke_summary["readouts"].append(readout_spec.id)
                if permutation_spec.name not in smoke_summary["permutations"]:
                    smoke_summary["permutations"].append(permutation_spec.name)

        del bundle
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    merged_score_by_unit = dict(existing_score_by_unit)
    merged_score_by_unit.update(new_score_by_unit)
    merged_audits_by_unit = dict(existing_audits_by_unit)
    merged_audits_by_unit.update(new_audits_by_unit)
    merged_failure_by_unit = {
        unit_key: row
        for unit_key, row in existing_failure_by_unit.items()
        if unit_key not in attempted_unit_keys and unit_key not in merged_score_by_unit
    }
    for unit_key, row in new_failure_by_unit.items():
        if unit_key not in merged_score_by_unit:
            merged_failure_by_unit[unit_key] = row

    score_rows = [merged_score_by_unit[key] for key in sorted(merged_score_by_unit)]
    audit_rows = [row for key in sorted(merged_audits_by_unit) for row in merged_audits_by_unit[key]]
    failure_rows = [merged_failure_by_unit[key] for key in sorted(merged_failure_by_unit)]

    write_jsonl_atomic(score_manifest_path, score_rows)
    write_jsonl_atomic(token_audits_path, audit_rows)
    write_jsonl_atomic(failure_log_path, failure_rows)
    if panel_name == "smoke":
        write_json_atomic(smoke_manifest_path, smoke_summary)

    update_run_manifest(
        paths,
        manifest,
        stage_name="run_controls",
        updates={
            "device": device,
            "dtype": str(args.dtype),
            "scorer_engine": str(args.scorer_engine),
            "score_row_count": len(score_rows),
            "token_audit_count": len(audit_rows),
            "failure_count": len(failure_rows),
            "cache_hits": smoke_summary["cache_hits"],
            "computed_units": smoke_summary["computed_units"],
            "batch_plan": smoke_summary["batch_plan"],
            "equivalence_checks": smoke_summary["equivalence_checks"],
            "oom_backoff_events": smoke_summary["oom_backoff_events"],
            "letter_label_fast_path": smoke_summary["letter_label_fast_path"],
        },
    )


if __name__ == "__main__":
    main()
