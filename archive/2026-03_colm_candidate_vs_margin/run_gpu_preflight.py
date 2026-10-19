from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_loader import CanonicalExample, CanonicalOption
from continuation_tokenizer import contextual_token_ids
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
from gpu_preflight import OOMErrorSignal, calibration_forecast, determine_safe_batch_size, free_disk_gb, verify_prompt_template_compatibility
from io_utils import read_jsonl, write_json_atomic
from layerwise_extractor import _find_final_norm, assert_final_layer_equivalence
from model_loader import capture_environment_manifest, load_model_bundle, validate_requested_device
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from prompt_builder import apply_permutation, build_prompt
from run_manifest import load_run_manifest, resolve_manifest_path, update_run_manifest, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_panel_name
from scoring_engine import count_unique_prefix_states, score_readout_group
from scoring_engine import find_equivalent_optimized_batch_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--scorer-engine", default="optimized_exact")
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--min-disk-gb", type=float, default=25.0)
    parser.add_argument("--hourly-rate-usd", type=float, default=0.0)
    return parser.parse_args()


def _load_examples(path: Path) -> list[CanonicalExample]:
    rows = []
    for row in read_jsonl(path):
        row["options"] = [CanonicalOption(**option) for option in row["options"]]
        rows.append(CanonicalExample(**row))
    return rows


def _validate_model_readout(bundle: Any) -> None:
    tokenizer = bundle.tokenizer
    prompt_ids = tokenizer.encode("Sanity check.\nFinal answer:", add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("sanity prompt tokenized to zero length")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
        outputs = bundle.model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    assert_final_layer_equivalence(
        outputs=outputs,
        output_embedding=bundle.model.get_output_embeddings(),
        final_norm=_find_final_norm(bundle.model),
        projection_mode=bundle.hidden_state_projection_mode,
        model_config=getattr(bundle.model, "config", None),
    )


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or exc.__class__.__name__ == "OutOfMemoryError"


def _candidate_batch_sizes(device: str, requested_max_batch_size: int) -> list[int]:
    if requested_max_batch_size > 0:
        values = [int(requested_max_batch_size)]
        current = int(requested_max_batch_size)
        while current > 1:
            current = max(1, current // 2)
            if current not in values:
                values.append(current)
        return values
    if str(device).startswith("cuda"):
        return [32, 16, 8, 4, 2, 1]
    return [1]


def _select_calibration_examples(examples: list[CanonicalExample]) -> list[CanonicalExample]:
    selected: list[CanonicalExample] = []
    seen_option_counts: set[int] = set()
    longest_example: CanonicalExample | None = None
    for example in examples:
        option_count = len(example.options)
        if option_count in {3, 4, 5} and option_count not in seen_option_counts:
            selected.append(example)
            seen_option_counts.add(option_count)
        if longest_example is None or len(example.question_raw) > len(longest_example.question_raw):
            longest_example = example
    if longest_example is not None and all(item.example_id != longest_example.example_id for item in selected):
        selected.append(longest_example)
    return selected[:4]


def _group_payloads(
    *,
    examples: list[CanonicalExample],
    spec: Any,
    prompt_template: Any,
    readout_specs: list[Any],
    permutation_specs: list[Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for example in examples:
        for permutation_spec in permutation_specs:
            permutation = permutation_for_spec(permutation_spec, option_count=len(example.options))
            rendered = build_prompt(
                example,
                template_id=prompt_template.template_id,
                readout_id=readout_specs[0].id,
                prompt_mode=spec.prompt_mode,
                answer_prefix=prompt_template.answer_prefix,
                permutation=permutation,
                tokenizer=tokenizer,
                system_prompt=prompt_template.system_prompt,
                system_prompt_handling=spec.system_prompt_handling,
            )
            permuted_options, _ = apply_permutation(example, permutation)
            readout_option_surfaces = {
                readout_spec.id: build_option_surfaces(readout_spec, permuted_options)
                for readout_spec in readout_specs
            }
            normalization_by_readout = {readout_spec.id: readout_spec.normalization for readout_spec in readout_specs}
            token_sequences = {}
            for readout_id, option_surfaces in readout_option_surfaces.items():
                for option_id, surface in option_surfaces.items():
                    token_sequences[f"{readout_id}:{option_id}"] = contextual_token_ids(
                        tokenizer,
                        prompt_text=rendered.rendered_prompt,
                        surface=surface,
                    )
            payloads.append(
                {
                    "example_id": example.example_id,
                    "permutation_id": permutation_spec.name,
                    "prompt_text": rendered.rendered_prompt,
                    "readout_option_surfaces": readout_option_surfaces,
                    "normalization_by_readout": normalization_by_readout,
                    "prefix_state_count": count_unique_prefix_states(token_sequences),
                    "option_count": len(example.options),
                }
            )
    return payloads


def _probe_batch_size(
    *,
    bundle: Any,
    scorer_engine: str,
    payload: dict[str, Any],
    batch_size: int,
) -> None:
    import torch  # noqa: PLC0415

    try:
        score_readout_group(
            bundle,
            prompt_text=payload["prompt_text"],
            readout_option_surfaces=payload["readout_option_surfaces"],
            normalization_by_readout=payload["normalization_by_readout"],
            scorer_engine=scorer_engine,
            max_batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise OOMErrorSignal(str(exc)) from exc
        raise


def main() -> None:
    args = _parse_args()
    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    explicit_panel = None
    if args.smoke or args.panel is not None:
        explicit_panel = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    manifest = load_run_manifest(paths, run_id=args.run_id, panel_name=explicit_panel)
    validate_run_manifest_config_hashes(paths, manifest)
    device = validate_requested_device(str(args.device or runtime_cfg.get("default_device", "cpu")))
    dtype = str(args.dtype)

    examples = _load_examples(resolve_manifest_path(paths, manifest, "prepared_examples"))
    selected_examples = _select_calibration_examples(examples)
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
    specs = {
        spec.name: spec
        for spec in load_model_specs(paths.configs_dir / "models.yaml")
        if spec.name in set(manifest.get("selections", {}).get("models", []))
    }

    disk_free_gb = free_disk_gb(paths.workspace_root)
    report = {
        "run_id": manifest["run_id"],
        "panel_name": manifest["panel_name"],
        "device": device,
        "dtype": dtype,
        "scorer_engine": str(args.scorer_engine),
        "environment": capture_environment_manifest(device=device),
        "disk_free_gb": disk_free_gb,
        "min_disk_gb_required": float(args.min_disk_gb),
        "model_checks": [],
    }
    failures: list[str] = []
    if disk_free_gb < float(args.min_disk_gb):
        failures.append(
            f"disk_free_gb_below_threshold: free={disk_free_gb:.2f} required={float(args.min_disk_gb):.2f}"
        )

    for spec in specs.values():
        prompt_template = resolve_prompt_template_for_mode(prompt_templates, prompt_mode=spec.prompt_mode)
        model_report: dict[str, Any] = {
            "model_name": spec.name,
            "model_id": spec.model_id,
            "device": device,
            "dtype": dtype,
            "prompt_compatibility": {"passed": False, "error": "not_run"},
            "model_load_and_equivalence": {"passed": False, "error": "not_run"},
            "optimized_equivalence": {"passed": False, "error": "not_run"},
            "batch_calibration": {"passed": False, "error": "not_run"},
        }
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                revision=spec.tokenizer_revision,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=spec.local_files_only,
            )
            if selected_examples:
                example = selected_examples[0]
                option_lines = [f"{option.option_id}. {option.raw_text}" for option in example.options]
                model_report["prompt_compatibility"] = verify_prompt_template_compatibility(
                    tokenizer=tokenizer,
                    prompt_mode=spec.prompt_mode,
                    question_text=example.question_raw,
                    options=option_lines,
                    answer_prefix=prompt_template.answer_prefix,
                    system_prompt=prompt_template.system_prompt,
                    system_prompt_handling=spec.system_prompt_handling,
                )
            else:
                model_report["prompt_compatibility"] = {"passed": False, "error": "no_examples_available"}
            if not model_report["prompt_compatibility"]["passed"]:
                failures.append(f"{spec.name}:prompt_compatibility")
                report["model_checks"].append(model_report)
                continue

            bundle = load_model_bundle(spec, device=device, dtype=dtype)
            _validate_model_readout(bundle)
            model_report["model_load_and_equivalence"] = {
                "passed": True,
                "dtype": bundle.dtype,
                "projection_mode": bundle.hidden_state_projection_mode,
            }

            calibration_groups = _group_payloads(
                examples=selected_examples,
                spec=spec,
                prompt_template=prompt_template,
                readout_specs=readout_specs,
                permutation_specs=permutation_specs,
                tokenizer=bundle.tokenizer,
            )
            if not calibration_groups:
                raise ValueError("no calibration groups available")

            if str(args.scorer_engine) == "optimized_exact":
                candidate_batch_sizes = _candidate_batch_sizes(device, int(args.max_batch_size or 0))
                safe_batch_size: int | None = None
                start = time.perf_counter()
                measured_states = 0
                for payload in calibration_groups:
                    reference, optimized, equivalence_result = find_equivalent_optimized_batch_size(
                        bundle,
                        prompt_text=payload["prompt_text"],
                        readout_option_surfaces=payload["readout_option_surfaces"],
                        normalization_by_readout=payload["normalization_by_readout"],
                        candidate_batch_sizes=(
                            _candidate_batch_sizes(device, int(safe_batch_size))
                            if safe_batch_size is not None
                            else candidate_batch_sizes
                        ),
                    )
                    safe_batch_size = equivalence_result.batch_size if safe_batch_size is None else min(
                        int(safe_batch_size),
                        int(equivalence_result.batch_size),
                    )
                    if equivalence_result.attempts:
                        model_report.setdefault("equivalence_batch_attempts", []).append(
                            {
                                "example_id": payload["example_id"],
                                "permutation_id": payload["permutation_id"],
                                "attempts": equivalence_result.attempts,
                            }
                        )
                    measured_states += int(payload["prefix_state_count"])
                elapsed_seconds = max(time.perf_counter() - start, 1.0e-9)
                total_prefix_states = sum(
                    item["prefix_state_count"]
                    for item in _group_payloads(
                        examples=examples,
                        spec=spec,
                        prompt_template=prompt_template,
                        readout_specs=readout_specs,
                        permutation_specs=permutation_specs,
                        tokenizer=bundle.tokenizer,
                    )
                )
                model_report["optimized_equivalence"] = {
                    "passed": True,
                    "calibration_groups": len(calibration_groups),
                    "drift_by_readout": equivalence_result.drift_by_readout,
                    "execution_modes": equivalence_result.execution_modes,
                    "feature_flags_by_readout": equivalence_result.feature_flags_by_readout,
                }
                model_report["batch_calibration"] = {
                    "passed": True,
                    "safe_batch_size": int(safe_batch_size or 1),
                    "measured_prefix_states": measured_states,
                    "total_prefix_states": total_prefix_states,
                    "disable_letter_label_fast_path": bool(equivalence_result.disable_letter_label_fast_path),
                    "forecast": calibration_forecast(
                        total_prefix_states=total_prefix_states,
                        measured_prefix_states=measured_states,
                        elapsed_seconds=elapsed_seconds,
                        hourly_rate_usd=float(args.hourly_rate_usd),
                    ),
                }
                model_report["letter_label_fast_path"] = {
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
                    "disabled": bool(equivalence_result.disable_letter_label_fast_path),
                    "execution_mode": equivalence_result.execution_modes.get("letter_label"),
                    "profile": equivalence_result.profile,
                }
            else:
                model_report["optimized_equivalence"] = {"passed": True, "skipped": True}
                model_report["batch_calibration"] = {"passed": True, "safe_batch_size": 1, "skipped": True}
        except Exception as exc:  # noqa: BLE001
            if model_report["model_load_and_equivalence"].get("passed") is False:
                model_report["model_load_and_equivalence"] = {"passed": False, "error": str(exc)}
            elif model_report["optimized_equivalence"].get("passed") is False:
                model_report["optimized_equivalence"] = {"passed": False, "error": str(exc)}
            else:
                model_report["batch_calibration"] = {"passed": False, "error": str(exc)}
            failures.append(f"{spec.name}:{exc}")
        report["model_checks"].append(model_report)
        try:
            del bundle
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    output_path = resolve_manifest_path(paths, manifest, "gpu_preflight_report")
    write_json_atomic(output_path, report)
    update_run_manifest(
        paths,
        manifest,
        stage_name="run_gpu_preflight",
        updates={
            "device": device,
            "dtype": dtype,
            "disk_free_gb": disk_free_gb,
            "min_disk_gb_required": float(args.min_disk_gb),
            "report_path": str(output_path),
            "scorer_engine": str(args.scorer_engine),
            "failed_checks": failures,
        },
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
