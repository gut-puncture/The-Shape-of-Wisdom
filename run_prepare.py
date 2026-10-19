from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_loader import load_canonical_examples
from dataset_registry import load_dataset_specs
from execution_config import load_controls_config, load_prompt_templates, load_readout_specs, resolve_permutation_specs, resolve_prompt_template_for_mode, resolve_readout_specs
from io_utils import write_json_atomic, write_jsonl_atomic
from model_loader import capture_environment_manifest
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from prepare_selection import build_selection_manifest, select_examples_for_dataset
from run_manifest import build_config_hashes, create_run_manifest, make_run_id, update_run_manifest
from runtime_panels import get_panel_config, load_runtime_config, resolve_named_specs, resolve_panel_name


def _max_examples_for_dataset(
    panel_cfg: dict[str, object],
    *,
    dataset_name: str,
    smoke_examples: int,
    smoke: bool,
) -> int:
    dataset_limits = panel_cfg.get("dataset_example_limits") or {}
    if dataset_limits and not isinstance(dataset_limits, dict):
        raise ValueError("dataset_example_limits must be a mapping of dataset_name -> nonnegative int")
    max_examples = int(dataset_limits.get(dataset_name, panel_cfg.get("max_examples_per_dataset", 0)))
    if max_examples < 0:
        raise ValueError(f"max_examples must be nonnegative for dataset={dataset_name}, got {max_examples}")
    if smoke and max_examples > 0:
        return min(int(smoke_examples), max_examples)
    return max_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    paths.ensure_required_dirs()

    all_dataset_specs = [spec for spec in load_dataset_specs(paths.configs_dir / "datasets.yaml") if spec.enabled]
    all_model_specs = load_model_specs(paths.configs_dir / "models.yaml")
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    panel_name = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    panel_cfg = get_panel_config(runtime_cfg, panel_name)
    dataset_specs = resolve_named_specs(
        all_dataset_specs,
        panel_cfg.get("datasets", []),
        require_enabled=True,
        require_family_ready=False,
        label=f"dataset:{panel_name}",
    )
    model_specs = resolve_named_specs(
        all_model_specs,
        panel_cfg.get("models", []),
        require_enabled=True,
        require_family_ready=False,
        label=f"model:{panel_name}",
    )
    prompt_templates = load_prompt_templates(paths.configs_dir / "prompts.yaml")
    readout_specs = resolve_readout_specs(list(panel_cfg.get("readouts", [])), load_readout_specs(paths.configs_dir / "readouts.yaml"))
    controls_cfg = load_controls_config(paths.configs_dir / "controls.yaml")
    permutation_specs = resolve_permutation_specs(list(panel_cfg.get("permutations", [])), controls_cfg)
    prompt_template_ids = sorted(
        {resolve_prompt_template_for_mode(prompt_templates, prompt_mode=spec.prompt_mode).template_id for spec in model_specs}
    )

    run_id = str(args.run_id or make_run_id(panel_name))
    run_paths = paths.run_paths(run_id)
    if run_paths.run_manifest_path.exists():
        raise SystemExit(f"run-id already exists: {run_id}")

    environment_manifest = capture_environment_manifest()
    manifest = create_run_manifest(
        paths,
        run_id=run_id,
        panel_name=panel_name,
        selections={
            "datasets": [spec.name for spec in dataset_specs],
            "models": [spec.name for spec in model_specs],
            "readouts": [spec.id for spec in readout_specs],
            "permutations": [spec.name for spec in permutation_specs],
            "prompt_templates": prompt_template_ids,
        },
        config_hashes=build_config_hashes(paths),
        environment_summary={
            "python_version": environment_manifest.get("python_version"),
            "platform": environment_manifest.get("platform"),
            "torch_version": environment_manifest.get("torch_version"),
            "transformers_version": environment_manifest.get("transformers_version"),
        },
    )

    selection_strategy = str(panel_cfg.get("selection_strategy") or "first_n")
    all_examples = []
    selection_records = []
    dataset_summaries = []
    for spec in dataset_specs:
        available_examples = load_canonical_examples(spec, max_examples=0)
        max_examples = _max_examples_for_dataset(
            panel_cfg,
            dataset_name=spec.name,
            smoke_examples=spec.smoke_examples,
            smoke=bool(args.smoke),
        )
        examples, dataset_selection_records = select_examples_for_dataset(
            available_examples,
            max_examples=max_examples,
            strategy=selection_strategy,
        )
        dataset_summaries.append(
            {
                "panel_name": panel_name,
                "dataset_name": spec.name,
                "dataset_id": spec.dataset_id,
                "revision": spec.revision,
                "split": spec.split,
                "available_example_count": len(available_examples),
                "example_count": len(examples),
                "selection_strategy": selection_strategy,
            }
        )
        selection_records.extend(dataset_selection_records)
        all_examples.extend(example.to_dict() for example in examples)

    run_paths.ensure_dirs()
    selection_manifest = build_selection_manifest(
        run_id=run_id,
        panel_name=panel_name,
        selection_strategy=selection_strategy,
        records=selection_records,
    )
    write_jsonl_atomic(run_paths.canonical_examples_path, all_examples)
    write_json_atomic(
        run_paths.prepared_manifest_path,
        {
            "run_id": run_id,
            "panel_name": panel_name,
            "selection_strategy": selection_strategy,
            "selection_manifest_sha256": selection_manifest["selection_manifest_sha256"],
            "datasets": dataset_summaries,
        },
    )
    write_json_atomic(run_paths.selection_manifest_path, selection_manifest)
    write_json_atomic(run_paths.environment_prepare_path, environment_manifest)
    update_run_manifest(
        paths,
        manifest,
        stage_name="prepare",
        updates={
            "example_count": len(all_examples),
            "dataset_count": len(dataset_summaries),
            "prepared_examples_path": str(run_paths.canonical_examples_path),
            "selection_strategy": selection_strategy,
            "selection_manifest_path": str(run_paths.selection_manifest_path),
            "selection_manifest_sha256": selection_manifest["selection_manifest_sha256"],
        },
    )
    print(run_id)


if __name__ == "__main__":
    main()
