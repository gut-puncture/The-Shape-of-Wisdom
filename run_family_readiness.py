from __future__ import annotations

import argparse
import multiprocessing
import sys
import warnings
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from family_readiness import check_model_readiness, summarize_family_readiness, timed_out_readiness_result
from execution_config import load_prompt_templates, resolve_prompt_template_for_mode
from io_utils import write_json_atomic
from model_loader import validate_requested_device
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from runtime_panels import get_panel_config, load_runtime_config, resolve_named_specs, resolve_panel_name

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown.*",
    category=UserWarning,
)


def _readiness_worker(spec, device: str, dtype: str, prompt_template, queue) -> None:
    queue.put(check_model_readiness(spec, device=device, dtype=dtype, prompt_template=prompt_template))


def _check_in_subprocess(spec, device: str, dtype: str, prompt_template, timeout_seconds: int) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_readiness_worker, args=(spec, device, dtype, prompt_template, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        queue.join_thread()
        return timed_out_readiness_result(spec, timeout_seconds=timeout_seconds)
    result = queue.get() if not queue.empty() else timed_out_readiness_result(spec, timeout_seconds=timeout_seconds)
    queue.close()
    queue.join_thread()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    args = parser.parse_args()

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    panel_name = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    panel_cfg = get_panel_config(runtime_cfg, panel_name)
    device = validate_requested_device(str(args.device or runtime_cfg.get("default_device", "cpu")))
    prompt_templates = load_prompt_templates(paths.configs_dir / "prompts.yaml")
    all_specs = load_model_specs(paths.configs_dir / "models.yaml")

    if args.model_name:
        specs = [spec for spec in all_specs if spec.name == args.model_name]
        if not specs:
            raise SystemExit(f"unknown model-name={args.model_name}")
    else:
        specs = resolve_named_specs(
            all_specs,
            panel_cfg.get("models", []),
            require_enabled=True,
            require_family_ready=False,
            label=f"model:{panel_name}",
        )

    results = []
    for spec in specs:
        prompt_template = resolve_prompt_template_for_mode(prompt_templates, prompt_mode=spec.prompt_mode)
        if args.timeout_seconds > 0:
            results.append(_check_in_subprocess(spec, device, str(args.dtype), prompt_template, args.timeout_seconds))
        else:
            results.append(check_model_readiness(spec, device=device, dtype=str(args.dtype), prompt_template=prompt_template))
    family_summary = summarize_family_readiness(results)
    payload = {
        "panel_name": panel_name,
        "device": device,
        "dtype": str(args.dtype),
        "model_results": results,
        "family_summary": family_summary,
    }
    suffix = args.model_name or panel_name
    write_json_atomic(paths.outputs_dir / "qc" / f"family_readiness_{suffix}.json", payload)


if __name__ == "__main__":
    main()
