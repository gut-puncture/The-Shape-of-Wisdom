from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from io_utils import read_yaml
from model_loader import validate_requested_device
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from run_manifest import make_run_id
from runtime_panels import get_panel_config, load_runtime_config, resolve_named_specs


def build_full_sweep_commands(
    *,
    run_id: str,
    device: str,
    dtype: str = "float32",
    scorer_engine: str = "optimized_exact",
    max_batch_size: int = 0,
) -> list[list[str]]:
    return [
        ["python3", "run_prepare.py", "--panel", "paper", "--run-id", run_id],
        [
            "python3",
            "run_gpu_preflight.py",
            "--panel",
            "paper",
            "--run-id",
            run_id,
            "--device",
            device,
            "--dtype",
            dtype,
            "--scorer-engine",
            scorer_engine,
            "--max-batch-size",
            str(max_batch_size),
        ],
        [
            "python3",
            "run_controls.py",
            "--panel",
            "paper",
            "--run-id",
            run_id,
            "--device",
            device,
            "--dtype",
            dtype,
            "--scorer-engine",
            scorer_engine,
            "--max-batch-size",
            str(max_batch_size),
        ],
        ["python3", "run_metrics.py", "--run-id", run_id],
        ["python3", "run_qc.py", "--run-id", run_id],
        ["python3", "run_stats.py", "--run-id", run_id],
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-full-sweep", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--scorer-engine", default="optimized_exact")
    parser.add_argument("--max-batch-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.confirm_full_sweep or os.environ.get("SCWI_ALLOW_FULL_SWEEP") != "1":
        raise SystemExit("full sweep blocked: require SCWI_ALLOW_FULL_SWEEP=1 and --confirm-full-sweep")

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    metrics_cfg = read_yaml(paths.configs_dir / "metrics.yaml")["metrics"]
    if not bool(metrics_cfg["positivity_criteria"]["authored"]):
        raise SystemExit("full sweep blocked: positivity criteria are not authored")

    panel_cfg = get_panel_config(runtime_cfg, "paper")
    try:
        resolve_named_specs(
            load_model_specs(paths.configs_dir / "models.yaml"),
            panel_cfg.get("models", []),
            require_enabled=True,
            require_family_ready=True,
            label="model:paper",
        )
    except ValueError as exc:
        raise SystemExit(f"full sweep blocked: {exc}") from exc

    device = validate_requested_device(str(args.device or runtime_cfg.get("default_device", "cpu")))
    run_id = str(args.run_id or make_run_id("paper"))
    env = dict(os.environ)
    env[str(runtime_cfg.get("inference_env_var", "SCWI_ALLOW_INFERENCE"))] = "1"
    for argv in build_full_sweep_commands(
        run_id=run_id,
        device=device,
        dtype=str(args.dtype),
        scorer_engine=str(args.scorer_engine),
        max_batch_size=int(args.max_batch_size),
    ):
        result = subprocess.run(argv, cwd=str(WORKSPACE_ROOT), check=False, env=env)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
