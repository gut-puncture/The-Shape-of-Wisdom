from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from path_manager import ExperimentPaths
from run_manifest import make_run_id
from runtime_panels import load_runtime_config


def build_smoke_commands(
    *,
    run_id: str,
    device: str,
    dtype: str = "float32",
    scorer_engine: str = "optimized_exact",
    max_batch_size: int = 0,
) -> list[list[str]]:
    return [
        ["python3", "run_prepare.py", "--smoke", "--run-id", run_id],
        [
            "python3",
            "run_gpu_preflight.py",
            "--smoke",
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
            "--smoke",
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


def _run(argv: list[str]) -> None:
    result = subprocess.run(argv, cwd=str(WORKSPACE_ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--scorer-engine", default="optimized_exact")
    parser.add_argument("--max-batch-size", type=int, default=0)
    args = parser.parse_args()

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    run_id = str(args.run_id or make_run_id("smoke"))
    device = str(args.device or runtime_cfg.get("default_device", "cpu"))
    for argv in build_smoke_commands(
        run_id=run_id,
        device=device,
        dtype=str(args.dtype),
        scorer_engine=str(args.scorer_engine),
        max_batch_size=int(args.max_batch_size),
    ):
        _run(argv)


if __name__ == "__main__":
    main()
