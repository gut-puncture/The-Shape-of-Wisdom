from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from io_utils import read_jsonl
from path_manager import ExperimentPaths
from resume_logic import unresolved_failure_unit_keys
from run_manifest import load_run_manifest, resolve_manifest_path, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_panel_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", default="run_controls")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage != "run_controls":
        raise SystemExit(f"unsupported resume stage={args.stage}")

    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")
    explicit_panel = None
    if args.smoke or args.panel is not None:
        explicit_panel = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    manifest = load_run_manifest(paths, run_id=args.run_id, panel_name=explicit_panel)
    validate_run_manifest_config_hashes(paths, manifest)

    failures = read_jsonl(resolve_manifest_path(paths, manifest, "run_controls_failures"))
    successes = read_jsonl(resolve_manifest_path(paths, manifest, "score_manifest"))
    unresolved = unresolved_failure_unit_keys(failures, successes)
    if not unresolved:
        print("no unresolved failures")
        return

    argv = ["python3", "run_controls.py", "--run-id", str(manifest["run_id"]), "--resume-failed-only"]
    if args.device is not None:
        argv.extend(["--device", str(args.device)])
    result = subprocess.run(argv, cwd=str(WORKSPACE_ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"resumed_units={len(unresolved)}")


if __name__ == "__main__":
    main()
