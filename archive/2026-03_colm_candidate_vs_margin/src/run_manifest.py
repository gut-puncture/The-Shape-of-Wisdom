from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json_atomic
from path_manager import ExperimentPaths


CONFIG_FILE_NAMES = [
    "runtime.yaml",
    "models.yaml",
    "datasets.yaml",
    "prompts.yaml",
    "readouts.yaml",
    "controls.yaml",
    "metrics.yaml",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(workspace_root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def resolve_manifest_path(paths: ExperimentPaths, manifest: dict[str, Any], key: str) -> Path:
    raw = ((manifest.get("paths") or {}).get(key))
    if not raw:
        raise ValueError(f"run manifest missing path entry: {key}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = paths.workspace_root / path
    return paths.assert_within_workspace(path)


def make_run_id(panel_name: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{panel_name}_{stamp}"


def build_config_hashes(paths: ExperimentPaths) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_name in CONFIG_FILE_NAMES:
        path = paths.configs_dir / file_name
        hashes[file_name] = _sha256_bytes(path.read_bytes()) if path.exists() else "missing"
    return hashes


def create_run_manifest(
    paths: ExperimentPaths,
    *,
    run_id: str,
    panel_name: str,
    selections: dict[str, Any],
    config_hashes: dict[str, str],
    environment_summary: dict[str, Any],
) -> dict[str, Any]:
    run_paths = paths.run_paths(run_id)
    run_paths.ensure_dirs()
    manifest = {
        "run_id": run_id,
        "panel_name": panel_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hashes": dict(config_hashes),
        "selections": dict(selections),
        "environment_summary": dict(environment_summary),
        "paths": {
            "run_root": _workspace_relative(run_paths.run_root, paths.workspace_root),
            "run_manifest": _workspace_relative(run_paths.run_manifest_path, paths.workspace_root),
            "prepared_examples": _workspace_relative(run_paths.canonical_examples_path, paths.workspace_root),
            "prepared_examples_manifest": _workspace_relative(run_paths.prepared_manifest_path, paths.workspace_root),
            "prepared_selection_manifest": _workspace_relative(run_paths.selection_manifest_path, paths.workspace_root),
            "environment_prepare": _workspace_relative(run_paths.environment_prepare_path, paths.workspace_root),
            "score_manifest": _workspace_relative(run_paths.score_manifest_path, paths.workspace_root),
            "primary_metrics": _workspace_relative(run_paths.primary_metrics_path, paths.workspace_root),
            "token_audits": _workspace_relative(run_paths.token_audits_path, paths.workspace_root),
            "gpu_preflight_report": _workspace_relative(run_paths.gpu_preflight_report_path, paths.workspace_root),
            "qc_summary_json": _workspace_relative(run_paths.qc_summary_json_path, paths.workspace_root),
            "qc_summary_markdown": _workspace_relative(run_paths.qc_summary_markdown_path, paths.workspace_root),
            "bootstrap_summary": _workspace_relative(run_paths.bootstrap_summary_path, paths.workspace_root),
            "model_dataset_positivity": _workspace_relative(run_paths.model_dataset_positivity_path, paths.workspace_root),
            "model_positivity_summary": _workspace_relative(run_paths.model_positivity_summary_path, paths.workspace_root),
            "family_positivity_summary": _workspace_relative(run_paths.family_positivity_summary_path, paths.workspace_root),
            "run_controls_failures": _workspace_relative(run_paths.failure_log_path("run_controls"), paths.workspace_root),
            "smoke_manifest": _workspace_relative(run_paths.smoke_manifest_path, paths.workspace_root),
        },
        "stages": {},
    }
    write_json_atomic(run_paths.run_manifest_path, manifest)
    latest_payload = {"run_id": run_id, "panel_name": panel_name}
    write_json_atomic(paths.latest_run_pointer_path, latest_payload)
    write_json_atomic(paths.latest_panel_run_pointer_path(panel_name), latest_payload)
    return manifest


def update_run_manifest(paths: ExperimentPaths, manifest: dict[str, Any], *, stage_name: str, updates: dict[str, Any] | None = None) -> dict[str, Any]:
    current = dict(manifest)
    stages = dict(current.get("stages") or {})
    stage_payload = dict(stages.get(stage_name) or {})
    stage_payload.update(updates or {})
    stage_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    stages[stage_name] = stage_payload
    current["stages"] = stages
    write_json_atomic(paths.run_paths(str(current["run_id"])).run_manifest_path, current)
    return current


def load_run_manifest(paths: ExperimentPaths, *, run_id: str | None = None, panel_name: str | None = None) -> dict[str, Any]:
    if run_id is None:
        pointer_path = paths.latest_panel_run_pointer_path(panel_name) if panel_name is not None else paths.latest_run_pointer_path
        if not pointer_path.exists():
            raise ValueError("no latest run pointer found")
        run_id = str(read_json(pointer_path)["run_id"])
    manifest_path = paths.run_paths(run_id).run_manifest_path
    if not manifest_path.exists():
        raise ValueError(f"run manifest not found for run_id={run_id}")
    manifest = read_json(manifest_path)
    if panel_name is not None and str(manifest.get("panel_name")) != panel_name:
        raise ValueError(f"run manifest panel mismatch: expected {panel_name}, got {manifest.get('panel_name')}")
    return manifest


def validate_run_manifest_config_hashes(paths: ExperimentPaths, manifest: dict[str, Any]) -> None:
    expected = dict(manifest.get("config_hashes") or {})
    current = build_config_hashes(paths)
    if expected != current:
        raise ValueError("run manifest config hashes do not match current workspace config files")
