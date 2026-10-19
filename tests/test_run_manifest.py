from __future__ import annotations

import pytest

from path_manager import ExperimentPaths
from run_manifest import (
    build_config_hashes,
    create_run_manifest,
    load_run_manifest,
    make_run_id,
    validate_run_manifest_config_hashes,
)


def test_create_and_load_run_manifest_via_latest_pointer(tmp_path) -> None:
    workspace = tmp_path / "shape-of-wisdom"
    (workspace / ".git").mkdir(parents=True)
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    for name in ["runtime.yaml", "models.yaml", "datasets.yaml", "prompts.yaml", "readouts.yaml", "controls.yaml", "metrics.yaml"]:
        (paths.configs_dir / name).write_text(f"{name}: test\n", encoding="utf-8")
    run_id = make_run_id("smoke")
    manifest = create_run_manifest(
        paths,
        run_id=run_id,
        panel_name="smoke",
        selections={"datasets": ["d1"], "models": ["m1"], "readouts": ["semantic_exact"], "permutations": ["identity"]},
        config_hashes=build_config_hashes(paths),
        environment_summary={"python_version": "3.x"},
    )

    loaded = load_run_manifest(paths, run_id=None, panel_name="smoke")
    assert loaded["run_id"] == manifest["run_id"]
    assert loaded["panel_name"] == "smoke"
    assert "prepared_selection_manifest" in loaded["paths"]


def test_validate_run_manifest_config_hashes_fails_on_drift(tmp_path) -> None:
    workspace = tmp_path / "shape-of-wisdom"
    (workspace / ".git").mkdir(parents=True)
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    for name in ["runtime.yaml", "models.yaml", "datasets.yaml", "prompts.yaml", "readouts.yaml", "controls.yaml", "metrics.yaml"]:
        (paths.configs_dir / name).write_text(f"{name}: test\n", encoding="utf-8")
    manifest = create_run_manifest(
        paths,
        run_id="smoke_20260327T000000Z",
        panel_name="smoke",
        selections={"datasets": [], "models": [], "readouts": [], "permutations": []},
        config_hashes=build_config_hashes(paths),
        environment_summary={},
    )
    (paths.configs_dir / "runtime.yaml").write_text("runtime: changed\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_run_manifest_config_hashes(paths, manifest)
