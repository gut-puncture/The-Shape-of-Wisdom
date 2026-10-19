from __future__ import annotations

from pathlib import Path

import pytest

from path_manager import ExperimentPaths, is_within_path


def test_is_within_path_accepts_nested_paths(tmp_path: Path) -> None:
    parent = tmp_path / "workspace"
    child = parent / "outputs" / "x.json"
    assert is_within_path(child, parent)


def test_is_within_path_rejects_escape(tmp_path: Path) -> None:
    parent = tmp_path / "workspace"
    other = tmp_path / "other" / "x.json"
    assert not is_within_path(other, parent)


def test_assert_within_workspace_rejects_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "v2" / "semantic_candidate_vs_winner_isolation"
    paths = ExperimentPaths.from_workspace_root(workspace)
    with pytest.raises(ValueError):
        paths.assert_within_workspace(tmp_path / "outside.txt")


def test_ensure_required_dirs_creates_expected_structure(tmp_path: Path) -> None:
    workspace = tmp_path / "v2" / "semantic_candidate_vs_winner_isolation"
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    assert (workspace / "configs").is_dir()
    assert (workspace / "outputs" / "caches").is_dir()
    assert (workspace / "outputs" / "logs").is_dir()


def test_from_workspace_root_prefers_nearest_git_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "shape-of-wisdom"
    (workspace / ".git").mkdir(parents=True)
    paths = ExperimentPaths.from_workspace_root(workspace)
    assert paths.repo_root == workspace
    assert paths.workspace_root == workspace
