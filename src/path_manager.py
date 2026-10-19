from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_within_path(candidate: Path, parent: Path) -> bool:
    candidate_r = _resolved(candidate)
    parent_r = _resolved(parent)
    try:
        candidate_r.relative_to(parent_r)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ExperimentPaths:
    repo_root: Path
    workspace_root: Path

    @classmethod
    def from_workspace_root(cls, workspace_root: Path) -> "ExperimentPaths":
        workspace_root = _resolved(workspace_root)
        repo_root = workspace_root
        for candidate in [workspace_root, *workspace_root.parents]:
            if (candidate / ".git").exists():
                repo_root = candidate
                break
        else:
            if workspace_root.parent.name == "v2":
                repo_root = workspace_root.parent.parent
        return cls(repo_root=repo_root, workspace_root=workspace_root)

    @property
    def configs_dir(self) -> Path:
        return self.workspace_root / "configs"

    @property
    def src_dir(self) -> Path:
        return self.workspace_root / "src"

    @property
    def tests_dir(self) -> Path:
        return self.workspace_root / "tests"

    @property
    def manifests_dir(self) -> Path:
        return self.workspace_root / "manifests"

    @property
    def outputs_dir(self) -> Path:
        return self.workspace_root / "outputs"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    @property
    def latest_run_pointer_path(self) -> Path:
        return self.manifests_dir / "latest_run.json"

    def latest_panel_run_pointer_path(self, panel_name: str) -> Path:
        return self.manifests_dir / f"latest_run_{panel_name}.json"

    def run_paths(self, run_id: str) -> "RunPaths":
        return RunPaths(paths=self, run_id=run_id)

    def assert_within_workspace(self, path: Path) -> Path:
        resolved = _resolved(path)
        if not is_within_path(resolved, self.workspace_root):
            raise ValueError(
                f"path escapes workspace root: candidate={resolved} workspace_root={self.workspace_root}"
            )
        return resolved

    def resolve_under_workspace(self, *parts: str) -> Path:
        path = self.workspace_root.joinpath(*parts)
        return self.assert_within_workspace(path)

    def ensure_required_dirs(self) -> None:
        for path in [
            self.configs_dir,
            self.src_dir,
            self.tests_dir,
            self.manifests_dir,
            self.outputs_dir,
            self.outputs_dir / "smoke_test",
            self.outputs_dir / "qc",
            self.outputs_dir / "prepared_data",
            self.outputs_dir / "caches",
            self.runs_dir,
            self.outputs_dir / "metrics",
            self.outputs_dir / "stats",
            self.outputs_dir / "figures",
            self.logs_dir,
        ]:
            self.assert_within_workspace(path).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunPaths:
    paths: ExperimentPaths
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.paths.resolve_under_workspace("outputs", "runs", self.run_id)

    @property
    def run_manifest_path(self) -> Path:
        return self.run_root / "run_manifest.json"

    @property
    def prepared_data_dir(self) -> Path:
        return self.run_root / "prepared_data"

    @property
    def canonical_examples_path(self) -> Path:
        return self.prepared_data_dir / "canonical_examples.jsonl"

    @property
    def prepared_manifest_path(self) -> Path:
        return self.run_root / "manifests" / "prepared_examples_manifest.json"

    @property
    def selection_manifest_path(self) -> Path:
        return self.run_root / "manifests" / "prepared_selection_manifest.json"

    @property
    def metrics_dir(self) -> Path:
        return self.run_root / "metrics"

    @property
    def score_manifest_path(self) -> Path:
        return self.metrics_dir / "score_manifest.jsonl"

    @property
    def primary_metrics_path(self) -> Path:
        return self.metrics_dir / "primary_metrics.jsonl"

    @property
    def qc_dir(self) -> Path:
        return self.run_root / "qc"

    @property
    def token_audits_path(self) -> Path:
        return self.qc_dir / "token_audits.jsonl"

    @property
    def qc_summary_json_path(self) -> Path:
        return self.qc_dir / "qc_summary.json"

    @property
    def qc_summary_markdown_path(self) -> Path:
        return self.qc_dir / "qc_summary.md"

    @property
    def gpu_preflight_report_path(self) -> Path:
        return self.qc_dir / "gpu_preflight.json"

    @property
    def stats_dir(self) -> Path:
        return self.run_root / "stats"

    @property
    def bootstrap_summary_path(self) -> Path:
        return self.stats_dir / "bootstrap_summary.json"

    @property
    def model_dataset_positivity_path(self) -> Path:
        return self.stats_dir / "model_dataset_positivity.json"

    @property
    def model_positivity_summary_path(self) -> Path:
        return self.stats_dir / "model_positivity_summary.json"

    @property
    def family_positivity_summary_path(self) -> Path:
        return self.stats_dir / "family_positivity_summary.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    def failure_log_path(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}_failures.jsonl"

    @property
    def environment_prepare_path(self) -> Path:
        return self.logs_dir / "environment_prepare.json"

    @property
    def smoke_dir(self) -> Path:
        return self.run_root / "smoke_test"

    @property
    def smoke_manifest_path(self) -> Path:
        return self.smoke_dir / "smoke_manifest.json"

    def ensure_dirs(self) -> None:
        for path in [
            self.run_root,
            self.prepared_data_dir,
            self.prepared_manifest_path.parent,
            self.selection_manifest_path.parent,
            self.metrics_dir,
            self.qc_dir,
            self.stats_dir,
            self.logs_dir,
            self.smoke_dir,
        ]:
            self.paths.assert_within_workspace(path).mkdir(parents=True, exist_ok=True)
