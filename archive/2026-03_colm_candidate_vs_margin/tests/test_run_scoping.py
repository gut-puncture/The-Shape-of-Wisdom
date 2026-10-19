from __future__ import annotations

import json

from io_utils import write_json_atomic, write_jsonl_atomic
from path_manager import ExperimentPaths
from run_manifest import create_run_manifest
from run_metrics import load_metric_rows_for_run


def _cache_payload(example_id: str, *, model_name: str) -> dict:
    return {
        "cache_key": {"unit_id": example_id},
        "bundle": {
            "example_id": example_id,
            "dataset_name": "dataset_x",
            "model_name": model_name,
            "model_family": "family_x",
            "model_size": "1b",
            "panel_name": "smoke",
            "readout_id": "semantic_exact",
            "permutation_id": "identity",
            "gold_answer": "A",
            "final_winner": "A",
            "layers": [
                {"layer_index": 0, "options": {"A": {"normalized_probability": 0.6}, "B": {"normalized_probability": 0.4}}},
                {"layer_index": 1, "options": {"A": {"normalized_probability": 0.8}, "B": {"normalized_probability": 0.2}}},
            ],
        },
    }


def test_run_metrics_reads_only_manifest_declared_cache_rows(tmp_path) -> None:
    workspace = tmp_path / "shape-of-wisdom"
    (workspace / ".git").mkdir(parents=True)
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    for name in ["runtime.yaml", "models.yaml", "datasets.yaml", "prompts.yaml", "readouts.yaml", "controls.yaml", "metrics.yaml"]:
        (paths.configs_dir / name).write_text(f"{name}: test\n", encoding="utf-8")

    run_a = create_run_manifest(
        paths,
        run_id="smoke_a",
        panel_name="smoke",
        selections={"datasets": [], "models": [], "readouts": [], "permutations": []},
        config_hashes={},
        environment_summary={},
    )
    create_run_manifest(
        paths,
        run_id="smoke_b",
        panel_name="smoke",
        selections={"datasets": [], "models": [], "readouts": [], "permutations": []},
        config_hashes={},
        environment_summary={},
    )
    cache_a = paths.outputs_dir / "caches" / "semantic_scores" / "aa" / "cache_a.json"
    cache_b = paths.outputs_dir / "caches" / "semantic_scores" / "bb" / "cache_b.json"
    write_json_atomic(cache_a, _cache_payload("ex-a", model_name="model_a"))
    write_json_atomic(cache_b, _cache_payload("ex-b", model_name="model_b"))
    score_manifest = paths.run_paths("smoke_a").score_manifest_path
    write_jsonl_atomic(
        score_manifest,
        [
            {
                "run_id": "smoke_a",
                "unit_key": "model_a:ex-a:identity:semantic_exact",
                "cache_key": "digest-a",
                "cache_path": str(cache_a),
                "example_id": "ex-a",
                "dataset_name": "dataset_x",
                "model_name": "model_a",
                "readout_id": "semantic_exact",
                "permutation_id": "identity",
                "gold_answer": "A",
                "source": "computed",
            }
        ],
    )

    rows = load_metric_rows_for_run(paths, run_a, metrics_cfg={"primary_tau": 0.8, "primary_delta": 0.15, "tie_epsilon": 1.0e-12}, model_specs={})
    assert len(rows) == 1
    assert rows[0]["example_id"] == "ex-a"
    assert rows[0]["model_name"] == "model_a"


def test_run_metrics_are_scientifically_stable_across_runs_sharing_cache(tmp_path) -> None:
    workspace = tmp_path / "shape-of-wisdom"
    (workspace / ".git").mkdir(parents=True)
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    for name in ["runtime.yaml", "models.yaml", "datasets.yaml", "prompts.yaml", "readouts.yaml", "controls.yaml", "metrics.yaml"]:
        (paths.configs_dir / name).write_text(f"{name}: test\n", encoding="utf-8")

    run_a = create_run_manifest(
        paths,
        run_id="smoke_a",
        panel_name="smoke",
        selections={"datasets": [], "models": [], "readouts": [], "permutations": []},
        config_hashes={},
        environment_summary={},
    )
    run_b = create_run_manifest(
        paths,
        run_id="smoke_b",
        panel_name="smoke",
        selections={"datasets": [], "models": [], "readouts": [], "permutations": []},
        config_hashes={},
        environment_summary={},
    )
    cache_path = paths.outputs_dir / "caches" / "semantic_scores" / "aa" / "shared.json"
    write_json_atomic(cache_path, _cache_payload("ex-shared", model_name="model_shared"))
    row = {
        "cache_key": "digest-shared",
        "cache_path": str(cache_path),
        "example_id": "ex-shared",
        "dataset_name": "dataset_x",
        "model_name": "model_shared",
        "readout_id": "semantic_exact",
        "permutation_id": "identity",
        "gold_answer": "A",
        "source": "cache_hit",
    }
    write_jsonl_atomic(paths.run_paths("smoke_a").score_manifest_path, [{"run_id": "smoke_a", "unit_key": "u", **row}])
    write_jsonl_atomic(paths.run_paths("smoke_b").score_manifest_path, [{"run_id": "smoke_b", "unit_key": "u", **row}])

    rows_a = load_metric_rows_for_run(
        paths,
        run_a,
        metrics_cfg={"primary_tau": 0.8, "primary_delta": 0.15, "tie_epsilon": 1.0e-12},
        model_specs={},
    )
    rows_b = load_metric_rows_for_run(
        paths,
        run_b,
        metrics_cfg={"primary_tau": 0.8, "primary_delta": 0.15, "tie_epsilon": 1.0e-12},
        model_specs={},
    )
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    rows_a[0].pop("run_id", None)
    rows_b[0].pop("run_id", None)
    assert rows_a == rows_b
