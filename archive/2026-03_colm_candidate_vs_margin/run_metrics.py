from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from io_utils import read_json, read_jsonl, read_yaml, write_jsonl_atomic
from metrics_candidate_vs_margin import compute_primary_metrics
from model_registry import load_model_specs
from path_manager import ExperimentPaths
from run_manifest import load_run_manifest, resolve_manifest_path, update_run_manifest, validate_run_manifest_config_hashes
from runtime_panels import load_runtime_config, resolve_panel_name


def _resolve_cache_path(paths: ExperimentPaths, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = paths.workspace_root / path
    return paths.assert_within_workspace(path)


def load_metric_rows_for_run(
    paths: ExperimentPaths,
    manifest: dict,
    *,
    metrics_cfg: dict,
    model_specs: dict,
) -> list[dict]:
    score_manifest_path = resolve_manifest_path(paths, manifest, "score_manifest")
    score_rows = read_jsonl(score_manifest_path)
    rows = []
    for score_row in score_rows:
        cache_path = _resolve_cache_path(paths, str(score_row["cache_path"]))
        payload = read_json(cache_path)
        bundle = payload["bundle"]
        spec = model_specs.get(bundle.get("model_name"))
        rows.append(
            {
                "run_id": manifest["run_id"],
                "cache_path": str(cache_path),
                "cache_key": score_row.get("cache_key"),
                "unit_key": score_row.get("unit_key"),
                "example_id": bundle["example_id"],
                "dataset_name": bundle.get("dataset_name"),
                "model_name": bundle.get("model_name"),
                "model_family": bundle.get("model_family") or (spec.family if spec is not None else None),
                "model_size": bundle.get("model_size") or (spec.size if spec is not None else None),
                "panel_name": bundle.get("panel_name"),
                "readout_id": bundle.get("readout_id"),
                "permutation_id": bundle.get("permutation_id"),
                "metrics": compute_primary_metrics(
                    bundle,
                    gold_answer=bundle.get("gold_answer"),
                    tau=float(metrics_cfg["primary_tau"]),
                    delta=float(metrics_cfg["primary_delta"]),
                    tie_eps=float(metrics_cfg["tie_epsilon"]),
                ),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = ExperimentPaths.from_workspace_root(WORKSPACE_ROOT)
    runtime_cfg = load_runtime_config(paths.configs_dir / "runtime.yaml")

    explicit_panel = None
    if args.smoke or args.panel is not None:
        explicit_panel = resolve_panel_name(runtime_cfg, smoke=args.smoke, explicit_panel=args.panel)
    manifest = load_run_manifest(paths, run_id=args.run_id, panel_name=explicit_panel)
    validate_run_manifest_config_hashes(paths, manifest)

    metrics_cfg = read_yaml(paths.configs_dir / "metrics.yaml")["metrics"]
    model_specs = {spec.name: spec for spec in load_model_specs(paths.configs_dir / "models.yaml")}
    rows = load_metric_rows_for_run(paths, manifest, metrics_cfg=metrics_cfg, model_specs=model_specs)
    primary_metrics_path = resolve_manifest_path(paths, manifest, "primary_metrics")
    write_jsonl_atomic(primary_metrics_path, rows)
    update_run_manifest(
        paths,
        manifest,
        stage_name="run_metrics",
        updates={"metric_row_count": len(rows)},
    )


if __name__ == "__main__":
    main()
