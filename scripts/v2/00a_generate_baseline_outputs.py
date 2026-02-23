#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import (
    base_parser,
    baseline_output_path,
    load_experiment_config,
    load_jsonl_rows,
    resolve_models,
    run_root_for,
    run_v2_root_for,
    write_json,
    write_text_atomic,
)
from sow.io_jsonl import iter_jsonl
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig
from sow.v2.baseline_inference import run_baseline_for_model, validate_baseline_rows


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_manifest(*, run_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_root = run_root_for(run_id)
    manifests_dir = run_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = manifests_dir / "ccc_baseline.jsonl"

    data_scope = cfg.get("data_scope") or {}
    source_path = Path(str(data_scope.get("baseline_manifest_source") or "")).expanduser() if data_scope.get("baseline_manifest_source") else None
    expected_sha = str(data_scope.get("baseline_manifest_sha256") or "").strip().lower()

    if run_manifest.exists():
        run_sha = _sha256(run_manifest)
        is_empty_placeholder = run_manifest.stat().st_size == 0
        hash_ok = bool((not expected_sha) or (run_sha == expected_sha))
        return {
            "manifest_path": run_manifest,
            "manifest_sha256": run_sha,
            "source_path": str(run_manifest),
            "hash_ok": bool(hash_ok),
            "manifest_empty_placeholder": bool(is_empty_placeholder),
            "hash_error": None
            if hash_ok
            else (
                "existing run manifest sha256 mismatch: "
                f"expected={expected_sha} got={run_sha} path={run_manifest}"
            ),
            "copied_from_source": False,
            "expected_sha256": expected_sha or None,
        }

    if source_path is None or (not source_path.exists()):
        raise SystemExit(f"missing data_scope.baseline_manifest_source: {source_path}")

    source_sha = _sha256(source_path)
    if expected_sha and source_sha != expected_sha:
        raise SystemExit(
            "baseline manifest sha256 mismatch: "
            f"expected={expected_sha} got={source_sha} source={source_path}"
        )

    write_text_atomic(run_manifest, source_path.read_text(encoding="utf-8"))
    run_sha = _sha256(run_manifest)
    hash_ok = (not expected_sha) or (run_sha == expected_sha)
    return {
        "manifest_path": run_manifest,
        "manifest_sha256": run_sha,
        "source_path": str(source_path),
        "hash_ok": bool(hash_ok),
        "manifest_empty_placeholder": False,
        "hash_error": None
        if hash_ok
        else (
            "copied run manifest sha256 mismatch: "
            f"expected={expected_sha} got={run_sha} path={run_manifest}"
        ),
        "copied_from_source": True,
        "expected_sha256": expected_sha or None,
    }


def _load_manifest_rows(path: Path, *, max_prompts: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(path), start=1):
        rows.append(dict(row))
        if max_prompts > 0 and idx >= int(max_prompts):
            break
    return rows


def _batch_chain_for_runtime(execution_cfg: Dict[str, Any]) -> List[int]:
    try:
        import torch
    except Exception:
        return [1]

    if torch.cuda.is_available():
        chain = execution_cfg.get("stage00_baseline_batch_chain_cuda") or [8, 6, 4, 2, 1]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        chain = execution_cfg.get("stage00_baseline_batch_chain_mps") or [6, 5, 4, 3, 2, 1]
    else:
        chain = [1]

    out = [int(x) for x in chain if int(x) > 0]
    return out if out else [1]


def main() -> int:
    ap = base_parser("V2: generate baseline outputs with deterministic fail-closed contract")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    if not models:
        raise SystemExit("no models selected")

    out_root = run_v2_root_for(args.run_id)
    report_path = out_root / "00a_generate_baseline_outputs.report.json"

    manifest_meta = _prepare_manifest(run_id=args.run_id, cfg=cfg)
    if not bool(manifest_meta.get("hash_ok")):
        report = {
            "pass": False,
            "run_id": str(args.run_id),
            "manifest": {
                "path": str(manifest_meta.get("manifest_path")),
                "source_path": str(manifest_meta.get("source_path")),
                "sha256": str(manifest_meta.get("manifest_sha256")),
                "expected_sha256": manifest_meta.get("expected_sha256"),
                "copied_from_source": bool(manifest_meta.get("copied_from_source")),
                "manifest_empty_placeholder": bool(manifest_meta.get("manifest_empty_placeholder")),
                "hash_error": str(manifest_meta.get("hash_error") or ""),
                "rows": 0,
                "rows_expected_full": int((cfg.get("data_scope") or {}).get("baseline_manifest_expected_rows_full", 0) or 0),
            },
            "execution": {},
            "stats_per_model": {},
            "rows_per_second": 0.0,
            "gates": {
                "models_selected": len(models) > 0,
                "manifest_hash_contract": False,
                "manifest_nonempty": False,
                "manifest_row_count_expected": False,
                "rows_complete_per_model": False,
                "all_model_validations_pass": False,
                "layer_index_contract_transformer_only": False,
                "thermal_checkpoint_not_triggered": True,
            },
            "failing_gates": [
                "manifest_hash_contract",
                "manifest_nonempty",
                "manifest_row_count_expected",
                "rows_complete_per_model",
                "all_model_validations_pass",
                "layer_index_contract_transformer_only",
            ],
            "layer_index_contract": "transformer_only",
            "stopped_early": False,
            "stop_reason": None,
            "thermal_action": None,
            "done_sentinel": None,
        }
        write_json(report_path, report)
        print(str(report_path))
        return 2
    manifest_rows = _load_manifest_rows(Path(manifest_meta["manifest_path"]), max_prompts=int(args.max_prompts))

    execution_cfg = cfg.get("execution") or {}
    data_scope_cfg = cfg.get("data_scope") or {}
    checkpoint_every = max(1, int(execution_cfg.get("stage00_baseline_checkpoint_every_prompts", 1)))
    batch_chain = _batch_chain_for_runtime(execution_cfg)
    deterministic_seed = int(execution_cfg.get("deterministic_seed", 12345))
    expected_rows_full = int(data_scope_cfg.get("baseline_manifest_expected_rows_full", 0) or 0)
    is_full_mode = int(args.max_prompts) == 0

    thermal_cfg_obj = ThermalHygieneConfig.from_cfg(cfg.get("thermal_policy"))
    thermal_events = out_root / "meta" / "thermal_events_stage00a.jsonl"
    governor = ThermalGovernor(cfg=thermal_cfg_obj, events_path=thermal_events) if thermal_cfg_obj.enabled else None

    stats_per_model: Dict[str, Dict[str, Any]] = {}
    model_pass_flags: List[bool] = []
    rows_complete_flags: List[bool] = []
    layer_contract_flags: List[bool] = []
    measured_rps: List[float] = []
    thermal_stop_action: Dict[str, Any] | None = None

    for model in models:
        model_id = str(model["model_id"])
        model_revision = str(model.get("revision") or "")
        out_path = baseline_output_path(args.run_id, model_id)
        if (not args.resume) and out_path.exists():
            out_path.unlink()

        def _thermal_check() -> Dict[str, Any]:
            if governor is None:
                return {}
            return governor.maybe_cooldown(
                stage="v2_baseline_inference",
                model_id=model_id,
                model_revision=model_revision,
            )

        run_stats = run_baseline_for_model(
            run_id=args.run_id,
            model=model,
            manifest_rows=manifest_rows,
            out_path=out_path,
            resume=bool(args.resume),
            checkpoint_every_prompts=checkpoint_every,
            batch_chain=batch_chain,
            deterministic_seed=deterministic_seed,
            thermal_check_fn=_thermal_check,
        )

        if bool(run_stats.get("stopped_early")):
            thermal_stop_action = {
                "model_id": model_id,
                "model_revision": model_revision,
                "action": run_stats.get("thermal_action"),
            }

        rows = load_jsonl_rows(out_path, max_rows=0) if out_path.exists() else []
        validation = validate_baseline_rows(
            rows=rows,
            expected_model_id=model_id,
            expected_model_revision=model_revision,
        )

        target_rows = int(len(manifest_rows))
        observed_rows = int(len(rows))
        layer_contract_ok = "invalid_layer_index_sequence" not in set(validation.get("errors") or [])
        rows_complete_ok = int(observed_rows) == int(target_rows)
        model_gates = {
            "schema_and_uniqueness": bool(validation.get("pass")),
            "rows_complete": bool(rows_complete_ok),
            "layer_index_contract_transformer_only": bool(layer_contract_ok),
        }
        model_failing = sorted([k for k, v in model_gates.items() if not bool(v)])
        model_pass = len(model_failing) == 0
        model_pass_flags.append(model_pass)
        rows_complete_flags.append(bool(rows_complete_ok))
        layer_contract_flags.append(bool(layer_contract_ok))

        rps = float(run_stats.get("rows_per_second") or 0.0)
        if rps > 0:
            measured_rps.append(rps)

        stats_per_model[model_id] = {
            **run_stats,
            "rows_in_output": observed_rows,
            "target_rows": target_rows,
            "validation": validation,
            "gates": model_gates,
            "failing_gates": model_failing,
            "pass": bool(model_pass),
        }
        if thermal_stop_action is not None:
            break

    gates = {
        "models_selected": len(models) > 0,
        "manifest_hash_contract": bool(manifest_meta.get("hash_ok")),
        "manifest_nonempty": int(len(manifest_rows)) > 0,
        "manifest_row_count_expected": (not is_full_mode)
        or (expected_rows_full <= 0)
        or (int(len(manifest_rows)) == int(expected_rows_full)),
        "rows_complete_per_model": bool(rows_complete_flags and all(rows_complete_flags)),
        "all_model_validations_pass": bool(model_pass_flags and all(model_pass_flags)),
        "layer_index_contract_transformer_only": bool(layer_contract_flags and all(layer_contract_flags)),
        "thermal_checkpoint_not_triggered": thermal_stop_action is None,
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    pass_flag = len(failing_gates) == 0
    done_sentinel = out_root / "00a_generate_baseline_outputs.done"
    if pass_flag:
        write_text_atomic(
            done_sentinel,
            json.dumps({"stage": "00a_generate_baseline_outputs", "pass": True}, ensure_ascii=False, sort_keys=True) + "\n",
        )

    rows_per_second = float(min(measured_rps)) if measured_rps else 0.0
    report = {
        "pass": bool(pass_flag and thermal_stop_action is None),
        "run_id": str(args.run_id),
        "manifest": {
            "path": str(manifest_meta.get("manifest_path")),
            "source_path": str(manifest_meta.get("source_path")),
            "sha256": str(manifest_meta.get("manifest_sha256")),
            "expected_sha256": manifest_meta.get("expected_sha256"),
            "copied_from_source": bool(manifest_meta.get("copied_from_source")),
            "manifest_empty_placeholder": bool(manifest_meta.get("manifest_empty_placeholder")),
            "rows": int(len(manifest_rows)),
            "rows_expected_full": int(expected_rows_full),
        },
        "execution": {
            "checkpoint_every_prompts": int(checkpoint_every),
            "batch_chain": batch_chain,
            "deterministic_seed": int(deterministic_seed),
        },
        "stats_per_model": stats_per_model,
        "rows_per_second": float(rows_per_second),
        "gates": gates,
        "failing_gates": failing_gates,
        "layer_index_contract": "transformer_only",
        "stopped_early": bool(thermal_stop_action is not None),
        "stop_reason": "thermal_checkpoint" if thermal_stop_action is not None else None,
        "thermal_action": thermal_stop_action,
        "done_sentinel": str(done_sentinel) if pass_flag else None,
    }
    write_json(report_path, report)
    print(str(report_path))
    if thermal_stop_action is not None:
        return 95
    return 0 if pass_flag else 2


if __name__ == "__main__":
    raise SystemExit(main())
