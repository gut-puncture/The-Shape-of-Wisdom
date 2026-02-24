#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _common import write_json  # noqa: E402
from sow.v2.runtime_policy import choose_backend, estimate_runtime  # noqa: E402


BASE_PIPELINE = [
    "01_extract_baseline.py",
    "02_compute_decision_metrics.py",
    "03_classify_trajectories.py",
    "04_region_analysis.py",
    "05_span_counterfactuals.py",
    "06_select_tracing_subset.py",
    "07_run_tracing.py",
    "08_attention_and_mlp_decomposition.py",
    "09_causal_tests.py",
    "10_causal_validation_tools.py",
    "14_readiness_audit.py",
    "11_generate_paper_assets.py",
]

HEAVY_STAGES = {"05_span_counterfactuals.py", "07_run_tracing.py"}
BACKEND_REASON_REQUIRED_KEYS = {
    "estimated_hours_all_models",
    "threshold_hours_all_models",
    "rows_per_second",
    "rows_per_second_source",
    "baseline_prompt_count_current",
    "backend_decision",
    "decision_rule",
}
DEPENDENCIES = {
    "08_attention_and_mlp_decomposition.py": {"07_run_tracing.py"},
    "09_causal_tests.py": {"07_run_tracing.py"},
    "10_causal_validation_tools.py": {"05_span_counterfactuals.py"},
    "11_generate_paper_assets.py": {
        "08_attention_and_mlp_decomposition.py",
        "09_causal_tests.py",
        "10_causal_validation_tools.py",
        "14_readiness_audit.py",
    },
    "14_readiness_audit.py": {
        "03_classify_trajectories.py",
        "05_span_counterfactuals.py",
        "06_select_tracing_subset.py",
        "07_run_tracing.py",
        "08_attention_and_mlp_decomposition.py",
        "09_causal_tests.py",
        "10_causal_validation_tools.py",
    },
}


def _pipeline_for_mode(mode: str) -> List[str]:
    if mode in {"full", "single_model"}:
        return ["00a_generate_baseline_outputs.py", *BASE_PIPELINE]
    return list(BASE_PIPELINE)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_contract_paths(*, cfg: dict, config_path: Path) -> list[Path]:
    exp = cfg.get("experiment") or {}
    paths = [
        Path(config_path),
        Path(str(exp.get("objective_doc") or (REPO_ROOT / "docs" / "PAPER_OBJECTIVE_V3.md"))),
        Path(str(exp.get("implementation_doc") or (REPO_ROOT / "docs" / "IMPLEMENTATION_PLAN_V3.md"))),
        REPO_ROOT / "docs" / "MODEL_NUANCES_V2.md",
        Path(str(exp.get("preregistration_doc") or (REPO_ROOT / "docs" / "PREREGISTERED_HYPOTHESES_V3.md"))),
    ]
    uniq = []
    seen = set()
    for p in paths:
        r = p.expanduser().resolve()
        s = str(r)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(r)
    return uniq


def _write_run_start_metadata_snapshot(*, run_id: str, cfg: dict, config_path: Path) -> Path:
    out_root = REPO_ROOT / "runs" / str(run_id) / "v2"
    snapshot_path = out_root / "meta" / "run_start_metadata_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for p in _metadata_contract_paths(cfg=cfg, config_path=config_path):
        if p.exists():
            records.append({"path": str(p), "sha256": _sha256(p)})
        else:
            records.append({"path": str(p), "sha256": None})
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "files": records,
    }
    write_json(snapshot_path, payload)
    return snapshot_path


def _baseline_prompt_count(run_id: str) -> int:
    ccc = REPO_ROOT / "runs" / run_id / "manifests" / "ccc_baseline.jsonl"
    base = REPO_ROOT / "runs" / run_id / "manifests" / "baseline_manifest.jsonl"
    p = ccc if ccc.exists() else base
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _pilot_rows_per_second(run_id: str) -> Tuple[float, str]:
    pilot_dir = REPO_ROOT / "runs" / run_id / "pilot"
    values = []
    if not pilot_dir.exists():
        return 0.2, "default_fallback_0.2"
    for p in pilot_dir.glob("*_pilot_report.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = float(obj.get("sample_size") or obj.get("metrics_overall", {}).get("n") or 0.0)
        t = float(obj.get("timing_seconds", {}).get("inference") or 0.0)
        if n > 0 and t > 0:
            values.append(n / t)
    if values:
        return float(min(values)), "pilot_report"
    return 0.2, "default_fallback_0.2"


def _stage00a_rows_per_second(run_id: str) -> Tuple[float, str]:
    path = REPO_ROOT / "runs" / run_id / "v2" / "00a_generate_baseline_outputs.report.json"
    if not path.exists():
        return 0.0, "stage00a_missing"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0, "stage00a_invalid"

    direct = float(obj.get("rows_per_second") or 0.0)
    if direct > 0:
        return direct, "stage00a_report"

    stats = obj.get("stats_per_model") or {}
    vals = []
    if isinstance(stats, dict):
        for entry in stats.values():
            try:
                v = float((entry or {}).get("rows_per_second") or 0.0)
            except Exception:
                continue
            if v > 0:
                vals.append(v)
    if vals:
        return float(min(vals)), "stage00a_report"
    return 0.0, "stage00a_report"


def _runtime_rows_per_second(run_id: str) -> Tuple[float, str]:
    rps_stage00a, src_stage00a = _stage00a_rows_per_second(run_id)
    if rps_stage00a > 0:
        return rps_stage00a, src_stage00a
    return _pilot_rows_per_second(run_id)


def _runtime_source_precedence(selected_source: str) -> dict:
    ordered = ["stage00a_report", "pilot_report", "default_fallback_0.2"]
    return {
        "selected_source": str(selected_source),
        "ordered_sources": ordered,
    }


def _run_script(script_name: str, argv: List[str]) -> int:
    script = REPO_ROOT / "scripts" / "v2" / script_name
    cmd = [sys.executable, str(script), *argv]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


def _heavy_stage_estimates(*, cfg: dict, baseline_count: int, rows_per_second: float) -> dict:
    runtime_cfg = cfg.get("runtime_estimator") or {}
    threshold_model_count = int(runtime_cfg.get("threshold_model_count", len(cfg.get("models") or [])) or 3)
    stage_row_multiplier = dict(runtime_cfg.get("stage_row_multiplier") or {})
    defaults = {"05_span_counterfactuals.py": 8.0, "07_run_tracing.py": 1.0}
    out = {}
    for stage in sorted(HEAVY_STAGES):
        mult = float(stage_row_multiplier.get(stage, defaults.get(stage, 1.0)))
        prompts_per_model = max(1, int(round(float(baseline_count) * mult)))
        est = estimate_runtime(
            task_name=stage,
            rows_per_second=max(float(rows_per_second), 1e-9),
            prompts_per_model=prompts_per_model,
            model_count=threshold_model_count,
        )
        out[stage] = est
    return out


def _enforce_runtime_rps_policy(*, cfg: dict, mode: str, rows_per_second: float, rps_source: str) -> None:
    runtime_cfg = cfg.get("runtime_estimator") or {}
    require_measured = bool(runtime_cfg.get("require_measured_rps_for_full", False))
    if mode == "full" and require_measured:
        if str(rps_source).startswith("default_fallback"):
            raise RuntimeError(
                "full mode requires measured throughput; default fallback RPS is disallowed by runtime_estimator.require_measured_rps_for_full"
            )
        if float(rows_per_second) <= 0.0:
            raise RuntimeError("full mode requires positive measured throughput")


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 experiment orchestrator")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", choices=["smoke", "single_model", "full"], default="full")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_v2.yaml"))
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_cfg(cfg_path)

    if args.mode == "single_model" and not args.model_name:
        raise SystemExit("--mode single_model requires --model-name")

    snapshot_path = _write_run_start_metadata_snapshot(run_id=args.run_id, cfg=cfg, config_path=cfg_path)

    out_root = REPO_ROOT / "runs" / args.run_id / "v2"
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "00_run_experiment.report.json"

    if args.snapshot_only:
        report = {
            "pass": True,
            "complete": False,
            "snapshot_only": True,
            "run_id": args.run_id,
            "mode": args.mode,
            "model_name": args.model_name,
            "executed_scripts": [],
            "skipped_scripts": [],
            "failed_script": None,
            "failed_exit_code": None,
            "requires_gpu_handoff": False,
            # Snapshot-only mode captures provenance and preflight-readiness before stage execution.
            "ready_to_execute_full_experiment": True,
            "runtime_source_precedence": _runtime_source_precedence("unset"),
            "run_start_metadata_snapshot": str(snapshot_path),
        }
        write_json(report_path, report)
        print(str(report_path))
        return 0

    threshold = float((cfg.get("runtime_estimator") or {}).get("threshold_hours_all_models", 36.0))

    mode_max_prompts = {"smoke": 25, "single_model": 300, "full": 0}[args.mode]

    base_argv = ["--run-id", args.run_id, "--config", str(cfg_path)]
    if args.model_name:
        base_argv.extend(["--model-name", args.model_name])
    if mode_max_prompts > 0:
        base_argv.extend(["--max-prompts", str(mode_max_prompts)])
    if args.resume:
        base_argv.append("--resume")

    pipeline = _pipeline_for_mode(args.mode)
    executed: List[str] = []
    skipped_names = set()
    skipped = []
    failed_script = None
    failed_exit_code = None

    rps = 0.0
    rps_source = "unset"
    baseline_prompt_count_current = 0
    stage_est = {}
    stage_backend = {}
    backend_reason = {}

    def _refresh_runtime_decisions() -> None:
        nonlocal rps, rps_source, stage_est, stage_backend, backend_reason, baseline_prompt_count_current
        baseline_prompt_count_current = _baseline_prompt_count(args.run_id)
        rps, rps_source = _runtime_rows_per_second(args.run_id)
        if int(baseline_prompt_count_current) > 0:
            _enforce_runtime_rps_policy(cfg=cfg, mode=args.mode, rows_per_second=rps, rps_source=rps_source)
        stage_est = _heavy_stage_estimates(cfg=cfg, baseline_count=baseline_prompt_count_current, rows_per_second=max(rps, 1e-9))
        stage_backend = {
            stage: choose_backend(estimated_hours_all_models=est.estimated_hours_all_models, threshold_hours=threshold)
            for stage, est in stage_est.items()
        }
        backend_reason = {}
        for stage, est in stage_est.items():
            payload = {
                "estimated_hours_all_models": float(est.estimated_hours_all_models),
                "threshold_hours_all_models": float(threshold),
                "rows_per_second": float(rps),
                "rows_per_second_source": str(rps_source),
                "baseline_prompt_count_current": int(baseline_prompt_count_current),
                "backend_decision": str(stage_backend.get(stage) or "unknown"),
                "decision_rule": "choose_backend(estimated_hours_all_models, threshold_hours_all_models)",
            }
            missing = sorted(k for k in BACKEND_REASON_REQUIRED_KEYS if k not in payload)
            if missing:
                raise RuntimeError(f"invalid heavy_stage_backend_reason payload for {stage}: missing={missing}")
            backend_reason[stage] = payload

    # Smoke/single-model may not run stage00a; establish decisions upfront.
    if args.mode != "full":
        _refresh_runtime_decisions()

    for script_name in pipeline:
        deps = DEPENDENCIES.get(script_name, set())
        missing_deps = sorted(d for d in deps if (d in skipped_names or d not in executed))
        if missing_deps:
            skipped.append({"script": script_name, "reason": f"dependency skipped: {', '.join(missing_deps)}"})
            skipped_names.add(script_name)
            continue

        if args.mode == "smoke" and script_name == "07_run_tracing.py":
            skipped.append({"script": script_name, "reason": "smoke mode skips model tracing stage"})
            skipped_names.add(script_name)
            continue

        if script_name in HEAVY_STAGES:
            try:
                _refresh_runtime_decisions()
            except RuntimeError as exc:
                failed_script = script_name
                failed_exit_code = 2
                skipped.append({"script": script_name, "reason": f"runtime policy failure: {exc}"})
                skipped_names.add(script_name)
                for rest in pipeline[pipeline.index(script_name) + 1 :]:
                    skipped.append({"script": rest, "reason": f"upstream stage failed: {script_name} (rc=2)"})
                    skipped_names.add(rest)
                break
            if args.mode == "full" and int(baseline_prompt_count_current) <= 0:
                failed_script = script_name
                failed_exit_code = 2
                skipped.append(
                    {
                        "script": script_name,
                        "reason": "runtime policy failure: full mode requires non-zero baseline prompt count before heavy stages",
                    }
                )
                skipped_names.add(script_name)
                for rest in pipeline[pipeline.index(script_name) + 1 :]:
                    skipped.append({"script": rest, "reason": f"upstream stage failed: {script_name} (rc=2)"})
                    skipped_names.add(rest)
                break
            if stage_backend.get(script_name) == "gpu":
                skipped.append({"script": script_name, "reason": "estimated runtime exceeds 36h threshold for all models"})
                skipped_names.add(script_name)
                continue

        script_argv = list(base_argv)
        if args.mode == "smoke" and script_name == "05_span_counterfactuals.py":
            script_argv.extend(["--counterfactual-mode", "proxy"])

        rc = _run_script(script_name, script_argv)
        if rc != 0:
            failed_script = script_name
            failed_exit_code = int(rc)
            for rest in pipeline[pipeline.index(script_name) + 1 :]:
                skipped.append({"script": rest, "reason": f"upstream stage failed: {script_name} (rc={rc})"})
                skipped_names.add(rest)
            break

        executed.append(script_name)

        # Stage00a updates measured throughput contract for full runs.
        if script_name == "00a_generate_baseline_outputs.py":
            try:
                _refresh_runtime_decisions()
            except RuntimeError as exc:
                failed_script = script_name
                failed_exit_code = 2
                for rest in pipeline[pipeline.index(script_name) + 1 :]:
                    skipped.append({"script": rest, "reason": f"upstream stage failed: {script_name} (runtime policy: {exc})"})
                    skipped_names.add(rest)
                break

    complete = (len(skipped) == 0) and (failed_script is None)
    ready_to_execute_full_experiment = bool(complete)

    report = {
        "pass": bool(complete),
        "complete": bool(complete),
        "snapshot_only": False,
        "run_id": args.run_id,
        "mode": args.mode,
        "model_name": args.model_name,
        "runtime_rows_per_second": float(rps),
        "runtime_rows_per_second_source": str(rps_source),
        "runtime_source_precedence": _runtime_source_precedence(str(rps_source)),
        "baseline_prompt_count_current": int(baseline_prompt_count_current),
        "runtime_policy_enforced": bool(
            args.mode == "full" and bool((cfg.get("runtime_estimator") or {}).get("require_measured_rps_for_full", False))
        ),
        "runtime_threshold_hours_all_models": float(threshold),
        "heavy_stage_estimates_hours_all_models": {
            stage: float(est.estimated_hours_all_models) for stage, est in stage_est.items()
        },
        "heavy_stage_backend_decision": stage_backend,
        "heavy_stage_backend_reason": backend_reason,
        "executed_scripts": executed,
        "skipped_scripts": skipped,
        "requires_gpu_handoff": any(
            (x.get("script") in HEAVY_STAGES) and ("exceeds 36h" in str(x.get("reason")))
            for x in skipped
        ),
        "failed_script": failed_script,
        "failed_exit_code": failed_exit_code,
        "ready_to_execute_full_experiment": bool(ready_to_execute_full_experiment),
        "run_start_metadata_snapshot": str(snapshot_path),
    }
    write_json(report_path, report)
    print(str(report_path))
    if failed_exit_code is not None:
        return int(failed_exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
