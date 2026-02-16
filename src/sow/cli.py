from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sow.config import default_run_config, read_yaml, validate_run_config, write_yaml
from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.judging.deterministic_parser import parse_choice
from sow.manifest.canonicalize import canonicalize_baseline_manifest, canonicalize_robustness_manifest_v2
from sow.manifest.schema import validate_baseline_manifest, validate_robustness_manifest
from sow.pca.membership import select_pca_membership, write_membership_file
from sow.pca.sample_inference import run_pca_sample_inference_for_model
from sow.pca.fit_pca import run_stage12_fit_for_model
from sow.pcc.build_pcc import build_pcc_manifests_for_model
from sow.ccc.build_ccc import (
    build_ccc_manifests,
    check_ccc_gates,
    compute_ccc_intersection,
    compute_retention_metrics,
    read_pcc_example_ids_and_domain_counts,
)
from sow.state import HashedPath, append_state_entry
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig
from sow.token_buckets.option_buckets import build_and_write_option_buckets_for_models, model_fs_id
from sow.stage0_env import collect_environment, run_smoke_test
from sow.pilot.pilot_inference import run_pilot_for_model, select_pilot_rows, stage7_gate
from sow.inference.stage13 import (
    batch_consistency_gate,
    run_stage13_inference_for_model,
)
from sow.analysis.stage14 import run_stage14_analysis


REPO_ROOT = Path(__file__).resolve().parents[2]


def _runs_root() -> Path:
    """
    Allow placing run artifacts on a separate disk (e.g., GPU VM attached volume).

    If `SOW_RUNS_ROOT` is set, it is treated as an absolute/relative path to the
    directory containing run subdirectories (i.e., `<runs_root>/<run_id>/...`).
    """
    env = os.environ.get("SOW_RUNS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "runs"


def _run_dir(run_id: str) -> Path:
    return _runs_root() / run_id


def _state_path() -> Path:
    return REPO_ROOT / "STATE.md"


def _config_snapshot_path(run_dir: Path) -> Path:
    return run_dir / "meta" / "config_snapshot.yaml"


def _config_snapshot_sha256(run_dir: Path) -> str | None:
    p = _config_snapshot_path(run_dir)
    return sha256_file(p) if p.exists() else None


def _next_available_path(path: Path) -> Path:
    """
    Return `path` if it doesn't exist; otherwise append a deterministic attempt suffix.
    """
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def cmd_init_run(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    if run_dir.exists():
        raise SystemExit(f"run dir already exists: {run_dir}")
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    (run_dir / "pca").mkdir(parents=True, exist_ok=True)
    (run_dir / "sentinels").mkdir(parents=True, exist_ok=True)

    cfg = default_run_config(run_id=args.run_id, random_seed=int(args.seed))
    validate_run_config(cfg)
    cfg_path = run_dir / "run_config.yaml"
    write_yaml(cfg_path, cfg)

    # Snapshot for meta (identical content but fixed location).
    snap_path = run_dir / "meta" / "config_snapshot.yaml"
    write_yaml(snap_path, cfg)

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 0 - init-run-config",
        status="PASS",
        command=f"python3 sow.py init-run --run-id {args.run_id} --seed {args.seed}",
        inputs=[],
        outputs=[
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
            HashedPath(path=str(snap_path), sha256=sha256_file(snap_path)),
        ],
        validators=[],
        notes="Pinned model revisions; greedy decoding; max_new_tokens=24; PCA sample size=1000.",
        next_step="Stage 2/3/4 - build-manifests",
    )
    print(str(cfg_path))
    return 0


def cmd_stage0(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    meta_dir = run_dir / "meta"
    v_dir = run_dir / "validation"
    s_dir = run_dir / "sentinels"
    meta_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    env = collect_environment(repo_root=REPO_ROOT)
    env_path = meta_dir / "environment.json"
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Default to the first model in run_config for smoke.
    model_name = args.model_name or cfg["models"][0]["name"]
    model = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if model is None:
        raise SystemExit(f"unknown model name: {model_name}")

    th_cfg = ThermalHygieneConfig.from_cfg(cfg.get("thermal_hygiene"))
    thermal_governor = ThermalGovernor(cfg=th_cfg, events_path=meta_dir / "thermal_events.jsonl") if th_cfg.enabled else None

    try:
        smoke = run_smoke_test(
            model_id=model["model_id"],
            revision=model["revision"],
            generation=cfg["generation"],
            seed=int(cfg["random_seed"]),
            preferred_device=args.device,
            thermal_governor=thermal_governor,
        )
    except Exception as exc:
        smoke = {
            "pass": False,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    smoke_path = meta_dir / "smoke_test.json"
    smoke_path.write_text(json.dumps(smoke, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    v_report = {
        "pass": bool(smoke.get("pass")),
        "run_id": args.run_id,
        "model_name": model_name,
        "environment_sha256": sha256_file(env_path),
        "smoke_test_sha256": sha256_file(smoke_path),
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = v_dir / "stage0_report.json"
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    s_path = s_dir / "stage0.done"
    s_path.write_text(json.dumps({"stage": "stage0", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n")

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 0 - environment lock + smoke test",
        status="PASS" if v_report["pass"] else "FAIL",
        command="python3 sow.py stage0 --run-id " + args.run_id + (f" --model-name {model_name}" if args.model_name else "") + (f" --device {args.device}" if args.device else ""),
        inputs=[HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path))],
        outputs=[
            HashedPath(path=str(env_path), sha256=sha256_file(env_path)),
            HashedPath(path=str(smoke_path), sha256=sha256_file(smoke_path)),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Smoke test attempts: tokenizer+model load, forward pass w/ hidden states, token bucket scoring, greedy generate.",
        next_step="Stage 7 - pilot inference" if v_report["pass"] else "Fix environment/smoke issues before any inference",
    )

    print(str(v_path))
    return 0 if v_report["pass"] else 2


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def cmd_build_manifests(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    in_baseline = REPO_ROOT / "data" / "experiment_inputs" / "main_prompts.jsonl"
    in_robust = REPO_ROOT / "data" / "experiment_inputs" / "robustness_prompts_v2.jsonl"

    out_baseline = run_dir / "manifests" / "baseline_manifest.jsonl"
    out_baseline_meta = run_dir / "manifests" / "baseline_manifest.meta.json"
    out_baseline_report = run_dir / "manifests" / "baseline_manifest.canonicalization_report.json"

    out_robust = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
    out_robust_meta = run_dir / "manifests" / "robustness_manifest_v2.meta.json"
    out_robust_report = run_dir / "manifests" / "robustness_manifest_v2.canonicalization_report.json"

    canonicalize_baseline_manifest(
        run_id=args.run_id,
        input_path=in_baseline,
        output_path=out_baseline,
        meta_path=out_baseline_meta,
        report_path=out_baseline_report,
    )

    try:
        canonicalize_robustness_manifest_v2(
            run_id=args.run_id,
            input_path=in_robust,
            output_path=out_robust,
            meta_path=out_robust_meta,
            report_path=out_robust_report,
            repair_missing=not args.validate_only,
        )
    except ValueError as exc:
        # Validate-only mode is expected to fail if the paid input has known issues.
        if args.validate_only:
            print(str(out_robust_report))
            print(f"validate-only failed: {exc}")
            return 2
        raise

    # Validate (strict stage gate).
    baseline_rows = _load_manifest_rows(out_baseline)
    robust_rows = _load_manifest_rows(out_robust)
    validate_baseline_manifest(baseline_rows)
    validate_robustness_manifest(robust_rows)

    v_report = {
        "pass": True,
        "baseline_rows": len(baseline_rows),
        "robustness_rows": len(robust_rows),
        "baseline_sha256": sha256_file(out_baseline),
        "robustness_sha256": sha256_file(out_robust),
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = run_dir / "validation" / "manifests_report.json"
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sentinel = {
        "stage": "manifests",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            str(out_baseline): sha256_file(out_baseline),
            str(out_robust): sha256_file(out_robust),
            str(out_baseline_meta): sha256_file(out_baseline_meta),
            str(out_robust_meta): sha256_file(out_robust_meta),
            str(out_baseline_report): sha256_file(out_baseline_report),
            str(out_robust_report): sha256_file(out_robust_report),
            str(v_path): sha256_file(v_path),
        },
    }
    s_path = run_dir / "sentinels" / "manifests.done"
    s_path.write_text(json.dumps(sentinel, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 2/3/4 - build+canonicalize+validate manifests",
        status="PASS",
        command=f"python3 sow.py build-manifests --run-id {args.run_id}",
        inputs=[
            HashedPath(path=str(in_baseline), sha256=sha256_file(in_baseline)),
            HashedPath(path=str(in_robust), sha256=sha256_file(in_robust)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            HashedPath(path=str(out_baseline), sha256=sha256_file(out_baseline)),
            HashedPath(path=str(out_robust), sha256=sha256_file(out_robust)),
            HashedPath(path=str(out_baseline_meta), sha256=sha256_file(out_baseline_meta)),
            HashedPath(path=str(out_robust_meta), sha256=sha256_file(out_robust_meta)),
            HashedPath(path=str(out_baseline_report), sha256=sha256_file(out_baseline_report)),
            HashedPath(path=str(out_robust_report), sha256=sha256_file(out_robust_report)),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Robustness v2: keep-last-line for duplicate (example_id, wrapper_id); drop out-of-wrapper-set; repair missing ascii_box for mmlu::test::12183; enforce suffix boundary.",
        next_step="Stage 6 - parser-regression",
    )
    print(str(v_path))
    return 0


def cmd_parser_regression(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    reg_path = REPO_ROOT / "artifacts" / "parser_edge_case_regression" / "regression_cases.json"
    cases = json.loads(reg_path.read_text(encoding="utf-8"))
    results = []
    passed = 0
    for case in cases:
        got = parse_choice(
            response_text=case["response_text"],
            first_token=case.get("first_token"),
            options=case["options"],
        )
        exp = case["expected"]
        ok = (got["parsed_choice"] == exp["choice"]) and (got["decision"] == exp["decision"])
        results.append(
            {
                "case_id": case["case_id"],
                "expected": exp,
                "got": {"parsed_choice": got["parsed_choice"], "decision": got["decision"]},
                "passed": ok,
            }
        )
        if ok:
            passed += 1

    report = {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass": passed == len(cases),
        "run_id": args.run_id,
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "results": results,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = run_dir / "validation" / "parser_regression_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    s_path = run_dir / "sentinels" / "parser_regression.done"
    s_path.write_text(
        json.dumps({"stage": "parser_regression", "output": str(out), "sha256": sha256_file(out)}, indent=2) + "\n",
        encoding="utf-8",
    )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 6 - parser regression suite",
        status="PASS" if report["pass"] else "FAIL",
        command=f"python3 sow.py parser-regression --run-id {args.run_id}",
        inputs=[HashedPath(path=str(reg_path), sha256=sha256_file(reg_path))],
        outputs=[
            HashedPath(path=str(out), sha256=sha256_file(out)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(out), sha256=sha256_file(out))],
        notes=None,
        next_step="Stage 10 - pca-membership" if report["pass"] else "Fix parser until regression passes",
    )

    print(str(out))
    return 0 if report["pass"] else 2


def cmd_token_buckets(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    out_dir = run_dir / "token_buckets"
    v_dir = run_dir / "validation"
    s_dir = run_dir / "sentinels"
    v_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    report = build_and_write_option_buckets_for_models(
        run_id=args.run_id,
        models=list(cfg["models"]),
        out_dir=out_dir,
    )
    report["run_config_path"] = str(cfg_path)
    report["run_config_sha256"] = sha256_file(cfg_path)
    report["config_snapshot_path"] = str(_config_snapshot_path(run_dir))
    report["config_snapshot_sha256"] = _config_snapshot_sha256(run_dir)

    v_path = v_dir / "token_buckets_report.json"
    v_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    s_path = s_dir / "token_buckets.done"
    s_path.write_text(
        json.dumps({"stage": "token_buckets", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n",
        encoding="utf-8",
    )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 5 - option token buckets (A/B/C/D) per model",
        status="PASS" if report.get("pass") else "FAIL",
        command=f"python3 sow.py token-buckets --run-id {args.run_id}",
        inputs=[HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path))],
        outputs=[
            *(HashedPath(path=f["path"], sha256=f["sha256"]) for f in report["files"]),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Built tokenizer-derived A/B/C/D token buckets with fixed variant templates; fail-fast if any bucket empty or overlaps detected.",
        next_step="Stage 7 - pilot inference (after confirming Stage 6 PASS already)" if report.get("pass") else "Fix bucket builder",
    )

    print(str(v_path))
    return 0 if report.get("pass") else 2


def cmd_pca_membership(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    baseline_manifest = run_dir / "manifests" / "baseline_manifest.jsonl"
    robustness_manifest = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
    if not baseline_manifest.exists() or not robustness_manifest.exists():
        raise SystemExit("manifests must be built before pca-membership")

    sample_obj = select_pca_membership(
        baseline_manifest=baseline_manifest,
        robustness_manifest=robustness_manifest,
        sample_size=1000,
        seed=int(cfg["random_seed"]),
    )

    # Determinism check: rerun and compare exact membership list.
    sample_obj2 = select_pca_membership(
        baseline_manifest=baseline_manifest,
        robustness_manifest=robustness_manifest,
        sample_size=1000,
        seed=int(cfg["random_seed"]),
    )
    same = sample_obj2["membership"] == sample_obj["membership"]

    out_paths: List[Path] = []
    for m in cfg["models"]:
        out_path = run_dir / "pca" / f"{m['name']}_sample_membership.json"
        write_membership_file(
            out_path=out_path,
            run_id=args.run_id,
            model_name=m["name"],
            model_id=m["model_id"],
            model_revision=m["revision"],
            baseline_manifest=baseline_manifest,
            robustness_manifest=robustness_manifest,
            membership_obj=sample_obj,
        )
        out_paths.append(out_path)

    report = {
        "pass": bool(same),
        "seed": int(cfg["random_seed"]),
        "sample_size": 1000,
        "deterministic_repeat_match": bool(same),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "robustness_manifest_sha256": sha256_file(robustness_manifest),
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "out_files": [str(p) for p in out_paths],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = run_dir / "validation" / "pca_membership_report.json"
    v_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    s_path = run_dir / "sentinels" / "pca_membership.done"
    s_path.write_text(
        json.dumps({"stage": "pca_membership", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n",
        encoding="utf-8",
    )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 10 - freeze PCA sample membership",
        status="PASS" if report["pass"] else "FAIL",
        command=f"python3 sow.py pca-membership --run-id {args.run_id}",
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            HashedPath(path=str(robustness_manifest), sha256=sha256_file(robustness_manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            *(HashedPath(path=str(p), sha256=sha256_file(p)) for p in out_paths),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Membership is stratified uniformly over (wrapper_id, coarse_domain) strata and is deterministic for the frozen seed.",
        next_step="Milestone 1 complete (no PCA fit / no inference yet)" if report["pass"] else "Fix determinism bug",
    )
    print(str(v_path))
    return 0 if report["pass"] else 2


def cmd_pca_sample_inference(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    # Dependencies
    for s in ["stage0.done", "manifests.done", "pca_membership.done"]:
        p = run_dir / "sentinels" / s
        if not p.exists():
            raise SystemExit(f"missing dependency sentinel: {p}")

    baseline_manifest = run_dir / "manifests" / "baseline_manifest.jsonl"
    robustness_manifest = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
    if not baseline_manifest.exists() or not robustness_manifest.exists():
        raise SystemExit("manifests must be built before pca-sample-inference")

    # Models to run.
    model_names = [args.model_name] if args.model_name else [m["name"] for m in cfg["models"]]
    models = [m for m in cfg["models"] if m["name"] in set(model_names)]
    if len(models) != len(model_names):
        known = sorted([m["name"] for m in cfg["models"]])
        raise SystemExit(f"unknown model_name in {model_names}; known={known}")

    per_model = []
    ok_all = True

    for m in models:
        model_key = model_fs_id(m["model_id"])
        sentinel = run_dir / "sentinels" / f"pca_sample_inference.{model_key}.done"
        if sentinel.exists():
            per_model.append(
                {
                    "model_name": m["name"],
                    "model_id": m["model_id"],
                    "model_revision": m["revision"],
                    "skipped": True,
                    "skip_reason": f"sentinel exists: {sentinel}",
                }
            )
            continue

        membership_path = run_dir / "pca" / f"{m['name']}_sample_membership.json"
        if not membership_path.exists():
            raise SystemExit(f"missing membership file for model {m['name']}: {membership_path}")

        res = run_pca_sample_inference_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            baseline_manifest=baseline_manifest,
            robustness_manifest=robustness_manifest,
            membership_path=membership_path,
            device_override=args.device,
            batch_size=str(args.batch_size),
            repro_check_k=int(args.repro_check_k),
            repro_atol=float(args.repro_atol),
            thermal_hygiene_cfg=cfg.get("thermal_hygiene"),
        )

        per_model_report = _next_available_path(run_dir / "validation" / f"pca_sample_inference_report.{model_key}.json")
        per_model_report.write_text(json.dumps(res["report"], indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

        if res["report"]["pass"]:
            sent = {
                "stage": "pca_sample_inference",
                "run_id": args.run_id,
                "model_id": m["model_id"],
                "model_revision": m["revision"],
                "hidden_path": str(res["hidden_path"]),
                "hidden_sha256": res["hidden_sha256"],
                "meta_path": str(res["meta_path"]),
                "meta_sha256": res["meta_sha256"],
                "report_path": str(per_model_report),
                "report_sha256": sha256_file(per_model_report),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            tmp = sentinel.with_name(f".{sentinel.name}.tmp")
            if tmp.exists():
                raise SystemExit(f"refusing to overwrite existing tmp sentinel: {tmp}")
            tmp.write_text(json.dumps(sent, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(sentinel)

        per_model.append({"model_name": m["name"], **res["report"], "hidden_path": str(res["hidden_path"]), "meta_path": str(res["meta_path"])})
        if not res["report"]["pass"]:
            ok_all = False

    v_report = {
        "pass": bool(ok_all),
        "run_id": args.run_id,
        "models": per_model,
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = _next_available_path(run_dir / "validation" / "pca_sample_inference_report.json")
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    stage_sentinel = run_dir / "sentinels" / "pca_sample_inference.done"
    if stage_sentinel.exists():
        # Append-only stage report; do not overwrite stage-level sentinel if it already exists.
        pass
    else:
        if ok_all:
            stage_sentinel.write_text(
                json.dumps({"stage": "pca_sample_inference", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n",
                encoding="utf-8",
            )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 11 - PCA sample extraction inference",
        status="PASS" if v_report["pass"] else "FAIL",
        command="python3 sow.py pca-sample-inference --run-id "
        + args.run_id
        + (f" --model-name {args.model_name}" if args.model_name else "")
        + (f" --device {args.device}" if args.device else "")
        + (f" --batch-size {args.batch_size}" if args.batch_size else "")
        + (f" --repro-check-k {args.repro_check_k}" if args.repro_check_k else "")
        + (f" --repro-atol {args.repro_atol}" if args.repro_atol else ""),
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            HashedPath(path=str(robustness_manifest), sha256=sha256_file(robustness_manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            *(HashedPath(path=m["hidden_path"], sha256=m["hidden_sha256"]) for m in per_model if "hidden_path" in m and "hidden_sha256" in m),
            *(HashedPath(path=m["meta_path"], sha256=m["meta_sha256"]) for m in per_model if "meta_path" in m and "meta_sha256" in m),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Extracted last-position hidden vectors for every transformer layer on the frozen PCA membership set. Includes a small reproducibility spot-check.",
        next_step="Stage 12 - pca-fit" if v_report["pass"] else "Fix PCA extraction until reproducibility/shape checks pass",
    )

    print(str(v_path))
    return 0 if v_report["pass"] else 2


def cmd_pca_fit(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    # Dependencies
    for s in ["pca_sample_inference.done"]:
        p = run_dir / "sentinels" / s
        if not p.exists():
            raise SystemExit(f"missing dependency sentinel: {p}")

    n_components = int(cfg["pca"]["n_components"])
    seed = int(cfg["random_seed"])

    model_names = [args.model_name] if args.model_name else [m["name"] for m in cfg["models"]]
    models = [m for m in cfg["models"] if m["name"] in set(model_names)]
    if len(models) != len(model_names):
        known = sorted([m["name"] for m in cfg["models"]])
        raise SystemExit(f"unknown model_name in {model_names}; known={known}")

    per_model = []
    ok_all = True

    for m in models:
        model_key = model_fs_id(m["model_id"])
        s_in = run_dir / "sentinels" / f"pca_sample_inference.{model_key}.done"
        if not s_in.exists():
            raise SystemExit(f"missing per-model Stage 11 sentinel: {s_in}")
        s_in_obj = json.loads(s_in.read_text(encoding="utf-8"))
        hidden_path = Path(s_in_obj["hidden_path"])
        meta_path = Path(s_in_obj["meta_path"])

        s_out = run_dir / "sentinels" / f"pca_fit.{model_key}.done"
        if s_out.exists():
            per_model.append(
                {
                    "model_name": m["name"],
                    "model_id": m["model_id"],
                    "model_revision": m["revision"],
                    "skipped": True,
                    "skip_reason": f"sentinel exists: {s_out}",
                }
            )
            continue

        res = run_stage12_fit_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            hidden_npz=hidden_path,
            hidden_meta=meta_path,
            n_components=n_components,
            seed=seed,
        )

        per_model_report = _next_available_path(run_dir / "validation" / f"pca_fit_report.{model_key}.json")
        per_model_report.write_text(json.dumps(res["report"], indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

        sent = {
            "stage": "pca_fit",
            "run_id": args.run_id,
            "model_id": m["model_id"],
            "model_revision": m["revision"],
            "basis_path": str(res["basis_path"]),
            "basis_sha256": res["basis_sha256"],
            "meta_path": str(res["meta_path"]),
            "meta_sha256": res["meta_sha256"],
            "report_path": str(per_model_report),
            "report_sha256": sha256_file(per_model_report),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = s_out.with_name(f".{s_out.name}.tmp")
        if tmp.exists():
            raise SystemExit(f"refusing to overwrite existing tmp sentinel: {tmp}")
        tmp.write_text(json.dumps(sent, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(s_out)

        per_model.append({"model_name": m["name"], **res["report"], "basis_path": str(res["basis_path"]), "meta_path": str(res["meta_path"])})
        if not res["report"]["pass"]:
            ok_all = False

    v_report = {
        "pass": bool(ok_all),
        "run_id": args.run_id,
        "models": per_model,
        "n_components": int(n_components),
        "seed": int(seed),
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = _next_available_path(run_dir / "validation" / "pca_fit_report.json")
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    stage_sentinel = run_dir / "sentinels" / "pca_fit.done"
    if not stage_sentinel.exists():
        stage_sentinel.write_text(
            json.dumps({"stage": "pca_fit", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n",
            encoding="utf-8",
        )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 12 - fit PCA basis per model (pooled across layers)",
        status="PASS" if v_report["pass"] else "FAIL",
        command="python3 sow.py pca-fit --run-id " + args.run_id + (f" --model-name {args.model_name}" if args.model_name else ""),
        inputs=[HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path))],
        outputs=[
            *(HashedPath(path=m["basis_path"], sha256=m["basis_sha256"]) for m in per_model if "basis_path" in m and "basis_sha256" in m),
            *(HashedPath(path=m["meta_path"], sha256=m["meta_sha256"]) for m in per_model if "meta_path" in m and "meta_sha256" in m),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Fit one PCA basis per model from pooled layer vectors and canonicalize component signs; includes an in-process reproducibility test (fit twice).",
        next_step="Stage 13 - full inference w/ on-the-fly PCA projection" if v_report["pass"] else "Fix PCA basis determinism until hash matches",
    )

    print(str(v_path))
    return 0 if v_report["pass"] else 2


def cmd_pilot_inference(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    baseline_manifest = run_dir / "manifests" / "baseline_manifest.jsonl"
    if not baseline_manifest.exists():
        raise SystemExit("baseline manifest must be built before pilot inference")

    # Deterministic prompt sample (stratified by coarse_domain).
    sample_rows = select_pilot_rows(
        baseline_manifest_path=baseline_manifest,
        sample_size=int(args.sample_size),
        seed=int(cfg["random_seed"]),
    )

    # Pilot-gate thresholds (required for PASS status).
    min_comp = float(args.min_one_token_compliance) if args.min_one_token_compliance is not None else None
    min_res = float(args.min_parser_resolved) if args.min_parser_resolved is not None else None

    # Models to run.
    model_names = [args.model_name] if args.model_name else [m["name"] for m in cfg["models"]]
    models = [m for m in cfg["models"] if m["name"] in set(model_names)]
    if len(models) != len(model_names):
        known = sorted([m["name"] for m in cfg["models"]])
        raise SystemExit(f"unknown model_name in {model_names}; known={known}")

    per_model = []
    ok_all = True
    gate_reasons_all = []
    for m in models:
        model_key = model_fs_id(m["model_id"])
        model_sentinel = run_dir / "sentinels" / f"pilot.{model_key}.done"
        if model_sentinel.exists():
            # Append-only: don't rerun a model's pilot unless user explicitly forces (not implemented yet).
            per_model.append(
                {
                    "model_name": m["name"],
                    "model_id": m["model_id"],
                    "model_revision": m["revision"],
                    "skipped": True,
                    "skip_reason": f"sentinel exists: {model_sentinel}",
                }
            )
            continue

        res = run_pilot_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            sample_rows=sample_rows,
            device_override=args.device,
            thermal_hygiene=cfg.get("thermal_hygiene"),
        )
        gate_ok, reasons = stage7_gate(
            overall_metrics=res["metrics_overall"],
            min_one_token_compliance_rate=min_comp,
            min_parser_resolved_rate=min_res,
        )
        per_model.append(
            {
                "model_name": m["name"],
                "model_id": m["model_id"],
                "model_revision": m["revision"],
                "outputs_path": str(res["outputs_path"]),
                "outputs_sha256": res["outputs_sha256"],
                "report_path": str(res["report_path"]),
                "report_sha256": res["report_sha256"],
                "metrics_overall": res["metrics_overall"],
                "gate": {
                    "pass": bool(gate_ok),
                    "min_one_token_compliance_rate": min_comp,
                    "min_parser_resolved_rate": min_res,
                    "reasons": reasons,
                },
            }
        )
        if not gate_ok:
            ok_all = False
            gate_reasons_all.extend([f"{m['name']}: {r}" for r in reasons])

    # Stage-level report is append-only; never overwrite.
    v_report = {
        "pass": bool(ok_all),
        "run_id": args.run_id,
        "sample_size": int(args.sample_size),
        "seed": int(cfg["random_seed"]),
        "gate": {
            "min_one_token_compliance_rate": min_comp,
            "min_parser_resolved_rate": min_res,
            "fail_reasons": gate_reasons_all,
        },
        "models": per_model,
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = _next_available_path(run_dir / "validation" / "pilot_report.json")
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    s_path = _next_available_path(run_dir / "sentinels" / "pilot.done")
    s_path.write_text(json.dumps({"stage": "pilot", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n")

    # Per-model sentinels for gating other stages.
    for entry in per_model:
        if entry.get("skipped"):
            continue
        model_key = model_fs_id(entry["model_id"])
        per_model_report = _next_available_path(run_dir / "validation" / f"pilot_report.{model_key}.json")
        per_model_report.write_text(json.dumps(entry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        per_model_sentinel = _next_available_path(run_dir / "sentinels" / f"pilot.{model_key}.done")
        per_model_sentinel.write_text(
            json.dumps({"stage": "pilot", "model_id": entry["model_id"], "output": str(per_model_report), "sha256": sha256_file(per_model_report)}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 7 - pilot inference (one-token compliance + viability)",
        status="PASS" if v_report["pass"] else "FAIL",
        command="python3 sow.py pilot-inference --run-id "
        + args.run_id
        + (f" --model-name {args.model_name}" if args.model_name else "")
        + (f" --sample-size {args.sample_size}" if args.sample_size else "")
        + (f" --min-one-token-compliance {args.min_one_token_compliance}" if args.min_one_token_compliance is not None else "")
        + (f" --min-parser-resolved {args.min_parser_resolved}" if args.min_parser_resolved is not None else "")
        + (f" --device {args.device}" if args.device else ""),
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            *(HashedPath(path=m["outputs_path"], sha256=m["outputs_sha256"]) for m in per_model if "outputs_path" in m),
            *(HashedPath(path=m["report_path"], sha256=m["report_sha256"]) for m in per_model if "report_path" in m),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            HashedPath(path=str(s_path), sha256=sha256_file(s_path)),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Pilot measures first-token one-token compliance and deterministic parser resolution/accuracy on a stratified sample.",
        next_step="Stage 8 - build PCC" if v_report["pass"] else "Fix prompting / token buckets / sampling before proceeding",
    )

    print(str(v_path))
    return 0 if v_report["pass"] else 2


def _require_sentinels(*, run_dir: Path, names: List[str]) -> None:
    for s in names:
        p = run_dir / "sentinels" / s
        if not p.exists():
            raise SystemExit(f"missing dependency sentinel: {p}")


def _write_jsonl_atomic_new(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def cmd_stage13_smoke(args: argparse.Namespace) -> int:
    """
    GPU-friendly smoke for Stage 13: run 20 prompts and enforce gates.

    This is intentionally tiny so we can validate CUDA/MPS correctness before scaling to 60k.
    """
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    # Dependencies: at minimum we need token buckets + PCA basis.
    _require_sentinels(
        run_dir=run_dir,
        names=[
            "stage0.done",
            "manifests.done",
            "token_buckets.done",
            "parser_regression.done",
            "ccc.done",
            "pca_fit.done",
        ],
    )

    ccc_baseline = run_dir / "manifests" / "ccc_baseline.jsonl"
    ccc_robust = run_dir / "manifests" / "ccc_robustness.jsonl"
    if not ccc_baseline.exists():
        raise SystemExit("CCC baseline manifest missing; run Stage 9 first")
    if (not bool(args.skip_robustness)) and (not ccc_robust.exists()):
        raise SystemExit("CCC robustness manifest missing; run Stage 9 first")

    # Smoke row selection.
    baseline_rows = select_pilot_rows(
        baseline_manifest_path=ccc_baseline,
        sample_size=int(args.sample_size),
        seed=int(cfg["random_seed"]),
    )
    # Robustness smoke: choose 1 example_id and take all 20 wrappers.
    robustness_rows: List[Dict[str, Any]] = []
    if not bool(args.skip_robustness):
        expected_wrappers = list(cfg["prompting"]["robustness_wrapper_ids_v2"])
        rob_rows_all = list(iter_jsonl(ccc_robust))
        by_ex = {}
        for r in rob_rows_all:
            by_ex.setdefault(str(r["example_id"]), {})[str(r["wrapper_id"])] = r
        ex = sorted(by_ex.keys())[0]
        robustness_rows = [by_ex[ex][wid] for wid in expected_wrappers]

    inf_cfg = cfg.get("inference") or {}
    thresholds = inf_cfg.get("commitment_margin_thresholds") or [0.05, 0.1, 0.2]
    atol = float(inf_cfg.get("batch_consistency_atol_candidate_logits") or 1e-3)

    model_names = [args.model_name] if args.model_name else [m["name"] for m in cfg["models"]]
    models = [m for m in cfg["models"] if m["name"] in set(model_names)]
    if len(models) != len(model_names):
        known = sorted([m["name"] for m in cfg["models"]])
        raise SystemExit(f"unknown model_name in {model_names}; known={known}")

    v_dir = run_dir / "validation"
    v_dir.mkdir(parents=True, exist_ok=True)

    per_model = []
    ok_all = True
    for m in models:
        mk = model_fs_id(m["model_id"])

        # Gate 1: batch consistency on baseline sample (bs=1 vs bs=4).
        gate = batch_consistency_gate(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            manifest_path=ccc_baseline,
            condition="baseline",
            rows=baseline_rows,
            device_override=args.device,
            atol_logits=atol,
        )

        # Gate 2: resume simulation (stop-after then resume) on baseline + robustness.
        base_manifest_smoke = _next_available_path(v_dir / f"stage13_smoke_manifest.baseline.{mk}.jsonl")
        _write_jsonl_atomic_new(base_manifest_smoke, baseline_rows)
        base_out = _next_available_path(v_dir / f"stage13_smoke_outputs.baseline.{mk}.jsonl")
        r_stop = run_stage13_inference_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            manifest_path=base_manifest_smoke,
            condition="baseline",
            batch_size=args.batch_size,
            device_override=args.device,
            output_path_override=base_out,
            stop_after_rows=min(7, int(args.sample_size)),
            limit_rows=None,
            commitment_margin_thresholds=[float(x) for x in thresholds],
            thermal_hygiene_cfg=None,
        )
        r_resume = run_stage13_inference_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            manifest_path=base_manifest_smoke,
            condition="baseline",
            batch_size=args.batch_size,
            device_override=args.device,
            output_path_override=base_out,
            stop_after_rows=None,
            limit_rows=None,
            commitment_margin_thresholds=[float(x) for x in thresholds],
            thermal_hygiene_cfg=None,
        )

        if bool(args.skip_robustness):
            rob_res = {"skipped": True}
        else:
            rob_manifest_smoke = _next_available_path(v_dir / f"stage13_smoke_manifest.robustness.{mk}.jsonl")
            _write_jsonl_atomic_new(rob_manifest_smoke, robustness_rows)
            rob_out = _next_available_path(v_dir / f"stage13_smoke_outputs.robustness.{mk}.jsonl")
            rob_res = run_stage13_inference_for_model(
                run_id=args.run_id,
                run_dir=run_dir,
                model=m,
                generation=cfg["generation"],
                manifest_path=rob_manifest_smoke,
                condition="robustness",
                batch_size=args.batch_size,
                device_override=args.device,
                output_path_override=rob_out,
                stop_after_rows=None,
                limit_rows=None,
                commitment_margin_thresholds=[float(x) for x in thresholds],
                thermal_hygiene_cfg=None,
            )

        ok = bool(gate["pass"] and r_resume.get("pass") and (bool(rob_res.get("pass")) if not bool(args.skip_robustness) else True))
        if not ok:
            ok_all = False
        per_model.append(
            {
                "model_name": m["name"],
                "model_id": m["model_id"],
                "model_revision": m["revision"],
                "batch_consistency_gate": gate,
                "resume_simulation": {"baseline_stop": r_stop, "baseline_resume": r_resume},
                "robustness_smoke": rob_res,
            }
        )

    report = {
        "pass": bool(ok_all),
        "run_id": args.run_id,
        "sample_size": int(args.sample_size),
        "models": per_model,
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = _next_available_path(v_dir / "stage13_smoke_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sent = None
    if report["pass"]:
        sent = _next_available_path(run_dir / "sentinels" / "stage13_smoke.done")
        sent.write_text(json.dumps({"stage": "stage13_smoke", "output": str(out), "sha256": sha256_file(out)}, indent=2) + "\n", encoding="utf-8")

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 13 - smoke (20 prompts, gates: batch-consistency + resume)",
        status="PASS" if report["pass"] else "FAIL",
        command="python3 sow.py stage13-smoke --run-id "
        + args.run_id
        + (f" --model-name {args.model_name}" if args.model_name else "")
        + (f" --sample-size {args.sample_size}" if args.sample_size else "")
        + (f" --batch-size {args.batch_size}" if args.batch_size else "")
        + (f" --device {args.device}" if args.device else ""),
        inputs=[
            HashedPath(path=str(ccc_baseline), sha256=sha256_file(ccc_baseline)),
            *(
                [HashedPath(path=str(ccc_robust), sha256=sha256_file(ccc_robust))]
                if (ccc_robust.exists() and (not bool(args.skip_robustness)))
                else []
            ),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            HashedPath(path=str(out), sha256=sha256_file(out)),
            *( [HashedPath(path=str(sent), sha256=sha256_file(sent))] if sent else [] ),
        ],
        validators=[HashedPath(path=str(out), sha256=sha256_file(out))],
        notes="Intended for GPU VMs: tiny run to validate Stage 13 correctness gates before scaling to 60k."
        + (" Robustness smoke skipped (baseline-only mode)." if bool(args.skip_robustness) else ""),
        next_step="Stage 13a - baseline inference" if report["pass"] else "Fix Stage 13 gates before full runs",
    )

    print(str(out))
    return 0 if report["pass"] else 2


def _cmd_stage13_condition(args: argparse.Namespace, *, condition: str) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    _require_sentinels(
        run_dir=run_dir,
        names=[
            "stage0.done",
            "manifests.done",
            "token_buckets.done",
            "parser_regression.done",
            "pilot.done",
            "pcc.done",
            "ccc.done",
            "pca_fit.done",
        ],
    )

    manifest = run_dir / "manifests" / ("ccc_baseline.jsonl" if condition == "baseline" else "ccc_robustness.jsonl")
    if not manifest.exists():
        raise SystemExit(f"missing CCC manifest for condition={condition}: {manifest}")

    inf_cfg = cfg.get("inference") or {}
    thresholds = inf_cfg.get("commitment_margin_thresholds") or [0.05, 0.1, 0.2]

    model_names = [args.model_name] if args.model_name else [m["name"] for m in cfg["models"]]
    models = [m for m in cfg["models"] if m["name"] in set(model_names)]
    if len(models) != len(model_names):
        known = sorted([m["name"] for m in cfg["models"]])
        raise SystemExit(f"unknown model_name in {model_names}; known={known}")

    per_model = []
    ok_all = True
    for m in models:
        res = run_stage13_inference_for_model(
            run_id=args.run_id,
            run_dir=run_dir,
            model=m,
            generation=cfg["generation"],
            manifest_path=manifest,
            condition=condition,
            batch_size=args.batch_size,
            device_override=args.device,
            output_path_override=None,
            stop_after_rows=None,
            limit_rows=None,
            commitment_margin_thresholds=[float(x) for x in thresholds],
            thermal_hygiene_cfg=cfg.get("thermal_hygiene"),
        )
        per_model.append(
            {
                "model_name": m["name"],
                "model_id": m["model_id"],
                "model_revision": m["revision"],
                **res,
            }
        )
        if not res.get("pass"):
            ok_all = False

    v_report = {
        "pass": bool(ok_all),
        "run_id": args.run_id,
        "condition": condition,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "models": per_model,
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    v_path = _next_available_path(run_dir / "validation" / f"inference_{condition}_report.json")
    v_path.write_text(json.dumps(v_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Per-model sentinels + stage-level sentinel.
    sent_dir = run_dir / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    sent_paths: List[Path] = []
    for entry in per_model:
        mk = model_fs_id(entry["model_id"])
        if not entry.get("pass"):
            continue
        s = _next_available_path(sent_dir / f"inference_{condition}.{mk}.done")
        s.write_text(
            json.dumps(
                {
                    "stage": f"inference_{condition}",
                    "run_id": args.run_id,
                    "model_id": entry["model_id"],
                    "model_revision": entry["model_revision"],
                    "output_path": entry["output_path"],
                    "output_sha256": entry["output_sha256"],
                    "report_path": str(v_path),
                    "report_sha256": sha256_file(v_path),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sent_paths.append(s)

    stage_sent = None
    if ok_all:
        stage_sent = _next_available_path(sent_dir / f"inference_{condition}.done")
        stage_sent.write_text(json.dumps({"stage": f"inference_{condition}", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n")
        sent_paths.append(stage_sent)

    append_state_entry(
        state_path=_state_path(),
        stage=f"Stage 13 - inference ({condition})",
        status="PASS" if ok_all else "FAIL",
        command=(
            f"python3 sow.py inference-{condition} --run-id {args.run_id}"
            + (f" --model-name {args.model_name}" if args.model_name else "")
            + (f" --batch-size {args.batch_size}" if args.batch_size else "")
            + (f" --device {args.device}" if args.device else "")
        ),
        inputs=[
            HashedPath(path=str(manifest), sha256=sha256_file(manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            *(HashedPath(path=str(Path(e["output_path"])), sha256=str(e["output_sha256"])) for e in per_model if e.get("output_path") and e.get("output_sha256")),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            *(HashedPath(path=str(p), sha256=sha256_file(p)) for p in sent_paths),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Full inference outputs are append-only JSONL; this stage is resumable and must pass strict validation before analysis.",
        next_step="Stage 13b - robustness inference" if condition == "baseline" and ok_all else ("Stage 14 - analysis" if ok_all else "Fix inference failures"),
    )

    print(str(v_path))
    return 0 if ok_all else 2


def cmd_inference_baseline(args: argparse.Namespace) -> int:
    return _cmd_stage13_condition(args, condition="baseline")


def cmd_inference_robustness(args: argparse.Namespace) -> int:
    return _cmd_stage13_condition(args, condition="robustness")


def cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    dep_sentinels = [
        "parser_regression.done",
        "ccc.done",
        "pca_fit.done",
        "inference_baseline.done",
    ]
    if not bool(args.skip_robustness):
        dep_sentinels.append("inference_robustness.done")
    _require_sentinels(run_dir=run_dir, names=dep_sentinels)

    baseline_manifest = run_dir / "manifests" / "ccc_baseline.jsonl"
    robustness_manifest = run_dir / "manifests" / "ccc_robustness.jsonl"
    if not baseline_manifest.exists():
        raise SystemExit("CCC baseline manifest missing; cannot analyze")
    if (not bool(args.skip_robustness)) and (not robustness_manifest.exists()):
        raise SystemExit("CCC robustness manifest missing; cannot analyze")

    inf_cfg = cfg.get("inference") or {}
    thresholds = [float(x) for x in (inf_cfg.get("commitment_margin_thresholds") or [0.05, 0.1, 0.2])]
    base_wid = str((cfg.get("prompting") or {}).get("baseline_wrapper_id") or "plain_exam")

    report = run_stage14_analysis(
        run_id=args.run_id,
        run_dir=run_dir,
        cfg=cfg,
        baseline_manifest=baseline_manifest,
        robustness_manifest=(robustness_manifest if not bool(args.skip_robustness) else None),
        baseline_wrapper_id=base_wid,
        thresholds=thresholds,
        include_robustness=(not bool(args.skip_robustness)),
        topology_layer=int(args.topology_layer),
    )

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = _next_available_path(analysis_dir / "final_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Stage-level validator report in validation/.
    v_dir = run_dir / "validation"
    v_dir.mkdir(parents=True, exist_ok=True)
    v = {
        "pass": bool(report.get("pass")),
        "run_id": args.run_id,
        "final_report_path": str(out),
        "final_report_sha256": sha256_file(out),
        "artifacts": report.get("artifacts"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    v_path = _next_available_path(v_dir / "analysis_report.json")
    v_path.write_text(json.dumps(v, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sent = None
    if v["pass"]:
        sent = _next_available_path(run_dir / "sentinels" / "analysis.done")
        sent.write_text(json.dumps({"stage": "analysis", "output": str(v_path), "sha256": sha256_file(v_path)}, indent=2) + "\n", encoding="utf-8")

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 14 - analysis and reports",
        status="PASS" if v["pass"] else "FAIL",
        command=f"python3 sow.py analyze --run-id {args.run_id}"
        + (" --skip-robustness" if bool(args.skip_robustness) else "")
        + (f" --topology-layer {args.topology_layer}" if int(args.topology_layer) != -1 else ""),
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            *(
                [HashedPath(path=str(robustness_manifest), sha256=sha256_file(robustness_manifest))]
                if (not bool(args.skip_robustness))
                else []
            ),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            HashedPath(path=str(out), sha256=sha256_file(out)),
            HashedPath(path=str(v_path), sha256=sha256_file(v_path)),
            *( [HashedPath(path=str(sent), sha256=sha256_file(sent))] if sent else [] ),
        ],
        validators=[HashedPath(path=str(v_path), sha256=sha256_file(v_path))],
        notes="Deterministic JSON/CSV analysis artifacts generated from Stage 13 outputs (no judge/adjudication applied here)."
        + (" Baseline-only mode: robustness analysis skipped." if bool(args.skip_robustness) else ""),
        next_step="(Optional) DeepSeek adjudication for unresolved-only, then re-run analysis; else proceed to paper writeup.",
    )

    print(str(out))
    return 0 if v["pass"] else 2


def _pilot_status_for_model(*, run_dir: Path, model_id: str) -> Dict[str, Any]:
    """
    Returns best-known pilot gate status for a model from run artifacts.
    Prefers per-model pilot reports if present; otherwise falls back to stage-level pilot reports.
    """
    model_key = model_fs_id(model_id)
    v_dir = run_dir / "validation"

    def attempt_idx(p: Path) -> int:
        name = p.name
        if ".attempt" not in name:
            return 0
        try:
            frag = name.split(".attempt", 1)[1]
            n = frag.split(".", 1)[0]
            return int(n)
        except Exception:
            return 0

    # 1) Per-model reports: pilot_report.<model_key>[.attemptN].json
    per_model = sorted(v_dir.glob(f"pilot_report.{model_key}*.json"), key=lambda p: (attempt_idx(p), p.name))
    if per_model:
        best = per_model[-1]
        obj = json.loads(best.read_text(encoding="utf-8"))
        gate = obj.get("gate") or {}
        return {
            "source": "per_model",
            "pass": bool(gate.get("pass")),
            "report_path": str(best),
            "report_sha256": sha256_file(best),
            "gate": gate,
        }

    # 2) Stage-level reports: pilot_report[.attemptN].json
    stage_reports = []
    for p in v_dir.glob("pilot_report*.json"):
        if p.name == "pilot_report.json" or p.name.startswith("pilot_report.attempt"):
            stage_reports.append(p)
    stage_reports = sorted(stage_reports, key=lambda p: (attempt_idx(p), p.name))
    for p in reversed(stage_reports):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for entry in obj.get("models", []) or []:
            if entry.get("model_id") == model_id:
                gate = entry.get("gate") or {}
                return {
                    "source": "stage_level",
                    "pass": bool(gate.get("pass")),
                    "report_path": str(p),
                    "report_sha256": sha256_file(p),
                    "gate": gate,
                    "metrics_overall": entry.get("metrics_overall"),
                }

    return {"source": "missing", "pass": False, "error": f"no pilot report found for model_id={model_id}"}


def cmd_build_pcc(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    # Dependencies (stage gates).
    for s in ["stage0.done", "manifests.done", "token_buckets.done", "parser_regression.done"]:
        p = run_dir / "sentinels" / s
        if not p.exists():
            raise SystemExit(f"missing dependency sentinel: {p}")

    baseline_manifest = run_dir / "manifests" / "baseline_manifest.jsonl"
    robustness_manifest = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
    if not baseline_manifest.exists() or not robustness_manifest.exists():
        raise SystemExit("manifests must be built before PCC")

    expected_wrappers = list(cfg["prompting"]["robustness_wrapper_ids_v2"])
    seed = int(cfg["random_seed"])
    target_size = int(args.target_size)
    max_new = int(cfg["generation"]["max_new_tokens"])

    # Require pilot gate PASS for every model we intend to run.
    pilot_info = {}
    missing = []
    for m in cfg["models"]:
        st = _pilot_status_for_model(run_dir=run_dir, model_id=m["model_id"])
        pilot_info[m["name"]] = st
        if not st.get("pass"):
            missing.append(f"{m['name']}: {st.get('error') or st.get('gate')}")
    if missing:
        raise SystemExit("pilot gate missing/failed for models: " + "; ".join(missing))

    out_dir = run_dir / "manifests"
    per_model = []
    for m in cfg["models"]:
        res = build_pcc_manifests_for_model(
            run_id=args.run_id,
            model_name=m["name"],
            model_id=m["model_id"],
            model_revision=m["revision"],
            baseline_manifest_path=baseline_manifest,
            robustness_manifest_path=robustness_manifest,
            expected_wrapper_ids=expected_wrappers,
            generation_max_new_tokens=max_new,
            target_size=target_size,
            seed=seed,
            out_dir=out_dir,
        )
        per_model.append(
            {
                "model_name": m["name"],
                "model_id": m["model_id"],
                "model_revision": m["revision"],
                "pilot": pilot_info.get(m["name"]),
                **res,
            }
        )

    report = {
        "pass": True,
        "run_id": args.run_id,
        "target_size": target_size,
        "seed": seed,
        "expected_wrapper_ids": expected_wrappers,
        "baseline_manifest_path": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "robustness_manifest_path": str(robustness_manifest),
        "robustness_manifest_sha256": sha256_file(robustness_manifest),
        "models": per_model,
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = _next_available_path(out_dir / "pcc_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sent_dir = run_dir / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    sent_path = _next_available_path(sent_dir / "pcc.done")
    sent_path.write_text(
        json.dumps({"stage": "pcc", "output": str(report_path), "sha256": sha256_file(report_path)}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Per-model sentinels.
    for entry in per_model:
        mk = model_fs_id(entry["model_id"])
        per_model_report = _next_available_path(out_dir / f"pcc_report.{mk}.json")
        per_model_report.write_text(json.dumps(entry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        per_model_sent = _next_available_path(sent_dir / f"pcc.{mk}.done")
        per_model_sent.write_text(
            json.dumps({"stage": "pcc", "model_id": entry["model_id"], "output": str(per_model_report), "sha256": sha256_file(per_model_report)}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 8 - build Primary Core Corpus (PCC)",
        status="PASS",
        command=f"python3 sow.py build-pcc --run-id {args.run_id} --target-size {target_size}",
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            HashedPath(path=str(robustness_manifest), sha256=sha256_file(robustness_manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            *(HashedPath(path=e["pcc_baseline_path"], sha256=e["pcc_baseline_sha256"]) for e in per_model),
            *(HashedPath(path=e["pcc_robustness_path"], sha256=e["pcc_robustness_sha256"]) for e in per_model),
            *(HashedPath(path=e["pcc_meta_path"], sha256=e["pcc_meta_sha256"]) for e in per_model),
            HashedPath(path=str(report_path), sha256=sha256_file(report_path)),
            HashedPath(path=str(sent_path), sha256=sha256_file(sent_path)),
        ],
        validators=[HashedPath(path=str(report_path), sha256=sha256_file(report_path))],
        notes="PCC filters: prompt-length safety (per-model context length), wrapper completeness (20/20), and pilot-gate PASS prerequisite.",
        next_step="Stage 9 - build CCC",
    )

    print(str(report_path))
    return 0


def cmd_build_ccc(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    validate_run_config(cfg)

    # Dependencies: manifests + PCC.
    for s in ["manifests.done", "pcc.done"]:
        p = run_dir / "sentinels" / s
        if not p.exists():
            raise SystemExit(f"missing dependency sentinel: {p}")

    baseline_manifest = run_dir / "manifests" / "baseline_manifest.jsonl"
    robustness_manifest = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
    if not baseline_manifest.exists() or not robustness_manifest.exists():
        raise SystemExit("canonical manifests must exist before CCC")

    expected_wrappers = list(cfg["prompting"]["robustness_wrapper_ids_v2"])

    # Load per-model PCC sets.
    per_model_example_ids: Dict[str, Set[str]] = {}
    per_model_counts_by_domain: Dict[str, Dict[str, int]] = {}
    domain_by_example: Dict[str, str] = {}

    # Domain mapping comes from canonical baseline (stable and common).
    for r in iter_jsonl(baseline_manifest):
        domain_by_example[str(r["example_id"])] = str(r.get("coarse_domain") or "unknown")

    for m in cfg["models"]:
        mk = model_fs_id(m["model_id"])
        pcc_sentinel = run_dir / "sentinels" / f"pcc.{mk}.done"
        if not pcc_sentinel.exists():
            raise SystemExit(f"missing per-model PCC sentinel: {pcc_sentinel}")
        s_obj = json.loads(pcc_sentinel.read_text(encoding="utf-8"))
        pcc_report_path = Path(str(s_obj.get("output") or ""))
        if not pcc_report_path.exists():
            raise SystemExit(f"per-model PCC report referenced by sentinel does not exist: {pcc_report_path}")
        if sha256_file(pcc_report_path) != str(s_obj.get("sha256") or ""):
            raise SystemExit(f"per-model PCC report sha256 mismatch for: {pcc_report_path}")
        pcc_report = json.loads(pcc_report_path.read_text(encoding="utf-8"))
        pcc_baseline = Path(str(pcc_report.get("pcc_baseline_path") or ""))
        if not pcc_baseline.exists():
            raise SystemExit(f"missing per-model PCC baseline manifest: {pcc_baseline}")
        if sha256_file(pcc_baseline) != str(pcc_report.get("pcc_baseline_sha256") or ""):
            raise SystemExit(f"per-model PCC baseline sha256 mismatch for: {pcc_baseline}")

        obj = read_pcc_example_ids_and_domain_counts(pcc_baseline)
        per_model_example_ids[m["name"]] = obj["example_ids"]
        per_model_counts_by_domain[m["name"]] = obj["counts_by_domain"]

    ccc_example_ids = compute_ccc_intersection(per_model_example_ids)
    retention = compute_retention_metrics(
        ccc_example_ids=set(ccc_example_ids),
        per_model_counts_by_domain=per_model_counts_by_domain,
        domain_by_example=domain_by_example,
    )
    ok, reasons = check_ccc_gates(
        retention_metrics=retention,
        min_overall=float(args.min_overall_retention),
        min_per_domain=float(args.min_per_domain_retention),
    )

    out_dir = run_dir / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write CCC manifests + report for audit; only write sentinel on PASS.
    ccc_baseline_existing = out_dir / "ccc_baseline.jsonl"
    ccc_robust_existing = out_dir / "ccc_robustness.jsonl"
    ccc_meta: Dict[str, Any]
    if ccc_baseline_existing.exists() and ccc_robust_existing.exists():
        try:
            base_rows = list(iter_jsonl(ccc_baseline_existing))
            rob_rows = list(iter_jsonl(ccc_robust_existing))
            from sow.pcc.build_pcc import validate_pcc_baseline_manifest as _vpb  # noqa: PLC0415
            from sow.pcc.build_pcc import validate_pcc_robustness_manifest as _vpr  # noqa: PLC0415

            _vpb(base_rows, expected_n=len(ccc_example_ids))
            _vpr(rob_rows, expected_n_examples=len(ccc_example_ids), expected_wrapper_ids=expected_wrappers)
            ex_existing = sorted({str(r["example_id"]) for r in base_rows})
            if ex_existing != ccc_example_ids:
                raise ValueError("existing CCC baseline example_id set does not match computed intersection")
            ccc_meta = {
                "run_id": args.run_id,
                "ccc_size": int(len(ccc_example_ids)),
                "selected_example_ids_sha256": sha256_text("\n".join(ccc_example_ids) + "\n"),
                "baseline_manifest_path": str(baseline_manifest),
                "baseline_manifest_sha256": sha256_file(baseline_manifest),
                "robustness_manifest_path": str(robustness_manifest),
                "robustness_manifest_sha256": sha256_file(robustness_manifest),
                "outputs": {
                    "ccc_baseline_path": str(ccc_baseline_existing),
                    "ccc_baseline_sha256": sha256_file(ccc_baseline_existing),
                    "ccc_robustness_path": str(ccc_robust_existing),
                    "ccc_robustness_sha256": sha256_file(ccc_robust_existing),
                },
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reused_existing_outputs": True,
            }
        except Exception:
            ccc_meta = build_ccc_manifests(
                run_id=args.run_id,
                ccc_example_ids=ccc_example_ids,
                baseline_manifest_path=baseline_manifest,
                robustness_manifest_path=robustness_manifest,
                expected_wrapper_ids=expected_wrappers,
                out_dir=out_dir,
            )
    else:
        ccc_meta = build_ccc_manifests(
            run_id=args.run_id,
            ccc_example_ids=ccc_example_ids,
            baseline_manifest_path=baseline_manifest,
            robustness_manifest_path=robustness_manifest,
            expected_wrapper_ids=expected_wrappers,
            out_dir=out_dir,
        )

    report = {
        "pass": bool(ok),
        "run_id": args.run_id,
        "min_overall_retention": float(args.min_overall_retention),
        "min_per_domain_retention": float(args.min_per_domain_retention),
        "gate_fail_reasons": reasons,
        "ccc_size": int(len(ccc_example_ids)),
        "ccc_example_ids_sha256": sha256_text("\n".join(ccc_example_ids) + "\n"),
        "retention": retention,
        "inputs": {
            "baseline_manifest_path": str(baseline_manifest),
            "baseline_manifest_sha256": sha256_file(baseline_manifest),
            "robustness_manifest_path": str(robustness_manifest),
            "robustness_manifest_sha256": sha256_file(robustness_manifest),
        },
        "outputs": ccc_meta["outputs"],
        "run_config_path": str(cfg_path),
        "run_config_sha256": sha256_file(cfg_path),
        "config_snapshot_path": str(_config_snapshot_path(run_dir)),
        "config_snapshot_sha256": _config_snapshot_sha256(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = _next_available_path(out_dir / "ccc_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sent_dir = run_dir / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    sent_path = None
    if ok:
        sent_path = _next_available_path(sent_dir / "ccc.done")
        sent_path.write_text(
            json.dumps({"stage": "ccc", "output": str(report_path), "sha256": sha256_file(report_path)}, indent=2) + "\n",
            encoding="utf-8",
        )

    append_state_entry(
        state_path=_state_path(),
        stage="Stage 9 - build Common Compatible Core (CCC)",
        status="PASS" if ok else "FAIL",
        command=f"python3 sow.py build-ccc --run-id {args.run_id} --min-overall-retention {args.min_overall_retention} --min-per-domain-retention {args.min_per_domain_retention}",
        inputs=[
            HashedPath(path=str(baseline_manifest), sha256=sha256_file(baseline_manifest)),
            HashedPath(path=str(robustness_manifest), sha256=sha256_file(robustness_manifest)),
            HashedPath(path=str(cfg_path), sha256=sha256_file(cfg_path)),
        ],
        outputs=[
            HashedPath(path=str(ccc_meta["outputs"]["ccc_baseline_path"]), sha256=str(ccc_meta["outputs"]["ccc_baseline_sha256"])),
            HashedPath(path=str(ccc_meta["outputs"]["ccc_robustness_path"]), sha256=str(ccc_meta["outputs"]["ccc_robustness_sha256"])),
            HashedPath(path=str(report_path), sha256=sha256_file(report_path)),
            *( [HashedPath(path=str(sent_path), sha256=sha256_file(sent_path))] if sent_path else [] ),
        ],
        validators=[HashedPath(path=str(report_path), sha256=sha256_file(report_path))],
        notes="CCC is the intersection of per-model PCC sets; gates enforce >=0.80 overall and >=0.60 per-domain retention (per model).",
        next_step="Stage 10 - PCA membership (already done) or Stage 11 - PCA sample extraction" if ok else "Fix PCC/thresholds before proceeding",
    )

    print(str(report_path))
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sow", description="Shape of Wisdom pipeline (Milestone 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-run", help="Create runs/<run_id>/run_config.yaml and meta snapshot")
    p_init.add_argument("--run-id", required=True)
    p_init.add_argument("--seed", type=int, default=12345)
    p_init.set_defaults(func=cmd_init_run)

    p_s0 = sub.add_parser("stage0", help="Environment lock + smoke test (model load, hidden states, short generate)")
    p_s0.add_argument("--run-id", required=True)
    p_s0.add_argument("--model-name", default=None, help="Model name from run_config.models[].name (default: first model)")
    p_s0.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_s0.set_defaults(func=cmd_stage0)

    p_m = sub.add_parser("build-manifests", help="Canonicalize + validate baseline and robustness manifests (run-scoped)")
    p_m.add_argument("--run-id", required=True)
    p_m.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate-only: do not repair known missing wrappers (expected to fail on current paid inputs).",
    )
    p_m.set_defaults(func=cmd_build_manifests)

    p_pr = sub.add_parser("parser-regression", help="Run deterministic parser regression suite")
    p_pr.add_argument("--run-id", required=True)
    p_pr.set_defaults(func=cmd_parser_regression)

    p_tb = sub.add_parser("token-buckets", help="Build per-model option token buckets (A/B/C/D)")
    p_tb.add_argument("--run-id", required=True)
    p_tb.set_defaults(func=cmd_token_buckets)

    p_pi = sub.add_parser("pilot-inference", help="Stage 7 pilot inference (200-500 prompts, stratified by coarse_domain)")
    p_pi.add_argument("--run-id", required=True)
    p_pi.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_pi.add_argument("--sample-size", type=int, default=200)
    p_pi.add_argument("--min-one-token-compliance", type=float, default=None)
    p_pi.add_argument("--min-parser-resolved", type=float, default=None)
    p_pi.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_pi.set_defaults(func=cmd_pilot_inference)

    p_pca = sub.add_parser("pca-membership", help="Freeze PCA sample membership (1000, stratified)")
    p_pca.add_argument("--run-id", required=True)
    p_pca.set_defaults(func=cmd_pca_membership)

    p_pcas = sub.add_parser("pca-sample-inference", help="Stage 11: PCA sample extraction inference (hidden states at decision position)")
    p_pcas.add_argument("--run-id", required=True)
    p_pcas.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_pcas.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_pcas.add_argument("--batch-size", default="auto", help="Integer batch size, or 'auto' (calibrate + adapt to OOM)")
    p_pcas.add_argument("--repro-check-k", type=int, default=8, help="Recompute first K prompts and compare to stored vectors (spot-check)")
    p_pcas.add_argument("--repro-atol", type=float, default=1e-3, help="Absolute tolerance for repro spot-check on float16 vectors")
    p_pcas.set_defaults(func=cmd_pca_sample_inference)

    p_pcaf = sub.add_parser("pca-fit", help="Stage 12: fit PCA basis once per model (pooled across layers)")
    p_pcaf.add_argument("--run-id", required=True)
    p_pcaf.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_pcaf.set_defaults(func=cmd_pca_fit)

    p_s13s = sub.add_parser("stage13-smoke", help="Stage 13 smoke (20 prompts) + strict gates (batch-consistency + resume)")
    p_s13s.add_argument("--run-id", required=True)
    p_s13s.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_s13s.add_argument("--sample-size", type=int, default=20)
    p_s13s.add_argument("--batch-size", default="auto", help="Integer batch size, or 'auto'")
    p_s13s.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_s13s.add_argument("--skip-robustness", action="store_true", help="Skip robustness smoke and validate only baseline gates")
    p_s13s.set_defaults(func=cmd_stage13_smoke)

    p_s13b = sub.add_parser("inference-baseline", help="Stage 13a: baseline full inference (CCC baseline)")
    p_s13b.add_argument("--run-id", required=True)
    p_s13b.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_s13b.add_argument("--batch-size", default="auto", help="Integer batch size, or 'auto'")
    p_s13b.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_s13b.set_defaults(func=cmd_inference_baseline)

    p_s13r = sub.add_parser("inference-robustness", help="Stage 13b: robustness full inference (CCC robustness)")
    p_s13r.add_argument("--run-id", required=True)
    p_s13r.add_argument("--model-name", default=None, help="Run for a single model name; default is all models")
    p_s13r.add_argument("--batch-size", default="auto", help="Integer batch size, or 'auto'")
    p_s13r.add_argument("--device", default=None, help="Override device (cuda/mps/cpu); default auto-pick")
    p_s13r.set_defaults(func=cmd_inference_robustness)

    p_an = sub.add_parser("analyze", help="Stage 14: deterministic analysis + report artifacts (JSON/CSV)")
    p_an.add_argument("--run-id", required=True)
    p_an.add_argument("--skip-robustness", action="store_true", help="Baseline-only mechanistic analysis (skip robustness deltas)")
    p_an.add_argument("--topology-layer", type=int, default=-1, help="Layer index for domain topology centroids (-1 means final layer)")
    p_an.set_defaults(func=cmd_analyze)

    p_pcc = sub.add_parser("build-pcc", help="Stage 8: build Primary Core Corpus (PCC) per model")
    p_pcc.add_argument("--run-id", required=True)
    p_pcc.add_argument("--target-size", type=int, default=3000)
    p_pcc.set_defaults(func=cmd_build_pcc)

    p_ccc = sub.add_parser("build-ccc", help="Stage 9: build Common Compatible Core (CCC) across models")
    p_ccc.add_argument("--run-id", required=True)
    p_ccc.add_argument("--min-overall-retention", type=float, default=0.80)
    p_ccc.add_argument("--min-per-domain-retention", type=float, default=0.60)
    p_ccc.set_defaults(func=cmd_build_ccc)

    return ap


def main(argv: List[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))
