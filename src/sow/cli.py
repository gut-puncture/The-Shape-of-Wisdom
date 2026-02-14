from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sow.config import default_run_config, read_yaml, validate_run_config, write_yaml
from sow.hashing import sha256_file
from sow.io_jsonl import iter_jsonl
from sow.judging.deterministic_parser import parse_choice
from sow.manifest.canonicalize import canonicalize_baseline_manifest, canonicalize_robustness_manifest_v2
from sow.manifest.schema import validate_baseline_manifest, validate_robustness_manifest
from sow.pca.membership import select_pca_membership, write_membership_file
from sow.state import HashedPath, append_state_entry


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / run_id


def _state_path() -> Path:
    return REPO_ROOT / "STATE.md"


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sow", description="Shape of Wisdom pipeline (Milestone 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-run", help="Create runs/<run_id>/run_config.yaml and meta snapshot")
    p_init.add_argument("--run-id", required=True)
    p_init.add_argument("--seed", type=int, default=12345)
    p_init.set_defaults(func=cmd_init_run)

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

    p_pca = sub.add_parser("pca-membership", help="Freeze PCA sample membership (1000, stratified)")
    p_pca.add_argument("--run-id", required=True)
    p_pca.set_defaults(func=cmd_pca_membership)

    return ap


def main(argv: List[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))

