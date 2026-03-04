#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _audit_common import default_paths, git_commit, require_paths, write_json


REQUIRED_INPUTS = [
    "decision_metrics.parquet",
    "prompt_types.parquet",
    "tracing_scalars.parquet",
    "ablation_results.parquet",
    "patching_results.parquet",
    "attention_mass_by_span.parquet",
    "attention_contrib_by_span.parquet",
    "span_effects.parquet",
    "span_labels.parquet",
    "span_deletion_causal.parquet",
    "negative_controls.parquet",
]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return {"cmd": cmd, "returncode": int(proc.returncode)}


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Reproduce vNext audit outputs from cached artifacts only.")
    ap.add_argument("--repo-root", type=Path, default=paths.repo)
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--reports-dir", type=Path, default=paths.reports)
    ap.add_argument("--audit-dir", type=Path, default=paths.audit)
    ap.add_argument("--figures-dir", type=Path, default=paths.figures_vnext)
    ap.add_argument("--build-paper", action="store_true", help="Attempt paper_publish_v3.pdf rebuild if latexmk exists.")
    ap.add_argument("--strict-paper-build", action="store_true", help="Fail if paper build toolchain is unavailable.")
    ap.add_argument("--out-json", type=Path, default=paths.audit / "reproduce_vnext_manifest.json")
    args = ap.parse_args()

    require_paths([args.parquet_dir / n for n in REQUIRED_INPUTS])

    env = dict(os.environ)
    env["SOW_ALLOW_INFERENCE"] = "0"
    python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise SystemExit("python interpreter not found")

    steps = [
        [python, "scripts/audit/extract_definitions.py", "--out-json", str(args.audit_dir / "definitions_extracted.json")],
        [
            python,
            "scripts/audit/check_artifacts_integrity.py",
            "--parquet-dir",
            str(args.parquet_dir),
            "--reports-dir",
            str(args.reports_dir),
            "--out-json",
            str(args.audit_dir / "artifact_integrity.json"),
        ],
        [
            python,
            "scripts/audit/spotcheck_trajectories.py",
            "--parquet-dir",
            str(args.parquet_dir),
            "--out-dir",
            str(args.audit_dir),
            "--n-prompts",
            "20",
            "--seed",
            "12345",
        ],
        [
            python,
            "scripts/audit/substitution_rederive.py",
            "--parquet-dir",
            str(args.parquet_dir),
            "--spans-jsonl",
            str(paths.results / "spans.jsonl"),
            "--out-csv",
            str(args.audit_dir / "substitution_pairs_vnext.csv"),
            "--out-json",
            str(args.audit_dir / "substitution_rederive_diagnostics.json"),
            "--seed",
            "12345",
        ],
        [
            python,
            "scripts/audit/substitution_sensitivity.py",
            "--pairs-csv",
            str(args.audit_dir / "substitution_pairs_vnext.csv"),
            "--out-csv",
            str(args.audit_dir / "substitution_sensitivity_summary.csv"),
        ],
        [
            python,
            "scripts/audit/drift_reconstruction_audit.py",
            "--parquet-dir",
            str(args.parquet_dir),
            "--out-dir",
            str(args.audit_dir),
            "--figures-dir",
            str(args.figures_dir),
        ],
        [
            python,
            "scripts/audit/contradiction_memo.py",
            "--out-json",
            str(args.audit_dir / "contradiction_memo.json"),
        ],
        [
            python,
            "scripts/audit/generate_figures_vnext.py",
            "--parquet-dir",
            str(args.parquet_dir),
            "--audit-dir",
            str(args.audit_dir),
            "--output-dir",
            str(args.figures_dir),
            "--strict",
        ],
    ]

    results: list[dict[str, Any]] = []
    for cmd in steps:
        rec = _run(cmd, cwd=args.repo_root, env=env)
        results.append(rec)
        if int(rec["returncode"]) != 0:
            write_json(
                args.out_json,
                {
                    "pass": False,
                    "failed_step": rec,
                    "steps": results,
                    "git_commit": git_commit(args.repo_root),
                    "inference_firewall_env": env.get("SOW_ALLOW_INFERENCE"),
                },
            )
            raise SystemExit(rec["returncode"])

    paper_status: dict[str, Any] = {"attempted": False, "built": False, "reason": "skipped"}
    if args.build_paper:
        paper_status["attempted"] = True
        latexmk = shutil.which("latexmk")
        if not latexmk:
            paper_status = {"attempted": True, "built": False, "reason": "latexmk_not_found"}
            if args.strict_paper_build:
                write_json(
                    args.out_json,
                    {
                        "pass": False,
                        "steps": results,
                        "paper_status": paper_status,
                        "git_commit": git_commit(args.repo_root),
                        "inference_firewall_env": env.get("SOW_ALLOW_INFERENCE"),
                    },
                )
                raise SystemExit(2)
        else:
            paper_cmd = [latexmk, "-pdf", "paper_publish_v3.tex"]
            proc = subprocess.run(
                paper_cmd,
                cwd=str(paths.paper_dir),
                env=env,
                check=False,
            )
            paper_status = {
                "attempted": True,
                "built": int(proc.returncode) == 0,
                "reason": "ok" if int(proc.returncode) == 0 else "latexmk_failed",
                "returncode": int(proc.returncode),
                "cmd": paper_cmd,
            }
            if int(proc.returncode) != 0 and args.strict_paper_build:
                write_json(
                    args.out_json,
                    {
                        "pass": False,
                        "steps": results,
                        "paper_status": paper_status,
                        "git_commit": git_commit(args.repo_root),
                        "inference_firewall_env": env.get("SOW_ALLOW_INFERENCE"),
                    },
                )
                raise SystemExit(int(proc.returncode))

    write_json(
        args.out_json,
        {
            "pass": True,
            "steps": results,
            "paper_status": paper_status,
            "git_commit": git_commit(args.repo_root),
            "inference_firewall_env": env.get("SOW_ALLOW_INFERENCE"),
        },
    )
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

