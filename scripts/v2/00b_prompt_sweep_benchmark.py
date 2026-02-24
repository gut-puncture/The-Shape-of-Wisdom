#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from _common import REPO_ROOT, write_json


def _parse_prompt_counts(raw: str) -> List[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("prompt-counts must include at least one positive integer")
    out: List[int] = []
    seen = set()
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid prompt count: {part}") from exc
        if value <= 0:
            raise ValueError(f"prompt count must be > 0: {value}")
        if value in seen:
            raise ValueError(f"duplicate prompt count: {value}")
        seen.add(value)
        out.append(value)
    return out


def _build_stage00a_command(
    *,
    python_bin: str,
    run_id: str,
    config_path: Path,
    model_name: str | None,
    prompt_count: int,
    cooldown_seconds: int,
    done_sentinel: Path,
) -> List[str]:
    cmd: List[str] = [
        str(REPO_ROOT / "scripts" / "v2" / "run_with_thermal_resume.sh"),
        "--done-sentinel",
        str(done_sentinel),
        "--cooldown-seconds",
        str(int(cooldown_seconds)),
        "--",
        str(python_bin),
        str(REPO_ROOT / "scripts" / "v2" / "00a_generate_baseline_outputs.py"),
        "--run-id",
        str(run_id),
        "--config",
        str(config_path),
        "--max-prompts",
        str(int(prompt_count)),
        "--resume",
    ]
    if model_name:
        cmd.extend(["--model-name", str(model_name)])
    return cmd


def _run_cmd(argv: List[str]) -> int:
    try:
        proc = subprocess.run(argv, cwd=str(REPO_ROOT), check=False)
        return int(proc.returncode)
    except FileNotFoundError:
        return 127
    except Exception:
        return 126


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object at {path}")
    return obj


def _positive_finite(value: Any) -> bool:
    try:
        num = float(value)
    except Exception:
        return False
    return bool(math.isfinite(num) and num > 0.0)


def _compute_diagnostics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_entries = [e for e in entries if _positive_finite(e.get("rows_per_second"))]
    if not valid_entries:
        return {
            "rows_per_second_5_to_50_ratio": None,
            "monotonic_nonincreasing_rows_per_second": False,
            "per_model_rows_per_second_5_to_50_ratio": {},
        }

    values = [float(e["rows_per_second"]) for e in valid_entries]
    monotonic_nonincreasing = all(values[idx] <= values[idx - 1] for idx in range(1, len(values)))
    ratio = None
    if values and values[0] > 0.0:
        ratio = float(values[-1] / values[0])

    first_model_rps = valid_entries[0].get("rows_per_second_per_model") or {}
    last_model_rps = valid_entries[-1].get("rows_per_second_per_model") or {}
    per_model_ratio: Dict[str, float] = {}
    for model_id in sorted(set(first_model_rps.keys()) & set(last_model_rps.keys())):
        first = float(first_model_rps[model_id])
        last = float(last_model_rps[model_id])
        if first > 0.0:
            per_model_ratio[str(model_id)] = float(last / first)

    return {
        "rows_per_second_5_to_50_ratio": ratio,
        "monotonic_nonincreasing_rows_per_second": bool(monotonic_nonincreasing),
        "per_model_rows_per_second_5_to_50_ratio": per_model_ratio,
    }


def run_prompt_sweep(
    *,
    run_id_prefix: str,
    prompt_counts: List[int],
    config_path: Path,
    model_name: str | None,
    cooldown_seconds: int,
    python_bin: str,
    exec_fn: Callable[[List[str]], int] = _run_cmd,
    output_report_path: Path | None = None,
) -> Tuple[int, Dict[str, Any]]:
    failures: List[str] = []
    entries: List[Dict[str, Any]] = []
    out_root = REPO_ROOT / "runs" / str(run_id_prefix) / "v2"
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = output_report_path or (out_root / "00b_prompt_sweep_benchmark.report.json")

    if not config_path.exists():
        failures.append(f"missing config path: {config_path}")
    elif not config_path.is_file():
        failures.append(f"config path must be a file: {config_path}")

    if int(cooldown_seconds) <= 0:
        failures.append(f"cooldown-seconds must be > 0: {cooldown_seconds}")

    if not prompt_counts:
        failures.append("prompt_counts cannot be empty")

    for prompt_count in prompt_counts:
        if failures:
            break
        run_id = f"{run_id_prefix}_p{int(prompt_count)}"
        done_sentinel = REPO_ROOT / "runs" / run_id / "v2" / "00a_generate_baseline_outputs.done"
        stage00a_report = REPO_ROOT / "runs" / run_id / "v2" / "00a_generate_baseline_outputs.report.json"
        entry: Dict[str, Any] = {
            "run_id": str(run_id),
            "prompt_count": int(prompt_count),
            "stage00a_report_path": str(stage00a_report),
            "done_sentinel": str(done_sentinel),
        }
        try:
            if done_sentinel.exists():
                if not done_sentinel.is_file():
                    failures.append(f"done sentinel exists but is not a file: {done_sentinel}")
                    entries.append(entry)
                    break
                done_sentinel.unlink()

            cmd = _build_stage00a_command(
                python_bin=python_bin,
                run_id=run_id,
                config_path=config_path,
                model_name=model_name,
                prompt_count=int(prompt_count),
                cooldown_seconds=int(cooldown_seconds),
                done_sentinel=done_sentinel,
            )
            start = time.monotonic()
            rc = int(exec_fn(cmd))
            elapsed = float(time.monotonic() - start)
            entry["command_rc"] = int(rc)
            entry["elapsed_seconds"] = float(elapsed)

            if rc != 0:
                failures.append(f"stage00a command failed rc={rc} run_id={run_id}")
                entries.append(entry)
                break

            if not stage00a_report.exists():
                failures.append(f"missing stage00a report: {stage00a_report}")
                entries.append(entry)
                break

            try:
                report = _load_json(stage00a_report)
            except Exception as exc:
                failures.append(f"invalid stage00a report JSON at {stage00a_report}: {exc}")
                entries.append(entry)
                break

            if not bool(report.get("pass")):
                failures.append(f"stage00a report pass=false run_id={run_id}")
                entries.append(entry)
                break

            rows_per_second = report.get("rows_per_second")
            if not _positive_finite(rows_per_second):
                failures.append(f"invalid non-positive rows_per_second run_id={run_id}: {rows_per_second}")
                entries.append(entry)
                break
            entry["rows_per_second"] = float(rows_per_second)

            manifest_rows = int(((report.get("manifest") or {}).get("rows") or 0))
            entry["manifest_rows"] = int(manifest_rows)
            if int(manifest_rows) != int(prompt_count):
                failures.append(
                    f"manifest rows mismatch run_id={run_id}: expected={prompt_count} observed={manifest_rows}"
                )
                entries.append(entry)
                break

            stats_per_model = report.get("stats_per_model") or {}
            if not isinstance(stats_per_model, dict) or not stats_per_model:
                failures.append(f"missing stats_per_model in stage00a report run_id={run_id}")
                entries.append(entry)
                break

            model_rps: Dict[str, float] = {}
            bad_model_rps: List[str] = []
            for model_id, model_stats in sorted(stats_per_model.items()):
                value = (model_stats or {}).get("rows_per_second")
                if not _positive_finite(value):
                    bad_model_rps.append(f"{model_id}:{value}")
                    continue
                model_rps[str(model_id)] = float(value)
            entry["rows_per_second_per_model"] = model_rps
            if bad_model_rps:
                failures.append(f"invalid per-model rows_per_second run_id={run_id}: {bad_model_rps}")
                entries.append(entry)
                break

            entries.append(entry)
        except Exception as exc:
            failures.append(f"unexpected benchmark error run_id={run_id}: {exc}")
            entries.append(entry)
            break

    pass_flag = not failures and len(entries) == len(prompt_counts)
    diagnostics = _compute_diagnostics(entries)
    final_report = {
        "pass": bool(pass_flag),
        "run_id_prefix": str(run_id_prefix),
        "config_path": str(config_path),
        "model_name": model_name,
        "prompt_counts": [int(x) for x in prompt_counts],
        "entries": entries,
        "diagnostics": diagnostics,
        "failures": failures,
        "report_path": str(report_path),
    }
    write_json(report_path, final_report)
    return (0 if pass_flag else 2), final_report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run deterministic stage00a benchmark sweep for prompt counts (default: 5,20,50)."
    )
    ap.add_argument("--run-id-prefix", required=True)
    ap.add_argument("--prompt-counts", default="5,20,50")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_v2.yaml"))
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--cooldown-seconds", type=int, default=1200)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--output-report", default="")
    args = ap.parse_args()

    try:
        prompt_counts = _parse_prompt_counts(str(args.prompt_counts))
    except ValueError as exc:
        raise SystemExit(str(exc))

    output_report = Path(str(args.output_report)).expanduser() if str(args.output_report).strip() else None
    rc, report = run_prompt_sweep(
        run_id_prefix=str(args.run_id_prefix),
        prompt_counts=prompt_counts,
        config_path=Path(str(args.config)).expanduser(),
        model_name=(str(args.model_name).strip() if args.model_name else None),
        cooldown_seconds=int(args.cooldown_seconds),
        python_bin=str(args.python_bin),
        output_report_path=output_report,
    )
    print(str(report["report_path"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
