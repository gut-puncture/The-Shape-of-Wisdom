#!/usr/bin/env python3
"""Audit cached parquet data for Paper 1. Outputs CHECKS.json.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/part1/audit_cached_data.py --parquet-dir results/parquet
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

EXPECTED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
REQUIRED_PARQUETS = ["decision_metrics.parquet", "layerwise.parquet"]
OPTIONAL_PARQUETS = ["prompt_types.parquet"]
MODEL_LAYERS = {"Qwen/Qwen2.5-7B-Instruct": 28, "meta-llama/Llama-3.1-8B-Instruct": 32, "mistralai/Mistral-7B-Instruct-v0.3": 32}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def run_checks(parquet_dir: Path) -> dict:
    checks = {}

    # A) File existence
    for name in REQUIRED_PARQUETS:
        p = parquet_dir / name
        checks[f"file_exists_{name}"] = {"pass": p.exists(), "path": str(p)}

    if not all(checks[f"file_exists_{n}"]["pass"] for n in REQUIRED_PARQUETS):
        checks["OVERALL"] = {"pass": False, "reason": "Missing required parquet files"}
        return checks

    dm = pd.read_parquet(parquet_dir / "decision_metrics.parquet")
    lw = pd.read_parquet(parquet_dir / "layerwise.parquet")

    # B) Row counts and uniqueness
    expected_rows = sum(3000 * MODEL_LAYERS[m] for m in EXPECTED_MODELS)
    actual_rows = len(dm)
    checks["row_count_decision_metrics"] = {
        "pass": actual_rows == expected_rows,
        "expected": expected_rows, "actual": actual_rows
    }
    dups = dm.duplicated(subset=["model_id", "prompt_uid", "layer_index"])
    checks["uniqueness_decision_metrics"] = {
        "pass": int(dups.sum()) == 0,
        "duplicates": int(dups.sum())
    }

    # C) Model coverage
    models = sorted(dm["model_id"].unique())
    checks["model_coverage"] = {
        "pass": len(models) == 3 and all(m in models for m in EXPECTED_MODELS),
        "found": models, "expected": EXPECTED_MODELS
    }

    # D) Prompt coverage
    for m in EXPECTED_MODELS:
        n = dm.loc[dm["model_id"] == m, "prompt_uid"].nunique()
        checks[f"prompt_coverage_{m.split('/')[-1]}"] = {
            "pass": n == 3000, "actual": n, "expected": 3000
        }

    # E) Numerical sanity
    for col in ["delta", "boundary", "drift"]:
        n_nan = int(dm[col].isna().sum())
        checks[f"no_nan_{col}"] = {"pass": n_nan == 0, "nan_count": n_nan}

    # Verify layerwise has logits
    checks["layerwise_has_logits"] = {
        "pass": "candidate_logits_json" in lw.columns,
        "columns": list(lw.columns)
    }

    # Score NaN check in layerwise
    sample = lw.head(1000)
    logits = sample["candidate_logits_json"].apply(json.loads)
    nan_count = sum(1 for d in logits if any(not np.isfinite(float(d.get(k, 0))) for k in "ABCD"))
    checks["no_nan_scores_sample"] = {"pass": nan_count == 0, "nan_in_sample_1000": nan_count}

    # F) SHA256 manifest
    file_hashes = {}
    for p in sorted(parquet_dir.glob("*.parquet")):
        file_hashes[p.name] = sha256(p)
    checks["sha256_manifest"] = {"pass": True, "hashes": file_hashes}

    # Overall
    all_pass = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))
    checks["OVERALL"] = {"pass": all_pass}
    return checks

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, default=Path("results/parquet"))
    ap.add_argument("--output", type=Path, default=Path("paper/part1/CHECKS.json"))
    args = ap.parse_args()

    checks = run_checks(args.parquet_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(f"Wrote {args.output}")

    passed = checks["OVERALL"]["pass"]
    n_checks = sum(1 for k, v in checks.items() if k != "OVERALL" and isinstance(v, dict))
    n_pass = sum(1 for k, v in checks.items() if k != "OVERALL" and isinstance(v, dict) and v.get("pass"))
    print(f"{'PASSED' if passed else 'FAILED'}: {n_pass}/{n_checks} checks")
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
