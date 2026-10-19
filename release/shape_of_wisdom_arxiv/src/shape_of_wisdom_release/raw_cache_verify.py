from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .build_assets import DERIVED_PATH, PRIMARY_DATASETS, load_jsonl
from .persistent_metrics import BundleValidationError, compute_example_metrics, metrics_to_row


PRIMARY_READOUT = "semantic_exact"
PRIMARY_PERMUTATION = "identity"
COMPARE_COLUMNS = [
    "final_winner",
    "d2_pos",
    "d1_pos",
    "d50_pos",
    "norm_d2",
    "norm_d1",
    "norm_d50",
    "norm_delta_21",
    "norm_delta_250",
    "write_share_after_d2",
    "write_share_after_d1",
    "write_share_last20",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical_examples(path: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in load_jsonl(path)}


def canonical_option_ids(example: dict[str, Any]) -> list[str]:
    options = list(example["options"])
    options.sort(key=lambda row: int(row["ordinal"]))
    return [str(row["option_id"]) for row in options]


def cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / cache_key[:2] / f"{cache_key}.json"


def close(lhs: Any, rhs: Any, *, atol: float) -> bool:
    if lhs is None and rhs is None:
        return True
    if isinstance(lhs, float) or isinstance(rhs, float):
        if lhs is None or rhs is None:
            return False
        if math.isnan(float(lhs)) and math.isnan(float(rhs)):
            return True
        return abs(float(lhs) - float(rhs)) <= atol
    return lhs == rhs


def verify_raw_cache(run_dir: Path, cache_root: Path, *, limit: int | None, atol: float) -> dict[str, Any]:
    expected = {
        (row["model_name"], row["example_id"]): row
        for row in load_jsonl(DERIVED_PATH)
    }
    examples = canonical_examples(run_dir / "prepared_data" / "canonical_examples.jsonl")
    rows = [
        row
        for row in load_jsonl(run_dir / "metrics" / "score_manifest.jsonl")
        if row["readout_id"] == PRIMARY_READOUT
        and row["permutation_id"] == PRIMARY_PERMUTATION
        and row["dataset_name"] in PRIMARY_DATASETS
    ]
    rows.sort(key=lambda row: (row["model_name"], row["example_id"]))
    if limit is not None:
        rows = rows[:limit]

    failures: list[dict[str, Any]] = []
    checked = 0
    for row in rows:
        key = (row["model_name"], row["example_id"])
        if key not in expected:
            failures.append({"key": key, "issue": "missing included derived row"})
            continue
        path = cache_path(cache_root, row["cache_key"])
        if not path.exists():
            failures.append({"key": key, "issue": "missing raw cache bundle", "path": str(path)})
            continue
        try:
            metrics = compute_example_metrics(
                bundle=load_json(path),
                cache_key=row["cache_key"],
                cache_relpath=str(path),
                canonical_option_ids=canonical_option_ids(examples[row["example_id"]]),
            )
        except (BundleValidationError, KeyError) as exc:
            failures.append({"key": key, "issue": "bundle validation failed", "detail": str(exc)})
            continue
        actual = metrics_to_row(metrics)
        checked += 1
        for column in COMPARE_COLUMNS:
            if not close(actual.get(column), expected[key].get(column), atol=atol):
                failures.append(
                    {
                        "key": key,
                        "issue": "value mismatch",
                        "column": column,
                        "expected": expected[key].get(column),
                        "actual": actual.get(column),
                    }
                )
                break
    return {"checked": checked, "failures": failures}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optionally verify included derived rows from raw cache bundles.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    summary = verify_raw_cache(args.run_dir, args.cache_root, limit=args.limit, atol=args.atol)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
