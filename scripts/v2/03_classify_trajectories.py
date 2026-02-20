#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import base_parser, run_v2_root_for, write_json, write_parquet
from sow.v2.trajectory_types import classify_trajectory_table, type_counts


def main() -> int:
    ap = base_parser("V2: classify prompts into four trajectory types")
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    in_path = out_root / "decision_metrics.parquet"
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    metrics = pd.read_parquet(in_path)
    if args.model_name:
        metrics = metrics[metrics["model_id"].str.contains(args.model_name, na=False, case=False)]
    if args.max_prompts > 0 and not metrics.empty:
        keep = set(metrics["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        metrics = metrics[metrics["prompt_uid"].isin(keep)]

    prompt_types_new = classify_trajectory_table(metrics)

    out_types = out_root / "prompt_types.parquet"
    if args.resume and out_types.exists():
        old = pd.read_parquet(out_types)
        prompt_types = pd.concat([old, prompt_types_new], ignore_index=True)
        prompt_types = prompt_types.drop_duplicates(subset=["model_id", "prompt_uid"], keep="last")
    else:
        prompt_types = prompt_types_new

    counts = type_counts(prompt_types)
    write_parquet(out_types, prompt_types)
    write_json(out_root / "type_counts.json", counts)
    write_json(
        out_root / "03_classify_trajectories.report.json",
        {"pass": True, "rows": int(prompt_types.shape[0]), "counts": counts, "out_path": str(out_types)},
    )
    print(str(out_types))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
