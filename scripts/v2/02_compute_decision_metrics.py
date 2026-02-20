#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import (
    base_parser,
    baseline_output_path,
    load_experiment_config,
    load_jsonl_rows,
    load_manifest_correct_keys,
    resolve_models,
    run_v2_root_for,
    write_json,
    write_parquet,
)
from sow.v2.metrics import build_decision_metrics_frame


def main() -> int:
    ap = base_parser("V2: compute decision metrics (delta, boundary, drift)")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    out_root = run_v2_root_for(args.run_id)
    out_path = out_root / "decision_metrics.parquet"

    correct_by_uid = load_manifest_correct_keys(args.run_id)
    all_rows = []
    for model in models:
        path = baseline_output_path(args.run_id, str(model["model_id"]))
        if not path.exists():
            raise SystemExit(f"missing baseline output: {path}")
        all_rows.extend(load_jsonl_rows(path, max_rows=int(args.max_prompts)))

    df_new = build_decision_metrics_frame(all_rows, correct_key_by_prompt_uid=correct_by_uid)
    if args.resume and out_path.exists():
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["model_id", "prompt_uid", "layer_index"], keep="last")
    else:
        df = df_new

    write_parquet(out_path, df)
    write_json(
        out_root / "02_compute_decision_metrics.report.json",
        {
            "pass": True,
            "rows": int(df.shape[0]),
            "prompts": int(df["prompt_uid"].nunique()) if not df.empty else 0,
            "out_path": str(out_path),
        },
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
