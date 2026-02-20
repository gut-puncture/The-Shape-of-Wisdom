#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _common import (
    base_parser,
    baseline_output_path,
    load_experiment_config,
    load_jsonl_rows,
    resolve_models,
    run_v2_root_for,
    write_json,
    write_parquet,
)


def main() -> int:
    ap = base_parser("V2: extract baseline layerwise rows to parquet")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    out_root = run_v2_root_for(args.run_id)
    out_path = out_root / "layerwise.parquet"

    rows = []
    for model in models:
        path = baseline_output_path(args.run_id, str(model["model_id"]))
        if not path.exists():
            raise SystemExit(f"missing baseline output: {path}")
        src_rows = load_jsonl_rows(path, max_rows=int(args.max_prompts))
        for row in src_rows:
            for layer in row.get("layerwise") or []:
                rows.append(
                    {
                        "model_id": str(row.get("model_id") or ""),
                        "model_revision": str(row.get("model_revision") or ""),
                        "prompt_uid": str(row.get("prompt_uid") or ""),
                        "example_id": str(row.get("example_id") or ""),
                        "wrapper_id": str(row.get("wrapper_id") or ""),
                        "coarse_domain": str(row.get("coarse_domain") or "unknown"),
                        "is_correct": bool(row.get("is_correct") is True),
                        "layer_index": int(layer.get("layer_index") or 0),
                        "candidate_logits_json": json.dumps(layer.get("candidate_logits") or {}, sort_keys=True),
                        "candidate_probs_json": json.dumps(layer.get("candidate_probs") or {}, sort_keys=True),
                        "candidate_entropy": float(layer.get("candidate_entropy") or 0.0),
                        "top_candidate": str(layer.get("top_candidate") or ""),
                        "top2_margin_prob": float(layer.get("top2_margin_prob") or 0.0),
                        "projected_hidden_128_json": json.dumps(layer.get("projected_hidden_128") or []),
                    }
                )

    df_new = pd.DataFrame.from_records(rows)
    if args.resume and out_path.exists():
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["model_id", "prompt_uid", "layer_index"], keep="last")
    else:
        df = df_new

    write_parquet(out_path, df)
    write_json(
        out_root / "01_extract_baseline.report.json",
        {
            "pass": True,
            "rows": int(df.shape[0]),
            "models": [str(m["model_id"]) for m in models],
            "out_path": str(out_path),
        },
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
