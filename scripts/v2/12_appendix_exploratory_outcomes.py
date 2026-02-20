#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, run_v2_root_for, write_json, write_parquet


def main() -> int:
    ap = base_parser("V2: optional appendix exploratory outcomes")
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    metrics_path = out_root / "decision_metrics.parquet"
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)

    if args.model_name:
        metrics = metrics[metrics["model_id"].str.contains(args.model_name, na=False)]
    if args.max_prompts > 0 and not metrics.empty:
        keep = set(metrics["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        metrics = metrics[metrics["prompt_uid"].isin(keep)]

    # Exploratory last-K ensembling proxy in logit-space.
    rows = []
    for (model_id, prompt_uid), g in metrics.groupby(["model_id", "prompt_uid"], sort=False):
        g = g.sort_values("layer_index")
        for k in [2, 4, 8]:
            tail = g.tail(k)
            score = float(np.mean(tail["delta"].to_numpy(dtype=np.float64)))
            rows.append(
                {
                    "model_id": str(model_id),
                    "prompt_uid": str(prompt_uid),
                    "k_last_layers": int(k),
                    "ensemble_delta": score,
                    "ensemble_predicts_correct": bool(score > 0.0),
                }
            )

    out = pd.DataFrame.from_records(rows)
    out_path = out_root / "appendix_exploratory_outcomes.parquet"
    write_parquet(out_path, out)
    write_json(
        out_root / "12_appendix_exploratory_outcomes.report.json",
        {"pass": True, "rows": int(out.shape[0]), "out_path": str(out_path)},
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
