#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import base_parser, run_v2_root_for, write_json, write_parquet
from sow.v2.causal.span_deletion import compare_evidence_vs_distractor, run_negative_controls, summarize_span_deletion_effects


def main() -> int:
    ap = base_parser("V2: run span deletion causal summaries and negative controls")
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    span_del_out = out_root / "span_deletion_causal.parquet"
    neg_out = out_root / "negative_controls.parquet"

    labels_path = out_root / "span_labels.parquet"
    if not labels_path.exists():
        raise SystemExit(f"missing span labels: {labels_path}")

    labels = pd.read_parquet(labels_path)
    if args.max_prompts > 0 and not labels.empty:
        keep = set(labels["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        labels = labels[labels["prompt_uid"].isin(keep)]

    span_del = summarize_span_deletion_effects(labels)
    neg = run_negative_controls(labels, seed=123)
    stats = compare_evidence_vs_distractor(labels)

    write_parquet(span_del_out, span_del)
    write_parquet(neg_out, neg)
    write_json(
        out_root / "10_causal_validation_tools.report.json",
        {
            "pass": True,
            "span_deletion_rows": int(span_del.shape[0]),
            "negative_control_rows": int(neg.shape[0]),
            "evidence_vs_distractor": stats,
        },
    )
    print(str(out_root / "span_deletion_causal.parquet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
