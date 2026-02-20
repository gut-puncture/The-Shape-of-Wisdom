#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _common import base_parser, baseline_manifest_path, load_experiment_config, resolve_models, run_v2_root_for, write_json
from sow.io_jsonl import iter_jsonl


def main() -> int:
    ap = base_parser("V2: select tracing subset balanced by trajectory type")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    per_model_target = int((cfg.get("sampling") or {}).get("tracing_prompts_per_model", 600))

    out_root = run_v2_root_for(args.run_id)
    types_path = out_root / "prompt_types.parquet"
    if not types_path.exists():
        raise SystemExit(f"missing input: {types_path}")
    types = pd.read_parquet(types_path)

    manifest_map = {}
    for row in iter_jsonl(baseline_manifest_path(args.run_id)):
        manifest_map[str(row.get("prompt_uid") or "")] = row

    report = {"pass": True, "models": {}}
    for model in models:
        model_id = str(model["model_id"])
        sub = types[types["model_id"] == model_id].copy()
        if sub.empty:
            report["models"][model_id] = {"selected": 0}
            continue

        groups = []
        for _, g in sub.groupby("trajectory_type", sort=True):
            take = max(1, per_model_target // max(1, sub["trajectory_type"].nunique()))
            groups.append(g.head(take))
        selected = pd.concat(groups, ignore_index=True).drop_duplicates(subset=["prompt_uid"])
        selected = selected.head(per_model_target)

        records = []
        for _, row in selected.iterrows():
            puid = str(row["prompt_uid"])
            m = manifest_map.get(puid, {})
            records.append(
                {
                    "prompt_uid": puid,
                    "example_id": str(m.get("example_id") or row.get("prompt_uid")),
                    "trajectory_type": str(row["trajectory_type"]),
                    "prompt_text": str(m.get("prompt_text") or ""),
                    "correct_key": str(m.get("correct_key") or "A"),
                    "coarse_domain": str(m.get("coarse_domain") or "unknown"),
                }
            )

        out_path = out_root / f"tracing_subset_{model_id.replace('/', '__')}.json"
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        report["models"][model_id] = {
            "selected": int(len(records)),
            "out_path": str(out_path),
            "trajectory_counts": selected["trajectory_type"].value_counts().to_dict(),
        }

    write_json(out_root / "06_select_tracing_subset.report.json", report)
    print(str(out_root / "06_select_tracing_subset.report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
