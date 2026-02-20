#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import base_parser, load_experiment_config, run_v2_root_for, write_json
from sow.v2.tracing.decomposition import drift_reconstruction_quality


def main() -> int:
    ap = base_parser("V2: validate attention+MLP decomposition against drift")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    r2_min = float(((cfg.get("causal") or {}).get("drift_decomposition_r2_min", 0.70)))

    out_root = run_v2_root_for(args.run_id)
    report_path = out_root / "08_attention_and_mlp_decomposition.report.json"

    tracing_path = out_root / "tracing_scalars.parquet"
    if not tracing_path.exists():
        raise SystemExit(f"missing tracing data: {tracing_path}")
    df = pd.read_parquet(tracing_path)

    if args.model_name:
        df = df[df["model_id"].str.contains(args.model_name, na=False)]
    if args.max_prompts > 0 and not df.empty:
        keep = set(df["prompt_uid"].drop_duplicates().head(int(args.max_prompts)).tolist())
        df = df[df["prompt_uid"].isin(keep)]

    model_reports = {}
    failing_models = []
    for model_id, sub in df.groupby("model_id", sort=False):
        q = drift_reconstruction_quality(
            observed_drift=sub["drift"].to_numpy(),
            attn_scalar=sub["s_attn"].to_numpy(),
            mlp_scalar=sub["s_mlp"].to_numpy(),
        )
        model_reports[str(model_id)] = q
        if float(q.get("r2", 0.0)) < float(r2_min):
            failing_models.append(str(model_id))

        layer = sub.groupby("layer_index", as_index=False).agg(s_attn=("s_attn", "mean"), s_mlp=("s_mlp", "mean"))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(layer["layer_index"], layer["s_attn"], label="s_attn", lw=2)
        ax.plot(layer["layer_index"], layer["s_mlp"], label="s_mlp", lw=2)
        ax.set_title(f"Motion Decomposition ({model_id})")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean scalar")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.savefig(out_root / f"fig_motion_decomposition_{model_id.replace('/', '__')}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    write_json(
        report_path,
        {
            "pass": len(failing_models) == 0,
            "drift_decomposition_r2_min": float(r2_min),
            "failing_models": failing_models,
            "models": model_reports,
        },
    )
    print(str(report_path))
    return 0 if len(failing_models) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
