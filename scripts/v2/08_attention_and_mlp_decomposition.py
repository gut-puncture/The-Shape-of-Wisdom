#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import base_parser, load_experiment_config, resolve_models, run_v2_root_for, write_json
from sow.v2.tracing.decomposition import drift_reconstruction_quality


def _split_model_rows(sub: pd.DataFrame, *, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tmp = sub.sort_values(["prompt_uid", "layer_index"]).copy()
    if tmp.empty:
        return tmp, tmp, "prompt_uid"
    frac = min(max(float(train_fraction), 0.05), 0.95)
    prompt_uids = sorted(tmp["prompt_uid"].astype(str).drop_duplicates().tolist()) if "prompt_uid" in tmp.columns else []
    if len(prompt_uids) >= 2:
        n_train_prompts = int(round(float(len(prompt_uids)) * frac))
        n_train_prompts = max(1, min(len(prompt_uids) - 1, n_train_prompts))
        train_uids = set(prompt_uids[:n_train_prompts])
        train = tmp[tmp["prompt_uid"].astype(str).isin(train_uids)].copy()
        test = tmp[~tmp["prompt_uid"].astype(str).isin(train_uids)].copy()
        return train, test, "prompt_uid"

    n = int(tmp.shape[0])
    if n <= 1:
        return tmp.copy(), tmp.copy(), "layer_row_fallback"
    n_train = int(round(float(n) * frac))
    n_train = max(1, min(n - 1, n_train))
    train = tmp.iloc[:n_train].copy()
    test = tmp.iloc[n_train:].copy()
    return train, test, "layer_row_fallback"


def main() -> int:
    ap = base_parser("V2: validate attention+MLP decomposition against drift")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]
    r2_min = float(((cfg.get("causal") or {}).get("drift_decomposition_r2_min", 0.70)))
    validators_cfg = cfg.get("validators") or {}
    stage08_cfg = validators_cfg.get("stage08_decomposition") or {}
    split_train_fraction = float(stage08_cfg.get("split_train_fraction", 0.7))
    min_train_rows = int(stage08_cfg.get("min_train_rows", 0))
    min_test_rows = int(stage08_cfg.get("min_test_rows", 0))
    require_split_r2 = bool(stage08_cfg.get("require_split_r2", False))

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
    split_contract = {
        "split_train_fraction": float(split_train_fraction),
        "min_train_rows": int(min_train_rows),
        "min_test_rows": int(min_test_rows),
        "require_split_r2": bool(require_split_r2),
        "expected_model_ids": expected_model_ids,
        "models": {},
    }
    failing_models = []
    split_train_ok = True
    split_test_ok = True
    split_r2_ok = True
    observed_model_ids = []
    for model_id, sub in df.groupby("model_id", sort=False):
        observed_model_ids.append(str(model_id))
        q = drift_reconstruction_quality(
            observed_drift=sub["drift"].to_numpy(),
            attn_scalar=sub["s_attn"].to_numpy(),
            mlp_scalar=sub["s_mlp"].to_numpy(),
        )
        model_reports[str(model_id)] = q
        if float(q.get("r2", 0.0)) < float(r2_min):
            failing_models.append(str(model_id))

        train_df, test_df, split_unit = _split_model_rows(sub, train_fraction=float(split_train_fraction))
        q_train = drift_reconstruction_quality(
            observed_drift=train_df["drift"].to_numpy(),
            attn_scalar=train_df["s_attn"].to_numpy(),
            mlp_scalar=train_df["s_mlp"].to_numpy(),
        )
        q_test = drift_reconstruction_quality(
            observed_drift=test_df["drift"].to_numpy(),
            attn_scalar=test_df["s_attn"].to_numpy(),
            mlp_scalar=test_df["s_mlp"].to_numpy(),
        )
        train_rows = int(train_df.shape[0])
        test_rows = int(test_df.shape[0])
        split_train_ok = bool(split_train_ok and (train_rows >= int(min_train_rows)))
        split_test_ok = bool(split_test_ok and (test_rows >= int(min_test_rows)))
        if require_split_r2:
            split_r2_ok = bool(split_r2_ok and (float(q_train.get("r2", 0.0)) >= float(r2_min)) and (float(q_test.get("r2", 0.0)) >= float(r2_min)))
        split_contract["models"][str(model_id)] = {
            "split_unit": str(split_unit),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
            "train_prompt_count": int(train_df["prompt_uid"].nunique()) if ("prompt_uid" in train_df.columns and not train_df.empty) else 0,
            "test_prompt_count": int(test_df["prompt_uid"].nunique()) if ("prompt_uid" in test_df.columns and not test_df.empty) else 0,
            "train_r2": float(q_train.get("r2", 0.0)),
            "test_r2": float(q_test.get("r2", 0.0)),
        }

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

    expected_models_present = all(mid in set(observed_model_ids) for mid in expected_model_ids) if expected_model_ids else len(observed_model_ids) > 0
    gates = {
        "expected_models_present": bool(expected_models_present),
        "drift_decomposition_r2": len(failing_models) == 0,
        "split_train_rows_min": bool(split_train_ok),
        "split_test_rows_min": bool(split_test_ok),
        "split_r2_min": bool(split_r2_ok),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    pass_flag = len(failing_gates) == 0

    write_json(
        report_path,
        {
            "pass": bool(pass_flag),
            "drift_decomposition_r2_min": float(r2_min),
            "failing_models": failing_models,
            "models": model_reports,
            "split_contract": split_contract,
            "gates": gates,
            "failing_gates": failing_gates,
        },
    )
    print(str(report_path))
    return 0 if bool(pass_flag) else 2


if __name__ == "__main__":
    raise SystemExit(main())
