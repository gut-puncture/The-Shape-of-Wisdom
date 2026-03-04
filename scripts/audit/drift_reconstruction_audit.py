#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _audit_common import default_paths, ensure_dir, read_parquet_required, write_csv, write_json


def _fit_ols_coefficients(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mid, g in df.groupby("model_id", sort=False):
        y = pd.to_numeric(g["drift"], errors="coerce").to_numpy(dtype=np.float64)
        a = pd.to_numeric(g["s_attn"], errors="coerce").to_numpy(dtype=np.float64)
        m = pd.to_numeric(g["s_mlp"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(m)
        if mask.sum() < 3:
            out[str(mid)] = {"coef_attn": 1.0, "coef_mlp": 1.0, "intercept": 0.0}
            continue
        x = np.stack([a[mask], m[mask], np.ones(mask.sum(), dtype=np.float64)], axis=1)
        beta, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
        out[str(mid)] = {
            "coef_attn": float(beta[0]),
            "coef_mlp": float(beta[1]),
            "intercept": float(beta[2]),
        }
    return out


def _reconstruct_delta(delta: np.ndarray, drift_hat: np.ndarray) -> np.ndarray:
    n = min(delta.size, drift_hat.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    out = np.zeros((n,), dtype=np.float64)
    out[0] = float(delta[0])
    for i in range(1, n):
        out[i] = out[i - 1] + float(drift_hat[i - 1])
    return out


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Audit linearized drift-model fidelity against cached trajectories.")
    ap.add_argument("--parquet-dir", type=Path, default=paths.parquet)
    ap.add_argument("--out-dir", type=Path, default=paths.audit)
    ap.add_argument("--figures-dir", type=Path, default=paths.figures_vnext)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.figures_dir)
    tracing = read_parquet_required(args.parquet_dir / "tracing_scalars.parquet")
    prompt_types = read_parquet_required(args.parquet_dir / "prompt_types.parquet")
    merged = tracing.merge(
        prompt_types[["model_id", "prompt_uid", "trajectory_type"]],
        on=["model_id", "prompt_uid"],
        how="left",
    ).sort_values(["model_id", "prompt_uid", "layer_index"], kind="stable")
    if merged.empty:
        raise SystemExit("tracing data is empty")

    ols_by_model = _fit_ols_coefficients(merged)

    rows: list[dict[str, Any]] = []
    for (mid, puid), g in merged.groupby(["model_id", "prompt_uid"], sort=False):
        model_id = str(mid)
        layer = pd.to_numeric(g["layer_index"], errors="coerce").to_numpy(dtype=np.int64)
        delta = pd.to_numeric(g["delta"], errors="coerce").to_numpy(dtype=np.float64)
        drift = pd.to_numeric(g["drift"], errors="coerce").to_numpy(dtype=np.float64)
        s_attn = pd.to_numeric(g["s_attn"], errors="coerce").to_numpy(dtype=np.float64)
        s_mlp = pd.to_numeric(g["s_mlp"], errors="coerce").to_numpy(dtype=np.float64)
        comp = g["competitor_key"].astype(str).to_numpy()
        n = min(layer.size, delta.size, drift.size, s_attn.size, s_mlp.size, comp.size)
        if n <= 1:
            continue
        layer = layer[:n]
        delta = delta[:n]
        drift = drift[:n]
        s_attn = s_attn[:n]
        s_mlp = s_mlp[:n]
        comp = comp[:n]
        coeff = ols_by_model.get(model_id, {"coef_attn": 1.0, "coef_mlp": 1.0, "intercept": 0.0})
        drift_unit = s_attn + s_mlp
        drift_ols = coeff["intercept"] + coeff["coef_attn"] * s_attn + coeff["coef_mlp"] * s_mlp
        delta_hat_unit = _reconstruct_delta(delta, drift_unit)
        delta_hat_ols = _reconstruct_delta(delta, drift_ols)
        switch = np.zeros((n,), dtype=np.int64)
        switch[1:] = (comp[1:] != comp[:-1]).astype(np.int64)
        ttype = str(g["trajectory_type"].iloc[0])
        for i in range(n):
            rows.append(
                {
                    "model_id": model_id,
                    "prompt_uid": str(puid),
                    "trajectory_type": ttype,
                    "layer_index": int(layer[i]),
                    "delta_true": float(delta[i]),
                    "delta_hat_unit": float(delta_hat_unit[i]),
                    "delta_hat_ols": float(delta_hat_ols[i]),
                    "err_unit": float(delta_hat_unit[i] - delta[i]),
                    "err_ols": float(delta_hat_ols[i] - delta[i]),
                    "abs_err_unit": float(abs(delta_hat_unit[i] - delta[i])),
                    "abs_err_ols": float(abs(delta_hat_ols[i] - delta[i])),
                    "competitor_switch": int(switch[i]),
                }
            )

    detail = pd.DataFrame.from_records(rows)
    if detail.empty:
        raise SystemExit("reconstruction detail is empty")
    detail_csv = args.out_dir / "drift_reconstruction_detail.csv"
    write_csv(detail_csv, detail)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < 3:
            return 0.0
        return float(np.corrcoef(aa[m], bb[m])[0, 1])

    overall = {
        "corr_unit": _corr(detail["delta_true"], detail["delta_hat_unit"]),
        "corr_ols": _corr(detail["delta_true"], detail["delta_hat_ols"]),
        "mae_unit": float(np.nanmean(detail["abs_err_unit"])),
        "mae_ols": float(np.nanmean(detail["abs_err_ols"])),
    }

    by_layer = (
        detail.groupby("layer_index", as_index=False)
        .agg(
            mae_unit=("abs_err_unit", "mean"),
            mae_ols=("abs_err_ols", "mean"),
            p90_unit=("abs_err_unit", lambda x: float(np.nanpercentile(x, 90))),
            p90_ols=("abs_err_ols", lambda x: float(np.nanpercentile(x, 90))),
        )
        .sort_values("layer_index")
    )
    by_type = (
        detail.groupby("trajectory_type", as_index=False)
        .agg(
            mae_unit=("abs_err_unit", "mean"),
            mae_ols=("abs_err_ols", "mean"),
            corr_unit=("delta_true", lambda x: 0.0),  # filled below
            corr_ols=("delta_true", lambda x: 0.0),
            n=("prompt_uid", "count"),
        )
        .sort_values("trajectory_type")
    )
    for i, row in by_type.iterrows():
        t = str(row["trajectory_type"])
        sub = detail[detail["trajectory_type"] == t]
        by_type.loc[i, "corr_unit"] = _corr(sub["delta_true"], sub["delta_hat_unit"])
        by_type.loc[i, "corr_ols"] = _corr(sub["delta_true"], sub["delta_hat_ols"])

    switch_agg = (
        detail.groupby("competitor_switch", as_index=False)
        .agg(mae_unit=("abs_err_unit", "mean"), mae_ols=("abs_err_ols", "mean"), n=("prompt_uid", "count"))
        .sort_values("competitor_switch")
    )

    # Figure 1: error vs layer.
    fig1, ax1 = plt.subplots(figsize=(8.2, 4.2))
    ax1.plot(by_layer["layer_index"], by_layer["mae_unit"], color="#1f77b4", lw=2.2, label="Unit-coefficient reconstruction")
    ax1.plot(by_layer["layer_index"], by_layer["mae_ols"], color="#d62728", lw=2.2, label="OLS-coefficient reconstruction")
    ax1.fill_between(
        by_layer["layer_index"],
        by_layer["mae_ols"],
        by_layer["p90_ols"],
        color="#d62728",
        alpha=0.14,
        linewidth=0,
    )
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Mean absolute reconstruction error (|Δδ| logits)")
    ax1.set_title("Drift-model reconstruction error vs layer")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    fig1_png = args.figures_dir / "drift_reconstruction_error_vs_layer.png"
    fig1_pdf = args.figures_dir / "drift_reconstruction_error_vs_layer.pdf"
    fig1.savefig(fig1_png, dpi=220)
    fig1.savefig(fig1_pdf)
    plt.close(fig1)

    # Figure 2: error distribution by trajectory type.
    order = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
    labels = [x for x in order if x in set(detail["trajectory_type"].astype(str))]
    data_unit = [detail.loc[detail["trajectory_type"] == x, "abs_err_unit"].to_numpy(dtype=np.float64) for x in labels]
    data_ols = [detail.loc[detail["trajectory_type"] == x, "abs_err_ols"].to_numpy(dtype=np.float64) for x in labels]
    x = np.arange(len(labels), dtype=np.float64)
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.4))
    b1 = ax2.boxplot(data_unit, positions=x - 0.17, widths=0.3, patch_artist=True, showfliers=False)
    b2 = ax2.boxplot(data_ols, positions=x + 0.17, widths=0.3, patch_artist=True, showfliers=False)
    for patch in b1["boxes"]:
        patch.set_facecolor("#6baed6")
        patch.set_alpha(0.75)
    for patch in b2["boxes"]:
        patch.set_facecolor("#fb6a4a")
        patch.set_alpha(0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "-") for s in labels], rotation=15)
    ax2.set_ylabel("Absolute reconstruction error (|Δδ| logits)")
    ax2.set_title("Reconstruction error by trajectory type")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(
        [b1["boxes"][0], b2["boxes"][0]],
        ["Unit-coefficient", "OLS-coefficient"],
        frameon=False,
        loc="upper right",
    )
    fig2.tight_layout()
    fig2_png = args.figures_dir / "drift_reconstruction_error_by_type.png"
    fig2_pdf = args.figures_dir / "drift_reconstruction_error_by_type.pdf"
    fig2.savefig(fig2_png, dpi=220)
    fig2.savefig(fig2_pdf)
    plt.close(fig2)

    write_csv(args.out_dir / "drift_reconstruction_by_layer.csv", by_layer)
    write_csv(args.out_dir / "drift_reconstruction_by_type.csv", by_type)
    write_csv(args.out_dir / "drift_reconstruction_by_competitor_switch.csv", switch_agg)

    payload = {
        "overall": overall,
        "ols_coefficients_by_model": ols_by_model,
        "by_layer_csv": str(args.out_dir / "drift_reconstruction_by_layer.csv"),
        "by_type_csv": str(args.out_dir / "drift_reconstruction_by_type.csv"),
        "by_competitor_switch_csv": str(args.out_dir / "drift_reconstruction_by_competitor_switch.csv"),
        "detail_csv": str(detail_csv),
        "figures": [
            str(fig1_png),
            str(fig1_pdf),
            str(fig2_png),
            str(fig2_pdf),
        ],
    }
    out_json = args.out_dir / "drift_reconstruction_audit.json"
    write_json(out_json, payload)
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

