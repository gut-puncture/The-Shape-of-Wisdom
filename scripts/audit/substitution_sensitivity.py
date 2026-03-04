#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _audit_common import bootstrap_mean_ci, default_paths, write_csv


BASELINE = {
    "pairing_mode": "all_pairs_within_model",
    "normalization_mode": "raw",
    "layer_range_mode": "paper-default",
    "failing_set_mode": "stable_wrong_plus_unstable_wrong",
}


def _summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean, lo, hi = bootstrap_mean_ci(arr, n_boot=2000, seed=12345, alpha=0.05)
    median = float(np.nanmedian(arr))
    frac_pos = float(np.mean(arr > 0))
    return mean, median, frac_pos, lo, hi


def _rows_for_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    baseline: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comps = sorted(df["component"].astype(str).unique().tolist())
    for setting_name in sorted(df[group_col].astype(str).unique().tolist()):
        filt = np.ones((df.shape[0],), dtype=bool)
        for c in ["pairing_mode", "normalization_mode", "layer_range_mode", "failing_set_mode"]:
            if c == group_col:
                continue
            filt &= (df[c].astype(str).to_numpy() == str(baseline[c]))
        filt &= (df[group_col].astype(str).to_numpy() == str(setting_name))
        sub = df[filt]
        for comp in comps:
            x = sub[sub["component"].astype(str) == comp]
            vals = pd.to_numeric(x["delta_shift"], errors="coerce").to_numpy(dtype=np.float64)
            mean, median, frac_pos, lo, hi = _summary(vals)
            rows.append(
                {
                    "setting_group": group_col,
                    "setting_name": str(setting_name),
                    "component": str(comp),
                    "n_pairs": int(x["pair_id"].nunique()),
                    "mean": float(mean),
                    "median": float(median),
                    "frac_positive": float(frac_pos),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "blocked_reason": "",
                }
            )
    return rows


def _blocked_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    components = ["attention", "mlp"]
    blocked_specs = [
        (
            "competitor_selection_mode",
            [
                "dynamic_per_layer",
                "fixed_final_layer_competitor",
                "fixed_target_final_competitor",
                "strongest_noncorrect_aggregated",
            ],
            "cached tracing scalars do not store per-layer logits for alternative competitor recomputation",
        ),
        (
            "option_token_set",
            ["strict_single_token", "inclusive_variant_bucket", "paper_default"],
            "substitution cache only stores scalar traces, not token-level option-logit decompositions",
        ),
    ]
    for group, names, reason in blocked_specs:
        for name in names:
            for comp in components:
                rows.append(
                    {
                        "setting_group": group,
                        "setting_name": name,
                        "component": comp,
                        "n_pairs": 0,
                        "mean": 0.0,
                        "median": 0.0,
                        "frac_positive": 0.0,
                        "ci_lo": 0.0,
                        "ci_hi": 0.0,
                        "blocked_reason": reason,
                    }
                )
    return rows


def main() -> int:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Compute substitution sensitivity summaries from per-pair cached results.")
    ap.add_argument("--pairs-csv", type=Path, default=paths.audit / "substitution_pairs_vnext.csv")
    ap.add_argument("--out-csv", type=Path, default=paths.audit / "substitution_sensitivity_summary.csv")
    args = ap.parse_args()

    if not args.pairs_csv.exists():
        raise SystemExit(f"missing substitution pairs csv: {args.pairs_csv}")
    df = pd.read_csv(args.pairs_csv)
    if df.empty:
        raise SystemExit("substitution pairs csv is empty")

    rows: list[dict[str, Any]] = []
    for group_col in ["pairing_mode", "layer_range_mode", "normalization_mode", "failing_set_mode"]:
        rows.extend(_rows_for_group(df, group_col=group_col, baseline=BASELINE))
    rows.extend(_blocked_rows())

    out = pd.DataFrame.from_records(rows)
    write_csv(args.out_csv, out)
    print(str(args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
