from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def run_activation_patching(
    failing_df: pd.DataFrame,
    success_df: pd.DataFrame,
    *,
    component: str,
    target_layers: Sequence[int],
) -> pd.DataFrame:
    if failing_df.empty or success_df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "prompt_uid_fail",
                "prompt_uid_success",
                "component",
                "delta_final_base",
                "delta_final_patched",
                "delta_shift",
            ]
        )

    comp_col = "s_attn" if component == "attention" else "s_mlp"
    target = set(int(x) for x in target_layers)

    successes = {
        (str(mid), str(uid)): g.sort_values("layer_index")
        for (mid, uid), g in success_df.groupby(["model_id", "prompt_uid"], sort=False)
    }

    rows = []
    for (mid, uid), fail_g in failing_df.groupby(["model_id", "prompt_uid"], sort=False):
        fail = fail_g.sort_values("layer_index")
        success_key = next((k for k in successes.keys() if k[0] == str(mid)), None)
        if success_key is None:
            continue
        succ = successes[success_key]

        fail_delta = fail["delta"].to_numpy(dtype=np.float64)
        fail_drift = fail["drift"].to_numpy(dtype=np.float64)
        fail_comp = fail[comp_col].to_numpy(dtype=np.float64)
        fail_li = fail["layer_index"].to_numpy(dtype=np.int64)
        succ_comp = succ[comp_col].to_numpy(dtype=np.float64)

        n = min(fail_drift.size, fail_comp.size, succ_comp.size, fail_li.size)
        if n == 0:
            continue

        patched = fail_drift[:n].copy()
        for i in range(n):
            if int(fail_li[i]) in target:
                # Remove failing component contribution, add success component contribution
                patched[i] = fail_drift[i] - fail_comp[i] + succ_comp[i]

        start = float(fail_delta[0])
        trace = np.zeros((n + 1,), dtype=np.float64)
        trace[0] = start
        for i in range(n):
            trace[i + 1] = trace[i] + patched[i]

        rows.append(
            {
                "model_id": str(mid),
                "prompt_uid_fail": str(uid),
                "prompt_uid_success": str(success_key[1]),
                "component": str(component),
                "target_layers": ",".join(str(x) for x in sorted(target)),
                "delta_final_base": float(fail_delta[min(fail_delta.size, n) - 1]),
                "delta_final_patched": float(trace[-1]),
                "delta_shift": float(trace[-1] - fail_delta[min(fail_delta.size, n) - 1]),
            }
        )

    return pd.DataFrame.from_records(rows)
