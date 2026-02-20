from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    component: str
    target_layers: Sequence[int]


def _simulate_counterfactual_delta(delta0: float, drift: np.ndarray) -> np.ndarray:
    out = np.zeros((drift.size + 1,), dtype=np.float64)
    out[0] = float(delta0)
    for i in range(drift.size):
        out[i + 1] = out[i] + float(drift[i])
    return out


def run_component_ablation(df: pd.DataFrame, *, component: str, target_layers: Sequence[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_id", "prompt_uid", "component", "delta_final_base", "delta_final_ablate", "delta_shift"])

    comp_col = "s_attn" if component == "attention" else "s_mlp"
    target = set(int(x) for x in target_layers)

    rows = []
    for (model_id, prompt_uid), g in df.sort_values("layer_index").groupby(["model_id", "prompt_uid"], sort=False):
        drift = g["drift"].to_numpy(dtype=np.float64)
        layer_idx = g["layer_index"].to_numpy(dtype=np.int64)
        comp = g[comp_col].to_numpy(dtype=np.float64)
        delta = g["delta"].to_numpy(dtype=np.float64)
        if delta.size == 0:
            continue

        drift_cf = drift.copy()
        for i, li in enumerate(layer_idx):
            if int(li) in target and i < drift_cf.size:
                drift_cf[i] = drift_cf[i] - float(comp[i])

        base_final = float(delta[-1])
        cf_final = float(_simulate_counterfactual_delta(float(delta[0]), drift_cf)[-1])
        rows.append(
            {
                "model_id": str(model_id),
                "prompt_uid": str(prompt_uid),
                "component": str(component),
                "target_layers": ",".join(str(x) for x in sorted(target)),
                "delta_final_base": base_final,
                "delta_final_ablate": cf_final,
                "delta_shift": cf_final - base_final,
            }
        )
    return pd.DataFrame.from_records(rows)
