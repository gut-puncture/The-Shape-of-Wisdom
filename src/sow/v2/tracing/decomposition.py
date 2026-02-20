from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


def decision_direction_from_logits(logits: Mapping[str, float], *, correct_key: str) -> np.ndarray:
    keys = ["A", "B", "C", "D"]
    values = np.asarray([float(logits.get(k, 0.0)) for k in keys], dtype=np.float64)
    ck = str(correct_key).strip().upper()
    if ck not in keys:
        ck = "A"
    ci = keys.index(ck)
    comp_idx = int(np.argmax([values[i] if i != ci else -1e18 for i in range(4)]))
    d = np.zeros((4,), dtype=np.float64)
    d[ci] = 1.0
    d[comp_idx] = -1.0
    return d


def _to_numpy_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def _to_numpy_attn(x: Any) -> np.ndarray:
    try:
        import torch  # noqa: PLC0415

        if torch.is_tensor(x):
            return x.detach().to("cpu").numpy()
    except Exception:
        pass
    return np.asarray(x)


def component_scalar(component_update: np.ndarray, *, decision_direction: np.ndarray | None = None) -> float:
    arr = _to_numpy_1d(component_update)
    if decision_direction is None:
        return float(np.linalg.norm(arr))
    d = _to_numpy_1d(decision_direction)
    n = min(arr.size, d.size)
    if n <= 0:
        return 0.0
    return float(np.dot(arr[:n], d[:n]))


def attention_mass_by_span_per_layer(attentions: Any, *, span_token_indices: Mapping[str, Iterable[int]]) -> list[Dict[str, float]]:
    """
    Compute attention mass from answer position to each span, per layer.
    """
    if attentions is None:
        return []

    masses_per_layer: list[Dict[str, float]] = []
    for layer_attn in attentions:
        arr = _to_numpy_attn(layer_attn)
        if arr.ndim != 4:
            continue
        # first batch, mean over heads, last query token
        q_mean = arr[0, :, -1, :].mean(axis=0)
        layer_mass: Dict[str, float] = {}
        for label, idxs in span_token_indices.items():
            idx = [int(i) for i in idxs if 0 <= int(i) < q_mean.shape[0]]
            layer_mass[str(label)] = float(np.sum(q_mean[idx])) if idx else 0.0
        masses_per_layer.append(layer_mass)
    return masses_per_layer


def attention_mass_by_span(attentions: Any, *, span_token_indices: Mapping[str, Iterable[int]]) -> Dict[str, float]:
    """Compute attention mass from answer position to each span."""
    masses_per_layer = attention_mass_by_span_per_layer(attentions, span_token_indices=span_token_indices)
    masses = {str(k): 0.0 for k in span_token_indices.keys()}
    if not masses_per_layer:
        return masses
    for layer_mass in masses_per_layer:
        for k, v in layer_mass.items():
            masses[k] = float(masses.get(k, 0.0) + float(v))
    total_layers = float(len(masses_per_layer))
    return {k: float(v / total_layers) for k, v in masses.items()}


def drift_series_from_deltas(delta_by_layer: np.ndarray) -> np.ndarray:
    """
    Given per-layer deltas on transformer-layer outputs, return drift g(l)=delta(l+1)-delta(l)
    aligned to the same layer index convention with a terminal 0.0 entry.
    """
    d = np.asarray(delta_by_layer, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return np.asarray([], dtype=np.float64)
    out = np.zeros_like(d, dtype=np.float64)
    if d.size > 1:
        out[:-1] = d[1:] - d[:-1]
    out[-1] = 0.0
    return out


def drift_reconstruction_quality(*, observed_drift: np.ndarray, attn_scalar: np.ndarray, mlp_scalar: np.ndarray) -> Dict[str, float]:
    y = np.asarray(observed_drift, dtype=np.float64).reshape(-1)
    a = np.asarray(attn_scalar, dtype=np.float64).reshape(-1)
    m = np.asarray(mlp_scalar, dtype=np.float64).reshape(-1)
    if not (y.size == a.size == m.size):
        raise ValueError(
            "observed_drift, attn_scalar, and mlp_scalar must have equal length "
            f"(got {y.size}, {a.size}, {m.size})"
        )
    n = y.size
    if n == 0:
        return {"r2": 0.0, "coef_attn": 0.0, "coef_mlp": 0.0, "intercept": 0.0}

    y = y[:n]
    x = np.stack([a[:n], m[:n], np.ones((n,), dtype=np.float64)], axis=1)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    if ss_tot > 0:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        # Constant-series edge case: exact reconstruction is a perfect fit.
        r2 = 1.0 if ss_res <= 1e-18 else 0.0
    return {
        "r2": float(r2),
        "coef_attn": float(beta[0]),
        "coef_mlp": float(beta[1]),
        "intercept": float(beta[2]),
    }
