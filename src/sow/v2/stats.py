from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def bootstrap_ci(values: Iterable[float], *, n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    rng = np.random.default_rng(int(seed))
    means = np.empty((int(n_bootstrap),), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))

    alpha = 1.0 - float(ci)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - (alpha / 2.0)))
    return {"mean": float(np.mean(arr)), "lo": lo, "hi": hi}


def permutation_test_mean_diff(a: Sequence[float], b: Sequence[float], *, n_permutations: int = 5000, seed: int = 0) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_arr = a_arr[np.isfinite(a_arr)]
    b_arr = b_arr[np.isfinite(b_arr)]
    if a_arr.size == 0 or b_arr.size == 0:
        return 1.0

    obs = float(np.mean(a_arr) - np.mean(b_arr))
    pooled = np.concatenate([a_arr, b_arr])
    n_a = int(a_arr.size)
    rng = np.random.default_rng(int(seed))

    count = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(pooled)
        test = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(test) >= abs(obs):
            count += 1
    return float((count + 1) / (int(n_permutations) + 1))


def benjamini_hochberg(p_values: Sequence[float], *, alpha: float = 0.05) -> List[bool]:
    p = np.asarray(list(p_values), dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    n = p.size
    thresh = (np.arange(1, n + 1, dtype=np.float64) / float(n)) * float(alpha)
    passed = ranked <= thresh
    max_i = np.where(passed)[0]
    keep = np.zeros((n,), dtype=bool)
    if max_i.size > 0:
        keep[: max_i[-1] + 1] = True
    out = np.zeros((n,), dtype=bool)
    out[order] = keep
    return [bool(x) for x in out.tolist()]
