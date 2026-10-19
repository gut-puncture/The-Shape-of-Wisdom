from __future__ import annotations

import random
from statistics import mean, median
from typing import Callable


def grouped_bootstrap(
    rows: list[dict],
    *,
    group_field: str,
    value_fn: Callable[[list[dict]], float],
    iterations: int = 200,
    seed: int = 12345,
) -> dict[str, float]:
    groups = {}
    for row in rows:
        groups.setdefault(row[group_field], []).append(row)
    keys = sorted(groups)
    if not keys:
        raise ValueError("rows must not be empty")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled_rows = []
        for _ in keys:
            key = rng.choice(keys)
            sampled_rows.extend(groups[key])
        estimates.append(float(value_fn(sampled_rows)))
    estimates.sort()
    lo_idx = max(0, int(0.025 * len(estimates)) - 1)
    hi_idx = min(len(estimates) - 1, int(0.975 * len(estimates)) - 1)
    return {
        "estimate_mean": mean(estimates),
        "estimate_median": median(estimates),
        "ci_lo": estimates[lo_idx],
        "ci_hi": estimates[hi_idx],
        "iterations": iterations,
    }

