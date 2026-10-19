from __future__ import annotations

from collections import Counter
from typing import Iterable

from sklearn.model_selection import GroupKFold


def assert_no_group_leakage(train_rows: Iterable[dict], test_rows: Iterable[dict], *, group_field: str) -> None:
    train_groups = {row[group_field] for row in train_rows}
    test_groups = {row[group_field] for row in test_rows}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(f"group leakage detected for {group_field}: {overlap[:10]}")


def make_grouped_splits(rows: list[dict], *, group_field: str, n_splits: int = 3) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not rows:
        raise ValueError("rows must not be empty")
    groups = [row[group_field] for row in rows]
    unique = Counter(groups)
    if len(unique) < n_splits:
        raise ValueError(f"need at least {n_splits} unique groups, got {len(unique)}")
    splitter = GroupKFold(n_splits=n_splits)
    indices = list(range(len(rows)))
    out: list[tuple[list[int], list[int]]] = []
    for train_idx, test_idx in splitter.split(indices, groups=groups):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        assert_no_group_leakage(train_rows, test_rows, group_field=group_field)
        out.append((list(train_idx), list(test_idx)))
    return out

