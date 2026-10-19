from __future__ import annotations

import pytest

from stats_grouped_cv import make_grouped_splits


def test_grouped_splits_have_no_leakage() -> None:
    rows = [
        {"example_id": f"e{i}", "group_id": f"g{i // 2}"}
        for i in range(8)
    ]
    splits = make_grouped_splits(rows, group_field="group_id", n_splits=2)
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        train_groups = {rows[i]["group_id"] for i in train_idx}
        test_groups = {rows[i]["group_id"] for i in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_grouped_splits_require_enough_groups() -> None:
    rows = [{"group_id": "g0"}, {"group_id": "g0"}]
    with pytest.raises(ValueError):
        make_grouped_splits(rows, group_field="group_id", n_splits=2)
