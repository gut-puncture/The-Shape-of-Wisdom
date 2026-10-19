from __future__ import annotations

from pathlib import Path

import pandas as pd

from io_utils import write_json_atomic, write_parquet_atomic, write_text_atomic


def test_write_text_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    write_text_atomic(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_write_json_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"b": 2, "a": 1})
    text = path.read_text(encoding="utf-8")
    assert '"a": 1' in text
    assert '"b": 2' in text


def test_write_parquet_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    df = pd.DataFrame.from_records([{"x": 1}, {"x": 2}])
    write_parquet_atomic(path, df)
    loaded = pd.read_parquet(path)
    assert loaded["x"].tolist() == [1, 2]

