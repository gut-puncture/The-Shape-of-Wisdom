from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


def _fsync_parent_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def stable_json_dumps(obj: Any, *, indent: int | None = 2) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=True)
    return text + ("\n" if not text.endswith("\n") else "")


def write_text_atomic(path: Path, text: str) -> None:
    atomic_write_bytes(path, str(text).encode("utf-8"))


def write_json_atomic(path: Path, obj: Any) -> None:
    write_text_atomic(path, stable_json_dumps(obj))


def write_jsonl_atomic(path: Path, rows: Iterable[Any]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    atomic_write_bytes(path, payload)


def write_parquet_atomic(path: Path, dataframe: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        dataframe.to_parquet(tmp_path, index=False)
        with tmp_path.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
