from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    REQUIRED_DECISION_COLUMNS,
    REQUIRED_LAYERWISE_COLUMNS,
    REQUIRED_MANIFEST_COLUMNS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return df.copy()


def load_layerwise(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [column for column in REQUIRED_LAYERWISE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"layerwise parquet missing required columns: {missing}")
    return df.copy()


def load_decision_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [column for column in REQUIRED_DECISION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"decision_metrics parquet missing required columns: {missing}")
    return df.copy()


def load_old_core(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path).copy()


def parse_json_column(series: pd.Series[str]) -> list[dict[str, Any]]:
    return [json.loads(value) for value in series.tolist()]


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "matplotlib", "scipy", "pyarrow", "pypdf"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "missing"
    versions["executable"] = sys.executable
    return versions

