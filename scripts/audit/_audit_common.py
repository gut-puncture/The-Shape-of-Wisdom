#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class Paths:
    repo: Path
    results: Path
    parquet: Path
    reports: Path
    audit: Path
    figures_vnext: Path
    paper_dir: Path
    paper_v3_tex: Path
    paper_v3_pdf: Path


def default_paths() -> Paths:
    repo = REPO_ROOT
    results = repo / "results"
    paper_dir = repo / "paper" / "final_paper"
    return Paths(
        repo=repo,
        results=results,
        parquet=results / "parquet",
        reports=results / "reports",
        audit=results / "audit",
        figures_vnext=repo / "figures_vNext",
        paper_dir=paper_dir,
        paper_v3_tex=paper_dir / "paper_publish_v3.tex",
        paper_v3_pdf=paper_dir / "paper_publish_v3.pdf",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fail(msg: str) -> None:
    raise SystemExit(msg)


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        fail("missing required artifact(s): " + ", ".join(missing))


def read_parquet_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"missing required parquet: {path}")
    return pd.read_parquet(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path | None = None) -> str | None:
    root = str(repo_root or REPO_ROOT)
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        return out or None
    except Exception:
        return None


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    means = np.empty((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    lo_q = 100.0 * (float(alpha) / 2.0)
    hi_q = 100.0 * (1.0 - float(alpha) / 2.0)
    lo, hi = np.percentile(means, [lo_q, hi_q]).tolist()
    return float(np.mean(arr)), float(lo), float(hi)


def model_layer_counts_from_frame(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    out: dict[str, int] = {}
    for mid, g in df.groupby("model_id", sort=False):
        try:
            out[str(mid)] = int(pd.to_numeric(g["layer_index"], errors="coerce").max()) + 1
        except Exception:
            out[str(mid)] = 0
    return out


def bool_env(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}

