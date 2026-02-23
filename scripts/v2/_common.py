from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sow.io_jsonl import iter_jsonl  # noqa: E402
from sow.token_buckets.option_buckets import model_fs_id  # noqa: E402


def load_experiment_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else (REPO_ROOT / "configs" / "experiment_v2.yaml")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def resolve_models(cfg: Dict[str, Any], *, model_name: str | None) -> List[Dict[str, Any]]:
    models = list(cfg.get("models") or [])
    if model_name is None:
        return models
    filtered = [m for m in models if str(m.get("name")) == str(model_name)]
    if not filtered:
        known = sorted(str(m.get("name")) for m in models)
        raise SystemExit(f"unknown --model-name={model_name}; known={known}")
    return filtered


def run_root_for(run_id: str) -> Path:
    return REPO_ROOT / "runs" / str(run_id)


def run_v2_root_for(run_id: str) -> Path:
    p = run_root_for(run_id) / "v2"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir_for(run_id: str) -> Path:
    return run_root_for(run_id) / "outputs"


def baseline_output_path(run_id: str, model_id: str) -> Path:
    return outputs_dir_for(run_id) / model_fs_id(model_id) / "baseline_outputs.jsonl"


def baseline_manifest_path(run_id: str) -> Path:
    ccc = run_root_for(run_id) / "manifests" / "ccc_baseline.jsonl"
    if ccc.exists():
        return ccc
    return run_root_for(run_id) / "manifests" / "baseline_manifest.jsonl"


def load_manifest_correct_keys(run_id: str) -> Dict[str, str]:
    path = baseline_manifest_path(run_id)
    out: Dict[str, str] = {}
    for row in iter_jsonl(path):
        uid = str(row.get("prompt_uid") or "")
        ck = str(row.get("correct_key") or "").strip().upper()
        if uid and ck:
            out[uid] = ck
    return out


def load_jsonl_rows(path: Path, *, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows = []
    for i, row in enumerate(iter_jsonl(path), start=1):
        rows.append(row)
        if max_rows > 0 and i >= int(max_rows):
            break
    return rows


def _fsync_parent_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    payload = (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_text_atomic(path: Path, text: str) -> None:
    payload = str(text).encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
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


def run_cmd(argv: List[str], *, cwd: Path | None = None) -> None:
    proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_v2.yaml"))
    return ap
