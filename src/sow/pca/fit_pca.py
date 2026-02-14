from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sow.hashing import sha256_file
from sow.token_buckets.option_buckets import model_fs_id


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def _write_json_atomic_new(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_npz_atomic_new(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    """
    Fix PCA sign ambiguity deterministically:
    for each component vector, ensure the entry with largest absolute value is positive.
    """
    if components.ndim != 2:
        raise ValueError("components must be 2D")
    comps = components.copy()
    for i in range(comps.shape[0]):
        v = comps[i]
        j = int(np.argmax(np.abs(v)))
        if float(v[j]) < 0.0:
            comps[i] = -v
    return comps


def _basis_hash(*, mean: np.ndarray, components: np.ndarray) -> str:
    """
    Hash the PCA basis in a stable, explicit way.
    """
    h = hashlib.sha256()
    h.update(mean.astype(np.float32).tobytes(order="C"))
    h.update(components.astype(np.float32).tobytes(order="C"))
    return h.hexdigest()


def fit_pca_basis_from_hidden(
    *,
    hidden_npz: Path,
    n_components: int,
    seed: int,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit a PCA basis to pooled hidden vectors across layers.

    Input hidden format: npz with key "hidden" of shape (n_prompts, n_layers, hidden_dim).
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    import sklearn  # noqa: PLC0415

    if n_components <= 0:
        raise ValueError("n_components must be positive")

    with np.load(hidden_npz) as z:
        if "hidden" not in z:
            raise ValueError(f"hidden_npz missing 'hidden' array: {hidden_npz}")
        hidden = z["hidden"]

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be 3D (n_prompts, n_layers, hidden_dim); got shape {hidden.shape}")

    n_prompts, n_layers, hidden_dim = (int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2]))
    x = hidden.reshape(n_prompts * n_layers, hidden_dim).astype(np.float32, copy=False)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        x = x[: int(max_rows), :]

    if n_components > min(int(x.shape[0]), int(x.shape[1])):
        raise ValueError("n_components exceeds min(n_samples, n_features) for PCA fit")

    pca = PCA(n_components=int(n_components), svd_solver="randomized", random_state=int(seed))
    t0 = time.perf_counter()
    pca.fit(x)
    t1 = time.perf_counter()

    mean = pca.mean_.astype(np.float32, copy=True)
    components = pca.components_.astype(np.float32, copy=True)
    components = _canonicalize_component_signs(components)
    basis_hash = _basis_hash(mean=mean, components=components)

    evr = getattr(pca, "explained_variance_ratio_", None)
    if evr is None:
        raise RuntimeError("PCA missing explained_variance_ratio_")
    evr = np.asarray(evr, dtype=np.float64)
    if evr.shape != (int(n_components),):
        raise RuntimeError("unexpected explained_variance_ratio_ shape")

    return {
        "mean": mean,
        "components": components,
        "explained_variance_ratio": evr.astype(np.float64),
        "basis_hash": basis_hash,
        "fit_meta": {
            "sklearn_version": str(sklearn.__version__),
            "svd_solver": "randomized",
            "random_state": int(seed),
            "n_components": int(n_components),
            "n_prompts": int(n_prompts),
            "n_layers": int(n_layers),
            "hidden_dim": int(hidden_dim),
            "pooled_rows": int(x.shape[0]),
            "fit_seconds": float(t1 - t0),
        },
    }


def run_stage12_fit_for_model(
    *,
    run_id: str,
    run_dir: Path,
    model: Dict[str, Any],
    hidden_npz: Path,
    hidden_meta: Path,
    n_components: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Stage 12: fit PCA basis once per model and validate reproducibility (fit twice).
    """
    model_id = str(model["model_id"])
    revision = str(model["revision"])

    if not hidden_npz.exists():
        raise FileNotFoundError(f"missing hidden_npz: {hidden_npz}")
    if not hidden_meta.exists():
        raise FileNotFoundError(f"missing hidden_meta: {hidden_meta}")

    t0 = time.perf_counter()
    fit1 = fit_pca_basis_from_hidden(hidden_npz=hidden_npz, n_components=int(n_components), seed=int(seed))
    fit2 = fit_pca_basis_from_hidden(hidden_npz=hidden_npz, n_components=int(n_components), seed=int(seed))
    same = bool(fit1["basis_hash"] == fit2["basis_hash"])
    t1 = time.perf_counter()

    out_dir = run_dir / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = _next_available_path(out_dir / f"{model_fs_id(model_id)}_pca_basis.npz")
    out_meta = _next_available_path(out_dir / f"{model_fs_id(model_id)}_pca_basis.meta.json")

    _write_npz_atomic_new(
        out_npz,
        mean=fit1["mean"],
        components=fit1["components"],
        explained_variance_ratio=fit1["explained_variance_ratio"],
    )

    meta = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "input_hidden_path": str(hidden_npz),
        "input_hidden_sha256": sha256_file(hidden_npz),
        "input_hidden_meta_path": str(hidden_meta),
        "input_hidden_meta_sha256": sha256_file(hidden_meta),
        "n_components": int(n_components),
        "seed": int(seed),
        "basis_hash": fit1["basis_hash"],
        "fit_meta": fit1["fit_meta"],
        "reproducibility": {"pass": bool(same), "basis_hash_repeat_match": bool(same)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic_new(out_meta, meta)

    report = {
        "pass": bool(same),
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "n_components": int(n_components),
        "basis_hash": fit1["basis_hash"],
        "basis_sha256": sha256_file(out_npz),
        "meta_sha256": sha256_file(out_meta),
        "fit_seconds_total": float(t1 - t0),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "report": report,
        "basis_path": out_npz,
        "basis_sha256": sha256_file(out_npz),
        "meta_path": out_meta,
        "meta_sha256": sha256_file(out_meta),
    }

