from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl
from sow.pcc.build_pcc import validate_pcc_baseline_manifest, validate_pcc_robustness_manifest


def _next_available_path(path: Path) -> Path:
    """
    Return `path` if it doesn't exist; otherwise append a deterministic attempt suffix.
    """
    if not path.exists():
        return path
    suffix = path.suffix
    base = path.name[: -len(suffix)] if suffix else path.name
    for i in range(2, 10_000):
        cand = path.with_name(f"{base}.attempt{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find available attempt path for: {path}")


def _write_jsonl_atomic_new(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write JSONL to a brand new path (refuse to overwrite) using a temp file + atomic rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to overwrite existing tmp file: {tmp}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def read_pcc_example_ids_and_domain_counts(pcc_baseline_path: Path) -> Dict[str, Any]:
    ex_ids: Set[str] = set()
    counts_by_domain: Dict[str, int] = defaultdict(int)
    for r in iter_jsonl(pcc_baseline_path):
        ex = str(r["example_id"])
        if ex in ex_ids:
            raise ValueError(f"duplicate example_id in PCC baseline: {ex}")
        ex_ids.add(ex)
        counts_by_domain[str(r.get("coarse_domain") or "unknown")] += 1
    return {"example_ids": ex_ids, "counts_by_domain": dict(counts_by_domain)}


def compute_ccc_intersection(per_model_example_ids: Dict[str, Set[str]]) -> List[str]:
    if not per_model_example_ids:
        raise ValueError("no models provided")
    it = iter(per_model_example_ids.values())
    inter = set(next(it))
    for s in it:
        inter &= set(s)
    return sorted(inter)


def compute_retention_metrics(
    *,
    ccc_example_ids: Set[str],
    per_model_counts_by_domain: Dict[str, Dict[str, int]],
    domain_by_example: Dict[str, str],
) -> Dict[str, Any]:
    ccc_counts_by_domain: Dict[str, int] = defaultdict(int)
    for ex in ccc_example_ids:
        ccc_counts_by_domain[domain_by_example.get(ex, "unknown")] += 1

    per_model = {}
    for model_name, pcc_counts in per_model_counts_by_domain.items():
        pcc_total = int(sum(pcc_counts.values()))
        ccc_total = int(len(ccc_example_ids))
        overall = (ccc_total / pcc_total) if pcc_total else 0.0

        per_domain = {}
        for domain, pcc_n in sorted(pcc_counts.items(), key=lambda kv: kv[0]):
            pcc_n = int(pcc_n)
            ccc_n = int(ccc_counts_by_domain.get(domain, 0))
            ratio = (ccc_n / pcc_n) if pcc_n else None
            per_domain[domain] = {"pcc": pcc_n, "ccc": ccc_n, "retention": ratio}

        per_model[model_name] = {
            "pcc_total": pcc_total,
            "ccc_total": ccc_total,
            "overall_retention": overall,
            "by_domain": per_domain,
        }

    return {
        "ccc_counts_by_domain": {k: int(v) for k, v in sorted(ccc_counts_by_domain.items(), key=lambda kv: kv[0])},
        "per_model": per_model,
    }


def check_ccc_gates(
    *,
    retention_metrics: Dict[str, Any],
    min_overall: float,
    min_per_domain: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True
    per_model = retention_metrics.get("per_model") or {}
    for model_name, m in sorted(per_model.items(), key=lambda kv: kv[0]):
        overall = float(m.get("overall_retention", 0.0))
        if overall < float(min_overall):
            ok = False
            reasons.append(f"{model_name}: overall_retention {overall:.4f} < min {float(min_overall):.4f}")
        by_domain = m.get("by_domain") or {}
        for domain, d in sorted(by_domain.items(), key=lambda kv: kv[0]):
            r = d.get("retention")
            if r is None:
                continue
            rr = float(r)
            if rr < float(min_per_domain):
                ok = False
                reasons.append(f"{model_name}:{domain}: retention {rr:.4f} < min {float(min_per_domain):.4f}")
    return ok, reasons


def build_ccc_manifests(
    *,
    run_id: str,
    ccc_example_ids: List[str],
    baseline_manifest_path: Path,
    robustness_manifest_path: Path,
    expected_wrapper_ids: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    selected = set(ccc_example_ids)
    if not selected:
        raise ValueError("CCC is empty")

    out_baseline = _next_available_path(out_dir / "ccc_baseline.jsonl")
    out_robust = _next_available_path(out_dir / "ccc_robustness.jsonl")

    baseline_rows = [r for r in iter_jsonl(baseline_manifest_path) if str(r["example_id"]) in selected]
    robust_rows = [r for r in iter_jsonl(robustness_manifest_path) if str(r["example_id"]) in selected]
    baseline_rows.sort(key=lambda r: str(r["prompt_uid"]))
    robust_rows.sort(key=lambda r: str(r["prompt_uid"]))

    _write_jsonl_atomic_new(out_baseline, baseline_rows)
    _write_jsonl_atomic_new(out_robust, robust_rows)

    # Validate.
    validate_pcc_baseline_manifest(baseline_rows, expected_n=len(selected))
    validate_pcc_robustness_manifest(
        robust_rows,
        expected_n_examples=len(selected),
        expected_wrapper_ids=expected_wrapper_ids,
    )

    meta = {
        "run_id": run_id,
        "ccc_size": int(len(selected)),
        "selected_example_ids_sha256": sha256_text("\n".join(sorted(selected)) + "\n"),
        "baseline_manifest_path": str(baseline_manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
        "robustness_manifest_path": str(robustness_manifest_path),
        "robustness_manifest_sha256": sha256_file(robustness_manifest_path),
        "outputs": {
            "ccc_baseline_path": str(out_baseline),
            "ccc_baseline_sha256": sha256_file(out_baseline),
            "ccc_robustness_path": str(out_robust),
            "ccc_robustness_sha256": sha256_file(out_robust),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return meta

