from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sow.hashing import sha256_file, sha256_text
from sow.io_jsonl import iter_jsonl


def _stratum_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row["wrapper_id"]), str(row.get("coarse_domain") or "unknown"))


def select_pca_membership(
    *,
    baseline_manifest: Path,
    robustness_manifest: Path,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rows.extend(iter_jsonl(baseline_manifest))
    rows.extend(iter_jsonl(robustness_manifest))

    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        strata.setdefault(_stratum_key(r), []).append(r)

    stratum_keys = sorted(strata.keys())
    n_strata = len(stratum_keys)
    if n_strata == 0:
        raise ValueError("No strata found")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size < n_strata:
        raise ValueError(f"sample_size={sample_size} is smaller than n_strata={n_strata}")

    base = sample_size // n_strata
    rem = sample_size % n_strata

    rng = random.Random(int(seed))
    keys_shuffled = list(stratum_keys)
    rng.shuffle(keys_shuffled)
    extra = set(keys_shuffled[:rem])

    membership: List[Dict[str, Any]] = []
    counts_by_stratum: Dict[str, int] = {}

    for k in stratum_keys:
        bucket = list(strata[k])
        # Deterministic shuffle within stratum:
        # 1) sort by stable key
        # 2) shuffle with per-stratum seed derived from (seed, wrapper_id, coarse_domain)
        bucket.sort(key=lambda r: str(r["prompt_uid"]))
        seed2 = int(sha256_text(f"{seed}|{k[0]}|{k[1]}"), 16)
        rng2 = random.Random(seed2)
        rng2.shuffle(bucket)

        want = base + (1 if k in extra else 0)
        if len(bucket) < want:
            raise ValueError(f"Stratum {k} has only {len(bucket)} rows, need {want}")

        picked = bucket[:want]
        for r in picked:
            membership.append(
                {
                    "prompt_uid": r["prompt_uid"],
                    "prompt_id": r["prompt_id"],
                    "example_id": r["example_id"],
                    "wrapper_id": r["wrapper_id"],
                    "coarse_domain": r.get("coarse_domain") or "unknown",
                }
            )

        counts_by_stratum[f"{k[0]}|{k[1]}"] = int(want)

    # Deterministic ordering for output file.
    membership.sort(key=lambda x: str(x["prompt_uid"]))

    return {
        "seed": int(seed),
        "sample_size": int(sample_size),
        "n_strata": int(n_strata),
        "base_per_stratum": int(base),
        "extra_strata": int(rem),
        "counts_by_stratum": counts_by_stratum,
        "membership": membership,
    }


def write_membership_file(
    *,
    out_path: Path,
    run_id: str,
    model_name: str,
    model_id: str,
    model_revision: str,
    baseline_manifest: Path,
    robustness_manifest: Path,
    membership_obj: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "model_id": model_id,
        "model_revision": model_revision,
        "sampling_policy": "stratified(wrapper_id, coarse_domain) uniform-over-strata",
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "robustness_manifest_sha256": sha256_file(robustness_manifest),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **membership_obj,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

