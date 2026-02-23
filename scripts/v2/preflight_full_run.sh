#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${REPO_ROOT}/configs/experiment_v2.yaml"

usage() {
  cat <<EOF
usage: $0 [--config <path>]
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

PYTHON_BIN="/opt/homebrew/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG_PATH}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


repo_root = Path(sys.argv[1]).resolve()
cfg_path = Path(sys.argv[2]).expanduser().resolve()
errors: list[str] = []
checks: dict[str, bool] = {}

if not cfg_path.exists():
    print(f"preflight failure: config missing: {cfg_path}", file=sys.stderr)
    raise SystemExit(2)

try:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
except Exception as exc:
    print(f"preflight failure: invalid yaml in {cfg_path}: {exc}", file=sys.stderr)
    raise SystemExit(2)

for key in ["models", "data_scope", "validators", "execution", "runtime_estimator", "experiment"]:
    ok = key in cfg and cfg.get(key) is not None
    checks[f"config.{key}.present"] = bool(ok)
    if not ok:
        errors.append(f"config missing required section: {key}")

models = list(cfg.get("models") or [])
checks["models.nonempty"] = len(models) > 0
if not models:
    errors.append("models list is empty")

data_scope = cfg.get("data_scope") or {}
manifest_source = Path(str(data_scope.get("baseline_manifest_source") or "")).expanduser()
expected_sha = str(data_scope.get("baseline_manifest_sha256") or "").strip().lower()
expected_rows = int(data_scope.get("baseline_manifest_expected_rows_full") or 0)

checks["manifest.source_exists"] = manifest_source.exists()
if not manifest_source.exists():
    errors.append(f"baseline manifest source missing: {manifest_source}")
else:
    actual_sha = sha256_file(manifest_source)
    checks["manifest.sha_matches"] = bool(expected_sha and actual_sha == expected_sha)
    if not expected_sha:
        errors.append("data_scope.baseline_manifest_sha256 missing")
    elif actual_sha != expected_sha:
        errors.append(
            f"baseline_manifest_sha256 mismatch: expected={expected_sha} got={actual_sha} source={manifest_source}"
        )
    row_count = read_jsonl_count(manifest_source)
    checks["manifest.rows_expected_full"] = expected_rows > 0 and row_count == expected_rows
    if expected_rows <= 0:
        errors.append("data_scope.baseline_manifest_expected_rows_full must be > 0")
    elif row_count != expected_rows:
        errors.append(
            f"baseline manifest row count mismatch: expected={expected_rows} got={row_count} source={manifest_source}"
        )

exp = cfg.get("experiment") or {}
docs = [
    Path(str(exp.get("objective_doc") or "")),
    Path(str(exp.get("implementation_doc") or "")),
    Path(str(exp.get("preregistration_doc") or "")),
    repo_root / "docs" / "MODEL_NUANCES_V2.md",
]
for p in docs:
    ok = p.expanduser().resolve().exists()
    checks[f"doc.{p.name}.exists"] = bool(ok)
    if not ok:
        errors.append(f"required document missing: {p}")

validators = cfg.get("validators") or {}
required_validators = [
    "stage03_trajectory",
    "stage05_paraphrase",
    "stage06_tracing_subset",
    "stage08_decomposition",
    "stage09",
    "stage10",
]
for k in required_validators:
    ok = isinstance(validators.get(k), dict) and len(validators.get(k) or {}) > 0
    checks[f"validators.{k}.present"] = bool(ok)
    if not ok:
        errors.append(f"validator contract missing or empty: validators.{k}")

execution = cfg.get("execution") or {}
for k in [
    "stage00_baseline_checkpoint_every_prompts",
    "stage05_checkpoint_every_prompts",
    "stage07_checkpoint_every_prompts",
]:
    v = int(execution.get(k) or 0)
    checks[f"execution.{k}.positive"] = v > 0
    if v <= 0:
        errors.append(f"execution contract invalid (must be >0): {k}")

runtime_cfg = cfg.get("runtime_estimator") or {}
require_measured = bool(runtime_cfg.get("require_measured_rps_for_full", False))
checks["runtime_estimator.require_measured_rps_for_full"] = bool(require_measured)
if not require_measured:
    errors.append("runtime_estimator.require_measured_rps_for_full must be true for full-mode rigor")

script_map = [
    "scripts/v2/00_run_experiment.py",
    "scripts/v2/00a_generate_baseline_outputs.py",
    "scripts/v2/14_readiness_audit.py",
    "scripts/v2/run_full_local_v2.sh",
]
for rel in script_map:
    p = repo_root / rel
    ok = p.exists()
    checks[f"script.{rel}.exists"] = bool(ok)
    if not ok:
        errors.append(f"required script missing: {p}")

if errors:
    print("preflight: FAIL", file=sys.stderr)
    for e in errors:
        print(f"- {e}", file=sys.stderr)
    print(json.dumps({"pass": False, "errors": errors, "checks": checks}, indent=2, sort_keys=True))
    raise SystemExit(2)

print("preflight: PASS")
print(json.dumps({"pass": True, "checks": checks}, indent=2, sort_keys=True))
raise SystemExit(0)
PY
