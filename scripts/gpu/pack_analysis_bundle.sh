#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 2
fi

if [[ -z "${SOW_RUNS_ROOT:-}" ]]; then
  echo "error: SOW_RUNS_ROOT must be set" >&2
  exit 2
fi

RUN_DIR="${SOW_RUNS_ROOT%/}/${RUN_ID}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "error: run dir not found: ${RUN_DIR}" >&2
  exit 2
fi

OUT_DIR="${RUN_DIR}/bundles"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/analysis_bundle_${RUN_ID}.tar.gz"

echo "[gpu] packing analysis bundle: ${OUT}"

tar -czf "${OUT}" -C "${RUN_DIR}" \
  --exclude='outputs/*' \
  --exclude='manifests/*.jsonl' \
  analysis validation meta sentinels run_config.yaml manifests

echo "[gpu] bundle sha256:"
python3 - <<PY
import hashlib, pathlib
p = pathlib.Path("${OUT}")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024*1024), b""):
        h.update(chunk)
print(h.hexdigest(), p)
PY

