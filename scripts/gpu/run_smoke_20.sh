#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/preflight.sh"
sow_preflight

mkdir -p "${SOW_RUNS_ROOT}"

echo "[gpu] runs_root=${SOW_RUNS_ROOT}"
echo "[gpu] run_id=${RUN_ID}"

python3 sow.py init-run --run-id "${RUN_ID}" --seed 12345
python3 sow.py stage0 --run-id "${RUN_ID}" --device cuda
python3 sow.py build-manifests --run-id "${RUN_ID}"
python3 sow.py parser-regression --run-id "${RUN_ID}"
python3 sow.py token-buckets --run-id "${RUN_ID}"
python3 sow.py pilot-inference --run-id "${RUN_ID}" --device cuda --sample-size 200 --min-one-token-compliance 0.8 --min-parser-resolved 0.9
python3 sow.py build-pcc --run-id "${RUN_ID}" --target-size 3000
python3 sow.py build-ccc --run-id "${RUN_ID}" --min-overall-retention 0.80 --min-per-domain-retention 0.60
python3 sow.py pca-membership --run-id "${RUN_ID}"
python3 sow.py pca-sample-inference --run-id "${RUN_ID}" --device cuda --batch-size auto --repro-check-k 8 --repro-atol 0.001
python3 sow.py pca-fit --run-id "${RUN_ID}"

# Stage 13 gates + smoke (20 prompts).
python3 sow.py stage13-smoke --run-id "${RUN_ID}" --device cuda --sample-size 20 --batch-size auto

echo "[gpu] smoke complete: run_id=${RUN_ID}"
