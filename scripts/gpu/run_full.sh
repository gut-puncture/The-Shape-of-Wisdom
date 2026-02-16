#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
MODE="${2:-full}"
if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id> [full|baseline_only]" >&2
  echo "note: run scripts/gpu/run_smoke_20.sh first (Stage 0..12 + Stage 13 smoke)." >&2
  exit 2
fi
if [[ "${MODE}" != "full" && "${MODE}" != "baseline_only" ]]; then
  echo "error: invalid mode '${MODE}' (expected full or baseline_only)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/preflight.sh"
sow_preflight

echo "[gpu] run_id=${RUN_ID}"
echo "[gpu] mode=${MODE}"

if [[ "${MODE}" == "baseline_only" ]]; then
  # Faster path: run baseline inference and baseline-only analysis.
  "${SOW_PYTHON}" sow.py inference-baseline --run-id "${RUN_ID}" --device cuda --batch-size auto
  "${SOW_PYTHON}" sow.py analyze --run-id "${RUN_ID}" --skip-robustness
else
  # Full baseline + robustness inference (Stage 13a/13b).
  "${SOW_PYTHON}" sow.py inference-baseline --run-id "${RUN_ID}" --device cuda --batch-size auto
  "${SOW_PYTHON}" sow.py inference-robustness --run-id "${RUN_ID}" --device cuda --batch-size auto

  # Analysis (Stage 14).
  "${SOW_PYTHON}" sow.py analyze --run-id "${RUN_ID}"
fi

# Bundle small artifacts for download (analysis + validation + meta + sentinels).
bash scripts/gpu/pack_analysis_bundle.sh "${RUN_ID}"

echo "[gpu] full run complete: run_id=${RUN_ID}"
