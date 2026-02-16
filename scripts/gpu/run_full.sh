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

run_with_oom_backoff() {
  # Usage: run_with_oom_backoff <step-name> <baseline|robustness>
  local step_name="$1"
  local condition="$2"
  local chain="${SOW_BATCH_RETRY_CHAIN:-16,12,8,6,4,2,1}"
  local logs_dir="${SOW_RUNS_ROOT}/${RUN_ID}/logs"
  mkdir -p "${logs_dir}"

  local success=0
  local bs
  IFS=',' read -r -a _BATCHES <<< "${chain}"
  for bs in "${_BATCHES[@]}"; do
    bs="$(echo "${bs}" | xargs)"
    if [[ -z "${bs}" ]]; then
      continue
    fi
    local attempt_log="${logs_dir}/${step_name}.bs${bs}.log"
    echo "[gpu] ${step_name}: attempt batch_size=${bs}"
    set +e
    "${SOW_PYTHON}" sow.py "inference-${condition}" --run-id "${RUN_ID}" --device cuda --batch-size "${bs}" 2>&1 | tee "${attempt_log}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -eq 0 ]]; then
      echo "[gpu] ${step_name}: success at batch_size=${bs}"
      success=1
      break
    fi
    if grep -Eqi "cuda out of memory|outofmemoryerror|cublas_status_alloc_failed" "${attempt_log}"; then
      echo "[gpu] ${step_name}: OOM at batch_size=${bs}; retrying lower batch"
      continue
    fi
    echo "[gpu] ${step_name}: non-OOM failure at batch_size=${bs}; see ${attempt_log}" >&2
    return "${rc}"
  done

  if [[ ${success} -ne 1 ]]; then
    echo "[gpu] ${step_name}: exhausted SOW_BATCH_RETRY_CHAIN=${chain} without a successful run" >&2
    return 1
  fi
}

if [[ "${MODE}" == "baseline_only" ]]; then
  # Faster path: run baseline inference and baseline-only analysis.
  run_with_oom_backoff "inference-baseline" "baseline"
  "${SOW_PYTHON}" sow.py analyze --run-id "${RUN_ID}" --skip-robustness
else
  # Full baseline + robustness inference (Stage 13a/13b).
  run_with_oom_backoff "inference-baseline" "baseline"
  run_with_oom_backoff "inference-robustness" "robustness"

  # Analysis (Stage 14).
  "${SOW_PYTHON}" sow.py analyze --run-id "${RUN_ID}"
fi

# Bundle small artifacts for download (analysis + validation + meta + sentinels).
bash scripts/gpu/pack_analysis_bundle.sh "${RUN_ID}"

echo "[gpu] full run complete: run_id=${RUN_ID}"
