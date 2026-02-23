#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/opt/homebrew/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

RUN_ID=""
CONFIG_PATH="${REPO_ROOT}/configs/experiment_v2.yaml"
COOLDOWN_SECONDS=1200
MODEL_NAME=""
MAX_PROMPTS=0
LOG_DIR=""

usage() {
  cat <<EOF
usage: $0 --run-id <id> [--config <path>] [--model-name <name>] [--max-prompts <n>] [--cooldown-seconds <n>] [--log-dir <path>]
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --max-prompts)
      MAX_PROMPTS="$2"
      shift 2
      ;;
    --cooldown-seconds)
      COOLDOWN_SECONDS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  usage
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${REPO_ROOT}/runs/${RUN_ID}/v2/logs/local_full_$(date -u '+%Y%m%dT%H%M%SZ')"
fi
mkdir -p "${LOG_DIR}"

export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

report_path_for_stage() {
  local stage_script="$1"
  local out_root="${REPO_ROOT}/runs/${RUN_ID}/v2"
  case "${stage_script}" in
    00_run_experiment.py) echo "${out_root}/00_run_experiment.report.json" ;;
    00a_generate_baseline_outputs.py) echo "${out_root}/00a_generate_baseline_outputs.report.json" ;;
    01_extract_baseline.py) echo "${out_root}/01_extract_baseline.report.json" ;;
    02_compute_decision_metrics.py) echo "${out_root}/02_compute_decision_metrics.report.json" ;;
    03_classify_trajectories.py) echo "${out_root}/03_classify_trajectories.report.json" ;;
    04_region_analysis.py) echo "${out_root}/04_region_analysis.report.json" ;;
    05_span_counterfactuals.py) echo "${out_root}/05_span_counterfactuals.report.json" ;;
    06_select_tracing_subset.py) echo "${out_root}/06_select_tracing_subset.report.json" ;;
    07_run_tracing.py) echo "${out_root}/07_run_tracing.report.json" ;;
    08_attention_and_mlp_decomposition.py) echo "${out_root}/08_attention_and_mlp_decomposition.report.json" ;;
    09_causal_tests.py) echo "${out_root}/09_causal_tests.report.json" ;;
    10_causal_validation_tools.py) echo "${out_root}/10_causal_validation_tools.report.json" ;;
    14_readiness_audit.py) echo "${out_root}/meta/readiness_audit.json" ;;
    11_generate_paper_assets.py) echo "${out_root}/11_generate_paper_assets.report.json" ;;
    *) echo "" ;;
  esac
}

done_sentinel_for_stage() {
  local stage_script="$1"
  local out_root="${REPO_ROOT}/runs/${RUN_ID}/v2"
  case "${stage_script}" in
    00a_generate_baseline_outputs.py) echo "${out_root}/00a_generate_baseline_outputs.done" ;;
    05_span_counterfactuals.py) echo "${out_root}/sentinels/05_span_counterfactuals.done" ;;
    07_run_tracing.py) echo "${out_root}/sentinels/07_run_tracing.done" ;;
    *) echo "" ;;
  esac
}

assert_report_pass() {
  local report_path="$1"
  local stage_script="$2"
  REPORT_PATH="${report_path}" STAGE_SCRIPT="${stage_script}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

report = Path(os.environ["REPORT_PATH"])
stage = os.environ["STAGE_SCRIPT"]
if not report.exists():
    raise SystemExit(f"missing report for {stage}: {report}")
obj = json.loads(report.read_text(encoding="utf-8"))
if not bool(obj.get("pass")):
    failing = obj.get("failing_gates")
    raise SystemExit(f"{stage} report pass=false failing_gates={failing}")
print(f"{stage} pass=true report={report}")
PY
}

run_stage() {
  local stage_script="$1"
  local stage_log="${LOG_DIR}/${stage_script%.py}.log"
  local stage_start
  stage_start="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "stage start script=${stage_script} log=${stage_log}"

  local -a cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/v2/${stage_script}" "--run-id" "${RUN_ID}" "--config" "${CONFIG_PATH}" "--resume")
  if [[ -n "${MODEL_NAME}" ]]; then
    cmd+=("--model-name" "${MODEL_NAME}")
  fi
  if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
    cmd+=("--max-prompts" "${MAX_PROMPTS}")
  fi

  local done_sentinel
  done_sentinel="$(done_sentinel_for_stage "${stage_script}")"
  if [[ -n "${done_sentinel}" && -f "${done_sentinel}" ]]; then
    # Never trust stale sentinels across resumed runs; stage scripts are resume-safe.
    rm -f "${done_sentinel}"
    log "removed stale sentinel for ${stage_script}: ${done_sentinel}"
  fi

  set +e
  if [[ -n "${done_sentinel}" ]]; then
    /usr/bin/time -p "${REPO_ROOT}/scripts/v2/run_with_thermal_resume.sh" \
      --done-sentinel "${done_sentinel}" \
      --cooldown-seconds "${COOLDOWN_SECONDS}" \
      -- "${cmd[@]}" 2>&1 | tee -a "${stage_log}"
    rc=${PIPESTATUS[0]}
  else
    /usr/bin/time -p "${cmd[@]}" 2>&1 | tee -a "${stage_log}"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  local stage_end
  stage_end="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "stage end script=${stage_script} rc=${rc} start=${stage_start} end=${stage_end}"
  if [[ ${rc} -ne 0 ]]; then
    return "${rc}"
  fi

  local report_path
  report_path="$(report_path_for_stage "${stage_script}")"
  if [[ -n "${report_path}" ]]; then
    assert_report_pass "${report_path}" "${stage_script}" 2>&1 | tee -a "${stage_log}"
  fi
}

run_stage00_snapshot() {
  local stage_script="00_run_experiment.py"
  local stage_log="${LOG_DIR}/${stage_script%.py}.log"
  local stage_start
  stage_start="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "stage start script=${stage_script} log=${stage_log}"

  local -a cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/v2/00_run_experiment.py"
    "--run-id" "${RUN_ID}"
    "--mode" "full"
    "--config" "${CONFIG_PATH}"
    "--snapshot-only"
  )
  if [[ -n "${MODEL_NAME}" ]]; then
    cmd+=("--model-name" "${MODEL_NAME}")
  fi

  set +e
  /usr/bin/time -p "${cmd[@]}" 2>&1 | tee -a "${stage_log}"
  rc=${PIPESTATUS[0]}
  set -e

  local stage_end
  stage_end="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "stage end script=${stage_script} rc=${rc} start=${stage_start} end=${stage_end}"
  if [[ ${rc} -ne 0 ]]; then
    return "${rc}"
  fi

  local report_path
  report_path="$(report_path_for_stage "${stage_script}")"
  if [[ -n "${report_path}" ]]; then
    assert_report_pass "${report_path}" "${stage_script}" 2>&1 | tee -a "${stage_log}"
  fi
}

main() {
  cd "${REPO_ROOT}"
  log "run_id=${RUN_ID} config=${CONFIG_PATH} model_name=${MODEL_NAME:-all} max_prompts=${MAX_PROMPTS} cooldown_seconds=${COOLDOWN_SECONDS}"
  log "logs=${LOG_DIR}"
  run_stage00_snapshot

  local -a stages=(
    "00a_generate_baseline_outputs.py"
    "01_extract_baseline.py"
    "02_compute_decision_metrics.py"
    "03_classify_trajectories.py"
    "04_region_analysis.py"
    "05_span_counterfactuals.py"
    "06_select_tracing_subset.py"
    "07_run_tracing.py"
    "08_attention_and_mlp_decomposition.py"
    "09_causal_tests.py"
    "10_causal_validation_tools.py"
    "14_readiness_audit.py"
    "11_generate_paper_assets.py"
  )

  local stage
  for stage in "${stages[@]}"; do
    run_stage "${stage}"
  done

  log "all stages completed successfully run_id=${RUN_ID}"
}

main "$@"
