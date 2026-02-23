#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="$(command -v python3)"

RUN_ID=""
CONFIG_PATH="${REPO_ROOT}/configs/experiment_v2.yaml"
MODEL_NAME=""
MAX_PROMPTS=0
COOLDOWN_SECONDS=1200
RETRY_DELAY_SECONDS=120
MAX_RETRIES=20
FINAL_PUBLICATION_RUN=0

usage() {
  cat <<EOF
usage: $0 --run-id <id> [--config <path>] [--model-name <name>] [--max-prompts <n>] [--cooldown-seconds <n>] [--retry-delay-seconds <n>] [--max-retries <n>] [--final-publication-run]
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --max-prompts) MAX_PROMPTS="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --retry-delay-seconds) RETRY_DELAY_SECONDS="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --final-publication-run) FINAL_PUBLICATION_RUN=1; shift 1 ;;
    *) usage ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  usage
fi

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

run_stage13_decision() {
  local -a cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/v2/13_baseline_rerun_decision.py"
    "--run-id" "${RUN_ID}"
    "--config" "${CONFIG_PATH}"
  )
  if [[ -n "${MODEL_NAME}" ]]; then
    cmd+=("--model-name" "${MODEL_NAME}")
  fi
  if [[ "${FINAL_PUBLICATION_RUN}" -eq 1 ]]; then
    cmd+=("--final-publication-run")
  fi
  log "stage13 decision: ${cmd[*]}"
  "${cmd[@]}"
}

run_full_once() {
  local -a cmd=(
    "${REPO_ROOT}/scripts/v2/run_full_local_v2.sh"
    "--run-id" "${RUN_ID}"
    "--config" "${CONFIG_PATH}"
    "--cooldown-seconds" "${COOLDOWN_SECONDS}"
  )
  if [[ -n "${MODEL_NAME}" ]]; then
    cmd+=("--model-name" "${MODEL_NAME}")
  fi
  if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
    cmd+=("--max-prompts" "${MAX_PROMPTS}")
  fi
  log "run_full_local_v2: ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  cd "${REPO_ROOT}"
  run_stage13_decision

  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    log "watchdog attempt=${attempt}/${MAX_RETRIES}"
    set +e
    run_full_once
    local rc=$?
    set -e

    if [[ "${rc}" -eq 0 ]]; then
      log "watchdog completed run_id=${RUN_ID}"
      exit 0
    fi

    if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
      log "watchdog exhausted retries rc=${rc}"
      exit "${rc}"
    fi

    log "watchdog retrying after rc=${rc} sleep=${RETRY_DELAY_SECONDS}s"
    sleep "${RETRY_DELAY_SECONDS}"
  done
}

main "$@"
