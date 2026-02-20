#!/bin/zsh
set -euo pipefail

DONE_SENTINEL=""
COOLDOWN_SECONDS=1200

usage() {
  echo "usage: $0 --done-sentinel <path> [--cooldown-seconds <n>] -- <command...>" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --done-sentinel)
      DONE_SENTINEL="$2"
      shift 2
      ;;
    --cooldown-seconds)
      COOLDOWN_SECONDS="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "${DONE_SENTINEL}" ]]; then
  usage
fi
if [[ $# -eq 0 ]]; then
  usage
fi

while true; do
  if [[ -f "${DONE_SENTINEL}" ]]; then
    echo "[thermal-resume] done sentinel exists: ${DONE_SENTINEL}"
    exit 0
  fi

  echo "[thermal-resume] running: $*"
  set +e
  "$@"
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[thermal-resume] command exited successfully"
    if [[ -f "${DONE_SENTINEL}" ]]; then
      exit 0
    fi
    # Success without done sentinel: stop to avoid accidental loop.
    exit 0
  fi

  if [[ ${rc} -eq 95 ]]; then
    echo "[thermal-resume] thermal checkpoint exit detected, cooling for ${COOLDOWN_SECONDS}s"
    sleep "${COOLDOWN_SECONDS}"
    continue
  fi

  echo "[thermal-resume] command failed rc=${rc}"
  exit ${rc}
done
