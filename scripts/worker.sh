#!/usr/bin/env bash
# Runs the listed models one after another; records each model's real exit status.
#   bash scripts/worker.sh RUN_DIR "llama1b llama8b" [extra args for wisdom.run]
set -uo pipefail
RUN_DIR=$1; MODELS=$2; shift 2
mkdir -p "$RUN_DIR"
for m in $MODELS; do
  stages=ff,mcq,hint,steer
  [[ $m == *untrained* ]] && stages=ff
  echo "== $m start $(date -u +%FT%TZ)" >> "$RUN_DIR/status"
  python3 -m wisdom.run --model "$m" --run-dir "$RUN_DIR" --stages "$stages" "$@" 2>&1 | tee -a "$RUN_DIR/$m.log"
  code=${PIPESTATUS[0]}
  echo "== $m exit $code $(date -u +%FT%TZ)" >> "$RUN_DIR/status"
done
echo "== worker finished $(date -u +%FT%TZ)" >> "$RUN_DIR/status"
