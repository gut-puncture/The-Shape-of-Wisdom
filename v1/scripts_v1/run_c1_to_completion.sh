#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# run_c1_to_completion.sh
# Runs C1 Qwen baseline to completion with 90-min run / 20-min cooldown thermal policy.
# Automatically starts C2 validation watcher.

BASE="${BASE:-/Users/shaileshrana/shape-of-wisdom}"
LOG_DIR="$BASE/logs"
DONE_SENTINEL="$BASE/artifacts/inference_v2_canonical/qwen_baseline/done.json"

# Thermal policy: 30 min run, 12 min cooldown
RUN_SECONDS=1800      # 30 minutes
COOLDOWN_SECONDS=720  # 12 minutes

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=== C1 Qwen Baseline Runner ==="
log "Thermal policy: ${RUN_SECONDS}s run / ${COOLDOWN_SECONDS}s cooldown"
log "Done sentinel: $DONE_SENTINEL"

# Check if already complete
if [[ -f "$DONE_SENTINEL" ]]; then
  log "Already complete (done.json exists). Exiting."
  exit 0
fi

# Start C2 watcher in background
log "Starting C2 queue watcher in background..."
nohup "$BASE/scripts/queue_c2_after_c1.sh" > "$LOG_DIR/c2_queue.log" 2>&1 &
C2_WATCHER_PID=$!
log "C2 watcher started with PID $C2_WATCHER_PID"

# Run the cooldown loop
log "Starting cooldown loop..."
"$BASE/scripts/run_with_cooldowns_loop.sh" \
  --done-sentinel "$DONE_SENTINEL" \
  --run-seconds "$RUN_SECONDS" \
  --cooldown-seconds "$COOLDOWN_SECONDS" \
  -- python3 "$BASE/scripts/run_inference.py" \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --model-name "qwen2.5-7b-instruct" \
    --prompts "$BASE/data/experiment_inputs/main_prompts.jsonl" \
    --out "$BASE/artifacts/inference_v2_canonical/qwen_baseline/main_outputs.jsonl" \
    --meta-out "$BASE/artifacts/inference_v2_canonical/qwen_baseline/meta.json" \
    --pca-out "$BASE/artifacts/inference_v2_canonical/qwen_baseline/pca.joblib" \
    --resume \
    --time-budget-seconds "$RUN_SECONDS" \
    --done-sentinel "$DONE_SENTINEL"

RC=$?

if (( RC == 0 )); then
  log "C1 baseline completed successfully!"
else
  log "C1 baseline failed with exit code $RC"
  # Kill C2 watcher if it's still running
  kill "$C2_WATCHER_PID" 2>/dev/null || true
fi

exit "$RC"
