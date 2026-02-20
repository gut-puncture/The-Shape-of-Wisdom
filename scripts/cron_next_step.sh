#!/bin/zsh
# Shape of Wisdom - Autopilot Pipeline Runner
# This script checks project state and triggers the next appropriate step
# Run from project root: /Users/shaileshrana/shape-of-wisdom

set -euo pipefail

PROJECT_DIR="/Users/shaileshrana/shape-of-wisdom"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
STATE_FILE="$ARTIFACTS_DIR/last_cron_next_step.json"
LOG_FILE="$PROJECT_DIR/logs/cron_next_step.log"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$PROJECT_DIR"

# Read current state
if [[ -f "$STATE_FILE" ]]; then
    LAST_RUN=$(jq -r '.last_run // "never"' "$STATE_FILE" 2>/dev/null || echo "never")
    LAST_STATUS=$(jq -r '.status // "unknown"' "$STATE_FILE" 2>/dev/null || echo "unknown")
    LAST_STAGE=$(jq -r '.stage // "none"' "$STATE_FILE" 2>/dev/null || echo "none")
else
    LAST_RUN="never"
    LAST_STATUS="unknown"
    LAST_STAGE="none"
fi

log "=== Autopilot Check ==="
log "Last run: $LAST_RUN | Status: $LAST_STATUS | Stage: $LAST_STAGE"

# Check if a run is currently active (look for running processes)
ACTIVE_INFERENCE=$(pgrep -f "sow.py inference" || true)
ACTIVE_ANALYSIS=$(pgrep -f "sow.py analyze" || true)

if [[ -n "$ACTIVE_INFERENCE" ]]; then
    log "⚠️  Inference is currently running (PID: $ACTIVE_INFERENCE). Skipping."
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg reason "inference_active" \
        '{timestamp: $timestamp, status: "skipped", reason: $reason, next_check: "+10 minutes"}' \
        > "$STATE_FILE"
    exit 0
fi

if [[ -n "$ACTIVE_ANALYSIS" ]]; then
    log "⚠️  Analysis is currently running (PID: $ACTIVE_ANALYSIS). Skipping."
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg reason "analysis_active" \
        '{timestamp: $timestamp, status: "skipped", reason: $reason, next_check: "+10 minutes"}' \
        > "$STATE_FILE"
    exit 0
fi

# Check current project state from STATE.md
# Look for the most recent entry to determine what's next
LATEST_ENTRY=$(grep "^### 20" STATE.md | tail -1 || echo "")

if [[ -z "$LATEST_ENTRY" ]]; then
    log "⚠️  No state entries found in STATE.md"
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg reason "no_state_entries" \
        '{timestamp: $timestamp, status: "error", reason: $reason}' \
        > "$STATE_FILE"
    exit 1
fi

# Extract the stage from the latest entry
CURRENT_STAGE=$(echo "$LATEST_ENTRY" | sed -n 's/.*- Stage \([0-9a-z_]*\).*/\1/p' || echo "unknown")
CURRENT_STATUS=$(echo "$LATEST_ENTRY" | grep -oE "(PASS|FAIL|IN PROGRESS)" | tail -1 || echo "unknown")

log "Current stage from STATE.md: $CURRENT_STAGE | Status: $CURRENT_STATUS"

# Check for completed run artifacts
RUN_1452A_DIR="$PROJECT_DIR/downloads/gpu_runs_20260216_full/rtx6000ada_baseline_20260216_1452_a"
ANALYSIS_DONE="$RUN_1452A_DIR/sentinels/analysis.done"
BUNDLE_FILE="$RUN_1452A_DIR/bundles/analysis_bundle_rtx6000ada_baseline_20260216_1452_a.tar.gz"

# Determine next action based on state
if [[ "$CURRENT_STAGE" == "14" && "$CURRENT_STATUS" == "PASS" ]]; then
    # Check if manuscript/paper stage has been started
    if [[ -f "$PROJECT_DIR/paper/manuscript.tex" ]]; then
        log "✅ Paper manuscript exists. Checking if it needs updates..."
        # Could trigger LaTeX compilation or paper validation here
        jq -n \
            --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --arg stage "$CURRENT_STAGE" \
            --arg status "complete" \
            '{timestamp: $timestamp, last_run: $timestamp, stage: $stage, status: "complete", message: "Analysis complete, manuscript exists"}' \
            > "$STATE_FILE"
    else
        log "📄 Analysis complete. Ready for manuscript generation."
        # Trigger manuscript generation via OpenClaw skill or codex
        # For now, just record the state
        jq -n \
            --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --arg stage "$CURRENT_STAGE" \
            '{timestamp: $timestamp, last_run: $timestamp, stage: $stage, status: "ready_for_manuscript", message: "Analysis complete - ready for paper generation"}' \
            > "$STATE_FILE"
        
        # Send notification via OpenClaw system event
        # openclaw system event "Shape of Wisdom: Analysis complete. Ready for manuscript generation."
    fi
elif [[ "$CURRENT_STATUS" == "IN PROGRESS" ]]; then
    log "⏳ Stage $CURRENT_STAGE is in progress. Waiting for completion."
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg stage "$CURRENT_STAGE" \
        --arg run_status "$CURRENT_STATUS" \
        '{timestamp: $timestamp, last_run: $timestamp, stage: $stage, status: "waiting", run_status: $run_status, next_check: "+20 minutes"}' \
        > "$STATE_FILE"
else
    log "ℹ️  Current state: Stage $CURRENT_STAGE ($CURRENT_STATUS). No automatic action required."
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg stage "$CURRENT_STAGE" \
        --arg run_status "$CURRENT_STATUS" \
        '{timestamp: $timestamp, last_run: $timestamp, stage: $stage, status: "idle", run_status: $run_status, next_check: "+1 hour"}' \
        > "$STATE_FILE"
fi

log "=== Autopilot Check Complete ==="
exit 0
