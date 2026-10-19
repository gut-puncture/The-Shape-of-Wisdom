#!/usr/bin/env bash
# Start the worker inside tmux so it survives a dropped SSH connection. Safe to rerun at any time:
# finished prompts are skipped. Stop with: tmux kill-session -t sow
#   bash scripts/run_models.sh RUN_DIR "llama1b llama8b qwen7b gemma9b llama8b_untrained" [extra args]
set -euo pipefail
REPO=${REPO:-/workspace/sow}
export HF_HOME=${HF_HOME:-/workspace/hf} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
RUN_DIR=$1; MODELS=$2; shift 2
tmux new-session -d -s sow "cd $REPO && HF_HOME=$HF_HOME PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF bash scripts/worker.sh '$RUN_DIR' '$MODELS' $*"
echo "started tmux session 'sow'; per-model exit codes in $RUN_DIR/status"
