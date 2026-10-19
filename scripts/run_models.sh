#!/usr/bin/env bash
# Run models one after another inside tmux, so it survives a dropped SSH connection.
#   bash scripts/run_models.sh RUN_DIR "llama1b llama8b qwen7b gemma9b" [extra args for wisdom.run]
# Safe to rerun at any time: finished prompts are skipped. Stop with: tmux kill-session -t sow
set -euo pipefail
RUN_DIR=$1; MODELS=$2; shift 2; EXTRA="$*"
REPO=${REPO:-/workspace/sow}
export HF_HOME=${HF_HOME:-/workspace/hf} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"; mkdir -p "$RUN_DIR"
CMD="cd $REPO; export HF_HOME=$HF_HOME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; for m in $MODELS; do \
  stages=ff,mcq,hint,steer; [[ \$m == *untrained* ]] && stages=ff; \
  python -m wisdom.run --model \$m --run-dir $RUN_DIR --stages \$stages $EXTRA 2>&1 | tee -a $RUN_DIR/\$m.log; done; \
  echo ALL_DONE >> $RUN_DIR/status"
tmux new-session -d -s sow "$CMD"
echo "started tmux session 'sow'; logs in $RUN_DIR/<model>.log"
