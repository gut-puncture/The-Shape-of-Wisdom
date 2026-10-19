#!/usr/bin/env bash
# One-time setup on a fresh GPU pod. Usage: bash scripts/pod_setup.sh   (logs to setup.log)
# Expects: this repo at $REPO (default /workspace/sow), the HF token at ~/.cache/huggingface/token.
set -euo pipefail
REPO=${REPO:-$HOME/sow}
export HF_HOME=${HF_HOME:-$HOME/hf}
cd "$REPO"
exec > >(tee -a setup.log) 2>&1
echo "== setup $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
mkdir -p "$HF_HOME"; cp -n ~/.cache/huggingface/token "$HF_HOME/token" 2>/dev/null || true
# Download all weights in parallel while nothing else needs the network.
python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
from wisdom.models import MODELS
specs = {(m.repo, m.revision) for m in MODELS.values()}
def get(rs):
    snapshot_download(rs[0], revision=rs[1], allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"])
    return rs[0]
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(get, specs): print("downloaded", r, flush=True)
PY
df -h "$REPO" | tail -1
echo "== setup done"
