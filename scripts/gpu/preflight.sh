#!/usr/bin/env bash
set -euo pipefail

# Intended to be sourced by other GPU run scripts.
# Exports sane defaults for keeping caches + runs on the attached disk.

sow_preflight() {
  if [[ -z "${SOW_RUNS_ROOT:-}" ]]; then
    echo "error: SOW_RUNS_ROOT must be set to a path on the attached disk (e.g. /data/shape-of-wisdom-runs)" >&2
    return 2
  fi

  # Prefer a repo-local virtualenv if present (RunPod base images can lack transformers).
  # Callers should use $SOW_PYTHON for all sow CLI invocations.
  if [[ -z "${SOW_PYTHON:-}" ]]; then
    if [[ -x "./.venv/bin/python" ]]; then
      SOW_PYTHON="./.venv/bin/python"
      export SOW_PYTHON
    else
      SOW_PYTHON="python3"
      export SOW_PYTHON
    fi
  fi

  # HuggingFace gated models: prefer a local token file on GPU VMs.
  # Do NOT echo the token.
  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f "/root/.secrets/hf_token.txt" ]]; then
      HF_TOKEN="$(tr -d '\n' < /root/.secrets/hf_token.txt)"
      export HF_TOKEN
    fi
  fi
  if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi

  # Default caches onto /data if present, but do not override user-provided values.
  if [[ -d "/data" ]]; then
    : "${HF_HOME:=/data/hf}"
    : "${TRANSFORMERS_CACHE:=${HF_HOME}/transformers}"
    : "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
    : "${TORCH_HOME:=/data/torch}"
    export HF_HOME TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE TORCH_HOME
  elif [[ -d "/workspace" ]]; then
    # RunPod often mounts attached storage at /workspace.
    : "${HF_HOME:=/workspace/hf}"
    : "${TRANSFORMERS_CACHE:=${HF_HOME}/transformers}"
    : "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
    : "${TORCH_HOME:=/workspace/torch}"
    export HF_HOME TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE TORCH_HOME
  fi

  # Determinism / reproducibility knobs (must be exported before Python starts).
  : "${PYTHONHASHSEED:=0}"
  : "${CUBLAS_WORKSPACE_CONFIG:=:4096:8}"
  : "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
  export PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG
  export PYTORCH_CUDA_ALLOC_CONF

  # Keep CPU noise down (spec section 23).
  : "${OMP_NUM_THREADS:=1}"
  : "${MKL_NUM_THREADS:=1}"
  : "${OPENBLAS_NUM_THREADS:=1}"
  : "${VECLIB_MAXIMUM_THREADS:=1}"
  : "${NUMEXPR_NUM_THREADS:=1}"
  export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS

  mkdir -p "${SOW_RUNS_ROOT}"

  echo "[gpu] preflight runs_root=${SOW_RUNS_ROOT}"
  if command -v df >/dev/null 2>&1; then
    df -h "${SOW_RUNS_ROOT}" || true
    avail_kb="$(df -Pk "${SOW_RUNS_ROOT}" | awk 'NR==2{print $4}')"
    if [[ -n "${avail_kb}" ]]; then
      # Conservative minimum: 50 GiB free. Full inference can use much more; this is a fail-fast floor.
      min_kb="$((50 * 1024 * 1024))"
      if [[ "${avail_kb}" -lt "${min_kb}" ]]; then
        echo "error: insufficient free space on runs_root filesystem (<50GiB). avail_kb=${avail_kb}" >&2
        return 2
      fi
    fi
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found; are you on a CUDA GPU VM?" >&2
    return 2
  fi
  nvidia-smi || true

  # Verify python + torch cuda availability.
  "${SOW_PYTHON}" - <<'PY'
import torch
print("[gpu] torch", torch.__version__)
print("[gpu] torch.version.cuda", getattr(torch.version, "cuda", None))
ok = torch.cuda.is_available()
print("[gpu] cuda_available", ok)
if not ok:
    raise SystemExit("torch.cuda.is_available() is False")
try:
    print("[gpu] cuda_device_name", torch.cuda.get_device_name(0))
except Exception as e:
    print("[gpu] cuda_device_name_error", type(e).__name__, str(e))
PY

  # Fail fast on gated-model auth/access problems (Llama).
  "${SOW_PYTHON}" - <<'PY'
from huggingface_hub import hf_hub_download

repo = "meta-llama/Llama-3.1-8B-Instruct"
rev = "0e9e39f249a16976918f6564b8830bc894c89659"
try:
    p = hf_hub_download(repo_id=repo, revision=rev, filename="config.json")
    print(f"[gpu] hf_access_ok {repo}@{rev} {p}")
except Exception as e:
    raise SystemExit(f"HF gated model access check failed for {repo}@{rev}: {type(e).__name__}: {e}")
PY

  # Paid inputs must be present on the GPU VM (repo clone does not include data/).
  local required_inputs=(
    "data/experiment_inputs/main_prompts.jsonl"
    "data/experiment_inputs/robustness_prompts_v2.jsonl"
    "data/experiment_inputs/build_summary.json"
    "configs/coarse_domain_mapping.json"
  )
  for p in "${required_inputs[@]}"; do
    if [[ ! -f "${p}" ]]; then
      echo "error: missing required input file: ${p}" >&2
      echo "note: paid inputs are gitignored; copy them onto the GPU VM under the same path." >&2
      return 2
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  sow_preflight
fi
