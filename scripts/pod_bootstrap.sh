#!/usr/bin/env bash
# First thing on a fresh pod (plain CUDA image): install Python packages, then run pod_setup.sh.
set -uo pipefail
cd "$REPO"
{ pip install -q torch==2.10.0 transformers==5.1.0 "huggingface_hub>=1.4" numpy safetensors scikit-learn; echo "PIP_EXIT $?"; } > setup_pip.log 2>&1
bash scripts/pod_setup.sh
