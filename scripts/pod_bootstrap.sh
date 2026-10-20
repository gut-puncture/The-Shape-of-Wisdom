#!/usr/bin/env bash
# First thing on a fresh pod (plain CUDA image): install Python packages, then run pod_setup.sh.
set -uo pipefail
cd "$REPO"
{ pip install -q -r requirements.txt; echo "PIP_EXIT $?"; } > setup_pip.log 2>&1
bash scripts/pod_setup.sh
