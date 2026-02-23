#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGETS=(
  "runs"
  "artifacts/final_result_v2"
  "logs"
  "sentinels"
)

for rel in "${TARGETS[@]}"; do
  p="${REPO_ROOT}/${rel}"
  mkdir -p "${p}"
  find "${p}" -mindepth 1 -exec rm -rf {} +
done

echo "runtime state reset complete"
