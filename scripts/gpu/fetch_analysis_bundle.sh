#!/usr/bin/env bash
set -euo pipefail

# Helper to fetch only the small analysis bundle tarball from the GPU VM back to the local machine.
#
# Usage:
#   bash scripts/gpu/fetch_analysis_bundle.sh <ssh_target> <run_id> [remote_runs_root] [dest_dir]
#
# Example:
#   bash scripts/gpu/fetch_analysis_bundle.sh ubuntu@1.2.3.4 myrun_20260215 /data/shape-of-wisdom-runs .
#
# Notes:
# - Requires `scp` locally.
# - Assumes the bundle exists at:
#     <remote_runs_root>/<run_id>/bundles/analysis_bundle_<run_id>.tar.gz

SSH_TARGET="${1:-}"
RUN_ID="${2:-}"
REMOTE_RUNS_ROOT="${3:-/data/shape-of-wisdom-runs}"
DEST_DIR="${4:-.}"

if [[ -z "${SSH_TARGET}" || -z "${RUN_ID}" ]]; then
  echo "usage: $0 <ssh_target> <run_id> [remote_runs_root] [dest_dir]" >&2
  exit 2
fi

REMOTE_PATH="${REMOTE_RUNS_ROOT%/}/${RUN_ID}/bundles/analysis_bundle_${RUN_ID}.tar.gz"
mkdir -p "${DEST_DIR}"

echo "[fetch] scp ${SSH_TARGET}:${REMOTE_PATH} -> ${DEST_DIR}/"
scp "${SSH_TARGET}:${REMOTE_PATH}" "${DEST_DIR}/"

OUT="${DEST_DIR%/}/analysis_bundle_${RUN_ID}.tar.gz"
if command -v shasum >/dev/null 2>&1; then
  echo "[fetch] sha256:"
  shasum -a 256 "${OUT}"
elif command -v sha256sum >/dev/null 2>&1; then
  echo "[fetch] sha256:"
  sha256sum "${OUT}"
fi

echo "[fetch] done: ${OUT}"

