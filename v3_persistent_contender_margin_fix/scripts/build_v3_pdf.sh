#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${ROOT_DIR}/paper"
BUILD_DIR="${PAPER_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${PAPER_DIR}"

tectonic --keep-logs --keep-intermediates --outdir "${BUILD_DIR}" main.tex

if command -v pdftoppm >/dev/null 2>&1; then
  mkdir -p "${BUILD_DIR}/page_pngs"
  rm -f "${BUILD_DIR}/page_pngs"/page-*.png
  pdftoppm -png "${BUILD_DIR}/main.pdf" "${BUILD_DIR}/page_pngs/page" >/dev/null
fi
