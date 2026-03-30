#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shaileshrana/shape-of-wisdom"
PAPER_DIR="$ROOT/paper/colm2026"

python3 "$ROOT/scripts/paper/build_colm2026_assets.py"

cd "$PAPER_DIR"
tectonic --keep-logs --keep-intermediates main.tex
tectonic --keep-logs --keep-intermediates main.tex

rm -f "$PAPER_DIR"/page_pngs/page-*.png
pdftoppm -png -r 170 "$PAPER_DIR/main.pdf" "$PAPER_DIR/page_pngs/page"
