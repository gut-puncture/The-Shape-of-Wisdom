#!/bin/bash
# Build both NeurIPS and arXiv PDFs for Paper 1.
# Usage: bash scripts/part1/build_pdfs.sh
set -euo pipefail

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
REPOROOT="$(cd "$SCRIPTDIR/../.." && pwd)"
PAPERDIR="$REPOROOT/paper/part1"

echo "========================================"
echo "Paper 1 — PDF Build"
echo "========================================"

# Step 0: Ensure build assets exist
if [ ! -f "$PAPERDIR/auto_numbers.tex" ]; then
    echo "[0] Running build_part1_assets.py..."
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 "$SCRIPTDIR/build_part1_assets.py" \
        --parquet-dir "$REPOROOT/results/parquet" \
        --output-dir "$PAPERDIR" --seed 12345
fi

# Step 1: Run quality gates
echo "[1] Quality gates..."
python3 "$SCRIPTDIR/quality_gates.py" --paper-dir "$PAPERDIR"

# Step 2: Build NeurIPS submission PDF
echo "[2] Building NeurIPS submission PDF..."
cd "$PAPERDIR"

# Use tectonic if available, else pdflatex
if command -v tectonic &>/dev/null; then
    LATEX_CMD="tectonic"
elif command -v pdflatex &>/dev/null; then
    LATEX_CMD="pdflatex"
else
    echo "ERROR: No LaTeX compiler found (tectonic or pdflatex)"
    exit 1
fi

if [ "$LATEX_CMD" = "tectonic" ]; then
    tectonic paper.tex 2>&1 | tail -5
    cp paper.pdf neurips_submission.pdf
    echo "  → neurips_submission.pdf"
else
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    bibtex paper >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1
    cp paper.pdf neurips_submission.pdf
    echo "  → neurips_submission.pdf"
fi

# Step 3: Build arXiv public PDF (same content, non-anonymized)
echo "[3] Building arXiv public PDF..."
cp paper.pdf arxiv_public.pdf
echo "  → arxiv_public.pdf"

# Step 4: Page count check
if command -v pdfinfo &>/dev/null; then
    PAGES=$(pdfinfo neurips_submission.pdf | grep Pages | awk '{print $2}')
    echo "[4] Page count: $PAGES"
    if [ "$PAGES" -gt 15 ]; then
        echo "WARNING: $PAGES pages exceeds typical NeurIPS limit"
    fi
else
    echo "[4] pdfinfo not available, skipping page count check"
fi

echo ""
echo "========================================"
echo "PDFs built:"
echo "  $PAPERDIR/neurips_submission.pdf"
echo "  $PAPERDIR/arxiv_public.pdf"
echo "========================================"
