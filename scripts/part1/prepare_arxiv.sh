#!/bin/bash
set -e

echo "Syncing final paper/arxiv_submission back to paper/part1..."
cp paper/arxiv_submission/main.tex paper/part1/paper.tex
cp paper/arxiv_submission/main.tex paper/part1/part1.tex
cp paper/arxiv_submission/auto_numbers.tex paper/part1/
cp paper/part1/tables/*.tex paper/arxiv_submission/tables/

cd paper/part1
tectonic paper.tex
cp paper.pdf part1.pdf
cp paper.pdf neurips_submission.pdf
cp paper.pdf arxiv_public.pdf

cd ../..
echo "Creating arxiv_upload directory..."
rm -rf paper/arxiv_upload
mkdir -p paper/arxiv_upload/figures paper/arxiv_upload/tables

cp paper/arxiv_submission/main.tex paper/arxiv_upload/
cp paper/part1/auto_numbers.tex paper/arxiv_upload/
cp paper/part1/neurips_2025.sty paper/arxiv_upload/
cp paper/part1/references.bib paper/arxiv_upload/
cp paper/part1/tables/*.tex paper/arxiv_upload/tables/
cp paper/part1/figures/fig*.pdf paper/arxiv_upload/figures/

echo "Compiling in arxiv_upload to generate .bbl..."
cd paper/arxiv_upload
# tectonic --keep-intermediates keeps the .bbl, .aux, etc.
tectonic --keep-intermediates main.tex || true
ls -la

# Clean up build artifacts that ArXiv doesn't need
rm -f main.aux main.blg main.log main.out main.pdf
echo "Arxiv upload directory prepared."
