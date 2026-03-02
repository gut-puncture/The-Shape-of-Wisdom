.PHONY: paper clean

paper: part1.pdf

part1.pdf: part1.tex references.bib tables/*.tex
	pdflatex -interaction=nonstopmode part1.tex || true
	bibtex part1 || true
	pdflatex -interaction=nonstopmode part1.tex || true
	pdflatex -interaction=nonstopmode part1.tex

clean:
	rm -f part1.aux part1.bbl part1.blg part1.log part1.out part1.pdf
