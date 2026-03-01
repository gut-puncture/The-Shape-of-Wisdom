PYTHON ?= python3
PAPER = paper_publish_v2
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || echo "../..")

all: figures paper

figures:
	$(PYTHON) -m sow.v2.figures.paper_figures \
		--parquet-dir $(REPO_ROOT)/results/parquet \
		--output-dir figures/

paper: figures
	latexmk -pdf $(PAPER).tex

quick-paper:
	latexmk -pdf $(PAPER).tex

clean:
	latexmk -C $(PAPER).tex
	rm -f figures/fig*.pdf

.PHONY: all figures paper quick-paper clean
