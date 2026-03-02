# Reproduce Paper 1 / Part I

## Prerequisites

- Python 3.10+
- `pip install pandas numpy matplotlib scikit-learn`
- LaTeX (pdflatex + bibtex) for PDF compilation

## Steps

### 1. Build all assets (data, figures, tables, metadata)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python scripts/part1/build_part1_assets.py \
    --parquet-dir results/parquet \
    --output-dir paper/part1 \
    --seed 12345
```

This will:
- Load `layerwise.parquet` and `decision_metrics.parquet`
- Compute all canonical metrics from raw option scores
- Run data integrity tests (hard fail on violation)
- Generate `paper/part1/data/part1_core.parquet`
- Generate 6 figure PDFs in `paper/part1/figures/`
- Generate 3 LaTeX tables in `paper/part1/tables/`
- Write `BUILD_INFO.json` and `CLAIM_EVIDENCE.md`
- Run figure quality control

### 2. Run tests

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    pytest tests/part1/ -q
```

### 3. Compile paper

```bash
make -C paper/part1 paper
```

This produces `paper/part1/part1.pdf`.

### 4. Figure quality control (standalone)

```bash
python scripts/part1/figure_qc.py paper/part1/figures/
```

## Data provenance

- Source: `results/parquet/layerwise.parquet` (276,000 rows, 3 models)
- SHA-256 hashes written to `BUILD_INFO.json`
- Git commit recorded at build time
