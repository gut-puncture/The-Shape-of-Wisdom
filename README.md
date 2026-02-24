# Shape of Wisdom

**State, Motion, Boundary: A Mechanistic Account of Convergence and Failure in Transformer Multiple-Choice Decisions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

We introduce three quantities — **state** (logit-margin δ), **motion** (per-layer drift g), and **boundary** (|δ|) — that fully describe the layer-by-layer trajectory of a transformer's answer on multiple-choice benchmarks. These primitives separate prompts into four trajectory types (*stable-correct*, *stable-wrong*, *unstable-correct*, *unstable-wrong*), producing a phase diagram that replicates across three 7–8B-parameter instruction-tuned transformers (Qwen 2.5-7B, Llama 3.1-8B, Mistral 7B-v0.3) and multiple knowledge domains.

We decompose motion into **attention-driven routing** and **MLP-driven injection**, and validate causal claims via component ablation, activation patching, and span-deletion experiments with multiple-comparison correction.

**Paper**: See [`paper/final_paper/paper_publish.pdf`](paper/final_paper/paper_publish.pdf) for the full manuscript.

---

## Repository Structure

```
shape-of-wisdom/
├── configs/
│   ├── experiment_v2.yaml          # Full experiment configuration (models, thresholds, validators)
│   └── coarse_domain_mapping.json  # MMLU domain → coarse category mapping
├── prompt_packs/
│   └── ccc_baseline_v1_3000.jsonl  # 3,000 MMLU four-choice prompts (SHA-256 locked)
├── src/sow/                        # Python package
│   ├── v2/                         # Core analysis modules
│   │   ├── baseline_inference.py   # Batched inference with layerwise logit extraction
│   │   ├── metrics.py              # δ, g, b computation
│   │   ├── trajectory_types.py     # Four-type trajectory classification
│   │   ├── span_parser.py          # Regex-based prompt span parsing
│   │   ├── span_counterfactuals.py # Span deletion / counterfactual effects
│   │   ├── tracing/                # Attention & MLP component extraction
│   │   └── causal/                 # Ablation, patching, negative controls
│   ├── thermal/                    # Thermal management for long GPU runs
│   ├── io_jsonl.py                 # JSONL I/O utilities
│   └── hashing.py                  # SHA-256 manifest verification
├── scripts/v2/                     # 15-stage numbered pipeline
│   ├── run_full_local_v2.sh        # Orchestrator: runs all stages sequentially
│   ├── 00_run_experiment.py        # Stage 0: experiment snapshot & validation
│   ├── 00a_generate_baseline_outputs.py  # Batched inference (GPU required)
│   ├── 01_extract_baseline.py      # Extract baseline logits from raw outputs
│   ├── 02_compute_decision_metrics.py    # Compute δ, g, b for all prompts
│   ├── 03_classify_trajectories.py       # Classify into 4 trajectory types
│   ├── 04_region_analysis.py       # Entry/exit/re-entry event analysis
│   ├── 05_span_counterfactuals.py  # Span deletion experiments (GPU required)
│   ├── 06_select_tracing_subset.py # Select 600 prompts/model for tracing
│   ├── 07_run_tracing.py           # Extract attention/MLP components (GPU required)
│   ├── 08_attention_and_mlp_decomposition.py  # Linear drift decomposition
│   ├── 09_causal_tests.py          # Ablation & patching experiments
│   ├── 10_causal_validation_tools.py     # Negative controls & statistical tests
│   ├── 11_generate_paper_assets.py       # Generate figures & tables
│   └── 14_readiness_audit.py       # Final quality gate
├── results/                        # Pre-computed analysis outputs
│   ├── parquet/                    # All analysis data (see below)
│   ├── figures/                    # Generated figures from the paper
│   └── reports/                    # Stage-level validation reports (JSON)
├── tests/                          # Test suite (pytest)
├── paper/
│   └── final_paper/                # Publish-ready PDF + LaTeX source
├── requirements.txt
└── LICENSE
```

---

## Pre-Computed Results

The `results/` directory contains all analysis outputs from our primary GPU run, so you can inspect and build on our findings without re-running inference.

### Analysis Parquets (`results/parquet/`)

| File | Size | Description |
|------|------|-------------|
| `decision_metrics.parquet` | 10 MB | Per-prompt × per-layer δ, g, b values |
| `basin_gap.parquet` | 7.2 MB | Region entry/exit/re-entry events |
| `attention_contrib_by_span.parquet` | 2.4 MB | Decision-aligned attention contributions per span |
| `tracing_scalars.parquet` | 764 KB | s_attn, s_mlp scalar projections per layer |
| `attention_mass_by_span.parquet` | 728 KB | Attention mass distribution across spans |
| `prompt_types.parquet` | 132 KB | Trajectory type classification per prompt × model |
| `ablation_results.parquet` | 72 KB | Component ablation delta shifts |
| `span_labels.parquet` | 60 KB | Evidence / distractor / neutral labels |
| `span_effects.parquet` | 60 KB | Per-span counterfactual effects on δ |
| `patching_results.parquet` | 36 KB | Activation patching results |
| `span_paraphrase_stability.parquet` | 8 KB | Paraphrase stability validation |
| `span_deletion_causal.parquet` | 4 KB | Span deletion causal test results |
| `negative_controls.parquet` | 2 KB | Shuffled & sign-flipped baselines |

> **Note**: Raw layerwise logits (`layerwise.parquet`, ~244 MB) are excluded for size. Re-running stages 00–01 on a GPU regenerates this file.

### Figures (`results/figures/`)

13 publication-quality figures including phase diagrams, trajectory bundles, drift decomposition, causal validation panels, and attention routing heatmaps.

---

## Reproducing the Experiment

### Requirements

- **Python 3.10+**
- **GPU**: NVIDIA GPU with ≥24 GB VRAM (tested on RTX 6000 Ada) for inference stages
- **CPU-only**: Stages 01–04, 06, 08–11, 14 run on CPU using the pre-computed outputs

### Setup

```bash
git clone https://github.com/<your-org>/shape-of-wisdom.git
cd shape-of-wisdom
pip install -r requirements.txt
```

### Running the Full Pipeline (GPU)

The entire experiment runs through a single orchestrator script with fail-closed stage gating:

```bash
bash scripts/v2/run_full_local_v2.sh --run-id my_run
```

Each stage produces a JSON validation report; the pipeline halts if any gate fails. Stages that require GPU inference:
- `00a_generate_baseline_outputs.py` — batched layerwise logit extraction (~3K prompts × 3 models)
- `05_span_counterfactuals.py` — span deletion re-inference
- `07_run_tracing.py` — attention & MLP component capture

### Running Analysis Only (CPU)

If you want to reproduce the analysis from our pre-computed data, copy `results/parquet/` into a run directory and run stages 02+ individually:

```bash
RUN_ID=from_precomputed
mkdir -p runs/${RUN_ID}/v2

# Copy pre-computed parquets
cp results/parquet/*.parquet runs/${RUN_ID}/v2/

# Run any analysis stage, e.g.:
python scripts/v2/03_classify_trajectories.py --run-id ${RUN_ID} --config configs/experiment_v2.yaml --resume
python scripts/v2/08_attention_and_mlp_decomposition.py --run-id ${RUN_ID} --config configs/experiment_v2.yaml --resume
python scripts/v2/11_generate_paper_assets.py --run-id ${RUN_ID} --config configs/experiment_v2.yaml --resume
```

---

## Models

All models are loaded at pinned HuggingFace revisions with deterministic decoding (temperature=0, seed=12345):

| Model | HuggingFace ID | Revision |
|-------|----------------|----------|
| Qwen 2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | `a09a354...` |
| Llama 3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f...` |
| Mistral 7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c70...` |

Full revision hashes are in [`configs/experiment_v2.yaml`](configs/experiment_v2.yaml).

---

## Data

**Prompt manifest**: `prompt_packs/ccc_baseline_v1_3000.jsonl` contains 3,000 four-choice questions from [MMLU](https://github.com/hendrycks/test) (MIT License), balanced across difficulty levels and knowledge domains. The manifest is SHA-256 locked (`bfe255...`).

**Domain mapping**: `configs/coarse_domain_mapping.json` maps MMLU fine-grained subjects to coarse categories (STEM, humanities, social sciences, other).

---

## Key Configuration

All experiment parameters — model revisions, classification thresholds, validator gates, sampling sizes — are defined in [`configs/experiment_v2.yaml`](configs/experiment_v2.yaml). Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Tail window | 8 layers | Layers inspected for stability classification |
| Min boundary | \|δ\| ≥ 0.3 | Minimum margin for tail stability |
| R² gate | ≥ 0.70 | Drift decomposition quality threshold |
| BH α | 0.05 | Benjamini–Hochberg correction level |
| Tracing subset | 600 / model | Prompts selected for component tracing |
| Seed | 12345 | Deterministic seed for all random operations |

---

## Tests

```bash
python -m pytest tests/ -x -q
```

The test suite covers span parsing, trajectory classification, thermal management, metric computation, and stage-level validators.

---

## Citation

```bibtex
@inproceedings{shapewisdom2026,
  title={State, Motion, Boundary: A Mechanistic Account of Convergence and Failure in Transformer Multiple-Choice Decisions},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
