# PAPER_BUILD_MAP.md — Source-of-truth mapping for Paper 1

## LaTeX Source
- `paper/arxiv_submission/main.tex` → produces `main.pdf`
- Auto-generated macros: `paper/arxiv_submission/auto_numbers.tex` (from `scripts/part1/compute_numbers.py`)

## Parquet → Figure/Table Mapping

| Output | Source Parquet | Columns Used |
|--------|--------------|-------------|
| Figure A (figA_why_delta_soft.pdf) | `results/parquet/layerwise.parquet` + `results/parquet/decision_metrics.parquet` | candidate_logits_json → sA/sB/sC/sD, correct_key, delta_hard_dyn, delta_default, switch_indicator |
| Figure 2 (fig2_examples.pdf) | `paper/part1/data/part1_core.parquet` | delta_default, depth_norm, regime, commitment_layer |
| Figure 3 (fig3_distributions.pdf) | `paper/part1/data/part1_core.parquet` | delta_default, depth_norm, regime, model_id |
| Figure 4 (fig4_decision_space.pdf) | `paper/part1/data/part1_core.parquet` | sA, sB, sC, sD, regime, model_id, layer_index |
| Figure 5 (fig5_flow_field.pdf) | `paper/part1/data/part1_core.parquet` | delta_default, depth_norm, drift_default |
| Figure 6 (fig6_commitment.pdf) | `paper/part1/data/part1_core.parquet` | commitment_layer, flip_count, last_flip_layer, switch_indicator, depth_norm |
| Figure 7 (fig7_robustness.pdf) | `paper/part1/data/part1_core.parquet` | delta_soft_tau_0_5, delta_soft_tau_1_0, delta_soft_tau_2_0 |
| Table 1 (table1_models.tex) | Static (model metadata) | — |
| Table 2 (table2_regimes.tex) | `paper/part1/data/part1_core.parquet` | regime, model_id, final_delta_default |
| Table 3 (table3_robustness.tex) | `paper/part1/data/part1_core.parquet` | delta_soft_tau_* columns |
| auto_numbers.tex | `paper/part1/data/part1_core.parquet` | All columns (PCA, regime proportions, commitment, flips) |

## Data Provenance

`part1_core.parquet` (276k rows, 27 columns) is derived from:
- `results/parquet/layerwise.parquet` — per-layer candidate_logits_json
- `results/parquet/decision_metrics.parquet` — correct_key, model_id, prompt_uid

Both originate from: `downloads/gpu_full_20260224T145646Z/` (GPU inference run, pinned model revisions).

## Build Command
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 scripts/part1/build_part1_assets.py \
    --parquet-dir results/parquet \
    --output-dir paper/part1 --seed 12345
```
