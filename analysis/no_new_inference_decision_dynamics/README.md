# No-New-Inference Decision Dynamics

This package re-analyzes the cached layerwise four-option readout only. It performs no inference, uses no GPU, downloads nothing, and writes only inside this directory.

## Run

```bash
python3 analysis/no_new_inference_decision_dynamics/run_analysis.py \
  --out-dir analysis/no_new_inference_decision_dynamics \
  --seed 12345 \
  --bootstrap-reps 1000 \
  --force
```

## Required outputs

- Markdown reports: README.md, repo_inventory.md, analysis_report.md, figure_manifest.md, paper_outline.md, methods_for_general_cs_reader.md, claim_discipline.md
- Parquet files: derived_data/layerwise_scores_canonical.parquet, derived_data/layerwise_scores_with_state_metrics.parquet, derived_data/trajectory_metrics.parquet, derived_data/layerwise_future_flip_metrics.parquet
- Tables: tables/table_data_validation.csv, tables/table_baseline_reproduction.csv, tables/table_empirical_commitment_summary.csv, tables/table_exemplar_prompts.csv, tables/table_boundary_correlations.csv
- Figures: fig01_state_decomposition.pdf/.png, fig02_depthwise_dynamics.pdf/.png, fig03_reversibility_map.pdf/.png, fig04_boundary_dwellers.pdf/.png, fig05_exemplar_trajectories.pdf/.png, appendix_baseline_margin_smoothing.pdf/.png, appendix_baseline_switch_commitment.pdf/.png, appendix_per_model_reversibility_maps.pdf/.png, appendix_boundary_width_robustness.pdf/.png, appendix_travel_metric_distributions.pdf/.png, appendix_old_vs_empirical_commitment.pdf/.png, appendix_competitor_dynamics.pdf/.png

## Reproducibility

- Fixed seed: `12345`
- Fixed alphabetical tie-break for all argmax operations
- Fixed `z` binning and `p_correct` binning
- Source hashes and output hashes written to `analysis_manifest.json`
