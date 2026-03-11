# Figure Manifest

| kind | filename | claim | inputs | panels | qc_status |
| --- | --- | --- | --- | --- | --- |
| main | fig01_state_decomposition.pdf | state decomposition and answer-readout coordinates | layerwise_scores_with_state_metrics.parquet | schematic + depth-colored paths + early/mid/late densities | generated |
| main | fig02_depthwise_dynamics.pdf | depth-wise competition and commitment summaries | bootstrap curve summaries | model-specific small multiples for a, q, switch rate, and r | generated |
| main | fig03_reversibility_map.pdf | empirical reversibility map over depth and current probability | layerwise_future_flip_metrics.parquet | pooled heatmap with boundary line | generated |
| main | fig04_boundary_dwellers.pdf | boundary dwelling tracks flip-heavy trajectories, with a mixed link to later stabilization | trajectory_metrics.parquet + table_empirical_commitment_summary.csv | distribution + scatter + commitment-depth summary | generated |
| main | fig05_exemplar_trajectories.pdf | representative stable, wrong, and boundary-dwelling trajectories | layerwise_scores_with_state_metrics.parquet + table_exemplar_prompts.csv | paired p_correct and a-q views for algorithmic exemplars | generated |
| appendix | appendix_baseline_margin_smoothing.pdf | hard-vs-soft margin appendix | layerwise_scores_with_state_metrics.parquet | representative prompt + jump histograms | generated |
| appendix | appendix_baseline_switch_commitment.pdf | baseline switch and old commitment appendix | trajectory_metrics.parquet | last-flip and old-commitment histograms | generated |
| appendix | appendix_per_model_reversibility_maps.pdf | per-model reversibility maps | layerwise_future_flip_metrics.parquet | one heatmap per model | generated |
| appendix | appendix_boundary_width_robustness.pdf | boundary width robustness | trajectory_metrics.parquet | occupancy distributions for multiple thresholds | generated |
| appendix | appendix_travel_metric_distributions.pdf | trajectory travel distributions | trajectory_metrics.parquet | path, angular, and radial travel histograms | generated |
| appendix | appendix_old_vs_empirical_commitment.pdf | old-threshold versus empirical commitment comparison | trajectory_metrics.parquet | scatter of old and empirical commitment depths | generated |
| appendix | appendix_competitor_dynamics.pdf | competitor-dynamics appendix | layerwise_scores_with_state_metrics.parquet + bootstrap curves | next-switch vs q and per-model switch curves | generated |

All figures are readout-space summaries only. Raw `a/q/r` panels are model-specific because score scales differ across models.
