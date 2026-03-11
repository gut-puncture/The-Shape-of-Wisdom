# Analysis Report

## Executive Summary

This package re-analyzes the cached four-option readout only. The no-new-inference evidence is strong enough for a substantially stronger observational paper on its own, while still motivating one follow-up intervention as future work.

The central objects are the soft margin, the current correct-option probability inside the 4-choice readout, the commitment coordinate `a`, the competitor-dispersion magnitude `q`, and the empirical future-flip probability map over normalized depth and current probability.

## Data Inventory And Validation

| scope | metric | actual | expected | pass | details |
| --- | --- | --- | --- | --- | --- |
| global | canonical_rows | 276000 | 276000 | 1 |  |
| global | manifest_rows | 3000 | 3000 | 1 |  |
| global | manifest_prompt_uid_duplicates | 0 | 0 | 1 |  |
| global | canonical_duplicate_rows | 0 | 0 | 1 |  |
| global | missing_values_total | 0 | 0 | 1 | question may be empty string but not NA |
| Qwen/Qwen2.5-7B-Instruct | prompt_count | 3000 | 3000 | 1 |  |
| Qwen/Qwen2.5-7B-Instruct | row_count | 84000 | 84000 | 1 |  |
| Qwen/Qwen2.5-7B-Instruct | min_layer_index | 0 | 0 | 1 |  |
| Qwen/Qwen2.5-7B-Instruct | max_layer_index | 27 | 27 | 1 |  |
| meta-llama/Llama-3.1-8B-Instruct | prompt_count | 3000 | 3000 | 1 |  |
| meta-llama/Llama-3.1-8B-Instruct | row_count | 96000 | 96000 | 1 |  |
| meta-llama/Llama-3.1-8B-Instruct | min_layer_index | 0 | 0 | 1 |  |
| meta-llama/Llama-3.1-8B-Instruct | max_layer_index | 31 | 31 | 1 |  |
| mistralai/Mistral-7B-Instruct-v0.3 | prompt_count | 3000 | 3000 | 1 |  |
| mistralai/Mistral-7B-Instruct-v0.3 | row_count | 96000 | 96000 | 1 |  |
| mistralai/Mistral-7B-Instruct-v0.3 | min_layer_index | 0 | 0 | 1 |  |
| mistralai/Mistral-7B-Instruct-v0.3 | max_layer_index | 31 | 31 | 1 |  |
| global | complete_layer_coverage | 1 | 1 | 1 |  |
| global | probability_sum_max_abs_error | 4.44089e-16 | 0 | 1 |  |
| global | decision_metrics_correct_key_mismatch | 0 | 0 | 1 |  |
| global | decision_metrics_p_correct_max_abs_error | 4.44089e-16 | 0 | 1 |  |
| global | final_predicted_option_defined | 1 | 1 | 1 |  |
| global | top_tied_rows | 117 | 0 | 1 | reported for sensitivity, not a failure |
| global | competitor_tied_rows | 96 | 0 | 1 | reported for sensitivity, not a failure |
| global | final_argmax_tied_trajectories | 18 | 0 | 1 | reported for sensitivity, not a failure |
| global | state_reconstruction_max_abs_error | 1.06581e-14 | 0 | 1 |  |
| global | state_reconstruction_mean_abs_error | 1.08432e-16 | 0 | 1 |  |
| global | p_correct_sigmoid_delta_soft_max_abs_error | 5.55112e-16 | 0 | 1 |  |
| global | q_below_theta_nan_threshold_count | 0 | 0 | 1 |  |
| global | q_below_angular_threshold_count | 0 | 0 | 1 |  |

State-space reconstruction checks:

- max reconstruction absolute error: 1.066e-14
- mean reconstruction absolute error: 1.084e-16
- max |sigmoid(delta_soft) - p_correct|: 5.551e-16

Per-model final correctness rates:

| model_id | final_correct_rate |
| --- | --- |
| Qwen/Qwen2.5-7B-Instruct | 0.67 |
| meta-llama/Llama-3.1-8B-Instruct | 0.616 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.547667 |

## Exact Metric Definitions

- `delta_hard`: correct score minus strongest incorrect score.
- `delta_soft`: correct score minus log-sum-exp of the three incorrect scores.
- `p_correct`: current correct-option probability in the 4-choice readout; exactly `sigmoid(delta_soft)` here.
- `a`: linear correct-vs-rest commitment coordinate after centering and fixed permutation.
- `q`: magnitude of disagreement among the incorrect answers.
- `future_flip`: whether the sign of `delta_soft` ever changes again later in depth.
- empirical commitment depth: first layer after which the estimated future-flip probability stays below the chosen threshold for the rest of the trajectory.

## Baseline Reproduction

| observation | scope | value | reproduced | details |
| --- | --- | --- | --- | --- |
| soft_margin_smoother_than_hard_margin_at_switch_layers | pooled | 0.00288637 | True | positive means hard jumps exceed soft jumps at switch layers |
| switch_rate_mid_or_earlier_peak_and_late_decline | pooled | 0.390933 | True | peak switch rate minus late-depth switch rate |
| old_regime_unstable_last_flips_are_late | pooled | 1 | True | median old last-flip depth in old unstable group |
| old_threshold_commitment_is_late | pooled | 0.888889 | True | median old-threshold commitment depth |

## Main Quantitative Results

Figures produced by this run:

| kind | filename | claim | inputs | panels | qc_status |
| --- | --- | --- | --- | --- | --- |
| main | fig01_state_decomposition.pdf | state decomposition and answer-readout coordinates | layerwise_scores_with_state_metrics.parquet | schematic + depth-colored paths + early/mid/late densities | generated |
| main | fig02_depthwise_dynamics.pdf | depth-wise competition and commitment summaries | bootstrap curve summaries | model-specific small multiples for a, q, switch rate, and r | generated |
| main | fig03_reversibility_map.pdf | empirical reversibility map over depth and current probability | layerwise_future_flip_metrics.parquet | pooled heatmap with boundary line | generated |
| main | fig04_boundary_dwellers.pdf | boundary dwelling tracks flip-heavy trajectories, with a mixed link to later stabilization | trajectory_metrics.parquet + table_empirical_commitment_summary.csv | distribution + scatter + commitment-depth summary | generated |
| main | fig05_exemplar_trajectories.pdf | representative stable, wrong, and boundary-dwelling trajectories | layerwise_scores_with_state_metrics.parquet + table_exemplar_prompts.csv | paired p_correct and a-q views for algorithmic exemplars | generated |

Boundary correlations:

| scope | x_metric | y_metric | n | spearman_rho | missing_y_count |
| --- | --- | --- | --- | --- | --- |
| pooled | boundary_occupancy_prob_010 | last_flip_depth | 5569 | -0.482014 | 3431 |
| pooled | boundary_occupancy_prob_010 | flip_count | 9000 | 0.554626 | 0 |
| pooled | boundary_occupancy_prob_010 | pooled_empirical_commitment_depth_005 | 4133 | -0.0292373 | 4867 |
| Qwen/Qwen2.5-7B-Instruct | boundary_occupancy_prob_010 | last_flip_depth | 2421 | -0.379013 | 579 |
| Qwen/Qwen2.5-7B-Instruct | boundary_occupancy_prob_010 | flip_count | 3000 | 0.442745 | 0 |
| Qwen/Qwen2.5-7B-Instruct | boundary_occupancy_prob_010 | pooled_empirical_commitment_depth_005 | 1543 | -0.237546 | 1457 |
| meta-llama/Llama-3.1-8B-Instruct | boundary_occupancy_prob_010 | last_flip_depth | 1582 | -0.119465 | 1418 |
| meta-llama/Llama-3.1-8B-Instruct | boundary_occupancy_prob_010 | flip_count | 3000 | 0.801781 | 0 |
| meta-llama/Llama-3.1-8B-Instruct | boundary_occupancy_prob_010 | pooled_empirical_commitment_depth_005 | 1163 | 0.640584 | 1837 |
| mistralai/Mistral-7B-Instruct-v0.3 | boundary_occupancy_prob_010 | last_flip_depth | 1566 |  | 1434 |
| mistralai/Mistral-7B-Instruct-v0.3 | boundary_occupancy_prob_010 | flip_count | 3000 | 0.220194 | 0 |
| mistralai/Mistral-7B-Instruct-v0.3 | boundary_occupancy_prob_010 | pooled_empirical_commitment_depth_005 | 1427 | 0.123135 | 1573 |

Boundary-dwelling interpretation:

- Mean boundary occupancy is higher for final-correct trajectories (0.0653) than for final-wrong trajectories (0.0188) in this cache, so boundary dwelling is not a synonym for final failure.
- Boundary occupancy rises with flip count in the pooled data (Spearman rho = 0.555), and that sign is positive in each model.
- The later-stabilization link is mixed rather than uniform: pooled occupancy versus last-flip depth has rho = -0.482, while pooled occupancy versus empirical commitment depth has rho = -0.029. The main supported claim is prolonged competition near the boundary, not a single monotone delay-of-commitment law.

Empirical commitment summary:

| scope | final_group | alpha | pooled_nonmissing | pooled_never_enter | pooled_median_depth | model_nonmissing | model_never_enter | model_median_depth | old_nonmissing | old_median_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | all | 0.05 | 4133 | 4867 | 1 | 5673 | 3327 | 1 | 7583 | 0.888889 |
| pooled | final_correct | 0.05 | 3150 | 2351 | 1 | 3919 | 1582 | 1 | 4482 | 1 |
| pooled | final_wrong | 0.05 | 983 | 2516 | 0.967742 | 1754 | 1745 | 0.774194 | 3101 | 0 |
| Qwen/Qwen2.5-7B-Instruct | all | 0.05 | 1543 | 1457 | 0.925926 | 1230 | 1770 | 0.925926 | 2679 | 0.851852 |
| Qwen/Qwen2.5-7B-Instruct | final_correct | 0.05 | 1323 | 687 | 0.925926 | 1230 | 780 | 0.925926 | 1808 | 0.888889 |
| Qwen/Qwen2.5-7B-Instruct | final_wrong | 0.05 | 220 | 770 | 1 | 0 | 990 |  | 871 | 0 |
| meta-llama/Llama-3.1-8B-Instruct | all | 0.05 | 1163 | 1837 | 1 | 1653 | 1347 | 1 | 2234 | 0.935484 |
| meta-llama/Llama-3.1-8B-Instruct | final_correct | 0.05 | 678 | 1170 | 1 | 1091 | 757 | 1 | 1249 | 1 |
| meta-llama/Llama-3.1-8B-Instruct | final_wrong | 0.05 | 485 | 667 | 0.935484 | 562 | 590 | 0.903226 | 985 | 0 |
| mistralai/Mistral-7B-Instruct-v0.3 | all | 0.05 | 1427 | 1573 | 1 | 2790 | 210 | 1 | 2670 | 1 |
| mistralai/Mistral-7B-Instruct-v0.3 | final_correct | 0.05 | 1149 | 494 | 1 | 1598 | 45 | 1 | 1425 | 1 |
| mistralai/Mistral-7B-Instruct-v0.3 | final_wrong | 0.05 | 278 | 1079 | 1 | 1192 | 165 | 0.612903 | 1245 | 0 |
| pooled | all | 0.1 | 7076 | 1924 | 0.925926 | 6574 | 2426 | 0.935484 | 7583 | 0.888889 |
| pooled | final_correct | 0.1 | 4373 | 1128 | 1 | 4024 | 1477 | 1 | 4482 | 1 |
| pooled | final_wrong | 0.1 | 2703 | 796 | 0.925926 | 2550 | 949 | 0.806452 | 3101 | 0 |
| Qwen/Qwen2.5-7B-Instruct | all | 0.1 | 2568 | 432 | 0.888889 | 1660 | 1340 | 0.888889 | 2679 | 0.851852 |
| Qwen/Qwen2.5-7B-Instruct | final_correct | 0.1 | 1791 | 219 | 0.888889 | 1230 | 780 | 0.888889 | 1808 | 0.888889 |
| Qwen/Qwen2.5-7B-Instruct | final_wrong | 0.1 | 777 | 213 | 0.925926 | 430 | 560 | 0.925926 | 871 | 0 |

## Per-Model Consistency Checks

- Raw `a`, `q`, and `r` are shown per model because the models have materially different logit scales.
- Scale-free summaries such as `p_correct`, switch rate, boundary occupancy, and future-flip probability are pooled only through equal model weighting.
- Tie-affected states and tied-final trajectories are retained with explicit flags and covered in appendix robustness figures.

## Tie And Backoff Sensitivity

- top-score tied rows: 117
- competitor-tied rows: 96
- tied final trajectories: 18
- pooled backoff counts: {'cell': 276000}
- per-model backoff counts: {'cell': 276000}

## Limitations

- This is a readout-space analysis only.
- The 4-choice probabilities are conditional inside the recorded answer subspace.
- Old threshold commitment is included only for comparison and is asymmetric for final-wrong trajectories.

## One Suggested Next Experimental Step

Run one targeted follow-up intervention on prompts that remain boundary-dwelling late in depth, and test whether shifting the readout away from the boundary late in the network reduces future flips without changing the prompt set.

## What Cannot Be Claimed

- Nothing here identifies hidden-state computation.
- Nothing here establishes causality.
- Nothing here licenses claims about internal mechanisms beyond the observed answer-readout trajectories.
