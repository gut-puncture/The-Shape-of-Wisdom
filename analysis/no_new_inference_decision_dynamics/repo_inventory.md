# Repo Inventory

## Files Used

- `results/parquet/layerwise.parquet`: cached layerwise four-option readout scores.
- `results/parquet/decision_metrics.parquet`: cached readout-derived decision metrics used only for cross-checks.
- `prompt_packs/ccc_baseline_v1_3000.jsonl`: source-of-truth prompt metadata and correct options.
- `paper/part1/data/part1_core.parquet`: comparison-only old-threshold/old-regime artifact when present.

## Source Schemas

### layerwise.parquet

Columns: model_id, model_revision, prompt_uid, example_id, wrapper_id, coarse_domain, is_correct, layer_index, candidate_logits_json, candidate_probs_json, candidate_entropy, top_candidate, top2_margin_prob, projected_hidden_128_json

### decision_metrics.parquet

Columns: model_id, model_revision, prompt_uid, example_id, wrapper_id, coarse_domain, is_correct, correct_key, layer_index, delta, boundary, drift, competitor, p_correct, prob_margin, entropy

### prompt manifest

Columns: coarse_domain, correct_key, dataset, example_id, manifest_sha256, module, options, prompt_id, prompt_text, prompt_uid, question, split, subject, wrapper_description, wrapper_id

### canonical layerwise table

Columns: model_id, model_revision, model_name, model_short, prompt_uid, example_id, subject, coarse_domain, wrapper_id, dataset, split, correct_option, layer_index, max_layer_index, n_layers_logged, z, score_A, score_B, score_C, score_D, prob_A, prob_B, prob_C, prob_D, top_candidate, top_tie_count, top_is_tied, competitor_tie_count, competitor_is_tied, final_predicted_option, final_argmax_tie, final_correct, question

## Verified Counts

| scope | metric | actual | expected | pass | details |
| --- | --- | --- | --- | --- | --- |
| global | canonical_rows | 276000 | 276000 | 1 |  |
| global | manifest_rows | 3000 | 3000 | 1 |  |
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

## Assumptions and Ambiguities

- `layer_index = 0` is the first logged transformer block output, not embeddings, following the repo contract.
- Final predicted option is derived from final-layer score argmax with alphabetical tie-break; tied-final trajectories are flagged explicitly.
- The 4-choice probabilities are conditional inside the four-option readout, not full-vocabulary generation probabilities.
- Old threshold commitment and old regime labels are treated as comparison-only because their sign convention is asymmetric for final-wrong trajectories.

## Totals

- Models found: 3
- Trajectories found: 9000
- Layerwise rows found: 276000
