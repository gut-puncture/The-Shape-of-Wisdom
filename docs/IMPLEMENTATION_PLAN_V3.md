# Implementation Plan V3

Version: 1.1 (v3 repo execution)

Change from prior framing:
- The paper section “Improving convergence” is removed from the core narrative.
- Interventions are used as causal validation tools; exploratory outcome-improvement analysis is appendix-only.

## 0) Design principles
1. Anchor all claims on state, motion, boundary.
2. Keep categories minimal and operational.
3. Separate descriptive, mechanistic, and causal claims.
4. Back visuals with numerical tests in decision-evidence space.

## 1) Repository scope
Active artifacts are v3-only. Legacy artifacts are archived in `v1/`.

## 2) Inputs
- Baseline logs are the source for scale-level results.
- A dedicated tracing run is required for attention + MLP decomposition.
- No paid external data is required.

## 3) Core derived metrics
Primary:
- logit margin delta(l)
- boundary proximity b(l) = |delta(l)|
- drift g(l) = delta(l+1) - delta(l)
- competitor identity k*(l)

Secondary:
- p_correct(l)
- probability margin
- entropy
- flip metrics
- projected-space basin gap

## 4) Phase 1: baseline-only validation
Outputs:
- `layerwise.parquet`
- `decision_metrics.parquet`
- `prompt_types.parquet`
- `type_counts.json`
- phase-diagram and trajectory plots
- `basin_gap.parquet`

## 5) Phase 2: minimal prompt span taxonomy
Outputs:
- `spans.jsonl`
- `span_effects.parquet`
- `span_labels.parquet`

Sampling:
- start with 200 balanced prompts per model
- scale to 1000 per model when stable

## 6) Phase 3: mechanistic tracing run
Per model tracing subset:
- 600 prompts balanced over four trajectory types and difficult domains

Record:
- attention mass by span
- decision-aligned attention contribution by span
- decision-aligned MLP contribution

Outputs:
- `tracing_scalars.parquet`
- `attention_mass_by_span.parquet`
- `attention_contrib_by_span.parquet`

## 7) Phase 4: causal validation
Required outputs:
- `ablation_results.parquet`
- `patching_results.parquet`
- `span_deletion_causal.parquet`
- `negative_controls.parquet`

## 8) Core paper assets
- Figure 1: primitives + four types
- Figure 2: phase diagram
- Figure 3: region entry/exit/re-entry
- Figure 4: routing mass vs contribution
- Figure 5: motion decomposition
- Figure 6: causal tests

## 9) Statistical discipline
- pre-register primary hypotheses
- bootstrap CIs
- train/test split for learned components
- multiple-comparison control
- cross-model replication for main figures

## 10) Optional appendix outcomes
Exploratory only:
- last-K-layer logit ensembling
- structured prompt rerun on small subset

## 11) Deliverables
Core:
- metrics + trajectory typing for all 3 models
- phase diagrams + trajectory bundles
- basin entry/exit analysis
- span parsing and operational labels
- tracing scalars and decomposition
- causal validations with negative controls
- methods file matching implementation

Stop conditions:
- if attention+MLP scalars do not explain drift, repair instrumentation before proceeding
- if span labels are unstable under paraphrase, reduce manipulation aggressiveness and standardize replacements

## 12) Script map
- `01_extract_baseline.py`
- `02_compute_decision_metrics.py`
- `03_classify_trajectories.py`
- `04_region_analysis.py`
- `05_span_counterfactuals.py`
- `06_select_tracing_subset.py`
- `07_run_tracing.py`
- `08_attention_and_mlp_decomposition.py`
- `09_causal_tests.py`
- `10_causal_validation_tools.py`
- `11_generate_paper_assets.py`
- `12_appendix_exploratory_outcomes.py` (optional)
