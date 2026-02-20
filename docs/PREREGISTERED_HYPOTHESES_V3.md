# Preregistered Hypotheses V3

This document records primary hypotheses before running full V3 experiments.
It is part of the required reproducibility bundle contract.

## Scope
- Pipeline: `scripts/v2/01` through `scripts/v2/11`
- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- Seed policy: deterministic seed from `configs/experiment_v2.yaml`

## Primary hypotheses
1. `H1_descriptive`:
State (`delta`), motion (`drift`), and boundary (`|delta|`) are sufficient to separate stable vs unstable trajectory outcomes.
2. `H2_mechanistic`:
Per-layer attention and MLP scalars jointly explain observed drift with model-level reconstruction quality above configured floor.
3. `H3_causal_component`:
Component ablation/patching produces directionally consistent finite `delta_shift` effects for both attention and MLP.
4. `H4_causal_span`:
Evidence-labeled spans have stronger causal effect than distractor-labeled spans after multiple-comparison control and negative-control margins.

## Planned validators
- Stage 05: paraphrase stability gates.
- Stage 06: difficult-domain + domain-replication gates.
- Stage 08: decomposition R2 + deterministic train/test split gates.
- Stage 09: component coverage + finite shift + expected-model coverage gates.
- Stage 10: statistical significance + control strength + deterministic split gates.
- Stage 11: fail-closed artifact bundle including this preregistration document.
