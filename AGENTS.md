# AGENTS Policy For Shape of Wisdom

This repository targets Anthropic-level research and engineering rigor.
The explicit objective is an award-winning research paper at ICML/NeurIPS caliber.
All experiment and analysis decisions must prioritize 100% methodological rigor over speed.

## Standards
- Paper quality target: competitive with top-tier ICML/NeurIPS award-level work.
- No claim without measurable evidence and reproducible artifacts.
- Distinguish descriptive, mechanistic, and causal claims.
- Every expensive stage must be stage-gated with deterministic validators.
- Tests-first for all net-new behavior.
- Preserve deterministic seeds, model revisions, and artifact hashes.

## Non-negotiables
- No hidden hand-tuning after seeing results.
- No unlogged data transformations.
- No skipping negative controls for causal experiments.
- No merging when any test or validator is failing.

## Execution discipline
- Prefer smallest clear change surface and reviewer-friendly structure.
- Keep all run contracts explicit in config files.
- Log thermal and runtime safety decisions in run metadata.
- Keep final artifacts in a single auditable bundle.
