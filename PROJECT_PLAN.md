# Shape of Wisdom - PROJECT_PLAN (Template)

This file is a stable, run-agnostic checklist of stages.

Rules:
- Keep this plan free of run-specific counts, run history, and hypotheses.
- Run history, blockers, and "what is running now" live in `STATE.md`.
- Never proceed to the next stage unless the prior stage's validator passes and a `.done` sentinel exists.
- Cost discipline: build cheap validators + smoke tests first; avoid GPU until all local gates pass.
- Local Mac mini discipline: all inference stages must use thermal hygiene (see `docs/IMPLEMENTATION_SPEC.md` section 23).

Canonical docs:
- `docs/IMPLEMENTATION_SPEC.md`
- `docs/METHODOLOGY_NARRATIVE_V2.md`
- `STATE.md`

## Stage Checklist (authoritative order)

- [x] Stage 0 - Environment lock + smoke test
  - Artifacts: `runs/<run_id>/meta/environment.json`, `runs/<run_id>/meta/smoke_test.json`
  - Gate: smoke test PASS

- [ ] Stage 1 - Build canonical examples dataset
  - Inputs: dataset source, `configs/coarse_domain_mapping.json`
  - Artifacts: `data/derived/examples.jsonl`, `data/derived/examples.meta.json`
  - Gate: schema + `example_id` uniqueness + 4-option invariant PASS

- [x] Stage 2 - Build baseline prompt manifest
  - Artifacts: `runs/<run_id>/manifests/baseline_manifest.jsonl` + `.meta.json`
  - Gate: 1 row per `example_id`, unique `prompt_uid`, prompt ends with `Answer:` + trailing space PASS

- [x] Stage 3 - Build robustness prompt manifest (20 wrappers)
  - Artifacts: `runs/<run_id>/manifests/robustness_manifest_v2.jsonl` + `.meta.json`
  - Gate: exactly 20 wrappers per `example_id`, wrapper set matches expected list exactly, unique `prompt_uid` PASS

- [x] Stage 4 - Validate manifests and canonicalize if needed
  - Artifacts: canonical manifest(s) (only if needed) + canonicalization report
  - Gate: join-key uniqueness + wrapper coverage invariants PASS

- [x] Stage 5 - Build per-model option token buckets (A/B/C/D)
  - Artifacts: `runs/<run_id>/token_buckets/<model_id>.json`
  - Gate: buckets non-empty and handled deterministically PASS

- [x] Stage 6 - Deterministic answer parser + regression suite
  - Inputs: `artifacts/parser_edge_case_regression/regression_cases.json`
  - Artifacts: `runs/<run_id>/validation/parser_regression_report.json`
  - Gate: exact-match regression PASS

- [x] Stage 7 - Pilot inference (small)
  - Artifacts: `runs/<run_id>/pilot/*`
  - Gate: viability + one-token compliance thresholds met PASS

- [x] Stage 8 - Build Primary Core Corpus (PCC)
  - Artifacts: `runs/<run_id>/manifests/pcc_baseline.<model>.jsonl`, `runs/<run_id>/manifests/pcc_robustness.<model>.jsonl`, `runs/<run_id>/manifests/pcc_report.json`
  - Gate: stratification + invariants PASS

- [x] Stage 9 - Build Common Compatible Core (CCC)
  - Artifacts: `runs/<run_id>/manifests/ccc_*.jsonl`, `runs/<run_id>/manifests/ccc_report.json`
  - Gate: retention thresholds met PASS

- [x] Stage 10 - Freeze PCA sample membership deterministically
  - Artifacts: `runs/<run_id>/pca/<model_id>_sample_membership.json`
  - Gate: deterministic membership for fixed seed PASS

- [x] Stage 11 - PCA sample extraction inference
  - Artifacts: `runs/<run_id>/pca/<model_id>_sample_hidden.*`
  - Gate: shape + reproducibility checks PASS

- [x] Stage 12 - Fit PCA basis once per model
  - Artifacts: `runs/<run_id>/pca/<model_id>_pca_basis.*`
  - Gate: basis hash reproducible for same input PASS

- [ ] Stage 13a - Baseline inference runs with layerwise readouts + PCA projection (GPU recommended; local possible)
  - Artifacts: `runs/<run_id>/outputs/<model_id>/baseline_outputs.jsonl`, `runs/<run_id>/outputs/<model_id>/run_meta.json`
  - Gates: batch-consistency + resume simulation + schema validation PASS

- [ ] Stage 13b - Robustness inference runs with layerwise readouts + PCA projection (optional / deferred)
  - Artifacts: `runs/<run_id>/outputs/<model_id>/robustness_outputs.jsonl`, `runs/<run_id>/outputs/<model_id>/run_meta.json`
  - Gates: batch-consistency + resume simulation + schema validation PASS
  - Note: current fast-track execution skips this stage unless robustness deltas are explicitly required.

- [ ] Stage 14 - Mechanistic analysis and report generation (commitment + convergence + domain topology)
  - Required artifacts: `runs/<run_id>/analysis/per_prompt_metrics.csv`, `runs/<run_id>/analysis/layerwise_aggregates.csv`, `runs/<run_id>/analysis/convergence_by_layer.csv`, `runs/<run_id>/analysis/commitment_hist.csv`, `runs/<run_id>/analysis/domain_topology_centroids.csv`, `runs/<run_id>/analysis/domain_topology_pairwise_distances.csv`, `runs/<run_id>/analysis/figures/*.png`, `runs/<run_id>/analysis/final_report.json`
  - Optional artifacts (only when robustness mode is enabled): `runs/<run_id>/analysis/robustness_deltas.csv`
  - Gate: baseline manifest row-count parity + no missing layers + basis-hash match + topology/figure artifacts present PASS

## Baseline-Only Paper Readiness Gate (Robustness Deferred)

- Use this gate when publishing baseline-only mechanistic findings (commitment + convergence + domain topology) without robustness claims.
- Required stop condition for a run:
  - `runs/<run_id>/sentinels/inference_baseline.done` exists (Stage 13 baseline completion sentinel).
  - `runs/<run_id>/sentinels/analysis.done` exists.
  - `runs/<run_id>/analysis/final_report.json` has `pass=true` and no unresolved errors.
  - `runs/<run_id>/bundles/analysis_bundle_<run_id>.tar.gz` exists for download.
- If any condition fails, resume the same run with `scripts/gpu/run_full.sh <run_id> baseline_only` (do not start a new run_id unless artifacts are irrecoverable).
- Reporting discipline:
  - Clearly state that robustness analysis was deferred.
  - Limit claims to baseline mechanistic results and sample size used in that run.

## After Each Stage (required bookkeeping)
- Append a dated entry to `STATE.md` containing:
  - command line
  - input file paths + SHA-256 hashes
  - output file paths + SHA-256 hashes
  - validator outputs + PASS/FAIL
  - the next step
- Write a `.done` sentinel at `runs/<run_id>/sentinels/<stage_name>.done` containing the same hashes.

## 2026-02-17 20:43 (IST) - Session Plan (Deep Convergence, Baseline-Only, run `1452_a`)
- [ ] Verify baseline integrity for run `rtx6000ada_baseline_20260216_1452_a`:
  - row parity (3000/model), layer completeness, manifest join integrity.
- [ ] Formalize and compute convergence definitions from baseline outputs only:
  - probability target, margin target, stability-through-end, basin-distance criteria with explicit formulas.
- [ ] Compute correct-token dynamics on all prompts:
  - `p_correct(l)`, `m_correct(l)`, first-passage/commit layer, hazard, late flips, oscillations; then stratify by final correctness.
- [ ] Compute latent-space dynamics from `projected_hidden_128`:
  - displacement, path length, straightness, curvature proxy, alignment to correct-direction vectors, divergence windows.
- [ ] Compute prompt-difficulty structure:
  - hard-prompt subsets by domain/subject/structure and early-layer uncertainty signatures.
  - fit interpretable early-layer predictors of final convergence failure.
- [ ] Synthesize intervention levers grounded in measured baseline evidence:
  - prompt-level, inference-time, training/objective-level, representation-steering windows, each with minimal validation experiment.
- [ ] Produce deliverables under:
  - `downloads/gpu_runs_20260216_full/rtx6000ada_baseline_20260216_1452_a/analysis/deep_convergence_1452a`
  - required report: `DEEP_CONVERGENCE_REPORT.md`
  - include clean figures, CSV/JSON tables, executive summary, technical report.

## 2026-02-17 20:50 (IST) - Session Completion (Deep Convergence, Baseline-Only, run `1452_a`)
- [x] Verify baseline integrity for run `rtx6000ada_baseline_20260216_1452_a`.
- [x] Formalize and compute convergence definitions (probability, margin, stability-through-end, basin distance).
- [x] Compute correct-token dynamics across all prompts and stratify by final outcome.
- [x] Compute latent-space trajectory metrics and divergence layer windows from `projected_hidden_128`.
- [x] Compute prompt-difficulty structure and early-layer interpretable failure predictors.
- [x] Produce intervention levers with mechanism, expected effect, and minimal validation experiments.
- [x] Deliver artifacts under:
  - `downloads/gpu_runs_20260216_full/rtx6000ada_baseline_20260216_1452_a/analysis/deep_convergence_1452a`
  - includes `DEEP_CONVERGENCE_REPORT.md`, `EXECUTIVE_SUMMARY.md`, `TECHNICAL_REPORT.md`, CSV/JSON tables, and publication-quality figures.
