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

- [ ] Stage 13a - Baseline inference runs with layerwise readouts + PCA projection (local-friendly)
  - Artifacts: `runs/<run_id>/outputs/<model_id>/baseline_outputs.jsonl`, `runs/<run_id>/outputs/<model_id>/run_meta.json`
  - Gates: batch-consistency + resume simulation + schema validation PASS

- [ ] Stage 13b - Robustness inference runs with layerwise readouts + PCA projection (GPU recommended)
  - Artifacts: `runs/<run_id>/outputs/<model_id>/robustness_outputs.jsonl`, `runs/<run_id>/outputs/<model_id>/run_meta.json`
  - Gates: batch-consistency + resume simulation + schema validation PASS

- [ ] Stage 14 - Analysis and report generation
  - Artifacts: `runs/<run_id>/analysis/*` (final reports + figures)
  - Gate: manifest row-count parity + no missing layers + basis-hash match PASS

## After Each Stage (required bookkeeping)
- Append a dated entry to `STATE.md` containing:
  - command line
  - input file paths + SHA-256 hashes
  - output file paths + SHA-256 hashes
  - validator outputs + PASS/FAIL
  - the next step
- Write a `.done` sentinel at `runs/<run_id>/sentinels/<stage_name>.done` containing the same hashes.


<!-- AUTO_STATUS_START -->
- **Last auto update:** 2026-02-14 08:02:00 IST
- **Qwen filter progress:** 14042/14042 (100.00%)
- **Nano sanity:** status=`completed` batch_status=`completed` last_polled=`2026-02-14 08:01:59 IST`
- **Mini sanity:** status=`not_submitted`
- **Final sanity decision:** `gpt-5-nano` / `primary_perfect`
- **Last runner status:** `ok` actions=[] errors=[]
- **This cycle actions:** ['none']
- **This cycle errors:** ['none']
<!-- AUTO_STATUS_END -->
