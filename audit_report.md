# Cached-Artifact Audit Report (No New Inference)

Date: 2026-03-02  
Repo: `/Users/shaileshrana/shape-of-wisdom`  
Canonical manuscript target: [paper_publish_v3.tex](/Users/shaileshrana/shape-of-wisdom/paper/final_paper/paper_publish_v3.tex)

## Executive Summary

| Claim | Status | Verdict |
|---|---|---|
| Three primitives (δ, drift, boundary) classify trajectories reproducibly | PASS | Re-derived labels match stored `prompt_types` exactly (0 mismatches). |
| Drift decomposition has usable explanatory signal | CONDITIONAL | Stage-08 drift-level OLS gate passes (`R^2 >= 0.70`), but trajectory-level recurrence reconstruction error is large late in depth. |
| Legacy substitution result: MLP shifts larger than attention | PASS | Exactly reproduced cached `patching_results.parquet` (attention: -0.31 mean, 35.8% positive; MLP: +5.75 mean, 85.0% positive). |
| Strong claim “attention substitution does not work” | FAIL | Under order-invariant all-pairs pairing, attention is positive on average (mean +1.18, 53.0% positive). |
| Substitution finding is robust to settings | CONDITIONAL | MLP > attention persists across pairing modes tested, but sign/magnitude vary materially with layer range and normalization. |
| Panel references and prose consistency in v3 | FAIL | Text referenced substitution as panel (b); caption defines substitution as panel (c). Fixed in v3 source. |

## Scope and Guardrails

- No base-model inference was run.
- New audit/figure computations read cached artifacts only.
- Inference firewall implemented: `SOW_ALLOW_INFERENCE=1` is now required for inference-capable paths.

## Phase 0: Orientation + Inference Firewall

### Located entry points

- Analysis/report scripts: `scripts/v2/01`–`11`, plus new `scripts/audit/*`.
- Figure generators: [paper_figures.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/figures/paper_figures.py), [generate_figures_vnext.py](/Users/shaileshrana/shape-of-wisdom/scripts/audit/generate_figures_vnext.py).
- Cached artifacts: `results/parquet/*.parquet`, `results/spans.jsonl`, `results/reports/*.json`.
- Paper sources: `paper/final_paper/paper_publish*.tex`.

### Firewall implementation

Added [inference_firewall.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/inference_firewall.py) and integrated calls in:

- [00a_generate_baseline_outputs.py](/Users/shaileshrana/shape-of-wisdom/scripts/v2/00a_generate_baseline_outputs.py)
- [05_span_counterfactuals.py](/Users/shaileshrana/shape-of-wisdom/scripts/v2/05_span_counterfactuals.py)
- [07_run_tracing.py](/Users/shaileshrana/shape-of-wisdom/scripts/v2/07_run_tracing.py)
- [baseline_inference.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/baseline_inference.py)

Contract: any inference path without `SOW_ALLOW_INFERENCE=1` raises with actionable error.

### Safe vs dangerous scripts

From [artifact_integrity.json](/Users/shaileshrana/shape-of-wisdom/results/audit/artifact_integrity.json):

- Dangerous (inference-capable): `scripts/v2/00a_generate_baseline_outputs.py`, `scripts/v2/05_span_counterfactuals.py`, `scripts/v2/07_run_tracing.py`.
- Safe (analysis-only in this audit): `scripts/v2/08_attention_and_mlp_decomposition.py`, `scripts/v2/regenerate_paper_figures.py`, all `scripts/audit/*`.

### Reproduce command

Implemented root target:

- `make audit-vnext`

It runs [reproduce_vnext.py](/Users/shaileshrana/shape-of-wisdom/scripts/audit/reproduce_vnext.py), regenerates audit outputs and `figures_vNext` from cache, and fails loudly if required cached artifacts are missing.

## Phase 1: Definition Extraction and Consistency Diff

Machine extract: [definitions_extracted.json](/Users/shaileshrana/shape-of-wisdom/results/audit/definitions_extracted.json)

Canonical code-grounded definitions:

1. `δ(l)`
- Source: [metrics.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/metrics.py:88)
- Definition: `delta = correct_logit - competitor_logit` at each layer.

2. `k*(l)` closest competitor
- Source: [_competitor_from_logits](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/metrics.py:34)
- Definition: per-layer argmax over non-correct options; can change with `l`.

3. Option token set
- Source: [option_buckets.py](/Users/shaileshrana/shape-of-wisdom/src/sow/token_buckets/option_buckets.py:17)
- Variants: `{L}`, ` {L}`, `\n{L}`, `({L})`, `{L}.`, `{L}:` with bucket disjointness checks.

4. Stable/unstable
- Source: [trajectory_types.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/trajectory_types.py:57)
- Stability gate: tail late-flip count and `min_abs_delta_tail >= floor`; computed on `delta`/`sign(delta)`.

5. Component scalars `s_attn(l)`, `s_mlp(l)`
- Source: [07_run_tracing.py](/Users/shaileshrana/shape-of-wisdom/scripts/v2/07_run_tracing.py:324)
- Definition: margin differences from `h_in -> h_in+a` and `h_in+a -> h_in+a+m` (single-token tracing readout).

6. Substitution recurrence
- Source: [patching.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/causal/patching.py:59)
- Update: `g_patched = g_fail - s_fail + s_source` on target layers, then cumulative drift integration.

### Definition consistency mismatches (resolved/flagged)

- Bucket-readout vs single-token readout mismatch:
  - `decision_metrics` (bucket) vs tracing scalars (single token).
  - Interpretation: decomposition/substitution claims apply directly to single-token tracing readout.
- Failing-set mismatch:
  - Code in [09_causal_tests.py](/Users/shaileshrana/shape-of-wisdom/scripts/v2/09_causal_tests.py:54) uses `stable_wrong + unstable_wrong` while prose often implied stable-wrong-only.
- Legacy source-pairing bug:
  - [patching.py](/Users/shaileshrana/shape-of-wisdom/src/sow/v2/causal/patching.py:40) selects first stable-correct source per model.
- Panel mismatch in v3 prose:
  - Substitution was referenced as panel (b), caption says panel (c). Fixed in v3 source.

## Phase 2: Data Integrity Checks

Artifacts: [artifact_integrity.json](/Users/shaileshrana/shape-of-wisdom/results/audit/artifact_integrity.json), [trajectory_spotcheck_summary.json](/Users/shaileshrana/shape-of-wisdom/results/audit/trajectory_spotcheck_summary.json)

Results:

- Required artifact presence: PASS.
- NaN/inf checks: PASS (0 NaN/inf in required numeric fields).
- Duplicate key checks: PASS (0 duplicates on expected keys).
- Layer completeness/order:
  - `decision_metrics`: complete, in-order.
  - `tracing_scalars`: complete, in-order.
- Reclassification agreement:
  - Stored vs re-derived `prompt_types`: 0 mismatches.
- Final-layer plausibility:
  - `sign(delta_final)` vs `is_correct`: 0.99944 agreement.

20-prompt spot-check summary:

- Identity `delta(l+1)-delta(l) == drift(l)`: exact in sampled trajectories.
- Accounting residual `drift - (s_attn + s_mlp)`: mean abs residual ~1.56 logit units (non-trivial omitted terms).

## Phase 3: Substitution Audit (Core Dispute)

Independent recomputation:

- Script: [substitution_rederive.py](/Users/shaileshrana/shape-of-wisdom/scripts/audit/substitution_rederive.py)
- Per-pair output: [substitution_pairs_vnext.csv](/Users/shaileshrana/shape-of-wisdom/results/audit/substitution_pairs_vnext.csv)
- Sensitivity summary: [substitution_sensitivity_summary.csv](/Users/shaileshrana/shape-of-wisdom/results/audit/substitution_sensitivity_summary.csv)

### Reproduction and diagnostics

- Legacy deterministic mode (`legacy_first_per_model`) exactly reproduces cached `patching_results`.
- Legacy row-order perturbation changes attention mean strongly (max abs mean shift diff: 3.258).
- `all_pairs_within_model` is order-invariant (exact same means and pair counts under perturbation).
- Attention and MLP are compared on identical pair keys per setting (asserted and tested).

### Main substitution outcomes

Legacy cached protocol (matches stage-09 output):

- Attention: mean `-0.3099`, frac positive `0.3578`.
- MLP: mean `+5.7479`, frac positive `0.8500`.

Order-invariant all-pairs protocol:

- Attention: mean `+1.1830`, frac positive `0.5302`.
- MLP: mean `+4.7419`, frac positive `0.7799`.

Interpretation:

- “MLP > attention” survives as a relative statement.
- “attention substitution does not work” does not survive under order-invariant pairing.

### Sensitivity highlights

- Pairing mode: all tested pairing modes keep MLP larger than attention on average.
- Layer range: sign flips by range (attention strongest mid; MLP strongest late under raw settings).
- Failing-set choice (`stable_wrong_only` vs `stable_wrong + unstable_wrong`) changes effect sizes.
- Normalization (`raw` vs per-layer standardization) materially changes sign/magnitude.

### Blocked sensitivity toggles (no new inference)

Blocked in this audit and explicitly encoded with `blocked_reason` in summary CSV:

- Competitor selection alternatives (dynamic/fixed variants).
- Option-token-set alternatives (strict/inclusive variants).

Reason: cached tracing scalars do not include the per-layer token-logit snapshots needed to recompute those choices.

## Contradictions Resolved

Primary memo: [contradiction_memo.json](/Users/shaileshrana/shape-of-wisdom/results/audit/contradiction_memo.json)

Quoted legacy/current claims that conflicted:

- v3 pre-audit phrasing (line 446 before this edit): “MLP substitution consistently improves failing trajectories while attention substitution does not.”
- legacy manuscript (`paper_publish.tex`, line 440): substitution summary reported “63%” attention-positive vs “82%” MLP-positive.
- v3 panel pointer bug: in-text referenced `\\cref{fig:causal}b` while caption defines substitution as panel (c).

### What caused the mismatch

Not one mechanism “proving opposite conclusions,” but multiple definition/protocol drifts:

1. Legacy source selection bug (first key per model) made substitution row-order sensitive.
2. Code/prose failing-set mismatch (stable+unstable wrong in code vs stable-wrong-only wording).
3. Panel reference mismatch in prose (`b` vs actual substitution panel `c`).
4. Conceptual mismatch risk: scalar substitution under linearized bookkeeping vs full in-graph activation patching.

### Corrected claim (for v3)

Under cached scalar-substitution accounting, MLP substitution yields larger positive margin shifts than attention in audited pair sets, but this is conditional on pairing protocol, layer range, normalization, and readout definition.

## Phase 4: Linearized Drift-Model Fidelity

Artifacts:

- [drift_reconstruction_audit.json](/Users/shaileshrana/shape-of-wisdom/results/audit/drift_reconstruction_audit.json)
- [drift_reconstruction_by_layer.csv](/Users/shaileshrana/shape-of-wisdom/results/audit/drift_reconstruction_by_layer.csv)
- [drift_reconstruction_by_type.csv](/Users/shaileshrana/shape-of-wisdom/results/audit/drift_reconstruction_by_type.csv)
- Figures in [figures_vNext](/Users/shaileshrana/shape-of-wisdom/figures_vNext)

Findings:

- Drift-level fit (stage-08 report) passes gate (`R^2 >= 0.70` by model).
- Trajectory reconstruction by recurrence shows large late-layer errors:
  - Unit-coefficient MAE rises from ~2.11 (early) to ~6.72 (late).
  - OLS-coefficient MAE rises from ~1.03 (early) to ~4.51 (late).
- Errors are higher without competitor switches than with switches in this cached set.

Implication:

- Use decomposition as descriptive/diagnostic accounting.
- Downweight strong causal claims that require faithful long-horizon recurrence under the linearized model.

## Phase 5: Figure Rebuild + Manuscript Updates

Generated vNext figure suite in [figures_vNext](/Users/shaileshrana/shape-of-wisdom/figures_vNext):

- trajectory primitives
- stability map
- decomposition aggregates
- substitution sensitivity small multiples
- linearization fidelity
- representative prompt journey

All generated by [generate_figures_vnext.py](/Users/shaileshrana/shape-of-wisdom/scripts/audit/generate_figures_vnext.py).

Manuscript edits applied in [paper_publish_v3.tex](/Users/shaileshrana/shape-of-wisdom/paper/final_paper/paper_publish_v3.tex):

- Fixed substitution panel reference to `\cref{fig:causal}c`.
- Fixed span-deletion panel reference to `\cref{fig:causal}d`.
- Reframed substitution claim conservatively with legacy vs all-pairs caveat.
- Added explicit note of cached stage-09 failing-set and legacy pairing protocol.
- Softened conclusion wording to conditional robustness framing.

## Phase 7: PDF/Visual QA Checklist

### Figure files (vNext)

Visual check status:

- `fig_vnext_1` primitives: PASS
- `fig_vnext_2` stability map: PASS
- `fig_vnext_3` decomposition: PASS
- `fig_vnext_4` substitution sensitivity: PASS (fixed a clipping/collapse bug during QA)
- `fig_vnext_5` linearization fidelity: PASS
- `fig_vnext_6` representative journey: PASS

### Compiled paper pages containing figures/tables

- Visually inspected current compiled v3 pages from `tmp/paper_pages/page_004`–`page_012` images.
- Checklist on currently compiled PDF: zero visible clipping, zero overlaps, readable labels, sane panel proportions.

Build caveat:

- `latexmk` is unavailable in this environment, so the edited `paper_publish_v3.tex` could not be recompiled during this run.
- Existing `paper_publish_v3.pdf` visual QA therefore reflects the pre-existing compiled PDF state.

## Required Code Changes (Implemented)

1. Added inference firewall module and integrated calls in all specified inference-capable scripts/functions.
2. Added independent substitution audit pipeline and CSV schemas:
   - `substitution_pairs_vnext.csv`
   - `substitution_sensitivity_summary.csv`
3. Added integrity checks, definition extraction, contradiction memo, spot-check tooling.
4. Added drift reconstruction audit and fidelity/error figures.
5. Added `figures_vNext` generation pipeline with consistent styling.
6. Added single-entry reproducibility runner and `make audit-vnext` target.
7. Added tests for firewall contract and substitution pairing invariants.

Verification tests run:

- `python3 -m unittest tests.v2.test_audit_vnext_repro_contract tests.v2.test_inference_firewall_contract tests.v2.test_substitution_rederive_contract`
- Result: 10 tests, all passed.

## Required Paper Changes (Implemented + Remaining)

Implemented in v3 source:

1. Corrected panel references (substitution and span-deletion).
2. Replaced unconditional substitution language with conditional robustness framing.
3. Added explicit protocol disclosure (failing set + legacy pairing behavior).

Still required before final camera-ready:

1. Rebuild `paper_publish_v3.pdf` from updated source once LaTeX toolchain is available.
2. Re-run page-level visual QA on the newly compiled PDF (not just existing compiled artifact).
3. Align in-paper figure set with `figures_vNext` if adopting vNext visuals.

## Minimal Additional Inference Needed (Not Run)

Blocked analyses (competitor-mode and option-token-set substitution sensitivity) require extra cached logits.

Minimal run to unlock:

- Extend tracing to save per-layer choice logits for `h_in`, `h_in+a`, and `h_in+a+m`, plus component norms.
- Run exactly 540 prompts total (180/model; 60 stable-correct, 60 stable-wrong, 60 unstable-wrong).

Suggested command pattern (after adding the logging flag):

1. `SOW_ALLOW_INFERENCE=1 python3 scripts/v2/06_select_tracing_subset.py --run-id <new_run_id> --config configs/experiment_v2.yaml --resume`
2. `SOW_ALLOW_INFERENCE=1 python3 scripts/v2/07_run_tracing.py --run-id <new_run_id> --config configs/experiment_v2.yaml --resume`

Why this is minimal:

- Uses the same instrumentation stage already in pipeline.
- Adds only the missing cached observables needed for blocked toggles.
- Keeps model count and prompt budget bounded while balancing key trajectory categories.

---

## Final Classification

### PASS

- Cached artifacts are internally consistent (shape/layers/NaN/dup/reclassification).
- Legacy substitution numbers are exactly reproducible from cached arrays.
- MLP substitution tends to exceed attention substitution across tested pairing settings.
- Negative controls and span-deletion artifacts remain coherent with prior interpretation caveats.

### CONDITIONAL

- Substitution effect direction/magnitude depends on pairing, layer range, failing-set scope, and normalization.
- Drift decomposition is useful descriptively but not strong evidence of faithful long-horizon mechanistic recurrence.

### FAIL

- Unqualified claim that attention substitution “does not work.”
- v3 prose panel reference mismatch (was incorrect before this edit).
- Any claim that ignores legacy row-order sensitivity in cached substitution protocol.
