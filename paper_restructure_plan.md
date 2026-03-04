# Paper v3 Restructure Plan (Claim-First, Cache-Verified)

Canonical manuscript: [paper_publish_v3.tex](/Users/shaileshrana/shape-of-wisdom/paper/final_paper/paper_publish_v3.tex)

## Narrative Goals

1. Lead with immediately visible phenomenon (single trajectory behavior).
2. Separate operational definitions from empirical findings.
3. Keep decomposition/substitution claims explicitly conditional where sensitivity is high.
4. Avoid causal overreach where linearized recurrence fidelity is weak.

## Proposed Section Outline

1. **Phenomenon First: Decision Trajectories Exist**
- Claim: intermediate layers show non-trivial state evolution before final answer.
- Evidence: representative prompt + trajectory primitives.
- Figures: `fig_vnext_6`, then `fig_vnext_1`.

2. **Operational Taxonomy: Stable vs Unstable Regimes**
- Claim: tail-window criteria define four reproducible trajectory classes.
- Evidence: exact reclassification agreement and stability map.
- Figures: `fig_vnext_2`.
- Caveat: this is an operational partition, not a discovered discontinuity.

3. **Mechanistic Decomposition as Accounting Tool**
- Claim: `s_attn`/`s_mlp` explain drift directionally and expose type-specific patterns.
- Evidence: decomposition aggregates; stage-08 drift-level fit gate.
- Figures: `fig_vnext_3`.
- Caveat: tracing readout is single-token proxy; not full bucket readout.

4. **Counterfactual Probes: What is Robust vs Protocol-Sensitive**
- Claim: substitution effects depend on pairing/layer/normalization choices; MLP tends to exceed attention, but absolute sign for attention is protocol-sensitive.
- Evidence: substitution sensitivity small multiples + legacy reproducibility.
- Figures: `fig_vnext_4`.
- Caveat: legacy first-source pairing is row-order sensitive and should be labeled diagnostic/legacy only.

5. **Linearization Fidelity Limits**
- Claim: recurrence-level reconstruction error grows late in depth; counterfactual bookkeeping claims should be phrased cautiously.
- Evidence: layer-wise and type-wise error plots.
- Figures: `fig_vnext_5`.

6. **Span-Level Input Intervention and Controls**
- Claim: span deletion is true intervention; evidence/distractor separation is operationally defined and supported by independent controls.
- Evidence: existing causal panel span-deletion + controls.
- Figures: existing Fig 5(d) panel (or split into dedicated figure for readability).

7. **Discussion: Robust Findings, Conditional Findings, Open Gaps**
- Robust: taxonomy reproducibility; MLP>attention tendency across tested pairings.
- Conditional: absolute attention substitution direction; layer-range effects.
- Gaps: blocked competitor-mode and option-token-set substitution sensitivity without additional inference.

## Figure Placement Map

1. Intro end: `fig_vnext_6_representative_prompt_journey`
- Proves: single-example interpretability and observable trajectory transitions.

2. Primitives section: `fig_vnext_1_trajectory_primitives`
- Proves: state/motion/boundary signatures by trajectory type.

3. Classification section: `fig_vnext_2_stability_map`
- Proves: operational regimes and threshold geometry.

4. Decomposition section: `fig_vnext_3_decomposition_aggregates`
- Proves: attention/MLP contribution patterns with uncertainty bands.

5. Counterfactual section: `fig_vnext_4_substitution_sensitivity`
- Proves: substitution sensitivity structure; robust relative trend and conditional absolute effects.

6. Limitations subsection (before discussion): `fig_vnext_5_linearization_fidelity`
- Proves: where linearized recurrence is weak; supports conservative causal wording.

## Claim Policy (Enforced)

- Every non-trivial claim must map to explicit artifact evidence in [claim_evidence_matrix.md](/Users/shaileshrana/shape-of-wisdom/claim_evidence_matrix.md).
- Unsupported statements must be softened or removed.
- “Phase transition” language should remain removed unless a threshold-robust discontinuity is demonstrated.
- Substitution claims must always state pairing mode, failing-set scope, layer range, and normalization mode.

## Required v3 Text Edits

Implemented already in source:

1. Fix substitution panel reference to panel (c).
2. Fix span-deletion reference to panel (d).
3. Replace unconditional substitution language with conditional robustness framing.
4. Add explicit methods disclosure: stage-09 uses failing `stable_wrong + unstable_wrong` and legacy first-source pairing.

Still required before final submission:

1. Recompile v3 PDF from updated TeX once LaTeX toolchain is available.
2. Re-run full figure/table page visual QA on that rebuilt PDF.
3. If adopting `figures_vNext`, update figure includes and captions accordingly.

## Minimal Additional Inference (Only if unblocking blocked toggles)

- Add per-layer cached logits for `h_in`, `h_in+a`, `h_in+a+m` and component norms in tracing outputs.
- Run 540 prompts total (180/model = 60 stable-correct, 60 stable-wrong, 60 unstable-wrong).
- This enables competitor-mode and option-token-set sensitivity without full pipeline rerun.
