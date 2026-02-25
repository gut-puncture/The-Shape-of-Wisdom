# CHANGELOG — Correctness Audit

All changes target logical correctness using only existing data artifacts. No new model runs were performed.

## 1. Circularity in span labeling (Issue #1)

**Problem:** Evidence/distractor labels are defined by Δ = δ_full − δ_deleted, and the paper then claimed "evidence spans have larger causal effects" using the same quantity—making the claim circular.

**Fix:**
- Rewrote §6.2 as "Span Labeling via Counterfactual Effect (Operational Definition)"
- Added explicit note that evidence/distractor separation is true by construction
- Removed "validation" framing from §8.3 span deletion tests
- Updated fig4 panel (c) caption to read "Distribution of counterfactual effect by operationally defined span type"
- Added circular span labeling to the Limitations section

## 2. Attention routing mis-indexed (Issue #2)

**Problem:** Attention heatmaps used absolute option_A/B/C/D columns. The text interpreted this as "routes to correct option" but columns were not indexed relative to correctness.

**Fix:**
- Re-indexed option spans to `option_correct`, `option_competitor`, `option_other`
- Normalized attention mass per token (using character-length proxy)
- Regenerated fig5 with corrected column labels and separate mass/contribution rows
- Updated §6 text and figure caption

## 3. Causal language too strong (Issue #3)

**Problem:** Simulated component removal/patching was called "causal ablation"/"causal patching" but was not a true in-graph intervention.

**Fix:**
- Replaced "causal" → "linearized counterfactual accounting" for simulated experiments
- Replaced "ablation" → "simulated component removal", "patching" → "simulated activation substitution"
- Added dedicated limitations paragraph explaining the drift-bookkeeping model
- Kept "causal" only for span deletions (true input-level interventions)
- Updated abstract, contributions, §8 title, conclusion

## 4. Phase diagram algebraic redundancy (Issue #4)

**Problem:** Plotting δ(L) vs Σg was partially redundant because Σg telescopes to δ(L) − δ(L−τ).

**Fix:**
- Changed axes to: x = |δ(L−τ)| (boundary proximity at tail entry), y = tail flip count
- These axes are algebraically independent
- Regenerated fig2 and updated §4.2 text

## 5. Layer indexing inconsistencies (Issue #5)

**Problem:** Qwen has 28 layers, Llama/Mistral have 32. Cross-model plots could be misleading.

**Fix:**
- Normalized all layer axes to depth fraction l/(L−1) ∈ [0,1]
- Regenerated fig1, fig3, fig5, fig6 with normalized depth
- Added "Layer indexing" section in Methods

## 6. Restricted-choice readout unclear (Issue #6)

**Problem:** Not stated whether logits/probabilities are over {A,B,C,D} only or full vocabulary.

**Fix:**
- Added explicit statement in §2 that this is a restricted-choice readout
- Added to Methods section
- Updated Limitations to mention restricted-choice readout

## Files changed

- `paper/final_paper/paper_publish.tex` — all text fixes
- `paper/final_paper/fig1_three_primitives.png` — regenerated
- `paper/final_paper/fig2_phase_diagram.png` — regenerated
- `paper/final_paper/fig3_decomposition.png` — regenerated
- `paper/final_paper/fig4_causal_validation.png` — regenerated
- `paper/final_paper/fig5_attention_routing.png` — regenerated
- `paper/final_paper/fig6_prompt_flow.png` — regenerated
- `scripts/v2/regenerate_figures.py` — new figure generation script
- `scripts/99_figure_qc.py` — new QC script
