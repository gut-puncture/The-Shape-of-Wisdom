# CHANGELOG — Correctness Audit

## Round 2 — Three targeted changes (2026-03-01)

### Change 1 — Fix Simulated Activation Substitution (rigor + correctness)

**Problem:** Both `src/sow/v2/causal/patching.py` and Algorithm 2 in the paper implemented a simple overwrite (`g_patched[l] ← s_success[l]`) instead of the correct remove-then-add: `g_patched[l] = g_fail[l] - s_fail[l] + s_succ[l]`.

**Fix:**
- `src/sow/v2/causal/patching.py`: Corrected overwrite to `patched[i] = fail_drift[i] - fail_comp[i] + succ_comp[i]`.
- `results/parquet/patching_results.parquet`: Recomputed from stored `tracing_scalars.parquet` (no model inference). Updated numbers: attention 35.78% positive (mean −0.31 logits); MLP 85.00% positive (mean +5.75 logits).
- `paper/final_paper/paper_publish_v2.tex`: Algorithm 2 KwIn now lists `s_f[0..L-1]`; if-branch now reads `g_f[l] - s_f[l] + s_s[l]`. Introductory sentence, result numbers, and Figure 5 panel (c) caption updated.
- `paper/final_paper/figures/fig5_counterfactuals.pdf`: Regenerated.

### Change 2 — Proxy Validation: Token-Bucket vs Single-Token Readout (rigor)

**Problem:** Decomposition scalars use a single-token tracing readout as a proxy for the token-bucket readout, but no numerical validation of how close the proxy is was reported.

**Fix:**
- Computed on full tracing subset (55,200 prompt × layer pairs): Pearson r = 0.55, sign agreement = 75.4%. Proxy is moderate; claims are now honestly scoped.
- `src/sow/v2/figures/paper_figures.py`: Added `fig_appendix_proxy_validation()`.
- `paper/final_paper/figures/figA1_proxy_validation.pdf`: New appendix figure (hexbin δ_bucket vs δ_single, y=x reference line).
- `paper/final_paper/paper_publish_v2.tex`: New appendix section "Tracing-Readout Proxy Validation"; Methods paragraph updated with r = 0.55, sign agreement = 75.4%, and restriction of decomposition claims to single-token readout.

### Change 3 — Visual Redesign of Figure 2 and Figure 6 (clarity)

**Figure 2:** Replaced 9,000 opaque scatter points with hexbin density (`YlOrRd`). Covariance ellipses kept as unfilled overlays. Four region labels added inside quadrants. Threshold lines thickened to lw=1.4. Legend uses patch proxies.

**Figure 6:** Removed ribbon-stack schematic and phase-space panel. New 3-panel layout:
- Panel A: prompt spans (unchanged content).
- Panel B (full height): δ(l) vs real layer index; tail shaded; sign flips marked ▼; competitor strip below x-axis.
- Panel C: s_attn and s_mlp vs layer (same x-axis); zero reference; tail shaded.
Caption updated to match new layout.

- `paper/final_paper/figures/fig2_phase_diagram.pdf`: Regenerated.
- `paper/final_paper/figures/fig6_prompt_journey.pdf`: Regenerated.

---

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
