# Paper Outline

## Title candidates
- From Competition to Late Commitment in Cached Layerwise Answer Readouts
- Boundary Dwelling and Empirical Reversibility in Layerwise Multiple-Choice Readouts
- Late Stabilization in Transformer Answer Readouts Without New Inference

## One-sentence thesis
Cached four-option readouts reveal answer formation as a trajectory from reversible competition to later low-reversibility commitment, with flip-heavy trajectories spending more depth near the decision boundary and with the timing link to stabilization treated as model-dependent.

## Main text
1. Problem: final answers hide the path.
2. Setup: cached layerwise four-option scores on MMLU.
3. Better observables: soft margin and current correct-option probability.
4. Better state description: commitment `a` versus competitor dispersion `q`.
5. Main finding: trajectories move from competition to later stabilization.
6. Strongest result: empirical future-flip map and commitment as low future-flip probability.
7. Boundary dwelling: boundary-near trajectories show more flips, while the later-stabilization link is mixed across models.
8. Limits: readout-space only.
9. Conclusion: answer formation is a depth-wise trajectory, not just a final-layer label.

## Appendix
- tie robustness
- old-threshold comparison
- per-model reversibility maps
- travel metrics
- boundary-width robustness
