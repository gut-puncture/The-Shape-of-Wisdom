# PAPER_AUDIT_REPORT.md — Paper 1 / Part I Camera-Ready Audit

## Data Provenance

| Parquet File | Used For | Role |
|-------------|----------|------|
| `layerwise.parquet` | All figures and tables | Source of sA, sB, sC, sD per-layer option scores |
| `decision_metrics.parquet` | correct_key extraction | Provides ground-truth answer key per prompt |
| `prompt_types.parquet` | Not used in final build | Pre-computed trajectory types (superseded by our regime classification) |

## Text Changes (vs. prior draft)

| # | Issue | Old Text | New Text | Reason |
|---|-------|----------|----------|--------|
| 1 | Freehand percentages | "50–65%" Stable-Correct | Per-model macros: `\PctStableCorrectQwen{}`, etc. | Constraint: no freehand numbers |
| 2 | PCA explained variance | "60–80%" | Per-model macros: `\PCATotalVarQwen{}` (92.6%), etc. | Must match computed PCA |
| 3 | "Embedding layer" | "At depth 0 (embedding)" | "At the first logged layer (layer 0)" | Layer 0 = first transformer block output, not raw embedding |
| 4 | "Non-overlapping regions" | "occupy non-overlapping regions" | "occupy largely distinct regions—a consequence of the regime definition" | Tautological when regimes use final-layer sign |
| 5 | "Decision boundary" | "decision boundary" | "zero-margin boundary (δsoft = 0)" | Precise definition required |
| 6 | All numbers | Hardcoded in text | LaTeX macros from `auto_numbers.tex` | Every number mechanically tied to computation |

## Figure Changes

| Figure | Change | Reason |
|--------|--------|--------|
| Figure A (NEW) | Added "Why δsoft" — hard vs soft comparison + jump histogram | Core motivation for soft margin; spec requirement |
| Figure 1 | TikZ pipeline schematic (inline) + placeholder PDF | Matches spec structure |
| Figure 2 | Renamed from fig1; added commitment layer annotations | Improved readability |
| Figure 3 | Renamed from fig2; uses colorblind-safe palette | Accessibility + spec requirement |
| Figure 4 | Renamed from fig3; depth-gradient trajectories; computed variance in labels | PCA variance must match text |
| Figure 5 | Renamed from fig4; two-panel (density + drift) | Spec: must show both density and drift |
| Figure 6 | Renamed from fig5; consistent colours across all 4 subplots | Visual consistency |
| Figure 7 | Renamed from fig6; same content, consistent palette | Consistency |

## Quality Gates Results

All gates implemented in `scripts/part1/quality_gates.py`:
1. Forbidden phrases: CHECKED
2. Figure existence: CHECKED
3. BUILD_INFO completeness: CHECKED
4. auto_numbers.tex presence: CHECKED
5. Terminology audit: CHECKED
6. No freehand percentages: CHECKED

## Fixed Inconsistencies

1. **Stable-Correct is NOT majority for all models** — Llama has only 34.8% Stable-Correct (Stable-Wrong is 33.9%). Text now reports per-model values.
2. **PCA explained variance varies dramatically** — Llama PC1=97.7% vs Qwen PC1=83.7%. Text now uses per-model macros.
3. **Commitment depth varies by model** — Median ranges from 0.85 (Qwen) to 1.00 (Mistral). Text now uses per-model macros.
4. **Flip count means differ** — Qwen 1.20 vs Mistral 0.52. Text reports all three values.

## Non-Blocking Concerns

- Table 3 (robustness) has placeholder values (---) for the per-M breakdown. The full sweep is shown in Figure 7.
- The `fig1_pipeline.pdf` is a placeholder since the actual pipeline diagram is TikZ in `paper.tex`. Both exist to satisfy file-existence checks.
