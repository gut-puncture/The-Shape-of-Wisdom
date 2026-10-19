# Archive

Earlier phases of the Shape of Wisdom project. Nothing here is used by the current experiment.

| Folder | What it is | Status |
|---|---|---|
| `2026-02_mmlu_decision_trajectories/release/` | Paper, figures and cached data for the arXiv paper *The Shape of Wisdom: Decision Trajectories in Language Models* (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B; 3,000 MMLU questions). The full pipeline that produced it is at git tag `mmlu-arxiv-v1`. | Superseded. A review in October 2026 found that its headline claims do not hold (see below). |
| `2026-03_colm_candidate_vs_margin/` | The 9-model ARC-Challenge / CommonsenseQA pipeline, the COLM 2026 submission source, a persistent-metric reanalysis, and a draft combined paper. | Superseded. Desk-rejected at COLM 2026. |

## Why both were superseded (review, 19 October 2026)

MMLU paper:
- Intermediate layers were read through the unembedding without the model's final normalisation.
- The "stable / unstable" trajectory types came from a fixed 8-layer tail window that, for some models, starts before the answer appears; with a 4-layer window Qwen's stable share goes from 29% to 80%.
- Span "evidence / distractor" labels were defined by the sign of the deletion effect, so the reported separation is true by construction.
- The attention + MLP "drift R²" is an arithmetic identity (the two scalars sum exactly to the drift whenever the competitor letter is unchanged).
- "Ablation" and "patching" were sums of cached scalars, across different questions, not forward passes.

COLM pipeline:
- The headline lag compares a first-hit event with a persistent one, so it cannot be negative.
- The primary option-text readout averages log-probability per token and picks the longest option in 72–96% of items (chance about 36%).

What the data did show: the letter readout sits at chance until 55–80% of depth and then the answer appears within about two layers; trajectory features add nothing over final confidence for predicting correctness; and errors are shared across models far more than chance.
