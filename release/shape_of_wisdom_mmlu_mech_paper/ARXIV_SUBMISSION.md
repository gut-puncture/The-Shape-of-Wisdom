# arXiv Submission Metadata

## Title

The Shape of Wisdom: Decision Trajectories in Language Models

## Authors

Shailesh Rana

## Abstract

Language models do not simply choose an answer at the output layer. In a 9,000-trajectory MMLU study across Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3, the score of the answer moves across depth in structured ways. We describe each trajectory with three quantities: the current answer margin, the next-layer change in that margin, and the distance from a decision flip. The main empirical picture is that correctness and stability are different: the largest group is unstable-correct, not stable-correct. A traced subset then asks what moves the margin. In stable-correct cases, the average attention scalar points in the correct direction, while the average MLP scalar does not; span deletion shows that removing answer-supporting text hurts the margin and removing distractor-like text helps it. The result is not a full circuit explanation. It is a reproducible way to see which answers are settled, which remain fragile, and which measured sources move them.

## Suggested arXiv Fields

- Primary subject class: cs.CL (Computation and Language)
- Cross-lists: cs.LG (Machine Learning), cs.AI (Artificial Intelligence)
- Comments: 6 pages, 5 figures. Code and derived artifacts: https://github.com/gut-puncture/The-Shape-of-Wisdom
- License: CC BY 4.0
- Journal reference: leave blank
- DOI: leave blank
- Report number: leave blank

## Files

- Upload source: `dist/shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip`
- Preview PDF: `dist/shape_of_wisdom_mmlu_mech_paper.pdf`
