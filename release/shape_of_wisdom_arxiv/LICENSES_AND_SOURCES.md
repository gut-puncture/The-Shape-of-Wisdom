# Licenses And Sources

This release contains derived analysis data and paper build code from the local `shape-of-wisdom` project.

## Benchmark Sources

- ARC-Challenge: Clark et al., 2018, arXiv:1803.05457.
- CommonsenseQA: Talmor et al., 2019, NAACL, DOI:10.18653/v1/N19-1421.
- MMLU instrumented panel: Hendrycks et al., 2021, ICLR, arXiv:2009.03300.

## Model Sources

The tested model revisions are listed in `data/manifests/models.yaml`.

- Qwen 2.5 Instruct family: Yang et al., 2024, arXiv:2412.15115.
- Gemma 2 IT family: Gemma Team, 2024, arXiv:2408.00118.
- Llama Instruct family: Grattafiori et al., 2024, arXiv:2407.21783.

## Included Data

The release includes derived trajectory and mechanistic metrics, not raw benchmark datasets, raw model outputs, or model weights.

## Code

The release code is a minimal reproducibility subset for regenerating tables, figures, paper macros, manifests, and QA artifacts from included derived data.
