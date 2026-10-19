# The Shape of Wisdom: Decision Trajectories in Language Models

This release builds a two-column MMLU mechanistic paper from cached artifacts only.

## Scope

- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
- Data: 3,000 MMLU prompts per model.
- Evidence: layerwise option-score trajectories, cached attention/MLP tracing scalars, span deletion, negative controls, and substitution sensitivity.
- Excluded: all artifacts outside the 7--8B MMLU experiment.

## Rebuild

```bash
cd release/shape_of_wisdom_mmlu_mech_paper
python3 scripts/build_all.py
python3 -m pytest -q
```

The script regenerates figures, tables, manuscript source, PDF, page renders, contact sheet, QA docs, the `arxiv_source/` folder, and the files in `dist/`.

## Submission files

- PDF preview: `dist/shape_of_wisdom_mmlu_mech_paper.pdf`
- arXiv source upload: `dist/shape_of_wisdom_mmlu_mech_paper_arxiv_source.zip`
- Checksums: `dist/SHA256SUMS`
