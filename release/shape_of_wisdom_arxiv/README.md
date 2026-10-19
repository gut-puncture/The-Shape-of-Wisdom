# The Shape Of Wisdom

This folder is the clean arXiv release for the layerwise answer-formation paper.

## What The Paper Studies

The paper asks how a model's multiple-choice answer forms across depth. It studies cached layerwise exact-option continuation scores for nine instruction-tuned open-weight models on ARC-Challenge and CommonsenseQA, then adds a cached mechanistic deep dive from an instrumented three-model MMLU panel.

## What Is Included

- `paper/`: LaTeX source, generated figures, generated tables, generated macros, and the built PDF.
- `data/derived/`: compact trajectory and mechanistic rows used to rebuild every reported number and figure.
- `data/manifests/`: run provenance, model registry, and release file hashes.
- `src/shape_of_wisdom_release/`: minimal code for metric verification, table/figure generation, PDF rendering, and QA.
- `docs/`: release provenance, claim-evidence matrix, and excluded findings.
- `qa/`: rendered page PNGs and QA reports after a rebuild.

## Rebuild

```bash
python3 scripts/rebuild_all.py
```

After TeX resources are cached, a cached-only PDF rebuild is:

```bash
python3 scripts/rebuild_all.py --offline
```

## What Is Not Required

The main rebuild does not require GPU access, model weights, Hugging Face credentials, raw cache bundles, or new inference.

Readers who do have the original raw cache bundles can run `scripts/verify_raw_cache.py` to recompute the persistence rows and compare them with the included derived data.

## Full Inference Rerun

The release documents the source run and model revisions, but it is not a full experiment launcher. A full inference rerun requires the broader development repo, the pinned models, the benchmark datasets, and GPU hardware.
