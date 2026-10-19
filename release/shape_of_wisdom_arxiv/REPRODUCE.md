# Reproduce The Release

## Tier 1: Rebuild From Included Derived Data

This is the expected reader path.

```bash
python3 scripts/verify_derived.py
python3 scripts/rebuild_all.py
python3 -m pytest -q
```

Expected outputs:

- `paper/generated/paper_numbers.tex`
- `paper/tables/*.tex`
- `paper/figures/*.pdf`
- `paper/build/main.pdf`
- `qa/page_pngs/*.png`
- `qa/render_report.md`
- `data/manifests/release_manifest.json`

## Tier 2: Optional Raw-Cache Verification

The included derived rows retain `cache_key` values but not absolute local paths. If raw cache bundles and the original merged run directory are available, verify the included rows with:

```bash
python3 scripts/verify_raw_cache.py \
  --run-dir /path/to/outputs/runs/paper_merged_9x1000_20260330T122258Z \
  --cache-root /path/to/outputs/caches/semantic_scores
```

This tier is optional because the release is designed to rebuild without raw bundles.

## Tier 3: Full Inference Rerun

A full rerun requires the development pipeline outside this release folder. It also requires the pinned model revisions, the benchmark datasets, Hugging Face access for gated models where applicable, and GPU hardware. The run provenance is preserved in `data/manifests/run_manifest.json` and `data/manifests/models.yaml`.

## QA Expectations

The rebuild fails on stale paper terms, missing generated assets, LaTeX errors, undefined references, undefined citations, and overfull boxes. After the PDF is rendered, inspect all PNGs in `qa/page_pngs/` before treating the PDF as final.
