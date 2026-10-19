# V3 Persistent Contender / Margin Reanalysis

This workspace performs a zero-new-inference reanalysis of `paper_merged_9x1000_20260330T122258Z`.

## Inputs
- Run manifest: `/Users/shaileshrana/shape-of-wisdom/outputs/runs/paper_merged_9x1000_20260330T122258Z`
- Primary manifest filter: `semantic_exact + identity`
- Raw bundle source: cache bundles resolved by `cache_key` under the configured cache root

## Zero-new-inference statement
No models were run for this workspace. The analysis uses already-produced per-layer exact-option score bundles.

## Exact command
```bash
python3 v3_persistent_contender_margin_fix/scripts/build_persistent_reanalysis.py \
  --run-dir /Users/shaileshrana/shape-of-wisdom/outputs/runs/paper_merged_9x1000_20260330T122258Z \
  --cache-root /Users/shaileshrana/shape-of-wisdom/outputs/caches/semantic_scores \
  --output-root /Users/shaileshrana/shape-of-wisdom/v3_persistent_contender_margin_fix/outputs
```

## Outputs
- Derived prompt-level rows: `outputs/derived_data/persistent_metrics_semantic_exact_identity.jsonl`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Paper inputs: `paper/figures/`, `paper/tables/`, `paper/generated/paper_numbers.tex`
- Paper PDF: `paper/build/main.pdf`

## What was cut from the old paper
- First-hit candidate-set entry as the main early event
- Control-based invariance claims
- Any main result built on transient-versus-persistent asymmetry

## Rebuild paper PDF
```bash
bash v3_persistent_contender_margin_fix/scripts/build_v3_pdf.sh
```
