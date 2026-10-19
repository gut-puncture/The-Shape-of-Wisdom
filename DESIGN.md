# Design

This pipeline is organized around explicit, serializable artifacts:

1. canonical dataset examples
2. rendered prompts
3. tokenization audits
4. per-readout layerwise score bundles
5. per-anchor metrics
6. QC summaries and failure registries

The main path is exact option-text scoring. Templated semantic and letter-label readouts are
separate controls, not alternate views of the same cached output.

All writes are routed through workspace-local paths under `outputs/` or `manifests/`.
The code does not write derived artifacts outside this workspace.

## Main Contracts

- Dataset canonicalization preserves raw fields and writes explicit exclusion reasons.
- Prompt construction preserves the exact rendered prompt and answer prefix.
- Tokenizer audit records the constructed surface actually scored.
- Layerwise scoring uses teacher forcing over exact option surfaces.
- Metrics are computed once per anchor and once per threshold configuration.
- QC reports separate warnings from hard failures.

