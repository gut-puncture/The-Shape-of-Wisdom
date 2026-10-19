# Claim Evidence Matrix

| Claim | Evidence class | Source data | Generated artifact | Status |
|---|---|---|---|---|
| A model's final multiple-choice answer is the endpoint of a layerwise score trajectory. | Framing / measurement | `data/derived/trajectory_rows.jsonl` | Setup section, Table 1 | Included |
| The final winner usually becomes a persistent contender before it reaches a persistent decisive margin. | Primary | `data/derived/trajectory_rows.jsonl` | Figure 1, Figure 2, generated macros | Included |
| The pooled median `Delta_250` is nontrivial in normalized depth. | Primary | `data/derived/trajectory_rows.jsonl` | Figure 2, Table 2, generated macros | Included |
| A substantial share of positive winner-gap growth is written after `d2`. | Primary | `data/derived/trajectory_rows.jsonl` | Figure 3, Table 2, generated macros | Included |
| Broad answer-formation trajectories can be grouped into stable, late-crystallizing, switch-heavy, and unstable regimes. | Descriptive | `data/derived/trajectory_rows.jsonl` | Figure 4, Table 3 | Included |
| Attention/MLP scalar contributions provide useful held-out accounting of margin drift in the instrumented panel. | Mechanistic accounting | `data/derived/mechanistic_rows.jsonl` | Figure 5, Table 4 | Included |
| Span deletion separates evidence from distractor spans and passes corrected controls. | Cached intervention | `data/derived/mechanistic_rows.jsonl` | Figure 5, Table 4 | Included |
| Cached substitution suggests MLP shifts tend to be larger than attention shifts, but the result is protocol-sensitive. | Conditional counterfactual accounting | `data/derived/mechanistic_rows.jsonl` | Figure 5, limitations section | Included with caveat |
| Ties and nonpositive final gaps are logged rather than silently removed. | QC | `data/derived/trajectory_rows.jsonl` | QC table, verification report | Included |
| Hidden-state circuits fully explain the trajectory regimes. | Excluded | Not supported by cached release evidence | None | Excluded |
