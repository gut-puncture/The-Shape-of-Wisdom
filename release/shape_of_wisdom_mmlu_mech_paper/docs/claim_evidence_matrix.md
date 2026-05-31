# Claim-Evidence Matrix

| Claim | Status | Evidence artifact |
|---|---|---|
| Decisions are depthwise trajectories, not only final endpoints. | pass | `data/cached/decision_metrics.parquet`, Figure 1 |
| State, motion, and boundary distance separate operational trajectory types. | pass | `data/cached/prompt_types.parquet`, `data/audit/artifact_integrity.json`, Figures 1-2 |
| Attention/MLP scalars reconstruct one-step margin drift with held-out R2 above 0.70. | conditional | `data/audit/08_attention_and_mlp_decomposition.report.json`, Figure 3 |
| Span deletion separates operational evidence from distractors with controls near zero. | pass | `data/cached/span_deletion_causal.parquet`, `data/cached/negative_controls.parquet`, Figure 4 |
| MLP substitution tends to exceed attention substitution in the tested settings. | pass, conditional | `data/audit/substitution_sensitivity_summary.csv`, Figure 5 |
| Counterfactual bookkeeping is protocol-sensitive and not full circuit discovery. | pass | `data/audit/drift_reconstruction_audit.json`, `data/audit/substitution_rederive_diagnostics.json`, Figure 5 |

Conditional claims are phrased conditionally in the manuscript. Paper-facing claims use only the 7--8B MMLU experiment and its cached mechanistic artifacts.
