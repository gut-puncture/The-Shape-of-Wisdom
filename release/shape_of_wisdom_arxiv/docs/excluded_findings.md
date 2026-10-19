# Excluded Findings

The release keeps the paper narrow on purpose. The following findings are not used in the main manuscript.

## Asymmetric Candidate-Set Metrics

Older drafts used a first-entry candidate event compared against forever-after commitment events. That comparison mixed transient and persistent definitions. The release replaces it with `d2`, `d1`, and `d50`, which all use persistence.

## Binary Positivity Framing

The older COLM-format draft emphasized positive model-dataset cells under a preregistered `Delta_margin` rule. The release is centered on lag magnitude and late-write concentration under persistence definitions, not a binary positivity count.

## Answer-Format Invariance

Templated and letter-label controls changed evaluability and sometimes changed the final winner. They are useful sensitivity context, but they are not clean evidence for answer-format invariance. The release therefore makes no invariance claim.

## Full Circuit Claims

The paper includes mechanistic accounting from cached attention/MLP traces, span deletion, and counterfactual summaries. It does not claim full hidden-state circuit discovery, because the strongest counterfactual substitution artifacts are linearized bookkeeping estimates rather than full forward-pass interventions.

## PCA And Flow-Field Geometry

Historical Part I artifacts included PCA views and flow fields over four-option scores. They are useful lineage, but they are not needed for the streamlined narrative and are excluded from the main manuscript.

## Universal Claims About Language Models

The release studies nine instruction-tuned open-weight models in the broad panel and three instrumented 7--8B models in the mechanistic panel. It does not claim universality across architectures, base models, closed systems, open-ended generation, or answer formats.
