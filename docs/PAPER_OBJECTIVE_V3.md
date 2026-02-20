# Paper Objective V3

Title:
State, Motion, Boundary: A Mechanistic Account of Convergence and Failure in Transformer Multiple-Choice Decisions

## Abstract (one paragraph)
A transformer’s decision is a trajectory. We show that where the state is, which way it is moving, and how close it is to a boundary fully describe convergence outcomes at scale. We then decompose motion into attention-driven routing and multi-layer-perceptron-driven injection, validate causal claims with interventions, and show mode-specific fixes that increase convergence-to-correct.

## 1. The three primitives
1.1 State: the hidden vector at the answer position
1.2 Motion: layer-by-layer change in decision evidence
1.3 Boundary: how close the state is to a decision flip
1.4 Convergence: entering and staying inside the correct region

Narrative purpose: establish a physics-like mental model before implementation detail.

## 2. Experimental setting and observability
2.1 Models, prompts, layers, and logged quantities
2.2 Directly observed versus inferred quantities
2.3 Why probability alone is unstable as an evidence unit

Narrative purpose: prevent interpretability theater.

## 3. Phase diagram of convergence
3.1 Define decision evidence and boundary distance in logit space
3.2 Four trajectory outcomes: stable-correct, stable-wrong, unstable-correct, unstable-wrong
3.3 Locate each type in evidence/drift/boundary space
3.4 Replicate across three models and domains

Narrative purpose: show sufficiency of the three primitives.

## 4. Regions and basins in representation space
4.1 Operational region definition
4.2 Entry/exit/re-entry events over layers
4.3 Trajectory bundles for grouped prompts
4.4 Explicit caveats for visualization claims

Narrative purpose: connect mental model to latent geometry without overclaiming.

## 5. Opening the box: two-force motion decomposition
5.1 Per-layer motion into attention output and MLP output
5.2 Which component drives right or wrong drift by outcome type
5.3 Where late instability is created versus revealed

Narrative purpose: convert descriptive analysis into mechanism.

## 6. Attention as routing
6.1 Minimal prompt span taxonomy
6.2 Operational evidence/distractor definitions via counterfactual effect
6.3 Attention routing mass versus attention contribution metrics
6.4 Routing signatures across the four trajectory types

Narrative purpose: make focus measurable and causal.

## 7. MLP as injection
7.1 MLP contribution to decision evidence per layer
7.2 Help versus harm regimes
7.3 Interaction with routing (garbage in, amplified out)

Narrative purpose: explain wrong-direction motion as mechanism, not metaphor.

## 8. Causal validation
8.1 Component ablations
8.2 Activation patching
8.3 Span deletion tests
8.4 Negative controls and falsification tests

Narrative purpose: move from correlation to causal evidence.

## 9. Discussion and limits
9.1 What improves when convergence improves
9.2 Optional implications for training objectives
9.3 Limits of projections, attention, and single-run claims

Narrative purpose: honest closure.

## Methods (fully reproducible)
Data processing, definitions, trajectory classification, span segmentation, tracing instrumentation, causal experiments, statistics, and visualization rules.

## Appendices
Additional plots, per-domain breakdowns, per-model replication tables, ablation details, and robustness checks.
