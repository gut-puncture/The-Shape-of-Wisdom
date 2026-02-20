const DEPTH_LABELS = {
  1: "Level 1 · Crisp intuition (fast overview)",
  2: "Level 2 · Mechanistic (rigorous but concise)",
  3: "Level 3 · Research-level depth (max detail)",
};

const MODEL_COLORS = {
  Qwen: "#2a9d8f",
  Llama: "#ee9b00",
  Mistral: "#ae2012",
};

const HERO_CHIPS = [
  "Baseline-only run",
  "3000 manifest rows + 3000 prompts/model",
  "Models: Qwen 2.5-7B, Llama 3.1-8B, Mistral 7B v0.3",
  "Primary analysis: no hard-prompt filtering",
  "Outcome-stratified analysis included",
  "Fact vs interpretation explicitly separated",
];

const HEADLINE_METRICS = [
  {
    model: "Qwen",
    display: "Qwen / Qwen2.5-7B-Instruct",
    layers: 28,
    accuracy: 67.27,
    pHigh: 46.47,
    pLow: 41.13,
    stableValid: 53.2,
    earlyCommitCorrect: 76.5,
    lateFlipOverall: 14.6,
    lateFlipNonConverged: 29.7,
  },
  {
    model: "Llama",
    display: "meta-llama / Llama-3.1-8B-Instruct",
    layers: 32,
    accuracy: 60.67,
    pHigh: 34.63,
    pLow: 54.4,
    stableValid: 40.5,
    earlyCommitCorrect: 61.7,
    lateFlipOverall: 21.6,
    lateFlipNonConverged: 35.7,
  },
  {
    model: "Mistral",
    display: "mistralai / Mistral-7B-Instruct-v0.3",
    layers: 32,
    accuracy: 54.23,
    pHigh: 43.53,
    pLow: 50.97,
    stableValid: 46.4,
    earlyCommitCorrect: 82.7,
    lateFlipOverall: 12.2,
    lateFlipNonConverged: 21.8,
  },
];

const DOMAIN_DIFFICULTY = [
  { domain: "mathematics", nonConvergence: 70.5 },
  { domain: "economics_finance_business", nonConvergence: 65.6 },
  { domain: "law", nonConvergence: 63.2 },
  { domain: "chemistry", nonConvergence: 62.1 },
  { domain: "physics_engineering", nonConvergence: 59.9 },
  { domain: "biology", nonConvergence: 47.2 },
  { domain: "computer_science", nonConvergence: 44.0 },
  { domain: "psychology_social", nonConvergence: 42.4 },
];

const SECTIONS = [
  {
    id: "A",
    title: "Measured Facts",
    objective: "Restate only what the run data directly supports.",
    cards: [
      {
        title: "Run Identity + Scope",
        kind: "fact",
        levels: {
          1: "Source-of-truth run is `rtx6000ada_baseline_20260216_1452_a`. Baseline-only analysis with 3000 rows and 3000 prompts/model.",
          2: "The analysis scope is tightly defined: one baseline run, one manifest of 3000 items, and 3000 prompts evaluated per model. No hard-prompt filtering was used in primary analysis; stratification by final outcome was performed after the main pass.",
          3: "Dataset identity is explicit and narrow: `rtx6000ada_baseline_20260216_1452_a` only. Models: Qwen2.5-7B-Instruct (28 layers), Llama-3.1-8B-Instruct (32 layers), and Mistral-7B-Instruct-v0.3 (32 layers). Primary analysis did not pre-filter hard prompts; then the same set was stratified by end outcome. This matters because all later claims are bound to this exact run and evaluation frame.",
        },
      },
      {
        title: "Logged Fields + Definitions",
        kind: "fact",
        levels: {
          1: "Per row and layer, logs include candidate probabilities/logits for A/B/C/D, top candidate, top-2 probability margin, projected_hidden_128, and candidate entropy.",
          2: "Decision variables were defined explicitly: `p_correct,i(l)` and `m_correct,i(l)=p_correct,i(l)-max_{k!=c} p_i^k(l)`. End-layer convergence indicators used thresholds `p_correct >= tau` for tau in {0.6,0.8}, and `margin >= 0.15`.",
          3: "Stability-through-end commitment is the earliest `l*` such that for all later layers both `p_correct >= 0.70` and `margin >= 0.15`; otherwise undefined. Basin gap uses projected_hidden_128 and per-layer class centroids `mu_c(l)`: `gap_i(l)=min_{k!=c} ||h_i(l)-mu_k(l)||_2 - ||h_i(l)-mu_c(l)||_2`, with basin-converged-at-end when `gap_i(L-1) >= 0`. Additional logged dynamics include first-passage layers, hazard curves, flip_count, late_flip_any (last two transitions), path-length/straightness/curvature proxies, alignment to correct-direction vectors, and Cohen's d divergence windows.",
        },
        equation: "p_correct,i(l), m_correct,i(l)=p_correct,i(l)-max_{k!=c} p_i^k(l)\n\ncommit if: p_correct>=0.70 and margin>=0.15 for all t>=l*\n\ngap_i(l)=min_{k!=c}||h_i(l)-mu_k(l)||_2 - ||h_i(l)-mu_c(l)||_2",
      },
      {
        title: "Headline Metrics",
        kind: "fact",
        levels: {
          1: "Accuracy: Qwen 67.27%, Llama 60.67%, Mistral 54.23%.",
          2: "Final confidence split: Qwen had 46.47% at `p(correct)>=0.8` and 41.13% below 0.6. Llama had 34.63% at `>=0.8` and 54.40% below 0.6. Mistral had 43.53% at `>=0.8` and 50.97% below 0.6.",
          3: "Interpretation is deliberately withheld in this card: data-only restatement. Qwen leads raw accuracy. Llama has the largest low-confidence tail (`p(correct)<0.6` at 54.40%). Mistral shows lower accuracy than Qwen and Llama, but still a sizable high-confidence fraction (43.53%) indicating confidence distribution and correctness are not identical.",
        },
      },
      {
        title: "Commitment + Instability",
        kind: "fact",
        levels: {
          1: "Stable-commit valid fraction: Qwen 53.2%, Llama 40.5%, Mistral 46.4%.",
          2: "Among final-correct prompts, committing before the last 2 layers occurred in Qwen 76.5%, Llama 61.7%, Mistral 82.7%. Late-flip rate overall: 14.6%, 21.6%, and 12.2% respectively.",
          3: "Late-flip rates inside non-converged subsets are much higher: Qwen 29.7%, Llama 35.7%, Mistral 21.8%. That number is measured, not inferred. It indicates that instability is concentrated among difficult examples rather than uniformly distributed.",
        },
      },
      {
        title: "Cross-Model Difficulty",
        kind: "fact",
        levels: {
          1: "Non-converged in all 3 models: 904 prompts. Non-converged in at least 2 models: 1491 prompts.",
          2: "Hardest domains by mean non-convergence: mathematics 70.5%, economics/finance/business 65.6%, law 63.2%, chemistry 62.1%, physics/engineering 59.9%.",
          3: "Remaining domain non-convergence rates: biology 47.2%, computer science 44.0%, psychology/social 42.4%. These are aggregate rates, not per-model decomposition in this card.",
        },
      },
      {
        title: "Failure Mode Breakdown + Early Prediction",
        kind: "fact",
        levels: {
          1: "Within non-converged, wrong_direction dominates (Qwen 64.5%, Llama 61.5%, Mistral 77.0%).",
          2: "Late_instability is second largest (Qwen 29.7%, Llama 35.7%, Mistral 21.8%). Insufficient_drift is rare (0.5%, 0.2%, 1.2%).",
          3: "Early-layer non-convergence predictors achieved ROC-AUC/AP: Qwen 0.649/0.566, Llama 0.708/0.755, Mistral 0.704/0.705. Highest divergence windows for `p_correct`: Qwen layers 24-27 (then 20-23,16-19); Llama and Mistral layers 27-31 (then 22-26,17-21).",
        },
      },
      {
        title: "Recorded Interpretation Cautions",
        kind: "fact",
        levels: {
          1: "Basin-gap separation between converged and non-converged is strong. Alignment signals are mixed and model-dependent.",
          2: "The report explicitly cautions against over-reading a single geometric direction: alignment does not hold with one sign across all models.",
          3: "So two things are simultaneously true in the source report: (1) basin-gap is strongly discriminative, and (2) directional alignment signals are not universal. That is a precise caution against simplistic one-vector stories.",
        },
      },
    ],
  },
  {
    id: "B",
    title: "First-Principles Mechanistic Decomposition",
    objective: "Explain the dynamics from transformer internals to observed convergence metrics.",
    cards: [
      {
        title: "Display-Space vs Metric-Space",
        kind: "interpretation",
        levels: {
          1: "Display-space is the emitted option token. Metric-space is internal trajectory data across layers.",
          2: "Display-space is discrete and lossy: many internal states can map to the same A/B/C/D output. Metric-space is continuous and high-dimensional (even projected), where trajectory properties like drift, instability, and basin entry are defined.",
          3: "Core teaching principle: output convergence is a thresholded readout event, but mechanistic convergence is a path property. If you only inspect final output, you collapse distinct internal stories: smooth correct drift, wrong-direction lock-in, or late boundary oscillation can all produce a single token. Your metrics recover these hidden distinctions.",
        },
      },
      {
        title: "Transformer as Non-Autonomous Dynamical System",
        kind: "interpretation",
        levels: {
          1: "Each layer adds an attention edit and an MLP edit to the residual state.",
          2: "For decision position `j`, write `x_{l+1}=x_l+Δ_attn_l+Δ_mlp_l`. Equivalent form: `x_{l+1}=F_l(x_l)=x_l+f_l(x_l)`. Since each layer has distinct weights, this is time-varying (non-autonomous), not one fixed map.",
          3: "Why this matters for your data: fixed-point attractor language is only approximate in transformers because `F_l` changes with `l`. So practical stability is better described as entering a decision funnel that later layers continue to reinforce. Your `l*` metric operationalizes this: earliest layer after which correct evidence remains above chosen thresholds through the end.",
        },
        equation: "x_{l+1} = x_l + Δ^{attn}_l(x_l, X_l) + Δ^{mlp}_l(x_l)\n\nF_l(x)=x+f_l(x) (layer-varying map)",
      },
      {
        title: "Attention as Evidence Routing",
        kind: "interpretation",
        levels: {
          1: "Attention chooses which tokens donate information to the decision position.",
          2: "A head computes query/key similarity, forms attention weights, and writes a weighted value sum back into the residual stream. If attention locks onto distractors, the model can become confidently wrong.",
          3: "Mechanistic consequence for failure modes: wrong_direction can emerge from clean routing of wrong features. Multiple heads can jointly route different signals (constraints, intermediate values, formatting priors). Strong routing quality does not guarantee correctness; correctness depends on what was routed.",
        },
        equation: "α(t)=softmax_t(q·k(t)/sqrt(d_k))\nhead_out=Σ_t α(t)v(t)",
      },
      {
        title: "MLP as Feature Creator + Associative Memory",
        kind: "interpretation",
        levels: {
          1: "MLP blocks inject learned feature directions based on current token state.",
          2: "MLPs can supply knowledge not explicit in prompt tokens, sharpen existing features, and influence future attention by changing the residual state that generates next-layer queries/keys.",
          3: "In domain-heavy tasks (law/chem/econ), MLP memory is often where latent priors enter. That is powerful but risky under superposition: overlapping features can interfere, causing either wrong-direction drift (strong but wrong association) or unstable competition that surfaces as late flips.",
        },
        equation: "Δ^{mlp}_l(x)=W_out σ(W_in x+b_in)+b_out",
      },
      {
        title: "Readout Dynamics: Logit Differences Are the Core Object",
        kind: "interpretation",
        levels: {
          1: "Final option preference is governed by logit differences, not raw probabilities alone.",
          2: "Let `δ_{c,k}(l)=z_c(l)-z_k(l)`. Then layerwise change is projection of each residual edit onto `(w_c-w_k)`. Correct convergence means these margins become positive and stay positive.",
          3: "This is the bridge from internals to metrics: `p_correct` and probability margin are nonlinear summaries of the `δ` vector. Near ties, tiny `δ` perturbations cause large probability swings; at saturation, large evidence changes can barely move `p`. So mechanistic diagnosis should prefer logit-margin trajectories over probability thresholds alone.",
        },
        equation: "z_k(l)=w_k^T x_l + b_k\nδ_{c,k}(l)=z_c(l)-z_k(l)\nδ_{c,k}(l+1)-δ_{c,k}(l)=(w_c-w_k)^T(Δ^{attn}_l+Δ^{mlp}_l)",
      },
      {
        title: "Convergence = Basin Entry + Barrier Growth",
        kind: "interpretation",
        levels: {
          1: "Crossing into the correct region is not enough; staying there requires margin growth.",
          2: "Your basin-gap proxy estimates whether hidden state is closer to correct class centroid than alternatives in projected space. Stability requires a robust barrier against future layer edits.",
          3: "Correctness-at-end is a terminal condition (`δ_margin(L-1)>0`). Stability-through-end is a path condition: once above threshold, later updates must not push state back across decision boundaries. That distinction explains why some examples are correct but fragile.",
        },
      },
      {
        title: "Minimal Mechanistic Theory",
        kind: "hypothesis",
        levels: {
          1: "Success: sustained positive drift in correct-vs-competitor margins. Failure: wrong-direction or late instability.",
          2: "Wrong-direction means consistent evidence accumulation toward a wrong option. Late instability means mixed evidence near boundary with final-layer reranking.",
          3: "Your decomposition fits this minimal theory: insufficient-drift is rare, so failures are usually decisive dynamics of the wrong kind, not lack of movement. Most effort should target route-correction and boundary stabilization, not merely stronger confidence.",
        },
      },
    ],
  },
  {
    id: "C",
    title: "Why Convergence Happens vs Fails",
    objective: "Map measured patterns to root-cause taxonomy in decision coordinates.",
    cards: [
      {
        title: "Decision Coordinates and Drift",
        kind: "interpretation",
        levels: {
          1: "Use per-layer correct-vs-competitor logit margins as the state variable.",
          2: "Define `Δ_i(l)` as vector of `δ_{c,k}(l)` across competitors and `g_i(l)=Δ_i(l+1)-Δ_i(l)`. Many observed metrics are proxies of these two sequences.",
          3: "`p_correct`, probability margin, flips, basin-gap, and hazard-style commitment all become interpretable through this lens. This unifies disparate diagnostics into one geometry: where the state is (`Δ`) and how each layer moves it (`g`).",
        },
        equation: "Δ_i(l)=[δ_{c,k}(l)]_{k!=c},  g_i(l)=Δ_i(l+1)-Δ_i(l)",
      },
      {
        title: "Basins and Boundaries in a Time-Varying System",
        kind: "interpretation",
        levels: {
          1: "Think in terms of decision funnels to final outputs, not strict fixed points.",
          2: "For layer `l`, define basin-to-final-decision `B_c(l)={x_l: forward pass from l to end predicts c}`. Exact sets are hard to compute; your centroid-gap is an approximation to basin membership.",
          3: "Failure mapping: wrong_direction corresponds to entering/reinforcing a wrong basin; late_instability corresponds to hovering near a basin boundary where small late edits decide the final side. This exactly mirrors your measured failure decomposition proportions.",
        },
      },
      {
        title: "Information Flow and Domain Difficulty",
        kind: "hypothesis",
        levels: {
          1: "Hard domains likely require either precise computation or precise exception-aware recall.",
          2: "Math/physics pressures algorithmic reliability; law/chem/econ pressures high-fidelity recall and disambiguation. Both are vulnerable to plausible-but-wrong associations.",
          3: "This is an interpretation, not a measured causal fact in your summary. But it aligns with your hardest-domain ranking and wrong_direction dominance: high-plausibility distractors can generate coherent but misaligned drift. The model is not idle; it is decisively selecting the wrong evidence path.",
        },
      },
      {
        title: "Metric-by-Metric What You Track vs Miss",
        kind: "action",
        levels: {
          1: "Probability metrics are useful but nonlinear; logit-space metrics are more mechanistically faithful.",
          2: "Upgrade stack: add logit margin trajectories, competitor identity across layers, weighted flip energy, and split late flips into to-correct vs from-correct.",
          3: "Projected-space path metrics are informative but confounded by projection basis and scale changes. Compute path diagnostics both in whitened projection and in decision-coordinate space (`Δ`). Keep basin-gap, but validate its stories against logit-space outcomes so projection artifacts cannot dominate conclusions.",
        },
      },
      {
        title: "Why Commitment Often Happens Before Final Layers",
        kind: "interpretation",
        levels: {
          1: "Many prompts internally commit before the end; final layers often sharpen, not decide from scratch.",
          2: "Measured early commitment among final-correct is high (Qwen 76.5%, Llama 61.7%, Mistral 82.7% before last 2 layers).",
          3: "Late layers still matter: they can enlarge margins (stability gain), enforce output format, or re-rank near ties. So 'knowing earlier' and 'final layers being decisive in edge cases' can both be true at once.",
        },
      },
      {
        title: "Why Insufficient Drift Is Rare",
        kind: "hypothesis",
        levels: {
          1: "The system tends to choose a direction rather than remain neutral.",
          2: "Measured insufficient-drift rates (0.2-1.2%) are tiny relative to wrong-direction and late-instability.",
          3: "A plausible mechanism is winner-take-all pressure from next-token training: even weak evidence often gets amplified into a committed continuation. So practical failures are mostly directional errors or late boundary flips, not indecision.",
        },
      },
    ],
  },
  {
    id: "D",
    title: "Interventions Without Training",
    objective: "Change trajectories at inference time so states enter and stay in correct basins.",
    cards: [
      {
        title: "D1 · Early-Warning Gating + Targeted Second Pass",
        kind: "action",
        levels: {
          1: "Use early-layer risk predictor to spend extra compute only where failure is likely.",
          2: "Pipeline: run once, score early features (entropy, early margin, basin-gap), trigger second-pass only for high-risk prompts, then select by stronger final logit margin.",
          3: "Why highest leverage: your early predictor AUC/AP is already meaningful. This converts passive diagnostics into an active controller that reduces tail failures without paying full two-pass cost on easy prompts. Evaluate not just accuracy, but change in late_flip_from_correct, stable-commit increase, and compute overhead.",
        },
      },
      {
        title: "D2 · Last-K Layer Logit Ensembling",
        kind: "action",
        levels: {
          1: "Average logits from final K layers to damp last-layer boundary noise.",
          2: "Decision rule: `z̄_k=(1/K) Σ z_k(l)` over last K layers; pick argmax of `z̄`. Best candidate K values: 2, 4, 8.",
          3: "Targeted benefit is late_instability reduction. Watch for failure mode: if late layers often perform genuine corrections, smoothing can regress those examples. Use per-mode evaluation, not only aggregate accuracy.",
        },
        equation: "z̄_k = (1/K) * Σ_{l=L-K}^{L-1} z_k(l)",
      },
      {
        title: "D3 · Structured Prompting (givens -> compute -> choose)",
        kind: "action",
        levels: {
          1: "Force explicit extraction before selection to reduce mis-parsing routes.",
          2: "Template: restate question, list givens, solve, output A/B/C/D. This adds anchor tokens attention can repeatedly retrieve.",
          3: "Mechanistic intent is route control: improve evidence routing quality before option competition sharpens. Expected signature is earlier threshold passage and stronger stable commitment, with possible token-cost tradeoff.",
        },
      },
      {
        title: "D4 · Verification Second Pass",
        kind: "action",
        levels: {
          1: "Have the model explicitly try to disprove its first answer.",
          2: "Run propose pass, then verification pass with critique instructions, then choose result by larger final logit margin or agreement rule.",
          3: "This is mainly for wrong-direction cases where first pass is coherent but wrong. It can still fail if model self-rationalizes error; measure net effect on wrong-direction share, not just total accuracy.",
        },
      },
      {
        title: "D6 · Domain Retrieval Injection",
        kind: "action",
        levels: {
          1: "Inject concise authoritative references for hard knowledge domains.",
          2: "Best candidates from your difficulty profile: law, chemistry, econ/finance, and selected math/physics items requiring formulas/definitions.",
          3: "Mechanism: change what attention can copy and which MLP associations activate. Keep context short to avoid dilution, and include negative controls with wrong references to confirm causal sensitivity.",
        },
      },
      {
        title: "D8/D9 · Option-By-Option Evaluation + Deferral Policy",
        kind: "action",
        levels: {
          1: "Evaluate each option explicitly; defer on low margin when reliability matters.",
          2: "Option-wise scoring reduces single-distractor dominance. Deferral with entropy/margin thresholds trades coverage for safety.",
          3: "If your objective is strict always-answer throughput, deferral is not suitable. But for high-stakes use, calibrated abstention can cut confident wrong outputs while gated retries recover some coverage.",
        },
      },
    ],
  },
  {
    id: "E",
    title: "Interventions With Training/Fine-Tuning",
    objective: "Alter internal dynamics so correct drift starts earlier and destabilizes less often.",
    cards: [
      {
        title: "E1 · Deep Supervision on Intermediate Layers",
        kind: "action",
        levels: {
          1: "Apply auxiliary CE losses at intermediate layers, not only final layer.",
          2: "Objective: encourage earlier linearly readable separation for correct option, raising early commitment and reducing late volatility.",
          3: "Risk is premature wrong lock-in. Mitigate with depth weighting and partial-layer supervision (e.g., last K blocks) so model retains revision capacity when early interpretation is wrong.",
        },
        equation: "L = Σ_l ω_l * CE(softmax(z(l)), y_correct)",
      },
      {
        title: "E2 · Monotonic Margin Growth Regularization",
        kind: "action",
        levels: {
          1: "Penalize late decreases in correct margin.",
          2: "Apply in final K layers, conditionally after margin becomes modestly positive.",
          3: "This directly targets late-instability. If applied too aggressively, it blocks beneficial late corrections. Monitor both stability gains and accuracy regressions on previously correctable items.",
        },
        equation: "L_stab = Σ_l max(0, δ_margin(l)-δ_margin(l+1)+ε)",
      },
      {
        title: "E3 · Contrastive Basin Shaping",
        kind: "action",
        levels: {
          1: "Push hidden states closer to correct clusters and farther from wrong clusters.",
          2: "Triplet-style geometry objectives can increase class separation in representation space.",
          3: "Use caution: if optimization space is projection-biased, you may shape the wrong geometry. Prefer readout-relevant subspaces or combine with logit objectives.",
        },
      },
      {
        title: "E4 · Hard-Negative Curriculum",
        kind: "action",
        levels: {
          1: "Train on plausible distractors and cross-model-hard prompts.",
          2: "Your natural hard set is prompts non-converged in >=2 models (1491) and especially all-3 failures (904).",
          3: "Primary target is wrong_direction reduction. Expect strongest domain gains where distractor plausibility is highest. Validate generalization to avoid overfitting distractor style.",
        },
      },
      {
        title: "E6 · Stability-Aware Preference Optimization",
        kind: "action",
        levels: {
          1: "Reward correctness plus margin/stability, not correctness alone.",
          2: "Candidate reward adds terms for high correct margin and penalties for late flips or confident wrong.",
          3: "Guard against reward hacking with strict calibration checks: if high-confidence rates rise without accuracy gains, the policy is inflating confidence, not improving reasoning.",
        },
      },
      {
        title: "E7 · Early-Exit Reliability Heads",
        kind: "action",
        levels: {
          1: "Train intermediate predictors for confidence/stability to allocate compute adaptively.",
          2: "Early exit on clearly stable cases, extra processing for unstable cases.",
          3: "This can reduce latency and improve effective convergence when coupled with fallback passes, but calibration errors can lock in wrong early exits. Requires robust confidence calibration.",
        },
      },
    ],
  },
  {
    id: "F",
    title: "Experiment Designs",
    objective: "Convert hypotheses into falsifiable experiments with controls and confound checks.",
    cards: [
      {
        title: "F1.1 Quick: Decision-Drift Decomposition",
        kind: "action",
        levels: {
          1: "Compute `δ_margin(l)` and `g_margin(l)` per prompt, stratified by outcome group.",
          2: "Expected: wrong-direction has persistent negative drift; late-instability oscillates around zero near final layers.",
          3: "Falsification criterion: if groups show similar drift signatures, failure labels are not dynamically distinct or decision coordinate is misdefined. Include domain/length controls and per-competitor margins.",
        },
      },
      {
        title: "F1.2 Quick: Last-K Logit Ensembling",
        kind: "action",
        levels: {
          1: "Test K in {2,4,8} against final-layer-only baseline.",
          2: "Primary endpoint: gain in late-instability subset with minimal regression in corrected-by-late-layer cases.",
          3: "Normalize logits per layer if scale drifts. Report confidence intervals, subset effects, and change in high-confidence wrong rate.",
        },
      },
      {
        title: "F1.3 Quick: Gated Second Pass",
        kind: "action",
        levels: {
          1: "Train early risk predictor, trigger intervention only when risk exceeds threshold.",
          2: "Compare against single-pass baseline and always-heavy-prompt baseline.",
          3: "Success = near always-heavy quality at lower compute. Check leakage strictly: no late-layer features in predictor.",
        },
      },
      {
        title: "F1.4 Quick: Retrieval by Hard Domain",
        kind: "action",
        levels: {
          1: "Inject concise references in law/chem/econ/math-physics slices.",
          2: "Measure whether wrong_direction share drops and thresholds are crossed earlier.",
          3: "Use negative controls with deliberately bad references to validate mechanism and detect prompt overfitting.",
        },
      },
      {
        title: "F1.5 Quick: Causal Patching/Ablation",
        kind: "action",
        levels: {
          1: "Patch or ablate attention/MLP outputs in high-divergence windows.",
          2: "Look for components with outsized causal effect on `δ_margin` and late flips.",
          3: "If effects are diffuse, behavior may be distributed; then prioritize macro interventions (prompt/decoding) over per-head steering.",
        },
      },
      {
        title: "F2 Heavy: Training Pilots",
        kind: "action",
        levels: {
          1: "Pilot E1/E2/E4 with pre-registered metrics before large runs.",
          2: "Evaluate both gains and regressions: calibration, premature wrong commitment, generalization outside benchmark.",
          3: "Only scale if pilot improves targeted failure mode and does not raise confident-wrong tail risk.",
        },
      },
    ],
  },
  {
    id: "G",
    title: "What Better Convergence Accomplishes",
    objective: "Translate mechanistic improvements into reliability and deployment implications.",
    cards: [
      {
        title: "Reliability Gain: Fewer Knife-Edge Decisions",
        kind: "interpretation",
        levels: {
          1: "More prompts staying deep in correct basins means fewer fragile boundary outcomes.",
          2: "Higher stable commitment lowers sensitivity to tiny perturbations in decoding or prompt phrasing.",
          3: "Operationally this reduces surprise reversals and improves consistency under minor format or ordering changes, provided gains come from evidence quality rather than mere confidence inflation.",
        },
      },
      {
        title: "Calibration Is Not Automatic",
        kind: "interpretation",
        levels: {
          1: "Higher convergence metrics can improve or worsen calibration depending on mechanism.",
          2: "If margins grow from better evidence, calibration improves. If margins grow from logit scaling alone, confident wrong increases.",
          3: "Always co-report: reliability diagrams, `P(correct | confidence bin)`, and high-confidence-wrong rate. Convergence improvements without calibration checks are incomplete.",
        },
      },
      {
        title: "Robustness Transfer Potential",
        kind: "hypothesis",
        levels: {
          1: "Stable early commitment can transfer to better prompt/decode robustness.",
          2: "Expected transfer axes: prompt paraphrases, decoding variance, and cross-model consistency in borderline cases.",
          3: "But strong wrong-direction commitment can also become robust wrongness. Robustness gains should be interpreted jointly with error-mode composition, not just variance reduction.",
        },
      },
      {
        title: "Tail-Risk Reduction + Controllability",
        kind: "interpretation",
        levels: {
          1: "Reducing late-instability and confident wrong lowers catastrophic error risk.",
          2: "Stable dynamics improve control policies like margin-threshold answering and risk-triggered verification.",
          3: "This yields safer deployment knobs: explicit abstain/verify routes become more trustworthy when confidence signals are behaviorally aligned with correctness.",
        },
      },
    ],
  },
  {
    id: "H",
    title: "Risks and Trade-Offs",
    objective: "Prevent regressions while improving convergence.",
    cards: [
      {
        title: "Stability vs Corrigibility",
        kind: "hypothesis",
        levels: {
          1: "Pushing early stability too hard can reduce late correction ability.",
          2: "Mechanistically, higher barriers help when basin is correct but hurt when basin is wrong.",
          3: "Mitigate with conditional penalties and last-K-only stabilization so model can still revise early misinterpretations.",
        },
      },
      {
        title: "Confidence Inflation Risk",
        kind: "hypothesis",
        levels: {
          1: "Models can appear more converged without being more correct.",
          2: "Probability and margin can rise through logit scaling effects; this can worsen calibration.",
          3: "Mandatory guardrail: track confidence-conditioned accuracy and confident-wrong tail rate on hard domains.",
        },
      },
      {
        title: "Robust Wrongness",
        kind: "hypothesis",
        levels: {
          1: "Wrong-direction failures can become more stable under naive stabilizers.",
          2: "Smoothing and monotonic objectives may entrench wrong basins if route correction is not addressed.",
          3: "Countermeasure is mode-specific policy: route-fixing interventions for wrong-direction, stabilizers for late-instability.",
        },
      },
      {
        title: "Projection and Metric Confounds",
        kind: "interpretation",
        levels: {
          1: "Projected-space geometry can mislead if treated as ground truth.",
          2: "Basin-gap is useful but should be cross-validated in decision-logit coordinates.",
          3: "Use multiple projections, whitening, and readout-space checks to ensure conclusions are not artifacts of embedding basis.",
        },
      },
      {
        title: "Prompt Overhead + Attention Dilution",
        kind: "interpretation",
        levels: {
          1: "Heavier prompting increases token count and can dilute attention in long contexts.",
          2: "Structured prompts can also amplify wrong assumptions if extracted givens are hallucinated.",
          3: "Gating is the practical mitigation: keep heavy interventions selective and concise.",
        },
      },
    ],
  },
  {
    id: "I",
    title: "Prioritized 90-Day Plan",
    objective: "Sequence analyses and interventions with explicit stop/go criteria.",
    cards: [
      {
        title: "Phase 1 (Days 1-21): High-Information Fast Work",
        kind: "action",
        levels: {
          1: "Ship logit-drift analysis, last-K ensembling, and early-warning gating prototype.",
          2: "Stop/go: failure modes must separate in drift signatures; ensembling should help late-instability subset with limited harm.",
          3: "Target outcomes: +3 to +5 pp in late-instability subset, meaningful reduction in `p(correct)<0.6`, and controlled compute overhead under gated policy.",
        },
      },
      {
        title: "Phase 2 (Days 22-45): Mechanism Validation",
        kind: "action",
        levels: {
          1: "Run causal patching/ablation in divergence windows + domain retrieval pilots.",
          2: "Stop/go: identify repeatable causal layers/components or accept distributed dynamics and move to macro controls.",
          3: "Use domain-level negative controls and component-level falsification to avoid storytelling from correlational windows alone.",
        },
      },
      {
        title: "Phase 3 (Days 46-75): Mode-Specific Inference Policy",
        kind: "action",
        levels: {
          1: "Deploy conditional policy: stabilizers for late-instability, route-correction for wrong-direction.",
          2: "Upgrade metric stack to logit-first reporting + directional late-flip metrics.",
          3: "Stop/go: convergence and accuracy must improve without increasing confident-wrong tails; compute profile must remain practical.",
        },
      },
      {
        title: "Phase 4 (Days 76-90): Training Pilot Decision",
        kind: "action",
        levels: {
          1: "Select one training pilot (E2 or E4 or E1-last-K) based on earlier evidence.",
          2: "Run pilot with pre-registered metrics and generalization checks.",
          3: "Scale only if pilot improves target failure mode, improves confidence-conditioned correctness, and avoids major regression on external suite.",
        },
      },
      {
        title: "Top 10 Immediate Recommendations",
        kind: "action",
        levels: {
          1: "Top priorities: logit-margin tracking, last-K ensembling, risk-gated second pass.",
          2: "Then add directional late-flip metrics, structured prompts, verification, and domain retrieval pilots.",
          3: "After mechanism validation: targeted patching analysis and one training pilot aligned to dominant failure mode. Ranking criterion should be impact x feasibility x confidence, not novelty.",
        },
      },
    ],
  },
];

const GLOSSARY = [
  {
    term: "Display-space",
    definition: "The emitted output token sequence (A/B/C/D here). Discrete and lossy with respect to internal state.",
  },
  {
    term: "Metric-space",
    definition: "Measured internal trajectory variables: logits, probabilities, entropy, margins, flips, and hidden projections.",
  },
  {
    term: "δ_margin",
    definition: "Logit margin between correct option and strongest competitor. Most direct decision-coordinate scalar.",
  },
  {
    term: "g_margin",
    definition: "Layerwise drift in decision margin: `δ_margin(l+1)-δ_margin(l)`. Interpretable as per-layer evidence contribution.",
  },
  {
    term: "Stable commitment (l*)",
    definition: "Earliest layer after which thresholds remain satisfied through end. Path condition, not just endpoint.",
  },
  {
    term: "Basin gap",
    definition: "Projected-space distance advantage to correct centroid over nearest wrong centroid.",
  },
  {
    term: "Wrong-direction",
    definition: "Trajectory decisively drifts toward wrong option, often with low flip count.",
  },
  {
    term: "Late-instability",
    definition: "Trajectory hovers near boundary and flips in late layers; high flip and late_flip rates.",
  },
  {
    term: "Insufficient drift",
    definition: "Failure mode where neither direction accumulates strongly. Rare in your measured results.",
  },
  {
    term: "Divergence window",
    definition: "Layer range where strong-converged vs non-converged groups show highest separation (e.g., Cohen's d).",
  },
  {
    term: "Calibration",
    definition: "How well confidence matches empirical correctness; required to interpret convergence gains safely.",
  },
  {
    term: "Robust wrongness",
    definition: "Model becomes stable and confident in incorrect basin. Stability without correctness.",
  },
];

const INTERVENTIONS = [
  {
    id: "D1",
    title: "Early-warning gating + targeted second pass",
    type: "no-training",
    modes: ["wrong_direction", "late_instability", "all"],
    domains: ["all"],
    cost: "medium",
    mechanism:
      "Use early-layer risk signals to route only risky prompts into heavier interventions (structured prompt, verify pass, retrieval).",
    signature:
      "Expect lower p(correct)<0.6 and higher stable-commit in flagged subset with bounded compute.",
    caveat:
      "Needs careful predictor calibration; biased predictor can miss stable-but-wrong trajectories.",
  },
  {
    id: "D2",
    title: "Last-K layer logit ensembling",
    type: "no-training",
    modes: ["late_instability"],
    domains: ["all"],
    cost: "low",
    mechanism:
      "Average logits from final layers to reduce boundary jitter and last-layer rerank noise.",
    signature:
      "Best gains in late-instability cohort; little effect on stable wrong-direction examples.",
    caveat:
      "Can suppress beneficial late corrections on some prompts.",
  },
  {
    id: "D3",
    title: "Structured prompt: givens -> compute -> choose",
    type: "no-training",
    modes: ["wrong_direction", "all"],
    domains: ["mathematics", "physics_engineering", "all"],
    cost: "medium",
    mechanism:
      "Forces explicit extraction of constraints, reducing early misrouting of evidence.",
    signature:
      "Earlier threshold crossing and higher stable commitment if parsing is main failure source.",
    caveat:
      "If extracted givens are wrong, model becomes confidently wrong.",
  },
  {
    id: "D4",
    title: "Verification second pass",
    type: "no-training",
    modes: ["wrong_direction"],
    domains: ["all"],
    cost: "high",
    mechanism:
      "Second pass is prompted to challenge first-pass answer and recompute before selecting.",
    signature:
      "Can reduce stable wrong-direction failures where error is self-detectable.",
    caveat:
      "Adds latency and can rationalize error when model critique quality is weak.",
  },
  {
    id: "D6",
    title: "Domain retrieval injection",
    type: "no-training",
    modes: ["wrong_direction", "all"],
    domains: ["law", "chemistry", "economics_finance_business", "mathematics", "physics_engineering"],
    cost: "medium",
    mechanism:
      "Inject concise trusted references to override weak or wrong latent associations.",
    signature:
      "Largest expected shift in hardest knowledge-heavy domains; lower wrong-direction share.",
    caveat:
      "Poor retrieval quality can worsen drift.",
  },
  {
    id: "E2",
    title: "Monotonic margin regularization (last-K)",
    type: "training",
    modes: ["late_instability"],
    domains: ["all"],
    cost: "high",
    mechanism:
      "Penalize late drops in correct logit margin once positive threshold is reached.",
    signature:
      "Lower late flips, higher stable commitment in end layers.",
    caveat:
      "Can block useful late corrections if applied globally.",
  },
  {
    id: "E4",
    title: "Hard-negative curriculum",
    type: "training",
    modes: ["wrong_direction"],
    domains: ["mathematics", "law", "chemistry", "economics_finance_business", "physics_engineering", "all"],
    cost: "high",
    mechanism:
      "Train specifically on plausible distractors and cross-model-hard prompt subsets.",
    signature:
      "Expected drop in wrong-direction failures, especially in hardest domains.",
    caveat:
      "Risk of overfitting distractor style; needs generalization tests.",
  },
  {
    id: "E1",
    title: "Deep supervision on intermediate layers",
    type: "training",
    modes: ["late_instability", "all"],
    domains: ["all"],
    cost: "high",
    mechanism:
      "Add auxiliary cross-entropy at intermediate layers to encourage earlier separability.",
    signature:
      "Earlier convergence and less late volatility.",
    caveat:
      "Risk of premature wrong commitment.",
  },
];

const ROADMAP = [
  {
    phase: "Phase 1 · Days 1-21",
    summary:
      "Run logit-drift decomposition, last-K ensembling trial, and risk-gated second-pass prototype. Stop if failure modes do not separate in decision coordinates.",
  },
  {
    phase: "Phase 2 · Days 22-45",
    summary:
      "Mechanistic validation with patching/ablation in late divergence windows plus domain retrieval pilots. Stop if no repeatable causal leverage emerges.",
  },
  {
    phase: "Phase 3 · Days 46-75",
    summary:
      "Deploy mode-specific inference policy and upgraded metrics (logit-first, directional late-flips). Stop if confident-wrong tail increases.",
  },
  {
    phase: "Phase 4 · Days 76-90",
    summary:
      "Select and run one training pilot (E2 or E4 or E1-last-K). Scale only if accuracy and calibration both improve on held-out sets.",
  },
];

const SCENARIOS = {
  stable_correct: {
    label: "Stable convergence",
    initial: -0.08,
    drift: 0.12,
    noise: 0.06,
    lateJolt: 0.03,
    description:
      "Healthy run: margin crosses zero early, then barrier grows. Flip count should stay low and stable-commit appears before the final layers.",
  },
  wrong_direction: {
    label: "Wrong-direction lock-in",
    initial: 0.06,
    drift: -0.10,
    noise: 0.04,
    lateJolt: 0.02,
    description:
      "Failure run: layers keep adding evidence in the wrong sign. You get decisive but incorrect convergence.",
  },
  late_instability: {
    label: "Late instability",
    initial: -0.01,
    drift: 0.04,
    noise: 0.10,
    lateJolt: 0.24,
    description:
      "Boundary run: near-tie dynamics survive late, then final layers can flip outcome with small perturbations.",
  },
};

const DEEP_PRESETS = {
  stable_correct: {
    label: "Stable Correct Convergence",
    params: {
      evidence: 0.86,
      routing: 0.9,
      distractor: 0.28,
      mlpCorrect: 0.1,
      mlpWrong: 0.02,
      noise: 0.07,
      lateBias: -0.02,
      barrierGrowth: 0.36,
      retrieval: 0.08,
      reasoning: 0.12,
    },
    explanation:
      "Correct evidence wins early and repeated consistent layers build a tall wall, so the path does not escape the correct basin.",
  },
  wrong_direction: {
    label: "Wrong-Direction Lock-In",
    params: {
      evidence: 0.58,
      routing: 0.34,
      distractor: 0.82,
      mlpCorrect: 0.01,
      mlpWrong: 0.14,
      noise: 0.06,
      lateBias: 0.07,
      barrierGrowth: 0.34,
      retrieval: 0.0,
      reasoning: 0.02,
    },
    explanation:
      "A distractor route dominates; the model still builds high walls, but around the wrong basin (robust wrongness).",
  },
  late_instability: {
    label: "Late Instability",
    params: {
      evidence: 0.72,
      routing: 0.66,
      distractor: 0.6,
      mlpCorrect: 0.07,
      mlpWrong: 0.06,
      noise: 0.14,
      lateBias: 0.15,
      barrierGrowth: 0.11,
      retrieval: 0.03,
      reasoning: 0.07,
    },
    explanation:
      "Competing evidence keeps the state near the boundary. Walls remain low enough that late layers can flip the winner.",
  },
  late_correction: {
    label: "Late Correction",
    params: {
      evidence: 0.65,
      routing: 0.57,
      distractor: 0.68,
      mlpCorrect: 0.05,
      mlpWrong: 0.09,
      noise: 0.1,
      lateBias: -0.18,
      barrierGrowth: 0.2,
      retrieval: 0.14,
      reasoning: 0.2,
    },
    explanation:
      "Early layers drift wrong, then stronger late evidence and intervention effects pull trajectory back across the boundary.",
  },
};

const DEEP_OUTCOME_GUIDE = [
  {
    title: "Stable Correct",
    body: "Positive logit margin grows, entropy drops, basin-gap becomes positive, and wall height rises. Metrics: high p(correct), low flip risk, defined early l*.",
  },
  {
    title: "Stable Wrong",
    body: "Margin grows with wrong sign and wall height still rises. Metrics: low p(correct), low late flips, but strong confidence in wrong option.",
  },
  {
    title: "Late Flip To Correct",
    body: "Near-boundary path for most layers, then boundary crossing late. Metrics: high flip_count or late_flip_any with final positive margin.",
  },
  {
    title: "Late Flip From Correct",
    body: "Prompt looks recovered, then late policy/noise reranks wrong option. Metrics: late_flip_any true with final negative margin.",
  },
];

const DEEP_DRILLS = [
  {
    title: "Drill 1 · Build High Correct Walls",
    body:
      "Set routing >= 0.85, distractor <= 0.30, barrier growth >= 0.30. Run. Verify: early positive logit margin, rising wall height, stable l* appears.",
  },
  {
    title: "Drill 2 · Create Robust Wrongness",
    body:
      "Set routing <= 0.40, distractor >= 0.75, MLP wrong prior >= 0.10. Run. Verify: negative logit margin with low flip risk and tall walls in wrong basin.",
  },
  {
    title: "Drill 3 · Force Late Instability",
    body:
      "Keep margin near boundary: routing ~0.6, distractor ~0.6, noise >= 0.12, low barrier growth <= 0.15, positive late bias. Watch late_flip_any become true.",
  },
  {
    title: "Drill 4 · Rescue With Interventions",
    body:
      "Start from unstable/wrong preset, then enable retrieval + structured prompt + ensembling. Compare first-passage layers, final margin, and outcome class.",
  },
];

const state = {
  sectionId: "A",
  depth: 3,
  factsOnly: false,
  showEquations: true,
  simResult: null,
  deepResult: null,
  deepLayerIndex: 0,
};

function formatInline(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function formatText(raw) {
  const text = raw || "";
  const blocks = text.split(/\n\n+/).map((b) => b.trim()).filter(Boolean);
  return blocks
    .map((block) => {
      const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
      const listLike = lines.every((line) => line.startsWith("- "));
      if (listLike) {
        return `<ul>${lines.map((line) => `<li>${formatInline(line.slice(2))}</li>`).join("")}</ul>`;
      }
      return `<p>${formatInline(lines.join(" "))}</p>`;
    })
    .join("");
}

function getDepthText(card) {
  return card.levels[state.depth] || card.levels[2] || card.levels[1] || "";
}

function getKindLabel(kind) {
  if (kind === "fact") return "Fact";
  if (kind === "interpretation") return "Interpretation";
  if (kind === "hypothesis") return "Hypothesis";
  if (kind === "action") return "Action";
  return "Note";
}

function renderHero() {
  const el = document.getElementById("heroChips");
  el.innerHTML = HERO_CHIPS.map((chip) => `<span class="chip">${formatInline(chip)}</span>`).join("");
}

function renderNav() {
  const nav = document.getElementById("sectionNav");
  nav.innerHTML = SECTIONS.map((section) => {
    const active = state.sectionId === section.id ? "active" : "";
    return `<button class="nav-btn ${active}" data-section-id="${section.id}">${section.id}. ${formatInline(section.title)}</button>`;
  }).join("");

  nav.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.sectionId = btn.dataset.sectionId;
      renderNav();
      renderLesson();
    });
  });
}

function renderMetrics() {
  const headline = document.getElementById("headlineMetrics");
  headline.innerHTML = HEADLINE_METRICS.map((m) => {
    return `
      <article class="metric-card">
        <h3>${m.model} · ${m.layers} layers</h3>
        <div class="metric-row"><span>Accuracy</span><strong>${m.accuracy.toFixed(2)}%</strong></div>
        <div class="metric-row"><span>Final p(correct) >= 0.8</span><strong>${m.pHigh.toFixed(2)}%</strong></div>
        <div class="metric-row"><span>Final p(correct) < 0.6</span><strong>${m.pLow.toFixed(2)}%</strong></div>
        <div class="metric-row"><span>Stable commit valid</span><strong>${m.stableValid.toFixed(1)}%</strong></div>
      </article>
    `;
  }).join("");

  const metricDefs = [
    { key: "accuracy", label: "Accuracy" },
    { key: "pHigh", label: "High confidence: final p(correct) >= 0.8" },
    { key: "pLow", label: "Low confidence: final p(correct) < 0.6" },
    { key: "lateFlipOverall", label: "Late-flip overall" },
    { key: "lateFlipNonConverged", label: "Late-flip inside non-converged" },
  ];

  const bars = document.getElementById("metricBars");
  bars.innerHTML = metricDefs
    .map((metric) => {
      const rows = HEADLINE_METRICS.map((m) => {
        const value = m[metric.key];
        const color = MODEL_COLORS[m.model];
        return `
          <div class="bar">
            <span style="width:${Math.min(100, Math.max(0, value)).toFixed(1)}%; background:${color}; opacity:0.72;"></span>
            <div class="bar-label"><span>${m.model}</span><span>${value.toFixed(2)}%</span></div>
          </div>
        `;
      }).join("");
      return `
        <div class="metric-track">
          <div class="metric-track-label">${formatInline(metric.label)}</div>
          <div class="bar-wrap">${rows}</div>
        </div>
      `;
    })
    .join("");

  const domains = document.getElementById("domainDifficulty");
  domains.innerHTML = DOMAIN_DIFFICULTY.map((d) => {
    return `
      <article class="domain-card">
        <h4>${formatInline(d.domain)}</h4>
        <p class="domain-value">${d.nonConvergence.toFixed(1)}% non-converged</p>
      </article>
    `;
  }).join("");
}

function renderLesson() {
  const section = SECTIONS.find((s) => s.id === state.sectionId);
  if (!section) return;

  document.getElementById("sectionTitle").textContent = `${section.id}. ${section.title}`;
  document.getElementById("sectionObjective").textContent = section.objective;

  const cards = state.factsOnly ? section.cards.filter((card) => card.kind === "fact") : section.cards;

  const metaKinds = [...new Set(section.cards.map((card) => card.kind))];
  document.getElementById("sectionMeta").innerHTML = metaKinds
    .map((kind) => `<span class="badge ${kind}">${getKindLabel(kind)}</span>`)
    .join("");

  const cardContainer = document.getElementById("sectionCards");

  if (cards.length === 0) {
    cardContainer.innerHTML = `
      <article class="card">
        <div class="card-body">
          <p>No ` + "`fact`" + ` cards in this section. Disable facts-only mode to see full teaching content.</p>
        </div>
      </article>
    `;
    return;
  }

  cardContainer.innerHTML = cards
    .map((card) => {
      const text = getDepthText(card);
      const equationHtml = state.showEquations && card.equation ? `<pre class="equation">${formatInline(card.equation)}</pre>` : "";
      return `
        <article class="card">
          <div class="card-head">
            <h3>${formatInline(card.title)}</h3>
            <span class="badge ${card.kind}">${getKindLabel(card.kind)}</span>
          </div>
          <div class="card-body">
            ${formatText(text)}
            ${equationHtml}
          </div>
        </article>
      `;
    })
    .join("");
}

function logistic(x) {
  return 1 / (1 + Math.exp(-x));
}

function simulateTrajectory({ layers, drift, noise, lateJolt, initial }) {
  const points = [];
  let margin = initial;
  let prevSign = Math.sign(margin);
  let flipCount = 0;
  let lateFlipAny = false;

  for (let l = 0; l < layers; l += 1) {
    const stochastic = (Math.random() * 2 - 1) * noise;
    const lateShock = l >= layers - 2 ? (Math.random() * 2 - 1) * lateJolt : 0;
    margin += drift + stochastic + lateShock;

    const p = logistic(margin * 2.8);
    const sign = margin >= 0 ? 1 : -1;

    if (l > 0 && sign !== prevSign) {
      flipCount += 1;
      if (l >= layers - 2) lateFlipAny = true;
    }
    prevSign = sign;

    points.push({ layer: l, margin, p });
  }

  let stableCommit = -1;
  for (let l = 0; l < points.length; l += 1) {
    const stable = points.slice(l).every((pt) => pt.p >= 0.7 && pt.margin >= 0.15);
    if (stable) {
      stableCommit = l;
      break;
    }
  }

  const finalPoint = points[points.length - 1];

  let status = "Mixed/uncertain";
  if (finalPoint.margin < 0) {
    status = "Wrong-direction end state";
  } else if (stableCommit >= 0) {
    status = "Stable convergence";
  } else if (lateFlipAny) {
    status = "Late-instability end state";
  }

  return {
    points,
    flipCount,
    lateFlipAny,
    stableCommit,
    finalMargin: finalPoint.margin,
    finalP: finalPoint.p,
    status,
  };
}

function drawSimulation(result) {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  const cssWidth = Math.max(320, Math.floor(rect.width));
  const cssHeight = 300;

  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padding = { top: 20, right: 16, bottom: 34, left: 48 };
  const w = cssWidth - padding.left - padding.right;
  const h = cssHeight - padding.top - padding.bottom;

  const margins = result.points.map((pt) => pt.margin);
  const minY = Math.min(-1.6, ...margins) - 0.1;
  const maxY = Math.max(1.6, ...margins) + 0.1;

  const xAt = (idx) => padding.left + (idx / (result.points.length - 1 || 1)) * w;
  const yAt = (value) => padding.top + ((maxY - value) / (maxY - minY || 1)) * h;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  ctx.strokeStyle = "rgba(29,42,45,0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (i / 4) * h;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(cssWidth - padding.right, y);
    ctx.stroke();
  }

  const zeroY = yAt(0);
  ctx.strokeStyle = "rgba(174,32,18,0.7)";
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);
  ctx.lineTo(cssWidth - padding.right, zeroY);
  ctx.stroke();

  const barrierY = yAt(0.15);
  ctx.strokeStyle = "rgba(42,157,143,0.8)";
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, barrierY);
  ctx.lineTo(cssWidth - padding.right, barrierY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#0a9396";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  result.points.forEach((pt, idx) => {
    const x = xAt(idx);
    const y = yAt(pt.margin);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const final = result.points[result.points.length - 1];
  const fx = xAt(result.points.length - 1);
  const fy = yAt(final.margin);
  ctx.fillStyle = final.margin >= 0 ? "#2a9d8f" : "#ae2012";
  ctx.beginPath();
  ctx.arc(fx, fy, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#1d2a2d";
  ctx.font = "12px Space Grotesk";
  ctx.fillText("δ_margin", 8, padding.top + 8);
  ctx.fillText("0 boundary", padding.left + 4, zeroY - 6);
  ctx.fillText("0.15 barrier", padding.left + 4, barrierY - 6);
  ctx.fillText("Layer", cssWidth - padding.right - 34, cssHeight - 10);

  ctx.font = "11px Space Grotesk";
  const lastLayer = result.points.length - 1;
  [0, Math.floor(lastLayer / 2), lastLayer].forEach((tick) => {
    const x = xAt(tick);
    ctx.fillText(String(tick), x - 5, cssHeight - 13);
  });
}

function runSimulation() {
  const scenarioKey = document.getElementById("scenarioSelect").value;
  const layers = Number(document.getElementById("layerCountSelect").value);
  const drift = Number(document.getElementById("driftInput").value);
  const noise = Number(document.getElementById("noiseInput").value);

  const scenario = SCENARIOS[scenarioKey];
  const result = simulateTrajectory({
    layers,
    drift,
    noise,
    lateJolt: scenario.lateJolt,
    initial: scenario.initial,
  });

  state.simResult = result;
  drawSimulation(result);

  const stableText = result.stableCommit >= 0 ? `layer ${result.stableCommit}` : "undefined";
  document.getElementById("simSummary").innerHTML = `
    <strong>${formatInline(scenario.label)}</strong> · ${formatInline(scenario.description)}<br>
    Outcome: <strong>${result.status}</strong> · final margin <strong>${result.finalMargin.toFixed(3)}</strong>
    · final p(correct proxy) <strong>${(result.finalP * 100).toFixed(1)}%</strong><br>
    flip_count: <strong>${result.flipCount}</strong>
    · late_flip_any: <strong>${result.lateFlipAny ? "true" : "false"}</strong>
    · stable commitment l*: <strong>${stableText}</strong>
  `;
}

const OPTION_LABELS = ["A", "B", "C", "D"];
const OPTION_COLORS = ["#2a9d8f", "#ee9b00", "#ae2012", "#5b5f97"];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function makeRng(seedInput) {
  let seed = Number(seedInput);
  if (!Number.isFinite(seed) || seed <= 0) seed = 1452;
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >> 17;
    s >>>= 0;
    s ^= s << 5;
    s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((acc, v) => acc + v, 0) || 1;
  return exps.map((v) => v / sum);
}

function argmax(values) {
  let bestIdx = 0;
  let bestVal = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

function topTwo(values) {
  const indices = values.map((value, idx) => ({ value, idx })).sort((a, b) => b.value - a.value);
  return [indices[0], indices[1]];
}

function normalizedEntropy(probs) {
  const ent = probs.reduce((acc, p) => (p > 0 ? acc - p * Math.log(p) : acc), 0);
  return ent / Math.log(probs.length);
}

function strongestWrongIndex(values, correctIdx) {
  let idx = -1;
  let val = -Infinity;
  values.forEach((v, i) => {
    if (i === correctIdx) return;
    if (v > val) {
      idx = i;
      val = v;
    }
  });
  return idx;
}

function computeStableCommit(records) {
  for (let i = 0; i < records.length; i += 1) {
    const stable = records.slice(i).every((row) => row.pCorrect >= 0.7 && row.marginProb >= 0.15);
    if (stable) return i;
  }
  return -1;
}

function firstPass(records, predicate) {
  const row = records.find((r) => predicate(r));
  return row ? row.layer : -1;
}

function classifyDeepOutcome(result) {
  const final = result.records[result.records.length - 1];
  const finalCorrect = final.topIdx === result.correctIdx;
  if (finalCorrect && result.stableCommit >= 0) return "Stable correct convergence";
  if (!finalCorrect && result.flipCount <= 2 && final.wallHeight >= 1.3) return "Stable wrong (wrong-direction)";
  if (result.lateFlipAny && finalCorrect) return "Late flip to correct";
  if (result.lateFlipAny && !finalCorrect) return "Late flip from correct";
  if (result.flipCount >= 4) return "Oscillatory non-converged";
  return finalCorrect ? "Fragile correct (non-converged)" : "Fragile wrong (non-converged)";
}

function simulatePromptJourney(config) {
  const {
    layers,
    correctIdx,
    distractorIdx,
    seed,
    evidence,
    routing,
    distractor,
    mlpCorrect,
    mlpWrong,
    noise,
    lateBias,
    barrierGrowth,
    retrieval,
    reasoning,
    useRetrieval,
    useStructured,
    useEnsembling,
    ensembleK,
  } = config;

  const rng = makeRng(seed);
  const logits = [
    (rng() * 2 - 1) * 0.08,
    (rng() * 2 - 1) * 0.08,
    (rng() * 2 - 1) * 0.08,
    (rng() * 2 - 1) * 0.08,
  ];
  logits[correctIdx] += 0.03;

  const records = [];
  let prevTop = argmax(logits);
  let topStreak = 1;
  let flipCount = 0;
  let lateFlipAny = false;

  for (let l = 0; l < layers; l += 1) {
    const phase = layers > 1 ? l / (layers - 1) : 0;
    const lateGate = clamp((phase - 0.78) / 0.22, 0, 1);

    const effectiveEvidence = clamp(evidence + retrieval * 0.45 + (useRetrieval ? 0.2 : 0), 0, 1);
    const effectiveRouting = clamp(routing + reasoning * 0.3 + (useStructured ? 0.16 : 0), 0, 1);
    const effectiveDistractor = clamp(distractor + (1 - effectiveRouting) * 0.24, 0, 1);
    const effectiveMlpWrong = clamp(mlpWrong - (useRetrieval ? 0.05 : 0), -0.3, 0.3);
    const effectiveNoise = clamp(noise - (useStructured ? 0.02 : 0), 0, 0.45);

    const activeWrongIdx = phase < 0.35 ? distractorIdx : strongestWrongIndex(logits, correctIdx);
    const deltas = [0, 0, 0, 0];

    const attnCorrect = 0.42 * effectiveEvidence * effectiveRouting * (0.55 + 0.65 * phase);
    const attnWrong = 0.41 * effectiveEvidence * effectiveDistractor * (0.65 + 0.35 * Math.sin((l + 1) * 0.7));
    const mlpCorrectDelta = 0.28 * (mlpCorrect + reasoning * 0.18) * (0.55 + 0.8 * phase);
    const mlpWrongDelta = 0.3 * (effectiveMlpWrong + effectiveDistractor * 0.16) * (0.55 + 0.85 * phase);

    deltas[correctIdx] += attnCorrect + mlpCorrectDelta;
    deltas[activeWrongIdx] += attnWrong + mlpWrongDelta;

    OPTION_LABELS.forEach((_, idx) => {
      if (idx !== correctIdx && idx !== activeWrongIdx) deltas[idx] += 0.02 * effectiveDistractor;
    });

    let lateCorrect = 0;
    let lateWrong = 0;
    if (lateGate > 0) {
      const lateStrength = Math.abs(lateBias) * (0.45 + lateGate);
      if (lateBias >= 0) {
        const wrongTarget = rng() < 0.5 ? activeWrongIdx : 0;
        deltas[wrongTarget] += lateStrength;
        deltas[correctIdx] -= lateStrength * 0.24;
        lateWrong = lateStrength;
        lateCorrect = -lateStrength * 0.24;
      } else {
        deltas[correctIdx] += lateStrength;
        deltas[activeWrongIdx] -= lateStrength * 0.2;
        lateCorrect = lateStrength;
        lateWrong = -lateStrength * 0.2;
      }
    }

    const noiseSigma = effectiveNoise * (1 + 0.9 * lateGate);
    const noiseTerms = [0, 0, 0, 0];
    for (let i = 0; i < 4; i += 1) {
      noiseTerms[i] = gaussian(rng) * noiseSigma;
      deltas[i] += noiseTerms[i];
    }

    for (let i = 0; i < 4; i += 1) logits[i] += deltas[i];

    const probs = softmax(logits);
    const topIdx = argmax(probs);
    const topPair = topTwo(probs);
    const strongestWrongLogitIdx = strongestWrongIndex(logits, correctIdx);
    const maxWrongProb = probs[strongestWrongLogitIdx];
    const maxWrongLogit = logits[strongestWrongLogitIdx];

    if (topIdx === prevTop) topStreak += 1;
    else {
      flipCount += 1;
      if (l >= layers - 2) lateFlipAny = true;
      topStreak = 1;
    }
    prevTop = topIdx;

    const pCorrect = probs[correctIdx];
    const marginProb = pCorrect - maxWrongProb;
    const logitMargin = logits[correctIdx] - maxWrongLogit;
    const entropy = normalizedEntropy(probs);
    const basinGap = clamp(0.88 * logitMargin + 0.12 * (1 - entropy) + gaussian(rng) * 0.03, -3, 3);
    const wallHeight = clamp(
      0.08
      + barrierGrowth * topStreak
      + 0.62 * Math.max(0, Math.abs(logitMargin))
      + 0.35 * Math.max(0, basinGap)
      - 0.78 * entropy
      - 0.7 * effectiveNoise,
      0,
      6,
    );

    const netEvidence = (attnCorrect + mlpCorrectDelta + lateCorrect) - (attnWrong + mlpWrongDelta + lateWrong);
    const flipRisk = clamp(
      Math.exp(-(wallHeight + Math.max(0, logitMargin + 0.28)) / Math.max(0.06, noiseSigma * 2.1)),
      0,
      1,
    );

    records.push({
      layer: l,
      logits: [...logits],
      probs,
      topIdx,
      topProb: topPair[0].value,
      runnerUpIdx: topPair[1].idx,
      pCorrect,
      marginProb,
      logitMargin,
      entropy,
      basinGap,
      wallHeight,
      flipRisk,
      topStreak,
      activeWrongIdx,
      components: {
        attnCorrect,
        attnWrong,
        mlpCorrectDelta,
        mlpWrongDelta,
        lateCorrect,
        lateWrong,
        noiseCorrect: noiseTerms[correctIdx],
        noiseWrong: noiseTerms[activeWrongIdx],
        netEvidence,
      },
    });
  }

  const stableCommit = computeStableCommit(records);
  const final = records[records.length - 1];
  const firstP60 = firstPass(records, (r) => r.pCorrect >= 0.6);
  const firstP80 = firstPass(records, (r) => r.pCorrect >= 0.8);
  const firstMargin = firstPass(records, (r) => r.marginProb >= 0.15);

  let ensembleDecision = null;
  if (useEnsembling) {
    const K = clamp(ensembleK, 1, records.length);
    const lastRows = records.slice(records.length - K);
    const avg = [0, 0, 0, 0];
    lastRows.forEach((r) => {
      r.logits.forEach((z, i) => {
        avg[i] += z / K;
      });
    });
    const topIdx = argmax(avg);
    ensembleDecision = {
      topIdx,
      topLabel: OPTION_LABELS[topIdx],
      correct: topIdx === correctIdx,
      logits: avg,
      K,
    };
  }

  return {
    records,
    correctIdx,
    distractorIdx,
    stableCommit,
    flipCount,
    lateFlipAny,
    finalTop: OPTION_LABELS[final.topIdx],
    finalCorrect: final.topIdx === correctIdx,
    firstP60,
    firstP80,
    firstMargin,
    outcomeLabel: classifyDeepOutcome({ records, correctIdx, stableCommit, flipCount, lateFlipAny }),
    ensembleDecision,
  };
}

function fitCanvas(canvas, targetHeight) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = Math.max(320, Math.floor(rect.width || canvas.width));
  const cssHeight = targetHeight;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function drawAxes(ctx, width, height, padding, yTicks = 5) {
  const w = width - padding.left - padding.right;
  const h = height - padding.top - padding.bottom;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(29,42,45,0.16)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= yTicks; i += 1) {
    const y = padding.top + (i / yTicks) * h;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
}

function drawLogitTrajectories(result) {
  const canvas = document.getElementById("logitTraceCanvas");
  const { ctx, width, height } = fitCanvas(canvas, 280);
  const padding = { top: 18, right: 22, bottom: 32, left: 42 };
  const w = width - padding.left - padding.right;
  const h = height - padding.top - padding.bottom;
  drawAxes(ctx, width, height, padding);

  const allLogits = result.records.flatMap((r) => r.logits);
  const minY = Math.min(...allLogits, -0.4) - 0.05;
  const maxY = Math.max(...allLogits, 0.4) + 0.05;
  const xAt = (idx) => padding.left + (idx / (result.records.length - 1 || 1)) * w;
  const yAt = (val) => padding.top + ((maxY - val) / (maxY - minY || 1)) * h;

  OPTION_LABELS.forEach((label, optionIdx) => {
    ctx.strokeStyle = OPTION_COLORS[optionIdx];
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    result.records.forEach((row, layerIdx) => {
      const x = xAt(layerIdx);
      const y = yAt(row.logits[optionIdx]);
      if (layerIdx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const markerLayer = clamp(state.deepLayerIndex, 0, result.records.length - 1);
  const markerX = xAt(markerLayer);
  ctx.strokeStyle = "rgba(29,42,45,0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(markerX, padding.top);
  ctx.lineTo(markerX, height - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#1d2a2d";
  ctx.font = "11px Space Grotesk";
  ctx.fillText("Layer", width - 42, height - 10);
  ctx.fillText("Logit", 8, 14);
  ctx.fillText(String(0), xAt(0), height - 11);
  ctx.fillText(String(result.records.length - 1), xAt(result.records.length - 1) - 10, height - 11);

  OPTION_LABELS.forEach((label, idx) => {
    ctx.fillStyle = OPTION_COLORS[idx];
    ctx.fillRect(padding.left + idx * 54, height - 22, 10, 10);
    ctx.fillStyle = "#1d2a2d";
    ctx.fillText(label, padding.left + 14 + idx * 54, height - 13);
    if (idx === result.correctIdx) {
      ctx.fillText("(correct)", padding.left + 23 + idx * 54, height - 13);
    }
  });
}

function drawMarginWall(result) {
  const canvas = document.getElementById("marginWallCanvas");
  const { ctx, width, height } = fitCanvas(canvas, 280);
  const padding = { top: 18, right: 42, bottom: 32, left: 42 };
  const w = width - padding.left - padding.right;
  const h = height - padding.top - padding.bottom;
  drawAxes(ctx, width, height, padding);

  const margins = result.records.map((r) => r.logitMargin);
  const walls = result.records.map((r) => r.wallHeight);
  const minMargin = Math.min(-1.0, ...margins) - 0.1;
  const maxMargin = Math.max(1.0, ...margins) + 0.1;
  const maxWall = Math.max(1, ...walls) + 0.2;

  const xAt = (idx) => padding.left + (idx / (result.records.length - 1 || 1)) * w;
  const yMargin = (val) => padding.top + ((maxMargin - val) / (maxMargin - minMargin || 1)) * h;
  const yWall = (val) => padding.top + ((maxWall - val) / (maxWall || 1)) * h;

  const zeroY = yMargin(0);
  ctx.strokeStyle = "rgba(174,32,18,0.7)";
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);
  ctx.lineTo(width - padding.right, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#0a9396";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  margins.forEach((v, idx) => {
    const x = xAt(idx);
    const y = yMargin(v);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = "#bb3e03";
  ctx.lineWidth = 2.2;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  walls.forEach((v, idx) => {
    const x = xAt(idx);
    const y = yWall(v);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  const markerLayer = clamp(state.deepLayerIndex, 0, result.records.length - 1);
  const markerX = xAt(markerLayer);
  ctx.strokeStyle = "rgba(29,42,45,0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(markerX, padding.top);
  ctx.lineTo(markerX, height - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#1d2a2d";
  ctx.font = "11px Space Grotesk";
  ctx.fillText("logit margin (left axis)", 8, 14);
  ctx.fillText("wall height (right axis)", width - 136, 14);
  ctx.fillText("0 boundary", padding.left + 4, zeroY - 6);
}

function drawLandscape(result, layerIdx) {
  const canvas = document.getElementById("landscapeCanvas");
  const { ctx, width, height } = fitCanvas(canvas, 280);
  const padding = { top: 16, right: 20, bottom: 32, left: 40 };
  const w = width - padding.left - padding.right;
  const h = height - padding.top - padding.bottom;

  const row = result.records[layerIdx];
  const wallScale = 0.65 + row.wallHeight * 0.28;
  const tilt = clamp(row.components.netEvidence * 1.2, -1.6, 1.6);
  const xMin = -2.0;
  const xMax = 2.0;

  const potential = (x) => wallScale * (0.25 * x ** 4 - 0.5 * x ** 2) - tilt * x;
  const samples = [];
  for (let i = 0; i <= 220; i += 1) {
    const x = xMin + (i / 220) * (xMax - xMin);
    samples.push({ x, y: potential(x) });
  }

  const yVals = samples.map((s) => s.y);
  const minY = Math.min(...yVals) - 0.15;
  const maxY = Math.max(...yVals) + 0.15;

  const xAt = (x) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * w;
  const yAt = (y) => padding.top + ((maxY - y) / (maxY - minY || 1)) * h;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const boundaryX = xAt(0);
  ctx.fillStyle = "rgba(174,32,18,0.08)";
  ctx.fillRect(padding.left, padding.top, boundaryX - padding.left, h);
  ctx.fillStyle = "rgba(42,157,143,0.08)";
  ctx.fillRect(boundaryX, padding.top, width - padding.right - boundaryX, h);

  ctx.strokeStyle = "rgba(29,42,45,0.16)";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (i / 4) * h;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(29,42,45,0.5)";
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(boundaryX, padding.top);
  ctx.lineTo(boundaryX, height - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#1f4f53";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  samples.forEach((s, idx) => {
    const x = xAt(s.x);
    const y = yAt(s.y);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const xState = clamp(row.logitMargin / 1.3, -1.8, 1.8);
  const yState = potential(xState);
  ctx.fillStyle = row.logitMargin >= 0 ? "#2a9d8f" : "#ae2012";
  ctx.beginPath();
  ctx.arc(xAt(xState), yAt(yState), 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#1d2a2d";
  ctx.font = "11px Space Grotesk";
  ctx.fillText("Wrong basin", padding.left + 4, padding.top + 12);
  ctx.fillText("Correct basin", boundaryX + 4, padding.top + 12);
  ctx.fillText("Barrier", boundaryX - 18, padding.top + 26);
  ctx.fillText(`Layer ${layerIdx} · wall ${row.wallHeight.toFixed(2)} · net drift ${row.components.netEvidence.toFixed(3)}`, padding.left, height - 10);
}

function explainLayer(row, result) {
  const runner = OPTION_LABELS[row.runnerUpIdx];
  const top = OPTION_LABELS[row.topIdx];
  const wrongComp = OPTION_LABELS[row.activeWrongIdx];
  const c = row.components;

  let behavior = "Trajectory is mixed with no decisive trend yet.";
  if (row.logitMargin > 0.4 && row.wallHeight > 1.2) {
    behavior = "The state is deep in the correct basin; walls are high enough that escaping is unlikely.";
  } else if (row.logitMargin < -0.4 && row.wallHeight > 1.2) {
    behavior = "The state is deep in a wrong basin; this is robust wrongness (stable but wrong).";
  } else if (Math.abs(row.logitMargin) < 0.18 && row.flipRisk > 0.4) {
    behavior = "The state is near the boundary; late flips are plausible because barrier is low relative to perturbations.";
  }

  return `
    <strong>Layer ${row.layer}</strong> · top option <strong>${top}</strong> (runner-up ${runner}) · strongest wrong competitor <strong>${wrongComp}</strong>.<br>
    Attention contribution (correct vs wrong): <strong>${c.attnCorrect.toFixed(3)}</strong> vs <strong>${c.attnWrong.toFixed(3)}</strong>.<br>
    MLP contribution (correct vs wrong): <strong>${c.mlpCorrectDelta.toFixed(3)}</strong> vs <strong>${c.mlpWrongDelta.toFixed(3)}</strong>.<br>
    Late-layer bias (correct vs wrong): <strong>${c.lateCorrect.toFixed(3)}</strong> vs <strong>${c.lateWrong.toFixed(3)}</strong>.<br>
    Noise (correct vs wrong): <strong>${c.noiseCorrect.toFixed(3)}</strong> vs <strong>${c.noiseWrong.toFixed(3)}</strong>.<br>
    Net evidence this layer: <strong>${c.netEvidence.toFixed(3)}</strong>. ${behavior}
  `;
}

function renderLayerMetrics(row) {
  const metricCards = [
    {
      title: "p(correct)",
      value: `${(row.pCorrect * 100).toFixed(1)}%`,
      text: row.pCorrect >= 0.8 ? "High confidence in correct option." : "Not yet strongly confident in correct option.",
    },
    {
      title: "Prob Margin",
      value: row.marginProb.toFixed(3),
      text: row.marginProb >= 0.15 ? "Above your stability margin threshold." : "Below stability margin threshold.",
    },
    {
      title: "Logit Margin",
      value: row.logitMargin.toFixed(3),
      text: row.logitMargin >= 0 ? "Correct basin side of boundary." : "Wrong basin side of boundary.",
    },
    {
      title: "Entropy",
      value: row.entropy.toFixed(3),
      text: row.entropy < 0.45 ? "Low uncertainty (distribution concentrated)." : "High uncertainty (distribution diffuse).",
    },
    {
      title: "Basin Gap",
      value: row.basinGap.toFixed(3),
      text: row.basinGap >= 0 ? "Closer to correct centroid region." : "Closer to wrong centroid region.",
    },
    {
      title: "Wall Height",
      value: row.wallHeight.toFixed(3),
      text: row.wallHeight >= 1.2 ? "High barrier: hard to leave current basin." : "Low barrier: easier to switch basins.",
    },
    {
      title: "Flip Risk",
      value: `${(row.flipRisk * 100).toFixed(1)}%`,
      text: row.flipRisk < 0.2 ? "Low risk of boundary crossing next." : "Meaningful risk of crossing boundary next.",
    },
    {
      title: "Top Streak",
      value: String(row.topStreak),
      text: "Consecutive layers with same top option; contributes to barrier growth.",
    },
  ];

  const container = document.getElementById("deepMetricExplainer");
  container.innerHTML = metricCards
    .map((m) => `
      <article class="metric-pill">
        <h4>${formatInline(m.title)}</h4>
        <p><strong>${formatInline(m.value)}</strong><br>${formatInline(m.text)}</p>
      </article>
    `)
    .join("");
}

function renderTraceTable(result) {
  const tbody = document.querySelector("#traceTable tbody");
  tbody.innerHTML = result.records
    .map((row) => `
      <tr data-layer="${row.layer}">
        <td>${row.layer}</td>
        <td>${OPTION_LABELS[row.topIdx]}</td>
        <td>${(row.pCorrect * 100).toFixed(1)}%</td>
        <td>${row.marginProb.toFixed(3)}</td>
        <td>${row.logitMargin.toFixed(3)}</td>
        <td>${row.entropy.toFixed(3)}</td>
        <td>${row.basinGap.toFixed(3)}</td>
        <td>${row.wallHeight.toFixed(3)}</td>
        <td>${(row.flipRisk * 100).toFixed(1)}%</td>
      </tr>
    `)
    .join("");

  tbody.querySelectorAll("tr").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const layer = Number(rowEl.dataset.layer);
      state.deepLayerIndex = layer;
      document.getElementById("layerScrubber").value = String(layer);
      renderDeepLayerFocus();
    });
  });
}

function renderPossibleOutcomes(result) {
  const current = result.outcomeLabel;
  const outcome = current.toLowerCase();
  const container = document.getElementById("possibleOutcomes");
  container.innerHTML = DEEP_OUTCOME_GUIDE
    .map((item) => {
      const itemLower = item.title.toLowerCase();
      let active = false;
      if (itemLower === "stable correct" && outcome.includes("stable correct")) active = true;
      if (itemLower === "stable wrong" && outcome.includes("stable wrong")) active = true;
      if (itemLower === "late flip to correct" && outcome.includes("late flip to correct")) active = true;
      if (itemLower === "late flip from correct" && outcome.includes("late flip from correct")) active = true;
      const style = active ? " style=\"border-color: rgba(10,147,150,0.5);\"" : "";
      return `
        <article class="outcome-card"${style}>
          <h4>${formatInline(item.title)}</h4>
          <p>${formatInline(item.body)}</p>
        </article>
      `;
    })
    .join("");
}

function renderDeepDrills() {
  const container = document.getElementById("deepDrills");
  container.innerHTML = DEEP_DRILLS.map((item) => `
    <article class="drill-card">
      <h4>${formatInline(item.title)}</h4>
      <p>${formatInline(item.body)}</p>
    </article>
  `).join("");
}

function renderDeepOutcomeSummary(result, config) {
  const finalRow = result.records[result.records.length - 1];
  const stableText = result.stableCommit >= 0 ? `layer ${result.stableCommit}` : "undefined";
  const p60 = result.firstP60 >= 0 ? result.firstP60 : "none";
  const p80 = result.firstP80 >= 0 ? result.firstP80 : "none";
  const m15 = result.firstMargin >= 0 ? result.firstMargin : "none";
  const ensembleText = result.ensembleDecision
    ? ` · ensembled(K=${result.ensembleDecision.K})=${result.ensembleDecision.topLabel} (${result.ensembleDecision.correct ? "correct" : "wrong"})`
    : "";

  document.getElementById("deepOutcomeSummary").innerHTML = `
    <strong>Outcome:</strong> ${formatInline(result.outcomeLabel)} · final top ${formatInline(result.finalTop)}
    (${result.finalCorrect ? "correct" : "wrong"})${ensembleText}<br>
    <strong>Core metrics:</strong> flip_count=${result.flipCount}, late_flip_any=${result.lateFlipAny ? "true" : "false"},
    l*=${stableText}, first p(correct)>=0.6 at ${p60}, first p(correct)>=0.8 at ${p80}, first margin>=0.15 at ${m15}.<br>
    <strong>Final layer values:</strong> p(correct) ${(finalRow.pCorrect * 100).toFixed(1)}%, prob-margin ${finalRow.marginProb.toFixed(3)},
    logit-margin ${finalRow.logitMargin.toFixed(3)}, entropy ${finalRow.entropy.toFixed(3)}, basin-gap ${finalRow.basinGap.toFixed(3)},
    wall-height ${finalRow.wallHeight.toFixed(3)}.
  `;
}

function highlightActiveLayerRow(layerIdx) {
  document.querySelectorAll("#traceTable tbody tr").forEach((el) => {
    if (Number(el.dataset.layer) === layerIdx) el.classList.add("active");
    else el.classList.remove("active");
  });
}

function renderDeepLayerFocus() {
  if (!state.deepResult) return;
  const layerIdx = clamp(state.deepLayerIndex, 0, state.deepResult.records.length - 1);
  state.deepLayerIndex = layerIdx;
  const row = state.deepResult.records[layerIdx];

  document.getElementById("layerScrubber").value = String(layerIdx);
  document.getElementById("layerScrubber").max = String(state.deepResult.records.length - 1);
  document.getElementById("layerScrubberLabel").textContent = `${layerIdx} / ${state.deepResult.records.length - 1}`;
  document.getElementById("layerNarrative").innerHTML = explainLayer(row, state.deepResult);
  renderLayerMetrics(row);
  highlightActiveLayerRow(layerIdx);

  drawLogitTrajectories(state.deepResult);
  drawMarginWall(state.deepResult);
  drawLandscape(state.deepResult, layerIdx);
}

function readDeepParamsFromControls() {
  const correctIdx = Number(document.getElementById("deepCorrectOption").value);
  let distractorIdx = Number(document.getElementById("deepDistractorOption").value);
  if (distractorIdx === correctIdx) {
    distractorIdx = (correctIdx + 1) % 4;
    document.getElementById("deepDistractorOption").value = String(distractorIdx);
  }

  return {
    layers: Number(document.getElementById("deepLayerCountSelect").value),
    correctIdx,
    distractorIdx,
    seed: Number(document.getElementById("deepSeedInput").value),
    evidence: Number(document.getElementById("factorEvidence").value),
    routing: Number(document.getElementById("factorRouting").value),
    distractor: Number(document.getElementById("factorDistractor").value),
    mlpCorrect: Number(document.getElementById("factorMlpCorrect").value),
    mlpWrong: Number(document.getElementById("factorMlpWrong").value),
    noise: Number(document.getElementById("factorNoise").value),
    lateBias: Number(document.getElementById("factorLateBias").value),
    barrierGrowth: Number(document.getElementById("factorBarrier").value),
    retrieval: Number(document.getElementById("factorRetrieval").value),
    reasoning: Number(document.getElementById("factorReasoning").value),
    useRetrieval: document.getElementById("toggleUseRetrieval").checked,
    useStructured: document.getElementById("toggleUseStructured").checked,
    useEnsembling: document.getElementById("toggleUseEnsembling").checked,
    ensembleK: Number(document.getElementById("ensembleKSelect").value),
  };
}

function refreshFactorValueLabels() {
  const mapping = [
    ["factorEvidence", "factorEvidenceValue"],
    ["factorRouting", "factorRoutingValue"],
    ["factorDistractor", "factorDistractorValue"],
    ["factorMlpCorrect", "factorMlpCorrectValue"],
    ["factorMlpWrong", "factorMlpWrongValue"],
    ["factorNoise", "factorNoiseValue"],
    ["factorLateBias", "factorLateBiasValue"],
    ["factorBarrier", "factorBarrierValue"],
    ["factorRetrieval", "factorRetrievalValue"],
    ["factorReasoning", "factorReasoningValue"],
  ];
  mapping.forEach(([inputId, valueId]) => {
    const input = document.getElementById(inputId);
    const out = document.getElementById(valueId);
    out.textContent = Number(input.value).toFixed(2);
  });
}

function applyPresetToControls(presetKey) {
  const preset = DEEP_PRESETS[presetKey];
  if (!preset) return;
  const p = preset.params;
  document.getElementById("factorEvidence").value = p.evidence.toFixed(2);
  document.getElementById("factorRouting").value = p.routing.toFixed(2);
  document.getElementById("factorDistractor").value = p.distractor.toFixed(2);
  document.getElementById("factorMlpCorrect").value = p.mlpCorrect.toFixed(2);
  document.getElementById("factorMlpWrong").value = p.mlpWrong.toFixed(2);
  document.getElementById("factorNoise").value = p.noise.toFixed(2);
  document.getElementById("factorLateBias").value = p.lateBias.toFixed(2);
  document.getElementById("factorBarrier").value = p.barrierGrowth.toFixed(2);
  document.getElementById("factorRetrieval").value = p.retrieval.toFixed(2);
  document.getElementById("factorReasoning").value = p.reasoning.toFixed(2);
  refreshFactorValueLabels();
}

function runDeepLab() {
  const config = readDeepParamsFromControls();
  const result = simulatePromptJourney(config);
  state.deepResult = result;
  state.deepLayerIndex = 0;
  renderDeepOutcomeSummary(result, config);
  renderTraceTable(result);
  renderPossibleOutcomes(result);
  renderDeepLayerFocus();
}

function initDeepLab() {
  const presetSelect = document.getElementById("deepPresetSelect");
  presetSelect.innerHTML = Object.entries(DEEP_PRESETS)
    .map(([key, preset]) => `<option value="${key}">${formatInline(preset.label)}</option>`)
    .join("");

  const layerSelect = document.getElementById("deepLayerCountSelect");
  layerSelect.innerHTML = [
    { label: "Qwen depth (28)", value: 28 },
    { label: "Llama/Mistral depth (32)", value: 32 },
    { label: "Stress test depth (40)", value: 40 },
  ]
    .map((item) => `<option value="${item.value}">${item.label}</option>`)
    .join("");

  const optionSelectHtml = OPTION_LABELS.map((opt, idx) => `<option value="${idx}">${opt}</option>`).join("");
  document.getElementById("deepCorrectOption").innerHTML = optionSelectHtml;
  document.getElementById("deepDistractorOption").innerHTML = optionSelectHtml;

  document.getElementById("deepCorrectOption").value = "2";
  document.getElementById("deepDistractorOption").value = "1";
  document.getElementById("deepLayerCountSelect").value = "32";

  applyPresetToControls("stable_correct");
  refreshFactorValueLabels();
  renderDeepDrills();

  presetSelect.addEventListener("change", () => {
    applyPresetToControls(presetSelect.value);
    runDeepLab();
  });

  ["deepLayerCountSelect", "deepCorrectOption", "deepDistractorOption", "deepSeedInput", "ensembleKSelect"]
    .forEach((id) => {
      document.getElementById(id).addEventListener("change", runDeepLab);
    });

  [
    "factorEvidence",
    "factorRouting",
    "factorDistractor",
    "factorMlpCorrect",
    "factorMlpWrong",
    "factorNoise",
    "factorLateBias",
    "factorBarrier",
    "factorRetrieval",
    "factorReasoning",
  ].forEach((id) => {
    const el = document.getElementById(id);
    el.addEventListener("input", () => {
      refreshFactorValueLabels();
      runDeepLab();
    });
  });

  ["toggleUseRetrieval", "toggleUseStructured", "toggleUseEnsembling"].forEach((id) => {
    document.getElementById(id).addEventListener("change", runDeepLab);
  });

  document.getElementById("deepRunBtn").addEventListener("click", runDeepLab);

  document.getElementById("layerScrubber").addEventListener("input", (event) => {
    state.deepLayerIndex = Number(event.target.value);
    renderDeepLayerFocus();
  });

  runDeepLab();
}

function renderInterventions() {
  const mode = document.getElementById("modeSelect").value;
  const domain = document.getElementById("domainSelect").value;
  const budget = document.getElementById("budgetSelect").value;

  const budgetRank = { low: 1, medium: 2, high: 3 };

  const scored = INTERVENTIONS.map((item) => {
    let score = 0;

    if (item.modes.includes("all") || item.modes.includes(mode)) score += 4;
    if (item.domains.includes("all") || item.domains.includes(domain)) score += 3;

    const itemCost = budgetRank[item.cost] || 2;
    const selectedBudget = budgetRank[budget] || 2;
    if (itemCost <= selectedBudget) score += 2;
    else score -= 1;

    if (item.type === "no-training" && budget !== "high") score += 1;

    return { ...item, score };
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  const container = document.getElementById("interventionCards");
  container.innerHTML = scored
    .map((item) => {
      return `
        <article class="card">
          <div class="card-head">
            <h3>${formatInline(item.id)} · ${formatInline(item.title)}</h3>
            <span class="badge action">Recommended</span>
          </div>
          <div class="card-body">
            <p><strong>Type:</strong> ${formatInline(item.type)} · <strong>Cost:</strong> ${formatInline(item.cost)}</p>
            <p><strong>Mechanism:</strong> ${formatInline(item.mechanism)}</p>
            <p><strong>Expected metric signature:</strong> ${formatInline(item.signature)}</p>
            <p><strong>Failure case:</strong> ${formatInline(item.caveat)}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderRoadmap() {
  const el = document.getElementById("roadmapCards");
  el.innerHTML = ROADMAP.map((phase) => {
    return `
      <article class="phase-card">
        <h3>${formatInline(phase.phase)}</h3>
        <p>${formatInline(phase.summary)}</p>
      </article>
    `;
  }).join("");
}

function renderGlossary() {
  const el = document.getElementById("glossaryGrid");
  el.innerHTML = GLOSSARY.map((item) => {
    return `
      <article class="glossary-card">
        <h4>${formatInline(item.term)}</h4>
        <p>${formatInline(item.definition)}</p>
      </article>
    `;
  }).join("");
}

function initControls() {
  const depthRange = document.getElementById("depthRange");
  const depthLabel = document.getElementById("depthLabel");
  const factsOnlyToggle = document.getElementById("factsOnlyToggle");
  const equationToggle = document.getElementById("equationToggle");

  const scenarioSelect = document.getElementById("scenarioSelect");
  scenarioSelect.innerHTML = Object.entries(SCENARIOS)
    .map(([key, val]) => `<option value="${key}">${formatInline(val.label)}</option>`)
    .join("");

  const layerSelect = document.getElementById("layerCountSelect");
  layerSelect.innerHTML = [
    { label: "Qwen depth (28)", value: 28 },
    { label: "Llama/Mistral depth (32)", value: 32 },
  ]
    .map((item) => `<option value="${item.value}">${item.label}</option>`)
    .join("");

  depthLabel.textContent = DEPTH_LABELS[state.depth];

  depthRange.addEventListener("input", () => {
    state.depth = Number(depthRange.value);
    depthLabel.textContent = DEPTH_LABELS[state.depth];
    renderLesson();
  });

  factsOnlyToggle.addEventListener("change", () => {
    state.factsOnly = factsOnlyToggle.checked;
    renderLesson();
  });

  equationToggle.addEventListener("change", () => {
    state.showEquations = equationToggle.checked;
    renderLesson();
  });

  scenarioSelect.addEventListener("change", () => {
    const scenario = SCENARIOS[scenarioSelect.value];
    document.getElementById("driftInput").value = scenario.drift.toFixed(2);
    document.getElementById("noiseInput").value = scenario.noise.toFixed(2);
    runSimulation();
  });

  ["driftInput", "noiseInput", "layerCountSelect"].forEach((id) => {
    document.getElementById(id).addEventListener("change", runSimulation);
  });

  document.getElementById("rerunSimBtn").addEventListener("click", runSimulation);

  const modeSelect = document.getElementById("modeSelect");
  modeSelect.innerHTML = `
    <option value="wrong_direction">wrong_direction</option>
    <option value="late_instability">late_instability</option>
    <option value="all">mixed/unknown</option>
  `;

  const domainSelect = document.getElementById("domainSelect");
  domainSelect.innerHTML = [
    "all",
    "mathematics",
    "economics_finance_business",
    "law",
    "chemistry",
    "physics_engineering",
    "biology",
    "computer_science",
    "psychology_social",
  ]
    .map((domain) => `<option value="${domain}">${domain}</option>`)
    .join("");

  ["modeSelect", "domainSelect", "budgetSelect"].forEach((id) => {
    document.getElementById(id).addEventListener("change", renderInterventions);
  });

  const stableScenario = SCENARIOS.stable_correct;
  document.getElementById("scenarioSelect").value = "stable_correct";
  document.getElementById("driftInput").value = stableScenario.drift.toFixed(2);
  document.getElementById("noiseInput").value = stableScenario.noise.toFixed(2);
}

function init() {
  renderHero();
  renderMetrics();
  renderNav();
  initControls();
  initDeepLab();
  renderLesson();
  runSimulation();
  renderInterventions();
  renderRoadmap();
  renderGlossary();

  window.addEventListener("resize", () => {
    if (state.simResult) drawSimulation(state.simResult);
    if (state.deepResult) {
      drawLogitTrajectories(state.deepResult);
      drawMarginWall(state.deepResult);
      drawLandscape(state.deepResult, state.deepLayerIndex);
    }
  });
}

init();
