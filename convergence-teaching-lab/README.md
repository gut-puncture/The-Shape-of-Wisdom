# Convergence Teaching Lab

Standalone teaching app for the convergence analysis concepts (Sections A-I).

## What it includes

- Fact-only dashboard from run `rtx6000ada_baseline_20260216_1452_a`
- Deep section-by-section teaching cards with explicit labels:
  - `Fact`
  - `Interpretation`
  - `Hypothesis`
  - `Action`
- Decision-drift simulator (`δ_margin` trajectory over layers)
- Basin Wall Mechanics Lab:
  - End-to-end prompt simulation through layers
  - Per-layer decomposition (attention, MLP, late bias, noise)
  - Metrics: `p(correct)`, probability margin, logit margin, entropy, basin-gap, wall-height, flip risk
  - Three linked visualizations (logits, margin-vs-wall, potential landscape)
  - Layer scrubber + full trace table + intuition drills
- Intervention selector mapped to failure mode, domain, and compute budget
- 90-day execution roadmap and glossary

## Run locally

From repo root:

```bash
cd /Users/shaileshrana/shape-of-wisdom/convergence-teaching-lab
python3 -m http.server 8765
```

Open:

- <http://localhost:8765>

You can also open `index.html` directly in a browser, but a local server is better for consistent behavior.
