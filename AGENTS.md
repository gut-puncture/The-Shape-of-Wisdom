# Agent instructions: Shape of Wisdom (answer places)

- The current experiment lives at the repository root (`wisdom/`, `dataset/`, `scripts/`, `results/`, `paper/`). Earlier phases are only in git history (tag `mmlu-arxiv-v1`).
- Every number in the paper comes from `results/summary.json` through `python -m wisdom.paper_numbers`; never type a result into the LaTeX by hand.
- Scores are held out by subject; never fit anything (places, whitening, radii, layer choices) on test subjects.
- A run directory is one task; `wisdom/run.py` refuses to resume if its invariants change. Use a new `--run-dir` for a different model, precision or recording.
- For paper and figure work: render the compiled PDF and inspect every page with a figure or table after each change; no overlaps, clipping, legends over data, or text outside the margins.
