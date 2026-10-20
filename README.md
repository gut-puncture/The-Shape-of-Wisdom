# The Shape of Wisdom: How Language Models Converge on an Answer

When a language model answers a factual question, the state from which it answers moves into a region shared by every question with the same answer, summarised by its *answer centroid*. In four open models the answer becomes readable within one layer, at a model-specific depth; centroids follow the model's answer across relations and wordings; moving a state to a centroid after that layer changes the answer; and where a wrong answer sits separates near misses from real gaps better than the model's confidence does.

Paper: [`paper/main.pdf`](paper/main.pdf).

## What is here

| Path | What it is |
|---|---|
| `dataset/` | 2,256 Wikidata facts with recurring answers (`facts.jsonl`, built by `build_facts.py`), hand-curated answer names (`answers.json`, `curate_names.py`), checked rewordings (`rewrites.jsonl`) and the 16,294 prompts each model sees (`prompts.jsonl`). |
| `wisdom/` | The experiment: `prompts.py` builds prompts; `run.py` records the residual stream at every layer on a GPU (stop and resume at any time); `steer.py` moves states between answer centroids; `analysis.py` computes every result; `figures.py` and `paper_numbers.py` produce the paper's figures and numbers. |
| `scripts/` | Setting up a rented GPU and running the models in order. |
| `results/summary.json` | Every measurement in the paper. |
| `results/*_anatomy.jsonl` | Per-question records for each model: answer given, grade, and where the state sits relative to answer centroids. |
| `paper/` | LaTeX source; all numbers come from `paper/generated/numbers.tex`, written from the results. |
| `archive/` | Earlier phases of this project, with a note on why they were superseded. |

## Reproduce

```bash
pip install -r requirements.txt
python -m wisdom.prompts                                   # prompts from the dataset
python -m wisdom.run --model llama8b --run-dir runs/main   # on a GPU; repeat for qwen7b, gemma9b, llama1b, llama8b_untrained
python -m wisdom.analysis --run-dir runs/main --out results
python -m wisdom.figures --run-dir runs/main --results results --out paper/figures
python -m wisdom.paper_numbers && (cd paper && tectonic main.tex)
```

The recorded states (about 19 GB) are not in the repository; the full run took about one hour on one 48 GB GPU.

## Licence

MIT. Wikidata content is CC0.
