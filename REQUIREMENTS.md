# Requirements

## Scientific

- No paid datasets
- No synthetic data generation
- No hidden external services
- No silent normalization, coercion, or dropping
- Exact option-text scoring is primary
- Preregistered primary claim uses the semantic-exact, identity-permutation readout only
- Cross-dataset summaries are descriptive unless a stricter preregistered rule is added later

## Python

- Targeted for Python `3.11+`
- Current observed local shell is newer than the original repo floor; see `ENVIRONMENT.md`

## Libraries

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `PyYAML`
- `scikit-learn`
- `huggingface_hub` for readiness metadata checks
