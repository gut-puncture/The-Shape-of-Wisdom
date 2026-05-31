# Reproduce

This paper release includes the cached MMLU artifacts needed to rebuild the paper-facing figures and tables. No model inference is required.

```bash
python3 scripts/build_all.py
python3 -m pytest -q
```

Primary output: `paper/build/main.pdf`.
