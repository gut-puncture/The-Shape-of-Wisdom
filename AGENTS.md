# Experiment Agent Instructions

- Work only inside `/Users/shaileshrana/shape-of-wisdom/`.
- Treat `/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/v2/BLOCKER_REPORT.md` as active context.
- Do not silence dataset, model, prompt, or tokenization anomalies. Log them.
- Do not run `run_full_sweep.py` unless the runtime guard and config blocker are both cleared.
- Update `TODO.md` and `BLOCKERS.md` whenever a phase state or blocker changes.
- Preserve exact option surfaces in the main semantic path.
- Keep controls paired with the semantic readout on the same valid example subset whenever
  comparisons are reported.
