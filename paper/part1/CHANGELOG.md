# CHANGELOG.md — Paper 1 / Part I Revision

## Section-Level Changes

### Abstract
- Removed: seed value, "cached artifacts" phrasing, directory names, offline flags, hash details
- Added: Three concrete findings stated in plain language
- Added: One-sentence limitation statement

### Introduction
- Restructured contributions as: measurement correction → trajectory regimes → decision-space structure
- Removed "defining a metric" as standalone contribution
- Moved all reproducibility references to Appendix A

### Methods and Definitions (was "Measurement Setup and Definitions")
- **Added** new subsection "What we measure at each layer" — explains option scores in plain words before equations
- Margin definitions kept purely definitional
- "Why soft?" motivation moved to Results §3.1

### Results (was "Findings")
- **Added** "Findings overview" paragraph listing three findings before details
- Each finding restructured into: measured → observed → matters → caveat
- Tautological statement about final-layer separation explicitly labeled as "partly definitional"
- "Non-overlapping regions" replaced with "far apart" plus definitional caveat
- **Finding 1** (competitor-switching artifact) now has its own subsection with individual + aggregate evidence
- **Finding 2** (regime separation) includes PCA interpretation diagnostics:
  - PC1 loadings (~0.5 on each option = overall confidence)
  - PC1 ↔ entropy correlation (r ∈ [-0.53, -0.89])
  - "How to read this figure" paragraph added before PCA plot
  - PC2 described as inter-option contrast (varies by model)
- **Finding 3** (late commitment) explicitly connects to flow-field convergence

### Robustness
- Removed "phase boundary" language (was smooth variation)
- Shortened; kept factual

### Limitations
- Removed seed mention (moved to Appendix A)
- Shortened from 7 to 6 items

### Conclusion
- Simplified; references model-specific macros for commitment depth

### Appendix
- **Moved** appendix BEFORE \bibliography to prevent figures appearing after references
- Added \label{sec:references_start} for automated float-placement checking
- Pipeline TikZ diagram moved to Appendix B (out of main body)
- Reproducibility details concentrated in Appendix A

## Figure-Level Changes

| Figure | Change |
|--------|--------|
| Figure A | Caption rewritten; "heavier right tail" framing instead of "larger jumps" |
| Fig 2 | Seed removed from caption |
| Fig 3 | Caption: added "partly definitional" qualifier |
| Fig 4 (PCA) | Caption: PC1 explicitly labeled as "overall confidence"; loading diagnostics added |
| Fig 5 (Flow) | No change to figure; text restructured around it |
| Fig 6 (Commit) | No change |
| Fig 7 (Robust) | No change |
| Pipeline (TikZ) | Moved to Appendix B; single-column format |

## Infrastructure Added

| File | Purpose |
|------|---------|
| `scripts/paper_checks/check_float_placement.py` | Verifies no figures after references |
| `scripts/paper_checks/check_paper_text.py` | Verifies abstract has no forbidden tokens; terminology audit |
| `PAPER_BUILD_MAP.md` | Parquet → figure/table provenance mapping |
| `CHANGELOG.md` | This file |
