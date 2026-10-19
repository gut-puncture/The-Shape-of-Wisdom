# Manual Visual QA

Rendered artifact inspected: `paper/build/main.pdf`

Rendered pages inspected via: `qa/contact_sheet.png` and `qa/page_pngs/page-*.png`

Result: PASS

Notes:

- No blank pages.
- No figures or tables protrude beyond margins.
- No visible text/figure overlap.
- Two-column text, constrained abstract, and full-width floats are aligned within page margins.
- Figure legends are outside the plotted data regions; the mechanistic trace panel now uses in-title encoding instead of a data-covering legend.
- Core figures appear near the section text that introduces them without float-only pages or large stranded column gaps.
- The staged CDF and model-lag plots are consolidated into one full-width result figure.
- The regime figure and exact-count table are merged into one local block.
- Appendix tables are anchored under A.1/A.2/A.3 headings; no detached blank B/C/D headings remain.
- Remaining LaTeX underfull warnings are visual non-issues in the rendered PDF.
