# Changelog

## Final presentation-only cleanup
- Abstract sentence changed from “positive winner-gap writing” to “winner-gap growth.”
- Figure 3 caption changed to remove “positive winner-runner separation.”
- Introduction closing sentence changed from a contender-before-margin statement to “the lag from persistent contender status to persistent decisive margin is typically nontrivial.”
- Setup sentence changed from “positive winner-gap writing” to “winner-gap growth.”
- Results sentence changed from “$\Delta_{21}$ is also positive” to “$\Delta_{21}$ is also nontrivial.”
- Heterogeneity section changed to remove the appendix positivity framing and replace it with a descriptive appendix-table reference.
- Appendix subsection renamed from “Model positivity table” to “Model summary table.”
- Appendix table caption changed from a composite positivity diagnostic to a descriptive model summary caption.
- Table 1 late-write row changed from “positive winner-gap increments” to “winner-gap growth.”
- Appendix table replaced: removed the Positive/Yes-No column and now report only median $\Delta_{250}$, CI low, CI high, and median write share after $d_2$.

## Sections replaced
- Replaced the title and abstract with a persistence-based center.
- Replaced Section 1 introduction so it frames the paper as a correction of an asymmetric observational setup.
- Replaced Section 2 with a combined setup and corrected operationalization section built around `d2`, `d1`, `d50`, `Delta_21`, `Delta_250`, and late-write metrics.
- Replaced the results framing so the main text emphasizes lag magnitude and late-write concentration rather than sign alone.
- Replaced the heterogeneity section so it focuses on model spread in `Delta_250` and write-share metrics.
- Replaced the limitations section so controls appear only as a short sensitivity note.
- Replaced the conclusion so it stays scoped to exact-option continuation scoring across the nine tested models.

## Figures replaced
- Replaced the flagship figure with pooled CDFs for `d2`, `d1`, and `d50`.
- Replaced the main lag figure so it shows only `Delta_21` and `Delta_250`.
- Replaced the exemplar figure with three algorithmically selected low-, median-, and high-lag trajectories.
- Removed any main-text figure whose argumentative role depended on controls.

## Old claims removed
- Removed the old paper center based on first-hit candidate entry.
- Removed `Delta_margin` and any headline framing built on it.
- Removed “plausibility comes early, commitment comes late” framing.
- Removed claims that controls demonstrate answer-format robustness or invariance.
- Removed language implying that models “know which answers are still possible.”
- Removed references to NaN-layer control stories from the main manuscript.

## Remaining limitations
- The claim is about exact-option continuation scoring only.
- The evidence base is nine tested models, not all language models.
- `d2 <= d1 <= d50` is partly nested by construction, so magnitude and late-write concentration are the substantive results.
- Controls are heterogeneous and therefore demoted to sensitivity context rather than primary evidence.
- The MMLU slice remains too small for a benchmark-level claim and is kept out of the main narrative.
