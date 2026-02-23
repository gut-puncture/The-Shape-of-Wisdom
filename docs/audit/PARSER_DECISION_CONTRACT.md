# Parser Decision Contract (V3)

This note defines deterministic option parsing precedence and fail-closed conflict behavior for stage00a artifacts.

## Resolution order
1. First-token letter signal (`A/B/C/D`) resolves as `resolved_letter_first_token` when no disqualifying conflict exists.
2. Response-level unique letter cues resolve as `resolved_letter`.
3. Unique numeric cues mapped to options resolve as `resolved_numeric`.
4. Unique option-text fallback resolves as `resolved_option_text`.
5. Otherwise resolve to `unresolved_no_signal`.

## Conflict policy
1. Letter-vs-letter disagreement keeps first-token precedence when a first-token letter exists.
2. Numeric-vs-letter disagreement is fail-closed, including first-token letter vs numeric signal disagreements.
3. Numeric-vs-option-text disagreement is fail-closed.
4. Conflicts must produce `decision=unresolved_conflicting_signals` and `parsed_choice=null`.

## Authority boundary
1. Parser output is an observed text signal only.
2. Final `is_correct` for V2 metrics is authoritative from final-layer top1 A/B/C/D logits.
