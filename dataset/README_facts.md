# Factual-recall facts from Wikidata

Built by `dataset/build_facts.py` (rerun it to rebuild; raw query results are cached in `dataset/cache/`). Seed 42. Source: the public Wikidata Query Service.

## What is in it

- `facts.jsonl`: 2273 facts, one per line. Each subject appears once.
- `answers.json`: the 62 answer entities with QID, English label, aliases, demonyms (P1549) and `leak_terms`.

Answers: 24 countries (P27 citizenship of humans, P17 country of cities/towns/villages, P495 country of origin of films), 16 cities (P19 birthplace of humans, P159 headquarters of businesses, companies and organizations), 12 languages (P1412 language of humans, P364 original language of films), 10 sports (P641 sport of humans). Chinese is Q7850, the QID Wikidata uses most for these relations. Netherlands is Q55; for P27 Wikidata almost always uses Q29999 (Kingdom of the Netherlands), so both are treated as the same answer.

Popularity bands by sitelink count: obscure 1-4, mid 5-19, known 20 or more. Target per (answer, relation): 6 obscure, 5 mid, 5 known (8/8/8 for sports); a short band is filled from its neighbour and every fact records its true band.

## Queries

1. Candidates, one query per (answer, relation, band), random order within the band:

```sparql
SELECT DISTINCT ?s ?n WHERE {
  VALUES ?ans { wd:<answer> }
  ?s wdt:<P> ?ans .
  ?s wdt:P31 wd:Q5 .            # or Q11424 film; Q515/Q3957/Q532 settlement;
                                # Q4830453/Q783794/Q43229 organisation (direct P31)
  ?s wikibase:sitelinks ?n .
  FILTER(?n >= 1 && ?n <= 4)
} ORDER BY MD5(CONCAT(STR(?s), "seed42|<P>|<answer>")) LIMIT 80
```

2. Details for the candidates (VALUES batches): English label, English description, and all non-deprecated values of the relation.
3. Namesake check: every other item with exactly the same English label and the same kind (human; film/TV; any item with coordinates for settlements; business/organisation or anything with P159 for organisations).

## Filters and how many candidates each removed

Each candidate is counted once, at the first filter it failed. Filters: English label present, not a QID, at most 60 characters, no parentheses; for humans, at least two words (one-word names such as "Murdock" or "GQ" are ambiguous); exactly one non-deprecated value for the relation; at least 5 statements on the item (thin items are often wrong); for P27, born in 1800 or later when a birth date is known (drops anachronisms such as an ancient Greek with citizenship of modern Greece); for P641, an occupation that is a kind of athlete (drops team owners, officials, commentators); not itself an answer; label does not contain the answer (leak); no other Wikidata item of the same kind with the same English label; label not used by two different subjects anywhere in the pulls; one fact per subject.

Candidates pulled: 29531.

| filter | removed |
|---|---|
| surplus_not_checked | 15643 |
| not_exactly_one_value | 4435 |
| no_english_label | 1843 |
| label_shared_with_other_wikidata_item | 1409 |
| human_label_is_one_word | 312 |
| too_few_statements | 225 |
| p27_born_before_1800 | 152 |
| label_leaks_answer | 111 |
| p641_but_not_an_athlete | 57 |
| label_has_parentheses | 54 |
| label_over_60_chars | 30 |
| subject_used_by_another_pair | 17 |
| subject_is_an_answer | 1 |

`surplus_not_checked` are valid candidates beyond what each band needed; they were not namesake-checked and are not errors. `label_leaks_answer` matches the answer's label, aliases, demonyms and adjectival forms as whole words (case-insensitive; case-sensitive for terms of 3 characters or less).

Removed per relation:

| filter | P1412 | P159 | P17 | P19 | P27 | P364 | P495 | P641 |
|---|---|---|---|---|---|---|---|---|
| surplus_not_checked | 1496 | 1462 | 2454 | 2782 | 3117 | 1198 | 1738 | 1396 |
| not_exactly_one_value | 663 | 237 | 154 | 84 | 935 | 632 | 1557 | 173 |
| no_english_label | 56 | 224 | 523 | 75 | 171 | 243 | 492 | 59 |
| label_shared_with_other_wikidata_item | 105 | 26 | 465 | 148 | 190 | 122 | 220 | 133 |
| human_label_is_one_word | 62 | 0 | 0 | 77 | 158 | 0 | 0 | 15 |
| too_few_statements | 0 | 38 | 148 | 1 | 19 | 1 | 16 | 2 |
| p27_born_before_1800 | 0 | 0 | 0 | 0 | 152 | 0 | 0 | 0 |
| label_leaks_answer | 0 | 73 | 4 | 6 | 5 | 5 | 18 | 0 |
| p641_but_not_an_athlete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 |
| label_has_parentheses | 3 | 12 | 17 | 2 | 2 | 5 | 12 | 1 |
| label_over_60_chars | 0 | 15 | 3 | 1 | 1 | 5 | 5 | 0 |
| subject_used_by_another_pair | 2 | 0 | 0 | 1 | 2 | 1 | 11 | 0 |
| subject_is_an_answer | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## Final counts per (answer_type, relation, band)

| answer_type | relation | obscure | mid | known | total |
|---|---|---|---|---|---|
| country | P17 | 149 | 128 | 92 | 369 |
| country | P27 | 144 | 120 | 120 | 384 |
| country | P495 | 144 | 152 | 88 | 384 |
| city | P159 | 96 | 90 | 70 | 256 |
| city | P19 | 96 | 80 | 80 | 256 |
| language | P1412 | 72 | 60 | 60 | 192 |
| language | P364 | 72 | 60 | 60 | 192 |
| sport | P641 | 80 | 80 | 80 | 240 |
| all | | 853 | 770 | 650 | 2273 |

## Shortfalls

(answer, relation) pairs below target, with the eligible pool per band after filters:

| answer | relation | got | target | obscure / mid / known pool |
|---|---|---|---|---|
| Indonesia | P17 | 10 | 16 | 1 / 9 / 0 |
| Japan | P17 | 11 | 16 | 7 / 3 / 1 |
| Brazil | P17 | 12 | 16 | 8 / 2 / 2 |

## Caveats

- `facts.jsonl` has one field beyond the requested schema: `description_mentions_answer`.
- English descriptions often state the answer (e.g. "French painter"). 1091 facts have `description_mentions_answer: true`; do not put descriptions in prompts unchanged.
- Wikidata itself is noisy. A hand check of 30 random facts found 2 doubtful answers (Imperial Bank of India -> Mumbai, whose headquarters was Calcutta; Taradevi Siddhartha -> Hindi). P1412 = Hindi is the weakest cell: Indian politicians and actors were given Hindi by bulk edits, so about 6 of its 16 facts are people from non-Hindi-speaking states (Karnataka, Odisha, Assam, Kerala). Consider dropping or hand-checking that cell.
