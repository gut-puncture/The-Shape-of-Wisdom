#!/usr/bin/env python3
"""Build a factual-recall dataset (subject, relation, answer) from Wikidata.

The same answer entity (a country, city, language or sport) recurs across many
subjects and several relations, and subjects are spread over popularity bands by
Wikipedia sitelink count. Raw SPARQL results are cached under dataset/cache/, so a
rerun only re-queries what is missing.

Usage:  python3 dataset/build_facts.py
Output: dataset/facts.jsonl, dataset/answers.json, dataset/README_facts.md
"""
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "shape-of-wisdom-research/1.0 (shaileshrana77@yahoo.com)"
SEED = 42
PULL = 80          # random candidates fetched per (answer, relation, band)
WORKERS = 2        # concurrent queries (WDQS allows 5 per client)
MIN_GAP_S = 1.0    # minimum pause between query starts
MIN_STATEMENTS = 5 # thinner items are often unreliable (e.g. a Ghanaian name whose only claim was P27 Turkey)

BANDS = [("obscure", 1, 4), ("mid", 5, 19), ("known", 20, None)]
QUOTA_16 = {"obscure": 6, "mid": 5, "known": 5}
QUOTA_24 = {"obscure": 8, "mid": 8, "known": 8}
NEIGHBOURS = {"obscure": ["mid", "known"], "mid": ["known", "obscure"], "known": ["mid", "obscure"]}

# ---------------------------------------------------------------- answer entities
# name -> (qid, extra leak terms: adjectival forms and common variants)
COUNTRIES = {
    "France": ("Q142", ["French"]),
    "Germany": ("Q183", ["German"]),
    "Italy": ("Q38", ["Italian"]),
    "Spain": ("Q29", ["Spanish", "Spaniard"]),
    "Portugal": ("Q45", ["Portuguese"]),
    "Netherlands": ("Q55", ["Dutch", "Holland", "Netherlandish"]),
    "Sweden": ("Q34", ["Swedish", "Swede"]),
    "Norway": ("Q20", ["Norwegian"]),
    "Poland": ("Q36", ["Polish"]),
    "Greece": ("Q41", ["Greek", "Hellenic"]),
    "Russia": ("Q159", ["Russian"]),
    "Turkey": ("Q43", ["Turkish", "Turk", "Türkiye"]),
    "Egypt": ("Q79", ["Egyptian"]),
    "Nigeria": ("Q1033", ["Nigerian"]),
    "South Africa": ("Q258", ["South African"]),
    "Kenya": ("Q114", ["Kenyan"]),
    "India": ("Q668", ["Indian"]),
    "China": ("Q148", ["China", "Chinese"]),
    "Japan": ("Q17", ["Japanese"]),
    "South Korea": ("Q884", ["Korea", "Korean"]),
    "Indonesia": ("Q252", ["Indonesian"]),
    "Brazil": ("Q155", ["Brazilian", "Brasil"]),
    "Argentina": ("Q414", ["Argentine", "Argentinian", "Argentinean"]),
    "Mexico": ("Q96", ["Mexican", "México"]),
}
CITIES = {
    "Paris": ("Q90", ["Parisian"]),
    "London": ("Q84", ["Londoner"]),
    "Berlin": ("Q64", ["Berliner"]),
    "Rome": ("Q220", ["Roman", "Roma"]),
    "Madrid": ("Q2807", ["Madrileño", "Madrilenian"]),
    "Vienna": ("Q1741", ["Viennese", "Wien"]),
    "Moscow": ("Q649", ["Muscovite", "Moskva"]),
    "Istanbul": ("Q406", ["Constantinople", "İstanbul"]),
    "Cairo": ("Q85", ["Cairene"]),
    "Tokyo": ("Q1490", ["Tokyoite"]),
    "Mumbai": ("Q1156", ["Bombay"]),
    "New Delhi": ("Q987", ["Delhi"]),
    "New York City": ("Q60", ["New York", "NYC", "New Yorker"]),
    "Mexico City": ("Q1489", ["Mexico"]),
    "Buenos Aires": ("Q1486", ["Porteño"]),
    "Chicago": ("Q1297", ["Chicagoan"]),
}
# Chinese: Q7850 "Chinese" is used ~23.6k times for P1412+P364 versus ~2.6k for
# Q727694 "Standard Chinese" and ~1.8k for Q9192 "Mandarin" (checked 2026-10-19).
LANGUAGES = {
    "French": ("Q150", []),
    "German": ("Q188", []),
    "Spanish": ("Q1321", []),
    "Italian": ("Q652", []),
    "Portuguese": ("Q5146", []),
    "Russian": ("Q7737", []),
    "Japanese": ("Q5287", []),
    "Chinese": ("Q7850", ["Mandarin"]),
    "Arabic": ("Q13955", []),
    "Hindi": ("Q1568", []),
    "Polish": ("Q809", []),
    "Turkish": ("Q256", []),
}
SPORTS = {
    "association football": ("Q2736", ["football", "soccer", "footballer"]),
    "basketball": ("Q5372", []),
    "tennis": ("Q847", []),
    "ice hockey": ("Q41466", ["hockey"]),
    "baseball": ("Q5369", []),
    "cricket": ("Q5375", ["cricketer"]),
    "rugby union": ("Q5849", ["rugby"]),
    "golf": ("Q5377", ["golfer"]),
    "boxing": ("Q32112", ["boxer"]),
    "volleyball": ("Q1734", []),
}
ANSWER_GROUPS = [("country", COUNTRIES), ("city", CITIES), ("language", LANGUAGES), ("sport", SPORTS)]

# Wikidata stores Dutch citizenship (P27) almost always as Q29999 "Kingdom of the
# Netherlands" (110k uses vs 3k for Q55), while P17 and P495 use Q55. We treat them
# as one answer: query both and count them as a single value.
EQUIVALENT_QIDS = {"Q55": ["Q29999"]}
CANONICAL = {e: q for q, es in EQUIVALENT_QIDS.items() for e in es}

# answer_type -> [(relation, relation_name, subject_kind)]
RELATIONS = {
    "country": [("P27", "country of citizenship", "human"),
                ("P17", "country", "settlement"),
                ("P495", "country of origin", "film")],
    "city": [("P19", "place of birth", "human"),
             ("P159", "headquarters location", "org")],
    "language": [("P1412", "languages spoken, written or signed", "human"),
                 ("P364", "original language of film or TV show", "film")],
    "sport": [("P641", "sport", "human")],
}
SUBJECT_CLASS = {   # direct P31 only, no subclass traversal
    "human": "?s wdt:P31 wd:Q5 .",
    "film": "?s wdt:P31 wd:Q11424 .",
    "settlement": "VALUES ?cls { wd:Q515 wd:Q3957 wd:Q532 } ?s wdt:P31 ?cls .",
    "org": "VALUES ?cls { wd:Q4830453 wd:Q783794 wd:Q43229 } ?s wdt:P31 ?cls .",
}
# Namesake check: other items with the same English label that a reader could
# confuse with the subject.
NAMESAKE_COND = {
    "human": "?o wdt:P31 wd:Q5 .",
    "film": ("VALUES ?oc { wd:Q11424 wd:Q506240 wd:Q24862 wd:Q5398426 wd:Q202866 wd:Q1261214 } "
             "?o wdt:P31 ?oc ."),
    "settlement": "?o wdt:P625 ?coord .",   # any item with coordinates (a place)
    "org": ("{ ?o wdt:P159 ?hq . } UNION { VALUES ?oc { wd:Q4830453 wd:Q783794 wd:Q43229 "
            "wd:Q6881511 wd:Q891723 wd:Q163740 wd:Q167037 } ?o wdt:P31 ?oc . }"),
}

# ---------------------------------------------------------------- SPARQL client
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
_gap_lock = threading.Lock()
_last_start = [0.0]
STATS = Counter()


def sparql(query):
    """Run a query with caching, pacing and retries. Returns bindings, or None on failure."""
    key = hashlib.sha1(query.encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        STATS["cache_hits"] += 1
        return json.loads(path.read_text())["bindings"]
    for attempt in range(6):
        with _gap_lock:
            wait = _last_start[0] + MIN_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_start[0] = time.time()
        try:
            r = _session.post(ENDPOINT, data={"query": query, "format": "json"}, timeout=90)
        except requests.RequestException as e:
            status, body = None, str(e)
        else:
            status, body = r.status_code, r.text
            if status == 200:
                rows = [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"query": query, "bindings": rows}))
                tmp.replace(path)
                STATS["queries_run"] += 1
                return rows
            if status == 400:
                raise RuntimeError(f"Bad query:\n{query}\n{body[:500]}")
        timed_out = status == 504 or "TimeoutException" in (body or "")
        if timed_out and attempt >= 1:
            break
        if status == 429:
            delay = int(r.headers.get("Retry-After", "60") or 60)
        else:
            delay = 10 * 2 ** attempt
        print(f"  retry {attempt + 1} after status {status} (sleep {delay}s)", flush=True)
        time.sleep(delay)
    STATS["queries_failed"] += 1
    print(f"  FAILED query (giving up):\n{query[:300]}", flush=True)
    return None


def qid(uri):
    return uri.rsplit("/", 1)[-1]


def lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"@en'


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ---------------------------------------------------------------- answers
def fetch_answers():
    entries = [(atype, name, q, extra) for atype, group in ANSWER_GROUPS for name, (q, extra) in group.items()]
    values = " ".join(f"wd:{q}" for _, _, q, _ in entries)
    rows = sparql(f"""SELECT ?x ?label ?alias ?demonym WHERE {{
  VALUES ?x {{ {values} }}
  ?x rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?x skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}
  OPTIONAL {{ ?x wdt:P1549 ?demonym . FILTER(LANG(?demonym) = "en") }}
}}""")
    meta = defaultdict(lambda: {"label": None, "aliases": set(), "demonyms": set()})
    for r in rows:
        m = meta[qid(r["x"])]
        m["label"] = r["label"]
        if r.get("alias"):
            m["aliases"].add(r["alias"])
        if r.get("demonym"):
            m["demonyms"].add(r["demonym"])
    answers = []
    for atype, name, q, extra in entries:
        m = meta[q]
        if m["label"] is None:
            raise RuntimeError(f"No English label for {name} {q}")
        # keep aliases that contain a letter (drops flag emoji and symbols)
        aliases = sorted(a for a in m["aliases"] if re.search(r"[^\W\d_]", a) and a != m["label"])
        demonyms = sorted(m["demonyms"])
        leak = []
        for t in [m["label"], name] + aliases + demonyms + extra:
            if t not in leak:
                leak.append(t)
        answers.append({
            "qid": q, "name": name, "answer_type": atype, "label": m["label"],
            "aliases": aliases, "demonyms": demonyms, "leak_terms": leak,
            "equivalent_qids": EQUIVALENT_QIDS.get(q, []),
            "relations": [rel for rel, _, _ in RELATIONS[atype]],
        })
    return answers


def leak_regex(terms):
    """Whole-word match; case-sensitive for short terms (codes like NYC, PRC, FR)."""
    long_terms = [re.escape(t) for t in terms if len(t) > 3]
    short_terms = [re.escape(t) for t in terms if len(t) <= 3]
    pats = []
    if long_terms:
        pats.append(re.compile(r"(?<!\w)(?:" + "|".join(long_terms) + r")(?!\w)", re.IGNORECASE))
    if short_terms:
        pats.append(re.compile(r"(?<!\w)(?:" + "|".join(short_terms) + r")(?!\w)"))
    return lambda s: any(p.search(s) for p in pats)


# ---------------------------------------------------------------- candidates
def candidate_query(rel, kind, answer_qids, band):
    _, lo, hi = band
    cond = f"?n >= {lo}" + (f" && ?n <= {hi}" if hi else "")
    vals = " ".join(f"wd:{q}" for q in answer_qids)
    salt = f"seed{SEED}|{rel}|{answer_qids[0]}"
    return f"""SELECT DISTINCT ?s ?n WHERE {{
  VALUES ?ans {{ {vals} }}
  ?s wdt:{rel} ?ans .
  {SUBJECT_CLASS[kind]}
  ?s wikibase:sitelinks ?n .
  FILTER({cond})
}} ORDER BY MD5(CONCAT(STR(?s), "{salt}")) LIMIT {PULL}"""


def details_query(rel, qids):
    vals = " ".join(f"wd:{q}" for q in qids)
    # For P641, also count occupations that are a kind of athlete (Q2066131), so team
    # owners, officials and commentators with a P641 value can be dropped.
    extra = ("\n  OPTIONAL { ?s wdt:P106 ?occ . ?occ wdt:P279* wd:Q2066131 . }" if rel == "P641" else "")
    # For P27, the earliest birth year, to drop anachronistic citizenships.
    extra += ("\n  OPTIONAL { ?s wdt:P569 ?born . }" if rel == "P27" else "")
    return f"""SELECT ?s (SAMPLE(?lab) AS ?label) (SAMPLE(?d) AS ?desc) (GROUP_CONCAT(DISTINCT STR(?v); separator="|") AS ?vals) (SAMPLE(?nst) AS ?statements) (COUNT(DISTINCT ?occ) AS ?athletic) (MIN(YEAR(?born)) AS ?birth_year) WHERE {{
  VALUES ?s {{ {vals} }}
  OPTIONAL {{ ?s rdfs:label ?lab . FILTER(LANG(?lab) = "en") }}
  OPTIONAL {{ ?s schema:description ?d . FILTER(LANG(?d) = "en") }}
  OPTIONAL {{ ?s wikibase:statements ?nst . }}
  OPTIONAL {{ ?s p:{rel} ?st . ?st ps:{rel} ?v . FILTER NOT EXISTS {{ ?st wikibase:rank wikibase:DeprecatedRank }} }}{extra}
}} GROUP BY ?s"""


def namesake_query(kind, labels):
    vals = " ".join(lit(l) for l in labels)
    return f"""SELECT DISTINCT ?lab ?o WHERE {{
  VALUES ?lab {{ {vals} }}
  ?o rdfs:label ?lab .
  {NAMESAKE_COND[kind]}
}}"""


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    print("Fetching answer entities ...", flush=True)
    answers = fetch_answers()
    ans_by_qid = {a["qid"]: a for a in answers}
    all_answer_qids = set(ans_by_qid) | {q for a in answers for q in a["equivalent_qids"]}

    pairs = []  # (answer, rel, rel_name, kind)
    for a in answers:
        for rel, rel_name, kind in RELATIONS[a["answer_type"]]:
            pairs.append((a, rel, rel_name, kind))

    # Stage A: random candidates per (answer, relation, band)
    jobs = [(i, b) for i in range(len(pairs)) for b in BANDS]
    print(f"Stage A: {len(jobs)} candidate queries ...", flush=True)

    def run_a(job):
        i, band = job
        a, rel, _, kind = pairs[i]
        rows = sparql(candidate_query(rel, kind, [a["qid"]] + a["equivalent_qids"], band))
        return job, rows

    cand = {}  # (i, band_name) -> [(qid, sitelinks)]
    failed_bands = []
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for (i, band), rows in ex.map(run_a, jobs):
            done += 1
            if rows is None:
                failed_bands.append((pairs[i][0]["name"], pairs[i][1], band[0]))
                rows = []
            cand[(i, band[0])] = [(qid(r["s"]), int(r["n"])) for r in rows]
            if done % 30 == 0:
                print(f"  {done}/{len(jobs)} done ({dict(STATS)})", flush=True)

    # Stage B: label, description and relation values for each candidate
    print("Stage B: details ...", flush=True)

    def run_b(i):
        a, rel, _, _ = pairs[i]
        qids = sorted({q for b in BANDS for q, _ in cand[(i, b[0])]})
        out = {}
        for ch in chunks(qids, 120):
            rows = sparql(details_query(rel, ch)) or []
            for r in rows:
                out[qid(r["s"])] = r
        return i, out

    details = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, out in ex.map(run_b, range(len(pairs))):
            details[i] = out

    # Per-subject filters, applied in random (MD5) order within each band
    removed = Counter()
    removed_by_rel = defaultdict(Counter)
    survivors = {}  # (i, band) -> list of subject dicts
    for i, (a, rel, rel_name, kind) in enumerate(pairs):
        leaks = leak_regex(a["leak_terms"])
        for b in BANDS:
            keep = []
            for q, n in cand[(i, b[0])]:
                d = details[i].get(q)
                label = (d or {}).get("label")
                vals = {qid(v) for v in (d or {}).get("vals", "").split("|") if v}
                vals = {CANONICAL.get(v, v) for v in vals}   # Q29999 counts as Q55
                if d is None:
                    reason = "details_query_failed"
                elif not label:
                    reason = "no_english_label"
                elif re.fullmatch(r"Q\d+", label):
                    reason = "label_is_qid"
                elif len(label) > 60:
                    reason = "label_over_60_chars"
                elif "(" in label or ")" in label:
                    reason = "label_has_parentheses"
                elif kind == "human" and len(label.split()) < 2:
                    reason = "human_label_is_one_word"
                elif vals != {a["qid"]}:
                    reason = "not_exactly_one_value"
                elif int(d.get("statements") or 0) < MIN_STATEMENTS:
                    reason = "too_few_statements"
                elif rel == "P27" and d.get("birth_year") and int(d["birth_year"]) < 1800:
                    reason = "p27_born_before_1800"
                elif rel == "P641" and int(d.get("athletic") or 0) == 0:
                    reason = "p641_but_not_an_athlete"
                elif q in all_answer_qids:
                    reason = "subject_is_an_answer"
                elif leaks(label):
                    reason = "label_leaks_answer"
                else:
                    reason = None
                if reason:
                    removed[reason] += 1
                    removed_by_rel[rel][reason] += 1
                    continue
                desc = d.get("desc") or ""
                keep.append({"qid": q, "label": label, "desc": desc, "n": n,
                             "desc_leak": bool(desc) and leaks(desc)})
            survivors[(i, b[0])] = keep

    # Stage C: namesake check on the first few survivors of each band
    print("Stage C: namesake check ...", flush=True)
    to_check = defaultdict(set)  # kind -> labels
    checked = {}
    for i, (a, rel, _, kind) in enumerate(pairs):
        quota = QUOTA_24 if a["answer_type"] == "sport" else QUOTA_16
        for b in BANDS:
            k = 2 * quota[b[0]] + 6
            checked[(i, b[0])] = survivors[(i, b[0])][:k]
            for s in checked[(i, b[0])]:
                to_check[kind].add(s["label"])
    owners = defaultdict(set)  # (kind, label) -> items with that label

    def run_c(job):
        kind, labels = job
        return sparql(namesake_query(kind, labels))

    c_jobs = [(kind, ch) for kind, labels in to_check.items() for ch in chunks(sorted(labels), 60)]
    namesake_failed = set()
    with ThreadPoolExecutor(WORKERS) as ex:
        for (kind, labels), rows in zip(c_jobs, ex.map(run_c, c_jobs)):
            if rows is None:
                namesake_failed |= {(kind, l) for l in labels}
                continue
            for r in rows:
                owners[(kind, r["lab"])].add(qid(r["o"]))

    eligible = {}
    for i, (a, rel, _, kind) in enumerate(pairs):
        for b in BANDS:
            keep = []
            for s in checked[(i, b[0])]:
                if (kind, s["label"]) in namesake_failed:
                    reason = "namesake_check_failed"
                elif owners[(kind, s["label"])] - {s["qid"]}:
                    reason = "label_shared_with_other_wikidata_item"
                else:
                    reason = None
                if reason:
                    removed[reason] += 1
                    removed_by_rel[rel][reason] += 1
                else:
                    keep.append(s)
            unchecked = len(survivors[(i, b[0])]) - len(checked[(i, b[0])])
            removed["surplus_not_checked"] += unchecked
            removed_by_rel[rel]["surplus_not_checked"] += unchecked
            eligible[(i, b[0])] = keep

    # Global label uniqueness: a label used by two different subjects anywhere is dropped
    label_owners = defaultdict(set)
    for ss in eligible.values():
        for s in ss:
            label_owners[s["label"].casefold()].add(s["qid"])
    for key, ss in eligible.items():
        rel = pairs[key[0]][1]
        keep = []
        for s in ss:
            if len(label_owners[s["label"].casefold()]) > 1:
                removed["label_duplicated_across_pulls"] += 1
                removed_by_rel[rel]["label_duplicated_across_pulls"] += 1
            else:
                keep.append(s)
        eligible[key] = keep

    # Selection: quota per band, fill from neighbouring bands, one fact per subject
    order = list(range(len(pairs)))
    rng.shuffle(order)
    used = set()
    facts = []
    shortfalls = []
    for i in order:
        a, rel, rel_name, kind = pairs[i]
        quota = QUOTA_24 if a["answer_type"] == "sport" else QUOTA_16
        pool = {}
        for b in BANDS:
            pool[b[0]] = []
            for s in eligible[(i, b[0])]:
                if s["qid"] in used:
                    removed["subject_used_by_another_pair"] += 1
                    removed_by_rel[rel]["subject_used_by_another_pair"] += 1
                else:
                    pool[b[0]].append(s)
        chosen = {b: pool[b][:quota[b]] for b in pool}
        for b in pool:
            deficit = quota[b] - len(chosen[b])
            for nb in NEIGHBOURS[b]:
                if deficit <= 0:
                    break
                spare = pool[nb][len(chosen[nb]):]
                take = spare[:deficit]
                chosen[nb] += take
                deficit -= len(take)
        picked = [s for b in pool for s in chosen[b]]
        target = sum(quota.values())
        if len(picked) < target:
            shortfalls.append({"answer": a["name"], "relation": rel, "got": len(picked), "target": target,
                               "pool_by_band": {b: len(pool[b]) for b in pool}})
        for s in picked:
            used.add(s["qid"])
            band = next(name for name, lo, hi in BANDS if s["n"] >= lo and (hi is None or s["n"] <= hi))
            facts.append({
                "answer_type": a["answer_type"], "relation": rel, "relation_name": rel_name,
                "subject_qid": s["qid"], "subject_label": s["label"], "subject_description": s["desc"],
                "subject_sitelinks": s["n"], "band": band,
                "answer_qid": a["qid"], "answer_label": a["label"],
                "answer_aliases": [a["label"]] + [x for x in a["aliases"] if x != a["label"]],
                "description_mentions_answer": s["desc_leak"],
            })

    type_order = {t: k for k, (t, _) in enumerate(ANSWER_GROUPS)}
    band_order = {b[0]: k for k, b in enumerate(BANDS)}
    facts.sort(key=lambda f: (type_order[f["answer_type"]], f["relation"], f["answer_label"],
                              band_order[f["band"]], f["subject_label"]))
    for k, f in enumerate(facts, 1):
        f["fact_id"] = f"F{k:04d}"
    keys = ["fact_id", "answer_type", "relation", "relation_name", "subject_qid", "subject_label",
            "subject_description", "subject_sitelinks", "band", "answer_qid", "answer_label",
            "answer_aliases", "description_mentions_answer"]
    with open(HERE / "facts.jsonl", "w") as fh:
        for f in facts:
            fh.write(json.dumps({k: f[k] for k in keys}, ensure_ascii=False) + "\n")
    answers_out = {a["qid"]: {k: v for k, v in a.items() if k != "qid"} for a in answers}
    (HERE / "answers.json").write_text(json.dumps(answers_out, ensure_ascii=False, indent=1))

    # sanity: labels unique
    assert len({f["subject_label"].casefold() for f in facts}) == len(facts)
    assert len({f["subject_qid"] for f in facts}) == len(facts)

    write_readme(facts, removed, removed_by_rel, shortfalls, failed_bands, pairs, cand)
    print(f"Wrote {len(facts)} facts. Query stats: {dict(STATS)}", flush=True)


def write_readme(facts, removed, removed_by_rel, shortfalls, failed_bands, pairs, cand):
    c = Counter((f["answer_type"], f["relation"], f["band"]) for f in facts)
    rows = sorted({(t, r) for t, r, _ in c}, key=lambda x: ([g for g, _ in ANSWER_GROUPS].index(x[0]), x[1]))
    n_cand = sum(len(v) for v in cand.values())
    desc_leaks = sum(f["description_mentions_answer"] for f in facts)
    L = []
    L.append("# Factual-recall facts from Wikidata\n")
    L.append("Built by `dataset/build_facts.py` (rerun it to rebuild; raw query results are cached in "
             "`dataset/cache/`). Seed 42. Source: the public Wikidata Query Service.\n")
    L.append("## What is in it\n")
    L.append(f"- `facts.jsonl`: {len(facts)} facts, one per line. Each subject appears once.")
    L.append("- `answers.json`: the 62 answer entities with QID, English label, aliases, demonyms (P1549) and "
             "`leak_terms`.\n")
    L.append("Answers: 24 countries (P27 citizenship of humans, P17 country of cities/towns/villages, P495 country of "
             "origin of films), 16 cities (P19 birthplace of humans, P159 headquarters of businesses, companies and "
             "organizations), 12 languages (P1412 language of humans, P364 original language of films), "
             "10 sports (P641 sport of humans). Chinese is Q7850, the QID Wikidata uses most for these relations. "
             "Netherlands is Q55; for P27 Wikidata almost always uses Q29999 (Kingdom of the Netherlands), so both "
             "are treated as the same answer.\n")
    L.append("Popularity bands by sitelink count: obscure 1-4, mid 5-19, known 20 or more. Target per (answer, "
             "relation): 6 obscure, 5 mid, 5 known (8/8/8 for sports); a short band is filled from its neighbour "
             "and every fact records its true band.\n")
    L.append("## Queries\n")
    L.append("1. Candidates, one query per (answer, relation, band), random order within the band:\n")
    L.append("```sparql\nSELECT DISTINCT ?s ?n WHERE {\n  VALUES ?ans { wd:<answer> }\n  ?s wdt:<P> ?ans .\n"
             "  ?s wdt:P31 wd:Q5 .            # or Q11424 film; Q515/Q3957/Q532 settlement;\n"
             "                                # Q4830453/Q783794/Q43229 organisation (direct P31)\n"
             "  ?s wikibase:sitelinks ?n .\n  FILTER(?n >= 1 && ?n <= 4)\n"
             "} ORDER BY MD5(CONCAT(STR(?s), \"seed42|<P>|<answer>\")) LIMIT 80\n```\n")
    L.append("2. Details for the candidates (VALUES batches): English label, English description, and all "
             "non-deprecated values of the relation.")
    L.append("3. Namesake check: every other item with exactly the same English label and the same kind "
             "(human; film/TV; any item with coordinates for settlements; business/organisation or anything "
             "with P159 for organisations).\n")
    L.append("## Filters and how many candidates each removed\n")
    L.append("Each candidate is counted once, at the first filter it failed. Filters: English label present, not a "
             "QID, at most 60 characters, no parentheses; for humans, at least two words (one-word names such as "
             "\"Murdock\" or \"GQ\" are ambiguous); exactly one non-deprecated value for the relation; at "
             f"least {MIN_STATEMENTS} statements on the item (thin items are often wrong); for P27, born in 1800 or "
             "later when a birth date is known (drops anachronisms such as an ancient Greek with citizenship of "
             "modern Greece); for P641, an occupation that is a kind of athlete (drops team owners, officials, "
             "commentators); not itself an answer; label does not contain the answer (leak); no other Wikidata "
             "item of the same kind with the same English label; label not used by two different subjects "
             "anywhere in the pulls; one fact per subject.\n")
    L.append(f"Candidates pulled: {n_cand}.\n")
    L.append("| filter | removed |\n|---|---|")
    for k, v in removed.most_common():
        L.append(f"| {k} | {v} |")
    L.append("\n`surplus_not_checked` are valid candidates beyond what each band needed; they were not namesake-"
             "checked and are not errors. `label_leaks_answer` matches the answer's label, aliases, demonyms and "
             "adjectival forms as whole words (case-insensitive; case-sensitive for terms of 3 characters or less).\n")
    L.append("Removed per relation:\n")
    rels = sorted(removed_by_rel)
    reasons = [k for k, _ in removed.most_common()]
    L.append("| filter | " + " | ".join(rels) + " |\n|---|" + "---|" * len(rels))
    for k in reasons:
        L.append(f"| {k} | " + " | ".join(str(removed_by_rel[r][k]) for r in rels) + " |")
    L.append("\n## Final counts per (answer_type, relation, band)\n")
    L.append("| answer_type | relation | obscure | mid | known | total |\n|---|---|---|---|---|---|")
    for t, r in rows:
        o, m, k = (c[(t, r, b)] for b in ("obscure", "mid", "known"))
        L.append(f"| {t} | {r} | {o} | {m} | {k} | {o + m + k} |")
    bt = Counter(f["band"] for f in facts)
    L.append(f"| all | | {bt['obscure']} | {bt['mid']} | {bt['known']} | {len(facts)} |\n")
    L.append("## Shortfalls\n")
    if shortfalls:
        L.append("(answer, relation) pairs below target, with the eligible pool per band after filters:\n")
        L.append("| answer | relation | got | target | obscure / mid / known pool |\n|---|---|---|---|---|")
        for s in sorted(shortfalls, key=lambda s: (s["relation"], s["got"])):
            p = s["pool_by_band"]
            L.append(f"| {s['answer']} | {s['relation']} | {s['got']} | {s['target']} | "
                     f"{p['obscure']} / {p['mid']} / {p['known']} |")
    else:
        L.append("None.")
    if failed_bands:
        L.append("\nCandidate queries that failed (timeout): " + ", ".join(f"{a} {r} {b}" for a, r, b in failed_bands))
    L.append("\n## Caveats\n")
    L.append("- `facts.jsonl` has one field beyond the requested schema: `description_mentions_answer`.")
    L.append(f"- English descriptions often state the answer (e.g. \"French painter\"). {desc_leaks} facts have "
             "`description_mentions_answer: true`; do not put descriptions in prompts unchanged.")
    L.append("- Wikidata itself is noisy. A hand check of 30 random facts found 2 doubtful answers (Imperial Bank "
             "of India -> Mumbai, whose headquarters was Calcutta; Taradevi Siddhartha -> Hindi). P1412 = Hindi is "
             "the weakest cell: Indian politicians and actors were given Hindi by bulk edits, so about 6 of its 16 "
             "facts are people from non-Hindi-speaking states (Karnataka, Odisha, Assam, Kerala). Consider dropping "
             "or hand-checking that cell.")
    (HERE / "README_facts.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
