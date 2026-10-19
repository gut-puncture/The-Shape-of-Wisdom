"""Hand-curated names for every answer: the display name (first) and the other names a correct reply
may use. Wikidata aliases include codes ("el", "DE"), nicknames ("Big Smoke") and historical names
that would make grading wrong, so grading and the answer-word control use only these lists.
The original Wikidata aliases are kept as `wikidata_aliases` (used only to detect answer leaks).

    python dataset/curate_names.py
"""

import json
from pathlib import Path

NAMES = {
    # countries
    "Q142": ["France"], "Q183": ["Germany"], "Q38": ["Italy"], "Q29": ["Spain"], "Q45": ["Portugal"],
    "Q55": ["Netherlands", "Holland"], "Q34": ["Sweden"], "Q20": ["Norway"], "Q36": ["Poland"],
    "Q41": ["Greece"], "Q159": ["Russia", "Russian Federation"], "Q43": ["Turkey", "Türkiye", "Turkiye"],
    "Q79": ["Egypt"], "Q1033": ["Nigeria"], "Q258": ["South Africa"], "Q114": ["Kenya"], "Q668": ["India"],
    "Q148": ["China", "People's Republic of China", "PRC"], "Q17": ["Japan"],
    "Q884": ["South Korea", "Korea", "Republic of Korea"], "Q252": ["Indonesia"], "Q155": ["Brazil", "Brasil"],
    "Q414": ["Argentina"], "Q96": ["Mexico"],
    # cities
    "Q90": ["Paris"], "Q84": ["London"], "Q64": ["Berlin"], "Q220": ["Rome"], "Q2807": ["Madrid"],
    "Q1741": ["Vienna", "Wien"], "Q649": ["Moscow", "Moskva"], "Q406": ["Istanbul"], "Q85": ["Cairo"],
    "Q1490": ["Tokyo"], "Q1156": ["Mumbai", "Bombay"], "Q987": ["New Delhi", "Delhi"],
    "Q60": ["New York City", "New York", "NYC"], "Q1489": ["Mexico City"], "Q1486": ["Buenos Aires"],
    "Q1297": ["Chicago"],
    # languages
    "Q150": ["French"], "Q188": ["German"], "Q1321": ["Spanish", "Castilian"], "Q652": ["Italian"],
    "Q5146": ["Portuguese"], "Q7737": ["Russian"], "Q5287": ["Japanese"],
    "Q7850": ["Chinese", "Mandarin", "Mandarin Chinese", "Standard Chinese"], "Q13955": ["Arabic"],
    "Q1568": ["Hindi"], "Q809": ["Polish"], "Q256": ["Turkish"],
    # sports
    "Q2736": ["association football", "football", "soccer"], "Q5372": ["basketball"], "Q847": ["tennis"],
    "Q41466": ["ice hockey", "hockey"], "Q5369": ["baseball"], "Q5375": ["cricket"],
    "Q5849": ["rugby union", "rugby"], "Q5377": ["golf"], "Q32112": ["boxing"], "Q1734": ["volleyball"],
}


def main():
    path = Path(__file__).parent / "answers.json"
    answers = json.loads(path.read_text())
    assert set(answers) == set(NAMES), set(answers) ^ set(NAMES)
    for q, a in answers.items():
        if "wikidata_aliases" not in a:
            a["wikidata_aliases"] = a.get("aliases", [])
            a["wikidata_label"] = a["label"]
        a["label"], a["aliases"] = NAMES[q][0], NAMES[q][1:]
    path.write_text(json.dumps(answers, indent=1, ensure_ascii=False))
    print(f"curated {len(answers)} answers")


if __name__ == "__main__":
    main()
