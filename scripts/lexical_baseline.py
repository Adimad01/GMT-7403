"""A bag-of-words control: the score reachable without understanding anything.

Multinomial naive Bayes over word counts, trained on the same train.csv the
adapters were fine-tuned on and tested on the same eval rows. No geometry, no
reasoning, no model of any kind -- and place names are masked, so it cannot
answer from geography either. Word order is ignored, which is the point: what
it reaches is what the wording gives away on its own.

This is the floor a reported accuracy has to clear before it means what it
appears to mean. On the topological family it reaches 77 percent against the
base model's 80, which says that family is close to a vocabulary task rather
than a spatial one. On cardinal it reaches 28 against 76, so there the model
is doing something a word count cannot.

    python3 scripts/lexical_baseline.py
"""
import csv, math, re, collections
from pathlib import Path

def tokens(row):
    t = row["corpus"].lower()
    for col in ("source_entity", "target_entity", "via_entity", "observer_entity"):
        v = (row.get(col) or "").strip().lower()
        if not v:
            continue
        for form in (v, re.sub(r"^(city|state|province|borough) of ", "", v)):
            t = t.replace(form, " <place> ")
    return re.findall(r"[a-z]+", t)

for rel in ("topological", "cardinal", "relative"):
    tr = [r for r in csv.DictReader(open(f"data/{rel}/train.csv", encoding="utf-8"))
          if r["ambiguity_level"] != "Level 6"]
    ev = [r for r in csv.DictReader(open(f"data/{rel}/eval.csv", encoding="utf-8"))
          if r["ambiguity_level"] != "Level 6"]

    counts = collections.defaultdict(collections.Counter)
    prior = collections.Counter()
    vocab = set()
    for r in tr:
        lab = r["relation_label"]
        prior[lab] += 1
        for w in tokens(r):
            counts[lab][w] += 1
            vocab.add(w)
    total = {l: sum(c.values()) for l, c in counts.items()}
    V = len(vocab)

    def predict(row):
        best, best_s = None, -1e18
        for lab in prior:
            s = math.log(prior[lab] / len(tr))
            for w in tokens(row):
                s += math.log((counts[lab][w] + 1) / (total[lab] + V))
            if s > best_s:
                best, best_s = lab, s
        return best

    by_level = collections.defaultdict(lambda: [0, 0])
    ok = 0
    for r in ev:
        hit = predict(r) == r["relation_label"]
        ok += hit
        b = by_level[r["ambiguity_level"]]
        b[0] += hit; b[1] += 1
    print(f"{rel:<13} sac de mots : {ok/len(ev)*100:>5.1f} %   n={len(ev)}")
    print("              par niveau : " + "  ".join(
        f"L{lv[-1]}={by_level[lv][0]/by_level[lv][1]*100:.0f}%"
        for lv in sorted(by_level)))
