"""Rewrite the topological training descriptions so no training sentence shares a core with evaluation.

The evaluation side is frozen: eval.csv, eval_manifest.json, fewshot_manifest_eval.json and the
"eval" pool in paraphrases_topological.json are not touched, so every result computed on them stays
comparable. Only training changes:

1. The "train" pool of each cell becomes the old training templates whose core never occurs in the
   evaluation pool, plus the sentences in templates_topological_train_v2.py, each offered bare and
   behind the four openers the evaluation pool uses (so an opener says nothing about the split).
   Every old `crosses` template is dropped: they all cast {A} as the line, and in the corpus the
   subject is often the area.
2. Each Level 1-5 row of train.csv keeps its places, label and level and gets a new description.
   Within a cell, cores are dealt in shuffled rounds, so a cell repeats a core only once all its
   cores are used. Level 6 rows are left as they are.
3. The same descriptions are written to the matching rows of corpus.csv.

Row order and count in train.csv are unchanged, so fewshot_manifest.json still points at valid rows
(the demonstrations' wording changes, which is why the few-shot runs have to be redone).

Do not run scripts/build_splits.py after this: it regenerates evaluation too.

    python3 data_generation/rebuild_topological_train.py
    python3 scripts/check_template_split.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_generation"))
sys.path.insert(0, str(REPO / "scripts"))
from apply_templates import short  # noqa: E402
from check_template_split import core  # noqa: E402
from templates_topological_train_v2 import NEW  # noqa: E402

OPENERS = ("Looking at the map", "In spatial terms", "Geographically", "Visually")
SEED = 20261004
HOP = "Level 6"


def variants(t: str) -> list[str]:
    head = t if t.startswith("{") else t[0].lower() + t[1:]
    return [t] + [f"{o}, {head}" for o in OPENERS]


def main() -> int:
    data = REPO / "data" / "topological"
    pools_path = REPO / "data_generation" / "paraphrases_topological.json"
    pools = json.loads(pools_path.read_text())
    eval_cores = {core(t) for ts in pools["eval"].values() for t in ts}

    train_pool: dict[str, list[str]] = {}
    for key in pools["eval"]:
        pred, lvl = key.split("|")
        kept = [] if pred == "crosses" else [
            t for t in pools["train"].get(key, []) if core(t) not in eval_cores]
        new = [v for t in NEW[pred][int(lvl)] for v in variants(t)]
        train_pool[key] = kept + new
    pools["train"] = train_pool
    pools_path.write_text(json.dumps(pools, indent=1, ensure_ascii=False) + "\n")

    with (data / "train.csv").open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields, train = rd.fieldnames, list(rd)
    with (data / "corpus.csv").open(newline="", encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))

    # corpus rows that went to training, found by their full content before rewriting
    sig = lambda r: tuple(r[c] for c in fields)
    where: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(corpus):
        where[sig(r)].append(i)

    rng = random.Random(SEED)
    by_cell: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        lvl = r["ambiguity_level"].strip()
        if lvl != HOP:
            by_cell[f"{r['relation_label'].strip()}|{lvl.split()[-1]}"].append(i)

    rewritten = 0
    for key, idx in sorted(by_cell.items()):
        groups: dict[str, list[str]] = defaultdict(list)
        for t in train_pool[key]:
            groups[core(t)].append(t)
        cores = sorted(groups)
        deck: list[str] = []
        rng.shuffle(idx)
        for i in idx:
            if not deck:
                deck = cores[:]
                rng.shuffle(deck)
            tpl = rng.choice(groups[deck.pop()])
            r = train[i]
            old = sig(r)
            r["corpus"] = (tpl.replace("{A}", short(r["source_entity"]))
                              .replace("{B}", short(r["target_entity"])))
            hits = where.get(old)
            if not hits:
                raise SystemExit(f"train row {i} has no matching corpus row")
            corpus[hits.pop(0)]["corpus"] = r["corpus"]
            rewritten += 1

    for name, rows in (("train.csv", train), ("corpus.csv", corpus)):
        with (data / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"  {rewritten} training rows rewritten, {len(train) - rewritten} Level 6 rows kept")
    print(f"  train pool: {sum(len(v) for v in train_pool.values())} templates, "
          f"{len({core(t) for v in train_pool.values() for t in v})} distinct cores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
