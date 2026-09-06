"""Give every verified triplet a vernacular description.

The corpus already holds the facts: subject, predicate, object, each predicate
computed from OpenStreetMap geometry. This writes the sentence the model
actually sees, by filling a paraphrase template with the triplet's places.

Two things decide which template a row gets.

The ambiguity level is now a property of the WORDING, not of the geometry, so
it is assigned here rather than carried over. Rows are spread evenly across the
five levels within each predicate, which keeps the grid balanced and makes the
level an experimental variable rather than a relabelling of how far apart two
places happen to be.

The template pool depends on which split the row will land in. Evaluation rows
are written from templates that never appear in training, so the evaluation
asks whether the model can read a wording it has not seen. Sharing them would
let a fine-tuned model recognise the phrase instead, and would put the answer
directly into few-shot demonstrations.

Level 6 rows keep the descriptions they already have: multi-hop states two
links and asks for their composition, which is a different mechanism from
paraphrase and is not what the templates cover.

    python3 data_generation/apply_templates.py --relation cardinal
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_PER_CELL = {"cardinal": 6, "relative": 10, "topological": 7}
HOP = "Level 6"


def short(name: str) -> str:
    """The place name as it reads inside a sentence."""
    for p in ("City of ", "State of ", "Province of ", "Borough of "):
        if name.startswith(p):
            return name[len(p):]
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", required=True,
                    choices=["cardinal", "relative", "topological"])
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out")
    args = ap.parse_args()

    rel = args.relation
    data = REPO / "data" / rel
    pools = json.loads(
        (REPO / "data_generation" / f"paraphrases_{rel}.json").read_text())
    rows = list(csv.DictReader((data / "corpus.csv").open(newline="",
                                                          encoding="utf-8")))
    fields = list(rows[0])

    hop = [r for r in rows if r["ambiguity_level"].strip() == HOP]
    flat = [r for r in rows if r["ambiguity_level"].strip() != HOP]
    rng = random.Random(args.seed)

    # Spread each predicate's rows evenly over the five levels.
    by_pred: dict[str, list[dict]] = defaultdict(list)
    for r in flat:
        by_pred[r["relation_label"]].append(r)

    n_eval = EVAL_PER_CELL[rel]

    def held_out(n_rows: int) -> int:
        """How many of a cell's rows the split builder will reserve.

        It must be the same rule on both sides. The builder falls back to half
        a cell when the cell is smaller than the usual reservation, and using
        the flat number here meant every row of a three-row equals cell was
        written from the evaluation pool while two of them ended up in
        training, carrying held-out wordings with them.
        """
        return n_eval if n_rows > n_eval else max(1, n_rows // 2)

    out, missing = [], defaultdict(int)
    # Two different templates can render to the same sentence once the places
    # are filled in, which would put one wording on both sides of the split
    # however carefully the pools were separated. Rendered evaluation wordings
    # are remembered so training rows can avoid reproducing them.
    eval_rendered: dict[str, set[str]] = defaultdict(set)
    for pred, items in sorted(by_pred.items()):
        rng.shuffle(items)
        cells: dict[int, list[dict]] = defaultdict(list)
        for i, r in enumerate(items):
            cells[i % 5 + 1].append(r)
        for lvl in range(1, 6):
            key = f"{pred}|{lvl}"
            tr_pool, ev_pool = pools["train"].get(key, []), pools["eval"].get(key, [])
            if not tr_pool or not ev_pool:
                missing[key] += len(cells[lvl])
                continue
            group = cells[lvl]
            n_here = held_out(len(group))
            # eval rows first, so the split builder's per-cell reservation
            # takes exactly the rows written from the evaluation pool
            for i, r in enumerate(group):
                is_eval = i < n_here
                pool = ev_pool if is_eval else tr_pool
                for _ in range(12):
                    tpl = pool[rng.randrange(len(pool))]
                    text = (tpl.replace("{A}", short(r["source_entity"]))
                               .replace("{B}", short(r["target_entity"])))
                    if "{V}" in text:
                        text = text.replace("{V}",
                                            short(r.get("observer_entity", "")))
                    if is_eval or text not in eval_rendered[pred]:
                        break
                if is_eval:
                    eval_rendered[pred].add(text)
                r = dict(r)
                r["corpus"] = text
                r["ambiguity_level"] = f"Level {lvl}"
                out.append(r)

    # Level 6 keeps its multi-hop phrasing, but that phrasing needs splitting
    # too: it came from a fixed bank applied to every row, so the same wording
    # sat on both sides of the split. Sorting rows by which bank they happened
    # to use is not enough -- the surplus has to be re-rendered, or it lands in
    # the wrong half.
    def to_shape(r):
        """The row's wording with its own places replaced by named slots."""
        t = r["corpus"]
        for col, slot in (("source_entity", "{A}"), ("target_entity", "{B}"),
                          ("via_entity", "{C}"), ("observer_entity", "{V}")):
            v = (r.get(col) or "").strip()
            if v:
                t = t.replace(short(v), slot)
        return t

    def render(shape, r):
        t = shape
        for col, slot in (("source_entity", "{A}"), ("target_entity", "{B}"),
                          ("via_entity", "{C}"), ("observer_entity", "{V}")):
            v = (r.get(col) or "").strip()
            if v:
                t = t.replace(slot, short(v))
        return t

    hop_by_pred = defaultdict(list)
    for r in hop:
        hop_by_pred[r["relation_label"]].append(r)

    hop_out = []
    for pred, items in sorted(hop_by_pred.items()):
        shapes = sorted({to_shape(r) for r in items})
        rng.shuffle(shapes)
        cut = max(1, min(len(shapes) - 1, round(len(shapes) * 0.6)))
        train_shapes, eval_shapes = shapes[:cut], shapes[cut:]
        rng.shuffle(items)
        n_here = held_out(len(items))
        for i, r in enumerate(items):
            pool = eval_shapes if i < n_here else train_shapes
            r = dict(r)
            r["corpus"] = render(pool[rng.randrange(len(pool))], r)
            hop_out.append(r)
    out.extend(hop_out)
    if missing:
        print(f"  {sum(missing.values())} rows had no template pool:")
        for k, n in list(missing.items())[:6]:
            print(f"    {k}: {n} rows")

    dest = Path(args.out) if args.out else data / "corpus.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"  {len(out)} rows written to {dest}")
    print(f"  {len(flat)} given a paraphrase, {len(hop)} multi-hop rows kept as they were")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
