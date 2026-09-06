"""Check paraphrase templates and split them so train and eval share none.

Each line is  PREDICATE | LEVEL | template.  A template is rejected when it
names its own predicate, when a placeholder is missing, or when it repeats
another. What survives is divided into two pools.

The split matters more than it looks. If a wording appears in both training and
evaluation, a fine-tuned model can learn that phrase maps to that predicate and
score well without reading anything, and few-shot demonstrations hand the
mapping over outright. Keeping the pools disjoint means the evaluation asks
whether the model generalises to wordings it has never seen, which is the claim
the experiment is meant to support.

    python3 data_generation/ingest_paraphrases.py gemini_cardinal.txt --relation cardinal
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LABELS = {
    "topological": ["contains", "within", "touches", "crosses", "overlaps",
                    "disjoint", "equals"],
    "cardinal": ["north_of", "south_of", "east_of", "west_of", "northeast_of",
                 "northwest_of", "southeast_of", "southwest_of"],
    "relative": ["left_of", "right_of", "in_front_of", "behind", "next_to"],
}

# Strings that make a template answerable by matching rather than reading.
BANNED = {
    "north_of": ["north of", "northward of", "north-of"],
    "south_of": ["south of", "southward of"],
    "east_of": ["east of", "eastward of"],
    "west_of": ["west of", "westward of"],
    "northeast_of": ["northeast of", "north-east of"],
    "northwest_of": ["northwest of", "north-west of"],
    "southeast_of": ["southeast of", "south-east of"],
    "southwest_of": ["southwest of", "south-west of"],
    "contains": ["contains", "contain "],
    "within": ["within", "inside of"],
    "touches": ["touches", "touch "],
    "crosses": ["crosses", "cross "],
    "overlaps": ["overlaps", "overlap "],
    "disjoint": ["disjoint"],
    "equals": ["equals", "equal to"],
    "left_of": ["left of", "to the left", "left-hand"],
    "right_of": ["right of", "to the right", "right-hand"],
    "in_front_of": ["in front of", "in-front-of"],
    "behind": ["behind"],
    "next_to": ["next to", "adjacent to"],
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w{}\s]", "", t.lower())).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--relation", required=True, choices=list(LABELS))
    ap.add_argument("--eval-share", type=float, default=0.4,
                    help="fraction of each cell reserved for evaluation")
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    labels = set(LABELS[args.relation])
    need_v = args.relation == "relative"

    kept: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[str] = set()
    reasons = Counter()
    examples: dict[str, str] = {}

    for raw in Path(args.file).read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if line.count("|") < 2:
            continue
        pred, lvl, text = (p.strip() for p in line.split("|", 2))
        pred = pred.lower()

        def drop(why):
            reasons[why] += 1
            examples.setdefault(why, f"{pred} L{lvl}: {text[:58]}")

        if pred not in labels:
            drop(f"predicate not one of the {args.relation} labels"); continue
        if not lvl.isdigit() or not 1 <= int(lvl) <= 5:
            drop("level is not 1-5"); continue
        # Whole words only. A plain substring test rejects "slices across" for
        # the predicate crosses, because "across" contains "cross ".
        low = text.lower()
        if any(re.search(r"\b" + re.escape(b.strip()) + r"\b", low)
               for b in BANNED.get(pred, [])):
            drop("names its own predicate"); continue
        if "{A}" not in text or "{B}" not in text:
            drop("missing {A} or {B}"); continue
        if need_v and "{V}" not in text:
            drop("relative template has no observer {V}"); continue
        if not need_v and "{V}" in text:
            drop("uses {V} where there is no observer"); continue
        if re.search(r"[A-Z][a-z]{3,}", re.sub(r"\{[AB V]\}", "", text)[1:]):
            # a capitalised word mid-sentence is usually a place name
            pass
        key = norm(text)
        if key in seen:
            drop("duplicate of another template"); continue
        seen.add(key)
        kept[(pred, int(lvl))].append(text)

    total = sum(len(v) for v in kept.values())
    print(f"  {total} templates accepted, {sum(reasons.values())} rejected")
    for why, n in reasons.most_common():
        print(f"    {n:>4}  {why}")
        print(f"          e.g. {examples[why]}")

    print(f"\n  per cell (need at least 6 to split usefully):")
    print("  " + "predicate".ljust(14) + "".join(f"L{i}".rjust(6) for i in range(1, 6)))
    short = []
    for lab in LABELS[args.relation]:
        row = [len(kept.get((lab, i), [])) for i in range(1, 6)]
        print("  " + lab.ljust(14) + "".join(str(n).rjust(6) for n in row))
        for i, n in enumerate(row, 1):
            if n < 6:
                short.append((lab, i, n))

    rng = random.Random(args.seed)
    pools = {"train": {}, "eval": {}}
    for (lab, lvl), items in kept.items():
        items = sorted(items)
        rng.shuffle(items)
        cut = max(1, round(len(items) * (1 - args.eval_share)))
        pools["train"][f"{lab}|{lvl}"] = items[:cut]
        pools["eval"][f"{lab}|{lvl}"] = items[cut:]

    out = REPO / "data_generation" / f"paraphrases_{args.relation}.json"
    out.write_text(json.dumps(pools, indent=1), encoding="utf-8")
    n_tr = sum(len(v) for v in pools["train"].values())
    n_ev = sum(len(v) for v in pools["eval"].values())
    print(f"\n  split into {n_tr} training and {n_ev} evaluation templates,")
    print(f"  with no template in both -> {out.name}")
    if short:
        print(f"\n  {len(short)} cell(s) below 6 templates; I will ask for more:")
        for lab, lvl, n in short[:12]:
            print(f"    {lab} level {lvl}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
