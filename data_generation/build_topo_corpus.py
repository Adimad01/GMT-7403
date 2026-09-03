"""Build the topological corpus, re-deriving every label from geometry.

Run with the geometry virtualenv:

    <venv>/bin/python data_generation/build_topo_corpus.py --out data/topological/corpus.csv

Nothing is trusted from the selection file. Each row's relation is recomputed
from the polygons, and multi-hop rows have both links recomputed as well, since
a chain whose steps do not hold composes to nothing.

Descriptions pair two identity clauses and stop. Neither clause may mention the
other place, so the relation has to come from knowing the geography rather than
from decoding the sentence -- the failure that left the old corpus answerable
without any knowledge at all.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_topo_relations import difficulty, load, relate       # noqa: E402
from topo_identity import IDENTITY                                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PICKS = REPO / "data_generation" / "topo_picks.json"
HEADER = ["source_entity", "source_geometry", "target_entity", "target_geometry",
          "corpus", "via_entity", "relation_type", "relation_label",
          "explanation", "ambiguity_level"]

JOINERS = [
    "{a}. {b}.",
    "{a}, and separately {b}.",
    "{a}; {b}.",
    "It is worth recalling that {a}. Meanwhile {b}.",
    "{a}. As a distinct matter, {b}.",
    "Historically {a}, while {b}.",
    "{a} — and quite apart from that, {b}.",
    "Two facts stand on their own: {a}, and {b}.",
]

HOP_TEXT = {
    "contains": "{a} takes in the whole of {c}, and {c} in turn takes in the whole of {b}.",
    "within": "{a} sits entirely inside {c}, and {c} sits entirely inside {b}.",
    "disjoint": "{a} sits entirely inside {c}, and {c} shares no ground at all with {b}.",
}


def clause(name: str) -> str | None:
    c = IDENTITY.get(name)
    return c[0].upper() + c[1:] if c else None


def geom_type(g) -> str:
    return "Line" if g.geom_type in ("LineString", "MultiLineString") else "Polygon"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    geoms, _ = load()
    picks = json.loads(PICKS.read_text())
    rows, failures, seen_text = [], [], set()

    for i, p in enumerate(picks):
        a, b, lab, lvl, via = (p["subject"], p["object"], p["label"],
                               p["level"], p.get("via"))
        probs = []
        for n in (a, b) + ((via,) if via else ()):
            if n not in geoms:
                probs.append(f"no geometry for {n}")
            elif not clause(n):
                probs.append(f"no identity clause for {n}")
        if probs:
            failures.append((lab, lvl, a, b, probs))
            continue

        got, info = relate(geoms[a][0], geoms[b][0])
        if got != lab:
            probs.append(f"geometry gives {got}, not {lab}")
        elif lvl != 6 and difficulty(lab, info) != lvl:
            probs.append(f"measurements put this at Level {difficulty(lab, info)}")

        if lvl == 6:
            first = "contains" if lab in ("contains", "disjoint") else "within"
            g1, _ = relate(geoms[a][0], geoms[via][0])
            g2, _ = relate(geoms[via][0], geoms[b][0])
            want1 = "contains" if lab == "contains" else "within"
            want2 = "contains" if lab == "contains" else ("within" if lab == "within" else "disjoint")
            if g1 != want1:
                probs.append(f"first link is {g1}, not {want1}")
            if g2 != want2:
                probs.append(f"second link is {g2}, not {want2}")
        if probs:
            failures.append((lab, lvl, a, b, probs))
            continue

        if lvl == 6:
            text = HOP_TEXT[lab].format(a=a, b=b, c=via)
        else:
            text = JOINERS[i % len(JOINERS)].format(a=clause(a), b=clause(b))
        if text in seen_text:
            failures.append((lab, lvl, a, b, ["duplicate description"]))
            continue
        seen_text.add(text)

        bits = ", ".join(f"{k}={v:g}" for k, v in info.items()
                         if isinstance(v, (int, float)))
        expl = (f"Computed from OpenStreetMap geometry: {a} {lab} {b} ({bits})."
                if lvl != 6 else
                f"Composed through {via}: the relation holds on both links, so it "
                f"holds end to end ({bits}).")

        rows.append({
            "source_entity": a, "source_geometry": geom_type(geoms[a][0]),
            "target_entity": b, "target_geometry": geom_type(geoms[b][0]),
            "corpus": text, "via_entity": via or "",
            "relation_type": "topological", "relation_label": lab,
            "explanation": expl, "ambiguity_level": f"Level {lvl}",
        })

    if failures:
        print(f"  {len(failures)} item(s) failed verification:\n")
        for lab, lvl, a, b, probs in failures[:20]:
            print(f"    L{lvl} {lab:<10} {a} | {b}")
            for pr in probs:
                print(f"         - {pr}")
        print()
    print(f"  {len(rows)}/{len(picks)} rows verified")

    if args.out and not failures:
        dest = Path(args.out)
        with dest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {dest}")
    elif args.out:
        print("  nothing written — fix the failures first")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
