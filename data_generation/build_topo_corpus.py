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
from compute_topo_relations import load, relate                  # noqa: E402
from topo_identity import IDENTITY                                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PICKS = REPO / "data_generation" / "topo_picks.json"
RELS = REPO / "data" / "topological" / "osm" / "relations.json"
HEADER = ["source_entity", "source_geometry", "target_entity", "target_geometry",
          "corpus", "via_entity", "relation_type", "relation_label",
          "explanation", "ambiguity_level"]

# Connectives are deliberately neutral. "and separately", "as a distinct
# matter" and "quite apart from that" all read as hints towards disjoint when
# the row happens to be a disjoint one, which is the leak this corpus exists to
# remove.
JOINERS = [
    "{a}. {b}.",
    "{a}, and {b}.",
    "{a}; {b}.",
    "It is worth recalling that {a}. Meanwhile {b}.",
    "{a}. On another note, {b}.",
    "Historically {a}, while {b}.",
    "{a} — and for its part, {b}.",
    "Two facts to hold in mind: {a}, and {b}.",
    "{a}. Elsewhere, {b}.",
    "Note that {a}, and that {b}.",
]

HOP_TEXT = {
    "contains": [
        "{a} takes in the whole of {c}, and {c} in turn takes in the whole of {b}.",
        "{a} covers all of {c}, and {c} covers all of {b}.",
        "Every part of {c} falls under {a}, and every part of {b} falls under {c}.",
        "{a} encompasses {c} completely, and {c} encompasses {b} completely.",
        "The whole of {c} lies under {a}, and the whole of {b} lies under {c}.",
        "{a} accounts for all of {c}, and {c} accounts for all of {b}.",
        "Nothing of {c} falls outside {a}, and nothing of {b} falls outside {c}.",
        "{a} spans the entirety of {c}, and {c} spans the entirety of {b}.",
    ],
    "within": [
        "{a} sits entirely inside {c}, and {c} sits entirely inside {b}.",
        "{a} falls wholly under {c}, and {c} falls wholly under {b}.",
        "No part of {a} lies outside {c}, and no part of {c} lies outside {b}.",
        "{a} is set completely in {c}, and {c} completely in {b}.",
        "The whole of {a} belongs to {c}, and the whole of {c} belongs to {b}.",
        "{a} lies fully in {c}, which in turn lies fully in {b}.",
        "{a} is held entirely by {c}, and {c} entirely by {b}.",
        "{a} rests wholly in {c}, and {c} rests wholly in {b}.",
    ],
    "disjoint": [
        "{a} sits entirely inside {c}, and {c} shares no ground at all with {b}.",
        "{a} falls wholly under {c}, and {c} has no land in common with {b}.",
        "No part of {a} lies outside {c}, and {c} meets {b} nowhere.",
        "{a} is set completely in {c}, and {c} and {b} share no territory.",
        "The whole of {a} belongs to {c}, and {c} holds nothing in common with {b}.",
        "{a} lies fully in {c}, and {c} and {b} have no ground between them.",
        "{a} rests wholly in {c}, and nothing of {c} coincides with {b}.",
        "{a} is held entirely by {c}, and {c} is separate from {b} altogether.",
    ],
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
    # Levels come from each label's own ranking, computed once in
    # compute_topo_relations. Recomputing them here with a second rule was how
    # the builder and the selector came to disagree.
    graded = {(r["subject"], r["object"]): (r["label"], r["level"])
              for r in json.loads(RELS.read_text())}
    rows, failures, seen_text = [], [], set()
    hop_n = {k: 0 for k in HOP_TEXT}

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
        elif lvl != 6:
            rec = graded.get((a, b))
            if rec is None:
                probs.append("pair is absent from the graded relation index")
            elif rec[1] != lvl:
                probs.append(f"the ranking places this at Level {rec[1]}")

        if lvl == 6:
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
            i = hop_n[lab]
            hop_n[lab] += 1
            text = HOP_TEXT[lab][i % len(HOP_TEXT[lab])].format(a=a, b=b, c=via)
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
