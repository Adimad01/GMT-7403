"""Build the cardinal corpus from selected pairs, verifying every claim.

Nothing is trusted from the selection file: each label is recomputed from real
coordinates, and multi-hop rows have both links recomputed too, since a chain
whose steps do not hold composes to nothing.

Descriptions pair two identity clauses and stop. Naming a city's country is
allowed -- knowing that Oslo is Norwegian and Rome Italian is the knowledge the
item tests -- but the bearing itself may never appear, in any form. That was
the failure that left the old corpus answerable at 97-100% without knowing
where anything was.

    python3 data_generation/build_cardinal_corpus.py --out data/cardinal/corpus.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import (COORDS, bearing, components_agree,  # noqa: E402
                                  key, reciprocal, sector, separation)
from city_identity import CITY                                        # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PICKS = REPO / "data_generation" / "cardinal_picks.json"
MARGIN, MAX_SEP, MIN_SEP = 8.0, 110.0, 8.0
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
    "{a}. Independently of that, {b}.",
    "Consider that {a}, and that {b}.",
]

VERB = {"north_of": "north", "south_of": "south", "east_of": "east",
        "west_of": "west", "northeast_of": "northeast",
        "northwest_of": "northwest", "southeast_of": "southeast",
        "southwest_of": "southwest"}

# Level 6 states both links. A paraphrase rather than the label keeps the row
# from being answerable by keyword match while leaving the chain to compose.
AXIS = {
    "north_of": ["{a} lies at a higher latitude than {c}, and {c} at a higher latitude than {b}.",
                 "{a} stands nearer the Arctic than {c}, and {c} in turn nearer it than {b}.",
                 "{a} is further up the meridian than {c}, and that city further up than {b}.",
                 "{a} sits closer to the top of the globe than {c}, and {c} closer than {b}.",
                 "{a} is set further from the equator on the same side than {c}, and {c} than {b}.",
                 "{a} reaches a greater latitude than {c}, and {c} a greater one than {b}.",
                 "{a} lies deeper into the northern half than {c}, and {c} deeper than {b}."],
    "south_of": ["{a} lies further from the North Pole than {c}, and {c} further from it than {b}.",
                 "{a} sits at a lower signed latitude than {c}, and {c} lower than {b}.",
                 "{a} is nearer the Antarctic than {c}, and {c} in turn nearer it than {b}.",
                 "{a} stands further down the meridian than {c}, and that city further down than {b}.",
                 "{a} lies deeper into the southern half than {c}, and {c} deeper than {b}.",
                 "{a} reaches a lesser latitude than {c}, and {c} a lesser one than {b}.",
                 "{a} sits closer to the base of the globe than {c}, and {c} closer than {b}."],
    "east_of": ["{a} has a more easterly longitude than {c}, and {c} a more easterly one than {b}.",
                "{a} lies further toward the dateline than {c}, and {c} further than {b}.",
                "{a} keeps a later solar time than {c}, and {c} in turn a later one than {b}.",
                "{a} sits at a greater longitude than {c}, and {c} a greater one than {b}.",
                "{a} is further round the globe in that sense than {c}, and {c} than {b}.",
                "{a} meets the day earlier than {c}, and {c} earlier than {b}.",
                "{a} stands further along the eastward count than {c}, and {c} than {b}."],
    "west_of": ["{a} has a more westerly longitude than {c}, and {c} a more westerly one than {b}.",
                "{a} lies further back along the meridians than {c}, and {c} further back than {b}.",
                "{a} keeps an earlier solar time than {c}, and that city an earlier one than {b}.",
                "{a} sits at a lesser longitude than {c}, and {c} a lesser one than {b}.",
                "{a} is further round the globe in that sense than {c}, and {c} than {b}.",
                "{a} meets the day later than {c}, and {c} later than {b}.",
                "{a} stands further along the westward count than {c}, and {c} than {b}."],
}
for _lab, _ns, _ew in (("northeast_of", "higher latitude", "more easterly longitude"),
                       ("northwest_of", "higher latitude", "more westerly longitude"),
                       ("southeast_of", "lower latitude", "more easterly longitude"),
                       ("southwest_of", "lower latitude", "more westerly longitude")):
    AXIS[_lab] = [
        f"{{a}} has both a {_ns} and a {_ew} than {{c}}, and {{c}} stands the same way to {{b}}.",
        f"{{a}} exceeds {{c}} on both counts, {_ns} and {_ew}, and {{c}} exceeds {{b}} on both.",
        f"{{a}} shows a {_ns} than {{c}} together with a {_ew}, and {{c}} shows the same against {{b}}.",
        f"{{a}} differs from {{c}} by a {_ns} and a {_ew}, and {{c}} differs from {{b}} the same way.",
        f"{{a}} sits at a {_ns} and a {_ew} relative to {{c}}, and {{c}} relative to {{b}}.",
        f"{{a}} carries a {_ns} over {{c}} and a {_ew} too, and {{c}} carries both over {{b}}.",
        f"{{a}} stands apart from {{c}} by a {_ns} and a {_ew}, and {{c}} from {{b}} likewise.",
    ]


def coord(name: str):
    k = key(name)
    if k not in COORDS:
        raise KeyError(name)
    return COORDS[k]


def check(a: str, b: str, label: str) -> list[str]:
    """Every reason the pair (a <label> b) is unusable."""
    try:
        pa, pb = coord(a), coord(b)
    except KeyError as e:
        return [f"no coordinates for {e.args[0]}"]
    problems = []
    deg = bearing(pb, pa)
    got, m = sector(deg)
    sep = separation(pa, pb)
    if got != label:
        problems.append(f"bearing {deg:.1f} is {got}, not {label}")
    elif m < MARGIN:
        problems.append(f"only {m:.1f} deg from a sector boundary")
    if sep > MAX_SEP:
        problems.append(f"{sep:.0f} deg apart, direction not well defined")
    if sep < MIN_SEP:
        problems.append(f"only {sep:.0f} deg apart, too close to be a clear item")
    if not reciprocal(pa, pb):
        problems.append("reverse bearing is not the opposite sector")
    if not components_agree(pa, pb, label):
        problems.append("cone label disagrees with the projection-based reading")
    return problems


def clause(city: str) -> str | None:
    c = CITY.get(key(city))
    return c[0].upper() + c[1:] if c else None


def fmt(n: str) -> str:
    return "City of " + " ".join(w.capitalize() for w in n.split())


def latlon(p) -> str:
    return (f"{abs(p[0]):.1f}{'N' if p[0] >= 0 else 'S'} "
            f"{abs(p[1]):.1f}{'E' if p[1] >= 0 else 'W'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    picks = json.loads(PICKS.read_text())
    rows, failures, seen = [], [], set()
    hop_n = {k: 0 for k in AXIS}
    flat_n = 0

    for p in picks:
        a, b, lab, lvl, via = (p["subject"], p["object"], p["label"],
                               p["level"], p.get("via"))
        probs = check(a, b, lab)
        for n in (a, b) + ((via,) if via else ()):
            if not clause(n):
                probs.append(f"no identity clause for {n}")
        if lvl == 6:
            for x, y, which in ((a, via, "first"), (via, b, "second")):
                for pr in check(x, y, lab):
                    probs.append(f"{which} hop ({x} -> {y}): {pr}")
        if probs:
            failures.append((lvl, lab, a, b, probs))
            continue

        if lvl == 6:
            i = hop_n[lab]
            hop_n[lab] += 1
            text = AXIS[lab][i % len(AXIS[lab])].format(
                a=fmt(a)[8:], b=fmt(b)[8:], c=fmt(via)[8:])
        else:
            text = JOINERS[flat_n % len(JOINERS)].format(a=clause(a), b=clause(b))
            flat_n += 1
        if text in seen:
            failures.append((lvl, lab, a, b, ["duplicate description"]))
            continue
        seen.add(text)

        pa, pb = coord(a), coord(b)
        deg = bearing(pb, pa)
        expl = ((f"Two {VERB[lab]} steps compose through {fmt(via)[8:]}. "
                 f"{fmt(a)[8:]} {latlon(pa)}, {fmt(via)[8:]} {latlon(coord(via))}, "
                 f"{fmt(b)[8:]} {latlon(pb)}.") if lvl == 6 else
                (f"{fmt(a)[8:]} {latlon(pa)}, {fmt(b)[8:]} {latlon(pb)}; the bearing "
                 f"from {fmt(b)[8:]} to {fmt(a)[8:]} is {deg:.0f} degrees, "
                 f"{sector(deg)[1]:.0f} degrees inside the {lab} sector."))

        rows.append({
            "source_entity": fmt(a), "source_geometry": "Polygon",
            "target_entity": fmt(b), "target_geometry": "Polygon",
            "corpus": text, "via_entity": fmt(via) if via else "",
            "relation_type": "cardinal_direction", "relation_label": lab,
            "explanation": expl, "ambiguity_level": f"Level {lvl}",
        })

    if failures:
        print(f"  {len(failures)} item(s) failed verification:\n")
        for lvl, lab, a, b, probs in failures[:15]:
            print(f"    L{lvl} {lab:<13} {a} -> {b}")
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
