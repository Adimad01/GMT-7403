"""Surface city pairs that are geometrically usable for a given cardinal label.

Writing a row starts with picking two places. Picking them from memory is how
the last three batches ended up with bearings sitting on a sector boundary, so
this does the picking against real coordinates instead: it only ever offers
pairs whose bearing sits comfortably inside the sector, whose separation keeps
a compass direction meaningful, and which are not already used.

    python3 data_generation/find_cardinal_pairs.py --label northwest_of --top 40
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import (COORDS, bearing, components_agree,  # noqa: E402
                                  key, reciprocal, sector, separation)

REPO = Path(__file__).resolve().parents[1]

# Rough region tags, used only to favour pairs that span regions -- those are
# the ones a reader is most likely to get wrong, so they make better items.
REGION = {
    "europe": """oslo rome stockholm helsinki athens prague berlin madrid lisbon paris
        london dublin edinburgh vienna warsaw budapest bucharest sofia belgrade zagreb
        sarajevo tirana copenhagen amsterdam brussels zurich munich milan naples venice
        genoa turin porto valencia faro marseille lyon nice riga tallinn vilnius minsk
        kyiv moscow st petersburg bergen gothenburg malmo bratislava chisinau odesa
        birmingham cardiff reykjavik istanbul""",
    "africa": """cairo lagos accra nairobi khartoum addis ababa johannesburg cape town
        maputo luanda kinshasa dakar algiers tunis tripoli casablanca harare windhoek
        antananarivo dar es salaam mogadishu kampala""",
    "asia": """tokyo osaka sapporo seoul pyongyang beijing shanghai wuhan chengdu
        hong kong taipei manila jakarta surabaya bangkok hanoi phnom penh vientiane
        yangon singapore kuala lumpur delhi mumbai kolkata chennai pune karachi dhaka
        colombo kathmandu thimphu islamabad kabul tashkent almaty bishkek dushanbe
        ashgabat vladivostok ulaanbaatar""",
    "mideast": """tehran baghdad riyadh dubai abu dhabi muscat kuwait city doha amman
        beirut damascus jerusalem baku yerevan tbilisi ankara nicosia""",
    "namerica": """new york boston philadelphia washington miami atlanta chicago detroit
        cleveland minneapolis denver dallas houston san antonio new orleans phoenix
        las vegas salt lake city reno los angeles san diego san francisco seattle
        portland toronto montreal quebec city halifax ottawa winnipeg calgary edmonton
        vancouver anchorage honolulu mexico city tijuana havana kingston san juan
        panama city managua windsor""",
    "samerica": """lima bogota quito guayaquil caracas santiago valparaiso buenos aires
        montevideo asuncion la paz sao paulo rio de janeiro curitiba salvador recife
        fortaleza""",
    "oceania": """sydney melbourne brisbane perth auckland wellington suva port moresby""",
}
CITY_REGION = {c: r for r, blob in REGION.items() for c in blob.split("\n") for c in [c]}
CITY_REGION = {}
for _r, _blob in REGION.items():
    for _line in _blob.strip().split("\n"):
        _line = _line.strip()
        # names can contain spaces, so split on two-or-more spaces is unreliable;
        # match greedily against the coordinate table instead
        _toks = _line.split()
        i = 0
        while i < len(_toks):
            for span in (3, 2, 1):
                cand = " ".join(_toks[i:i + span])
                if cand in COORDS:
                    CITY_REGION[cand] = _r
                    i += span
                    break
            else:
                i += 1


def used_pairs() -> set[tuple[str, str]]:
    out = set()
    for rel_file in [REPO / "data" / "cardinal" / "corpus.csv"]:
        if not rel_file.exists():
            continue
        for r in csv.DictReader(rel_file.open(newline="", encoding="utf-8")):
            a, b = key(r["source_entity"]), key(r["target_entity"])
            out.add((a, b))
            out.add((b, a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--max-sep", type=float, default=110.0)
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--cross-region", action="store_true",
                    help="only pairs spanning two different regions")
    args = ap.parse_args()

    skip = used_pairs()
    names = sorted(COORDS)
    found = []
    for a in names:
        for b in names:
            if a == b or (a, b) in skip:
                continue
            pa, pb = COORDS[a], COORDS[b]
            sep = separation(pa, pb)
            if not (args.min_sep <= sep <= args.max_sep):
                continue
            got, m = sector(bearing(pb, pa))
            if got != args.label or m < args.margin:
                continue
            if not reciprocal(pa, pb):
                continue
            if not components_agree(pa, pb, args.label):
                continue
            ra, rb = CITY_REGION.get(a, "?"), CITY_REGION.get(b, "?")
            if args.cross_region and ra == rb:
                continue
            found.append((m, sep, a, b, ra, rb))

    found.sort(key=lambda t: -t[0])
    print(f"{len(found)} usable pairs for {args.label} "
          f"(margin >= {args.margin}, separation {args.min_sep}-{args.max_sep})")
    print(f"{'margin':>7} {'sep':>6}  pair")
    for m, sep, a, b, ra, rb in found[:args.top]:
        print(f"{m:7.1f} {sep:6.0f}  {a} {args.label} {b}   [{ra} -> {rb}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
