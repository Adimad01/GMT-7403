"""Take a list of NAME | clause lines, keep what OpenStreetMap can confirm.

    python3 data_generation/ingest_places.py gemini_places.txt --kind topological
    python3 data_generation/ingest_places.py gemini_cities.txt  --kind cities

Every place is geocoded and checked three ways: it must resolve, it must
resolve to the right kind of object, and the result must actually bear the name
asked for. Nominatim matches fuzzily, so without the last check a search for
Sahara returns New York State and outranks the desert on importance.

Clauses are screened for the words that would hand over an answer -- parentage,
neighbours, direction, area -- and anything caught is reported rather than
silently kept.

What survives is appended to the catalogue and the clause bank. What does not
is written to a rejects file, which is the basis for the follow-up prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from osm_resolve import LookupFailed, resolve       # noqa: E402

GEOM = REPO / "data" / "topological" / "osm" / "geometry.json"

# Words that put the answer in the description instead of the geography.
# Each pattern has to catch a positional claim without catching ordinary prose.
# "neighbourhood" is not a claim about neighbours, and "scattered across the
# landscape" is not a claim about being across a border from something.
LEAKS = [
    (r"\b(north|south|east|west)(ern|ward|erly)?\s+(of|from|coast|shore|edge|"
     r"end|tip|bank|side)\b", "direction"),
    (r"\bin the (north|south|east|west)\b", "direction"),
    (r"\b(capital of|city in|town in|located in|situated in|lies in|sits in)\b",
     "parentage"),
    (r"\bborders?\s+(on|with|the)\b|\bbordering\b|\badjacent to\b|"
     r"\bneighbou?ring\b|\bacross the (border|strait|channel|frontier|bay|"
     r"gulf|sound)\b", "neighbours"),
    (r"\bsquare (kilometres|kilometers|miles)\b|\b\d+[,\d]*\s*(km2|sq km)\b",
     "size"),
    (r"\b(latitude|longitude|equator|the pole|arctic circle|tropic of)\b",
     "coordinates"),
]

# Order matters. "Crater Lake National Park" contains the word Lake, and asking
# OSM for a lake by that name rejects the nature reserve it actually is; the
# same happened to Salt Lake County and Glacier Bay National Park. The most
# specific kind has to be tested first.
KIND_GUESS = [
    (re.compile(r"National Park|National Forest|Nature Reserve|"
                r"Conservation Area|Protected Area|National Monument", re.I), "park"),
    (re.compile(r"County$|County,|Parish$|Parish,|Prefecture|Province|"
                r"^Canton of |^State of |^Province of |Territory$|"
                r"^Republic|^Kingdom|^Federal Republic", re.I), "admin"),
    (re.compile(r"\b(River|Creek|Canal)\b", re.I), "river"),
    (re.compile(r"\b(Lake|Loch|Lough|Reservoir)\b", re.I), "lake"),
    (re.compile(r"\b(Sea|Gulf|Bay|Strait|Ocean|Sound)\b", re.I), "sea"),
    (re.compile(r"\b(Island|Islands|Isle|Archipelago|Peninsula)\b", re.I), "island"),
    (re.compile(r"\b(Desert|Forest|Mountains|Range|Massif|Alps)\b", re.I), "physical"),
]

# Anything unmatched could be an island, a region or a natural feature, so the
# fallback accepts all of them rather than insisting on an administrative
# boundary -- which is what discarded Sumatra, Sulawesi, Tahiti and Hispaniola.
BROAD = "broad"

def guess_kind(name: str, default: str) -> str:
    for pat, k in KIND_GUESS:
        if pat.search(name):
            return k
    return default


def screen(clause: str) -> list[str]:
    return [why for pat, why in LEAKS if re.search(pat, clause, re.I)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--kind", choices=["topological", "cities"], required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="screen the clauses without geocoding")
    ap.add_argument("--coords-out",
                    help="cities: write name,lat,lon here for the coordinate table")
    args = ap.parse_args()

    entries = []
    for raw in Path(args.file).read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if not line or "|" not in line:
            continue
        name, clause = (p.strip() for p in line.split("|", 1))
        if name and clause:
            entries.append((name, clause))
    print(f"  {len(entries)} places parsed from {Path(args.file).name}")

    default_kind = "city" if args.kind == "cities" else BROAD
    leaky, kept, rejected = [], [], []

    for name, clause in entries:
        why = screen(clause)
        if why:
            leaky.append((name, clause, ",".join(sorted(set(why)))))
    print(f"  {len(leaky)} clauses mention something that gives an answer away")
    for n, c, w in leaky[:8]:
        print(f"    [{w}] {n}: {c[:60]}")
    if args.dry_run:
        return 0

    cache = json.loads(GEOM.read_text()) if GEOM.exists() else {}
    leaky_names = {n for n, _, _ in leaky}
    todo = [(n, c) for n, c in entries if n not in leaky_names]
    print(f"\n  geocoding {len(todo)} places (about {len(todo) * 3 / 60:.0f} min)")

    for i, (name, clause) in enumerate(todo, 1):
        if name in cache and cache[name]:
            kept.append((name, clause))
            continue
        rec = None
        for attempt, simp in enumerate((0.0005, 0.003, 0.02)):
            try:
                rec = resolve(name, simplify=simp, timeout=90 + 30 * attempt,
                              kind=guess_kind(name, default_kind))
                break
            except LookupFailed:
                time.sleep(3)
        # Cardinal and relative work from a single coordinate, so a city that
        # OSM holds only as a node is perfectly usable. Requiring a polygon
        # there was a topological rule applied where it does not belong, and it
        # threw away most of the African and Central Asian cities.
        ok_geom = (rec and rec.get("geojson") and (
            args.kind == "cities"
            or rec["geojson"].get("type") in ("Polygon", "MultiPolygon",
                                              "LineString", "MultiLineString")))
        if ok_geom:
            cache[name] = {"osm_type": rec.get("osm_type"), "osm_id": rec.get("osm_id"),
                           "class": rec.get("class"), "type": rec.get("type"),
                           "display_name": rec.get("display_name"),
                           "importance": rec.get("importance"),
                           "lat": rec.get("lat"), "lon": rec.get("lon"),
                           "geojson": rec["geojson"]}
            kept.append((name, clause))
        else:
            kind = (rec or {}).get("geojson", {}).get("type", "nothing")
            rejected.append((name, f"resolved to {kind}"))
        if i % 20 == 0:
            GEOM.write_text(json.dumps(cache))
            print(f"  [{i}/{len(todo)}] kept {len(kept)}, rejected {len(rejected)}",
                  flush=True)

    GEOM.write_text(json.dumps(cache))
    if args.kind == "cities" and args.coords_out:
        lines = []
        for name, _ in kept:
            rec = cache.get(name)
            if rec and rec.get("lat") is not None:
                lines.append(f"{name},{rec['lat']},{rec['lon']}")
        Path(args.coords_out).write_text("\n".join(lines), encoding="utf-8")
        print(f"  coordinates for {len(lines)} cities -> {args.coords_out}")
    out = REPO / "data_generation" / f"accepted_{args.kind}.txt"
    out.write_text("\n".join(f"{n} | {c}" for n, c in kept), encoding="utf-8")
    rej = REPO / "data_generation" / f"rejected_{args.kind}.txt"
    rej.write_text("\n".join(f"{n} | {w}" for n, w in rejected)
                   + "\n" + "\n".join(f"{n} | clause leaks {w}" for n, _, w in leaky),
                   encoding="utf-8")

    print(f"\n  kept     {len(kept)}")
    print(f"  rejected {len(rejected)} unresolvable + {len(leaky)} leaky clauses")
    print(f"  -> {out.name} and {rej.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
