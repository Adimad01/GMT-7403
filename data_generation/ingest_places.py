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
LEAKS = [
    (r"\b(north|south|east|west|northern|southern|eastern|western)\b", "direction"),
    (r"\b(capital of|city in|town in|located in|situated in|part of)\b", "parentage"),
    (r"\bborder\w*\b|\badjacent\b|\bneighbou?r\w*\b|\bacross the\b", "neighbours"),
    (r"\bsquare (kilometres|kilometers|miles)\b|\blargest\b|\bsmallest\b", "size"),
    (r"\b(latitude|longitude|equator|pole)\b", "coordinates"),
]

KIND_GUESS = [
    (re.compile(r"\b(River|Creek|Canal)\b", re.I), "river"),
    (re.compile(r"\b(Lake|Loch|Reservoir)\b", re.I), "lake"),
    (re.compile(r"\b(Sea|Gulf|Bay|Strait|Ocean|Sound)\b", re.I), "sea"),
    (re.compile(r"National Park|Nature Reserve|Protected Area", re.I), "park"),
    (re.compile(r"\b(Island|Isle|Archipelago|Peninsula)\b", re.I), "island"),
    (re.compile(r"\b(Desert|Forest|Mountains|Range|Massif)\b", re.I), "physical"),
    (re.compile(r"County|Parish|Province|District|Canton|Prefecture|"
                r"^State of |^Republic|^Kingdom", re.I), "admin"),
]


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

    default_kind = "city" if args.kind == "cities" else "admin"
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
        if rec and rec.get("geojson") and rec["geojson"].get("type") in (
                "Polygon", "MultiPolygon", "LineString", "MultiLineString"):
            cache[name] = {"osm_type": rec.get("osm_type"), "osm_id": rec.get("osm_id"),
                           "class": rec.get("class"), "type": rec.get("type"),
                           "display_name": rec.get("display_name"),
                           "importance": rec.get("importance"),
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
