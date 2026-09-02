"""Offline truth check for cardinal rows.

The schema validator can tell you a row is well-formed; it cannot tell you the
row is FALSE. Geocoding every batch through Nominatim is slow and rate-limited,
and the OSM cache is currently known-wrong for a quarter of cardinal pairs, so
neither is a trustworthy referee right now.

This uses a fixed table of city centroids instead. It only covers well-known
cities, but that is exactly what these batches are made of, and it runs in a
second with no network.

    python3 data_generation/check_cardinal_truth.py new_cardinal.csv

For each row it computes the initial great-circle bearing from the TARGET to
the SOURCE (the row asserts "source is <label> of target") and compares the
resulting compass sector to the stated label.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# lat, lon  (east-positive)
COORDS = {
    "accra": (5.55, -0.20), "addis ababa": (9.03, 38.74), "algiers": (36.75, 3.06),
    "almaty": (43.24, 76.89), "anchorage": (61.22, -149.90), "asuncion": (-25.28, -57.63),
    "athens": (37.98, 23.73), "atlanta": (33.75, -84.39), "auckland": (-36.85, 174.76),
    "baghdad": (33.31, 44.36), "baku": (40.41, 49.87), "bangkok": (13.76, 100.50),
    "beijing": (39.90, 116.41), "beirut": (33.89, 35.50), "berlin": (52.52, 13.40),
    "bogota": (4.71, -74.07), "boston": (42.36, -71.06), "brisbane": (-27.47, 153.03),
    "buenos aires": (-34.60, -58.38), "cairo": (30.04, 31.24), "calgary": (51.05, -114.07),
    "cape town": (-33.92, 18.42), "caracas": (10.48, -66.90), "casablanca": (33.57, -7.59),
    "chennai": (13.08, 80.27), "chicago": (41.88, -87.63), "colombo": (6.93, 79.86),
    "dakar": (14.72, -17.47), "dallas": (32.78, -96.80), "damascus": (33.51, 36.29),
    "delhi": (28.70, 77.10), "denver": (39.74, -104.99), "dhaka": (23.81, 90.41),
    "dubai": (25.20, 55.27), "dublin": (53.35, -6.26), "edinburgh": (55.95, -3.19),
    "guayaquil": (-2.17, -79.92), "halifax": (44.65, -63.57), "hanoi": (21.03, 105.85),
    "havana": (23.11, -82.37), "ho chi minh city": (10.82, 106.63), "hong kong": (22.32, 114.17),
    "honolulu": (21.31, -157.86), "houston": (29.76, -95.37), "istanbul": (41.01, 28.98),
    "jerusalem": (31.77, 35.21), "johannesburg": (-26.20, 28.05), "kabul": (34.56, 69.21),
    "kampala": (0.35, 32.58), "karachi": (24.86, 67.01), "khartoum": (15.50, 32.56),
    "kingston": (17.97, -76.79), "kyiv": (50.45, 30.52), "la paz": (-16.50, -68.15),
    "lagos": (6.52, 3.38), "lima": (-12.05, -77.04), "lisbon": (38.72, -9.14),
    "london": (51.51, -0.13), "los angeles": (34.05, -118.24), "luanda": (-8.84, 13.23),
    "madrid": (40.42, -3.70), "manila": (14.60, 120.98), "maputo": (-25.97, 32.57),
    "melbourne": (-37.81, 144.96), "miami": (25.76, -80.19), "montevideo": (-34.90, -56.16),
    "montreal": (45.50, -73.57), "moscow": (55.76, 37.62), "nairobi": (-1.29, 36.82),
    "new york": (40.71, -74.01), "oslo": (59.91, 10.75), "panama city": (8.98, -79.52),
    "paris": (48.86, 2.35), "perth": (-31.95, 115.86), "phnom penh": (11.56, 104.92),
    "port moresby": (-9.44, 147.18), "pyongyang": (39.04, 125.76), "quito": (-0.18, -78.47),
    "reykjavik": (64.15, -21.94), "rio de janeiro": (-22.91, -43.17), "riyadh": (24.71, 46.68),
    "rome": (41.90, 12.50), "san francisco": (37.77, -122.42), "san juan": (18.47, -66.11),
    "santiago": (-33.45, -70.67), "sao paulo": (-23.55, -46.63), "seattle": (47.61, -122.33),
    "seoul": (37.57, 126.98), "shanghai": (31.23, 121.47), "singapore": (1.35, 103.82),
    "stockholm": (59.33, 18.07), "suva": (-18.14, 178.44), "sydney": (-33.87, 151.21),
    "taipei": (25.03, 121.57), "tbilisi": (41.72, 44.79), "tehran": (35.69, 51.39),
    "thimphu": (27.47, 89.64), "tokyo": (35.68, 139.65), "toronto": (43.65, -79.38),
    "tripoli": (32.89, 13.19), "tunis": (36.81, 10.18), "ulaanbaatar": (47.89, 106.91),
    "vancouver": (49.28, -123.12), "vienna": (48.21, 16.37), "vientiane": (17.97, 102.60),
    "warsaw": (52.23, 21.01), "yangon": (16.87, 96.20), "yerevan": (40.18, 44.51),
}

SECTORS = ["north_of", "northeast_of", "east_of", "southeast_of",
           "south_of", "southwest_of", "west_of", "northwest_of"]


def key(name: str) -> str:
    n = name.strip().lower()
    for p in ("city of ", "the city of "):
        if n.startswith(p):
            n = n[len(p):]
    return n.strip()


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial great-circle bearing from a to b, degrees clockwise from north."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlo = lo2 - lo1
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return math.degrees(math.atan2(y, x)) % 360.0


def separation(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle angular distance in degrees."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


def sector(deg: float) -> tuple[str, float]:
    """Compass sector plus distance in degrees to the nearest sector boundary."""
    idx = int((deg + 22.5) % 360 // 45)
    centre = idx * 45.0
    off = abs((deg - centre + 180) % 360 - 180)
    return SECTORS[idx], 22.5 - off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="flag rows whose bearing sits within this many degrees "
                         "of a sector boundary (default 5)")
    ap.add_argument("--far", type=float, default=140.0,
                    help="flag pairs separated by more than this many degrees, "
                         "where 'direction' stops being well defined (default 140)")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv_file).open(newline="", encoding="utf-8")))
    wrong, borderline, antipodal, unknown = [], [], [], []

    for i, r in enumerate(rows):
        ln = i + 2
        s, t = key(r["source_entity"]), key(r["target_entity"])
        lab = r["relation_label"].strip().lower()
        if s not in COORDS or t not in COORDS:
            unknown.append((ln, s if s not in COORDS else t))
            continue
        ps, pt = COORDS[s], COORDS[t]
        got, margin = sector(bearing(pt, ps))
        sep = separation(ps, pt)
        if got != lab:
            wrong.append((ln, r, lab, got, bearing(pt, ps)))
        elif sep > args.far:
            antipodal.append((ln, r, sep))
        elif margin < args.margin:
            borderline.append((ln, r, lab, margin))

    n = len(rows)
    print("=" * 78)
    print(f"  TRUTH CHECK  {Path(args.csv_file).name}   {n} rows")
    print("=" * 78)

    if unknown:
        print(f"\n  SKIPPED {len(unknown)} rows — city not in the coordinate table:")
        for ln, name in unknown[:15]:
            print(f"    line {ln}: {name}")

    if wrong:
        print(f"\n  FALSE — {len(wrong)} rows state the wrong direction:")
        for ln, r, lab, got, deg in wrong:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}")
            print(f"              says {lab}, actual bearing {deg:6.1f} deg = {got}")
    if antipodal:
        print(f"\n  UNSTABLE — {len(antipodal)} rows put the two places more than "
              f"{args.far:.0f} deg apart,")
        print("             where a single compass direction is not well defined:")
        for ln, r, sep in antipodal:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}"
                  f"   separation {sep:.0f} deg")
    if borderline:
        print(f"\n  BORDERLINE — {len(borderline)} rows sit within {args.margin:.0f} deg "
              f"of a sector boundary:")
        for ln, r, lab, m in borderline:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}"
                  f"   {lab}, {m:.1f} deg from the boundary")

    bad = len(wrong) + len(antipodal)
    print("\n" + "=" * 78)
    checked = n - len(unknown)
    print(f"  {checked - bad}/{checked} checked rows are sound   "
          f"({len(wrong)} false, {len(antipodal)} unstable, "
          f"{len(borderline)} borderline)")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
