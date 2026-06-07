"""
analyze_osm_reliability.py
================================================================================
Audit the reliability of the OSM / Nominatim evidence layer used across all
CoT / ToT / GoT experiments.

Answers three questions:
  1. Retrieval rate  — how often does Nominatim return a result at all?
  2. Data completeness — of returned results, how many carry all required fields?
  3. Semantic adequacy — are the returned entities the right kind (boundary /
     place / waterway …) and do their bounding boxes have meaningful extent?

Failure cases are classified into actionable categories with repair notes.

Usage:
    python analyze_osm_reliability.py
    python analyze_osm_reliability.py --cache results/osm_cache.json --out results/osm_reliability_report.txt
"""

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_CACHE = "results/osm_cache.json"
DEFAULT_OUT   = "results/osm_reliability_report.txt"

EXPECTED_FIELDS = ["lat", "lon", "boundingbox", "osm_type", "class", "type", "hierarchy"]

# OSM classes that represent well-defined geographic features (reliable for topology)
RELIABLE_CLASSES = {"boundary", "waterway", "natural", "highway", "place", "water", "leisure", "landuse"}

# A bbox smaller than this (km²) may be a point/node — unreliable for polygon-based topology
BBOX_AREA_MIN_KM2 = 0.01


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def bbox_area_km2(bb: list) -> float:
    """Approximate area of an OSM bounding box in km²."""
    try:
        s, n, w, e = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        lat_km = (n - s) * 111.0
        lon_km = (e - w) * 111.0 * math.cos(math.radians((s + n) / 2))
        return max(0.0, lat_km * lon_km)
    except (TypeError, ValueError, IndexError):
        return 0.0


def hierarchy_depth(h: dict) -> int:
    return len(h) if isinstance(h, dict) else 0


def classify_failure(key: str) -> str:
    """Assign a failure category to a None cache entry."""
    if re.search(r"[\x80-\xff]|‚Äì|â€|‚Äú", key):
        return "encoding_error"
    if re.search(r"Metropolitan Statistical Area|metropolitan area", key, re.I):
        return "composite_msa"
    if re.search(r"[A-Za-z]{3,}/[A-Za-z]{3,}", key):
        return "composite_slash"
    if re.search(
        r"(Highways?\s+\d|Routes?\s+\d|FM\s*\d|US\s*\d|State High)",
        key, re.I
    ) and (" and " in key.lower() or "," in key):
        return "multi_road"
    # Heuristic for plain typos / overly long descriptive strings
    if len(key) > 80:
        return "overly_long_or_descriptive"
    return "typo_or_not_found"


FAILURE_REPAIR = {
    "encoding_error":            "Fix Unicode corruption in source data (e.g. em-dash → '-').",
    "composite_msa":             "MSA strings are not Nominatim-searchable; split into component counties.",
    "composite_slash":           "Slash-joined city lists (A/B/C) are not valid queries; pick the primary entity.",
    "multi_road":                "Conjunctive road names ('Hwy 35 and 16') are not searchable; query each road separately.",
    "overly_long_or_descriptive":"Trim to the canonical place name before querying Nominatim.",
    "typo_or_not_found":         "Correct spelling or try a broader search term (city name without county).",
}

SEVERITY = {
    # How much does this failure type hurt topological reasoning?
    "encoding_error":            "HIGH   — entity completely missing from evidence",
    "composite_msa":             "MEDIUM — MSAs rarely appear as subjects/objects; low impact",
    "composite_slash":           "HIGH   — compound locations are semantically ambiguous anyway",
    "multi_road":                "HIGH   — missing road geometry blocks crosses/touches decisions",
    "overly_long_or_descriptive":"MEDIUM — descriptive strings seldom resolve to a single polygon",
    "typo_or_not_found":         "HIGH   — entity completely missing from evidence",
}


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def analyze(cache: dict) -> dict:
    total   = len(cache)
    none_entries  = {k: v for k, v in cache.items() if v is None}
    valid_entries = {k: v for k, v in cache.items() if v is not None}

    # --- 1. Retrieval rate -----------------------------------------------
    retrieval_rate = len(valid_entries) / total * 100 if total else 0.0

    # --- 2. Field completeness -------------------------------------------
    missing_fields: dict[str, list] = defaultdict(list)
    for k, v in valid_entries.items():
        for f in EXPECTED_FIELDS:
            if not v.get(f):
                missing_fields[f].append(k)

    fully_complete = sum(
        1 for v in valid_entries.values()
        if all(v.get(f) for f in EXPECTED_FIELDS)
    )
    completeness_rate = fully_complete / len(valid_entries) * 100 if valid_entries else 0.0

    # --- 3. Semantic adequacy --------------------------------------------
    class_counts = Counter(v.get("class", "unknown") for v in valid_entries.values())
    osm_type_counts = Counter(v.get("osm_type", "?") for v in valid_entries.values())

    reliable_class_count = sum(
        1 for v in valid_entries.values() if v.get("class") in RELIABLE_CLASSES
    )
    semantic_rate = reliable_class_count / len(valid_entries) * 100 if valid_entries else 0.0

    # Bbox area distribution (percentiles)
    areas = sorted(
        bbox_area_km2(v["boundingbox"])
        for v in valid_entries.values()
        if v.get("boundingbox")
    )
    point_like = sum(1 for a in areas if a < BBOX_AREA_MIN_KM2)

    def pct(lst, p):
        if not lst:
            return 0.0
        idx = min(int(p / 100 * len(lst)), len(lst) - 1)
        return lst[idx]

    bbox_stats = {p: pct(areas, p) for p in [0, 5, 25, 50, 75, 95, 100]}

    # Hierarchy depth distribution
    depths = Counter(hierarchy_depth(v.get("hierarchy", {})) for v in valid_entries.values())

    # Low-quality flags (non-blocking but worth noting)
    low_hierarchy = [k for k, v in valid_entries.items() if hierarchy_depth(v.get("hierarchy", {})) < 4]
    unreliable_class = [k for k, v in valid_entries.items() if v.get("class") not in RELIABLE_CLASSES]

    # --- 4. Failure classification ----------------------------------------
    failure_cats: dict[str, list] = defaultdict(list)
    for k in none_entries:
        failure_cats[classify_failure(k)].append(k)

    return {
        "total":                total,
        "valid_count":          len(valid_entries),
        "none_count":           len(none_entries),
        "retrieval_rate":       retrieval_rate,
        "fully_complete":       fully_complete,
        "completeness_rate":    completeness_rate,
        "missing_fields":       dict(missing_fields),
        "class_counts":         dict(class_counts),
        "osm_type_counts":      dict(osm_type_counts),
        "reliable_class_count": reliable_class_count,
        "semantic_rate":        semantic_rate,
        "bbox_stats_km2":       bbox_stats,
        "point_like_count":     point_like,
        "depth_distribution":   dict(depths),
        "low_hierarchy_entries":low_hierarchy,
        "unreliable_class_entries": unreliable_class,
        "failure_categories":   dict(failure_cats),
        "none_entries":         list(none_entries.keys()),
    }


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def _bar(value: float, total: float = 100.0, width: int = 30, fill: str = "█") -> str:
    filled = int(round(value / total * width))
    return fill * filled + "░" * (width - filled)


def generate_report(res: dict, cache_path: str) -> str:
    lines = []
    W = 80

    def h1(title):
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def h2(title):
        lines.append("")
        lines.append(f"  ── {title} " + "─" * max(0, W - len(title) - 6))

    def row(label, value, note=""):
        pad = 35
        note_str = f"  ← {note}" if note else ""
        lines.append(f"  {label:<{pad}}{value}{note_str}")

    # Header
    h1("OSM / Nominatim Evidence Reliability Report")
    lines.append(f"  Cache  : {cache_path}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── 1. Retrieval Rate ──────────────────────────────────────────────────
    h2("1. Retrieval Rate")
    row("Total cache entries",    f"{res['total']:>6,}")
    row("Returned a result",      f"{res['valid_count']:>6,}  {_bar(res['retrieval_rate'])}")
    row("Returned None (failed)", f"{res['none_count']:>6,}")
    row("Retrieval rate",         f"{res['retrieval_rate']:>6.2f}%",
        "fraction of queries answered by Nominatim")

    # ── 2. Data Completeness ───────────────────────────────────────────────
    h2("2. Data Completeness  (of returned results)")
    row("All required fields present", f"{res['fully_complete']:>6,}  {_bar(res['completeness_rate'])}")
    row("Completeness rate",           f"{res['completeness_rate']:>6.2f}%",
        "lat + lon + bbox + osm_type + class + type + hierarchy")
    lines.append("")
    if res["missing_fields"]:
        lines.append("  Fields with gaps:")
        for field, keys in res["missing_fields"].items():
            lines.append(f"    • {field:<20} missing in {len(keys)} entries")
    else:
        lines.append("  All returned entries carry every required field.  ✅")

    # ── 3. Semantic Adequacy ───────────────────────────────────────────────
    h2("3. Semantic Adequacy")

    lines.append("  OSM class distribution:")
    for cls, cnt in sorted(res["class_counts"].items(), key=lambda x: -x[1]):
        bar = _bar(cnt, res["valid_count"])
        reliable = "✅" if cls in RELIABLE_CLASSES else "⚠️ "
        lines.append(f"    {reliable} {cls:<22} {cnt:>5}  {bar}")

    lines.append("")
    row("Reliable-class entries",  f"{res['reliable_class_count']:>6,}  {_bar(res['semantic_rate'])}")
    row("Semantic adequacy rate",  f"{res['semantic_rate']:>6.2f}%",
        "boundary / waterway / natural / highway / place / …")
    lines.append("")

    lines.append("  OSM object type distribution:")
    for otype, cnt in sorted(res["osm_type_counts"].items(), key=lambda x: -x[1]):
        note = {
            "relation": "(polygon / multipolygon — ideal for topology)",
            "way":      "(single closed way — acceptable)",
            "node":     "(point — no polygon geometry ⚠️)",
        }.get(otype, "")
        lines.append(f"    {otype:<12} {cnt:>5}   {note}")

    lines.append("")
    row("Point-like entries (bbox < 0.01 km²)",
        f"{res['point_like_count']:>6,}",
        "no usable polygon extent for bbox analysis")

    lines.append("")
    lines.append("  Administrative hierarchy depth (# of fields):")
    for depth in sorted(res["depth_distribution"]):
        cnt = res["depth_distribution"][depth]
        bar = _bar(cnt, res["valid_count"], width=20)
        quality = "shallow ⚠️" if depth < 4 else ("normal" if depth < 9 else "deep")
        lines.append(f"    depth {depth:>2}: {cnt:>5}  {bar}  {quality}")

    if res["low_hierarchy_entries"]:
        lines.append(f"\n  Entries with shallow hierarchy (depth < 4) — {len(res['low_hierarchy_entries'])} total:")
        for k in res["low_hierarchy_entries"][:10]:
            lines.append(f"    • {k}")
        if len(res["low_hierarchy_entries"]) > 10:
            lines.append(f"    … and {len(res['low_hierarchy_entries']) - 10} more")

    if res["unreliable_class_entries"]:
        lines.append(f"\n  Entries with non-standard OSM class — {len(res['unreliable_class_entries'])} total:")
        for k in res["unreliable_class_entries"][:10]:
            lines.append(f"    • {k}")
        if len(res["unreliable_class_entries"]) > 10:
            lines.append(f"    … and {len(res['unreliable_class_entries']) - 10} more")

    # ── 4. Bounding Box Quality ────────────────────────────────────────────
    h2("4. Bounding Box Quality")
    lines.append("  Area percentiles (km²):")
    for p, val in res["bbox_stats_km2"].items():
        lines.append(f"    p{p:>3}: {val:>15,.1f} km²")

    lines.append("")
    lines.append("  Interpretation guide:")
    lines.append("    p0  – p5  : points / very small features (nodes, buildings)")
    lines.append("    p25 – p50 : city districts, small municipalities")
    lines.append("    p50 – p75 : cities, large parks, waterways")
    lines.append("    p75 – p95 : counties, regions, large rivers")
    lines.append("    p95 – p100: states, countries, national rivers")

    # ── 5. Failure Analysis ────────────────────────────────────────────────
    h2("5. Failure Analysis  (None entries)")
    if not res["none_entries"]:
        lines.append("  No failures — all queries returned a result.  ✅")
    else:
        lines.append(f"  Total failures : {res['none_count']} / {res['total']}  ({res['none_count']/res['total']*100:.1f}%)")
        lines.append("")
        for cat, keys in sorted(res["failure_categories"].items(), key=lambda x: -len(x[1])):
            lines.append(f"  [{cat}]  ({len(keys)} entries)")
            lines.append(f"    Severity : {SEVERITY.get(cat, 'UNKNOWN')}")
            lines.append(f"    Repair   : {FAILURE_REPAIR.get(cat, 'N/A')}")
            lines.append("    Entries  :")
            for k in keys:
                lines.append(f"      • {k}")
            lines.append("")

    # ── 6. Summary ────────────────────────────────────────────────────────
    h2("6. Summary & Recommendations")

    # Overall score: average of retrieval, completeness, semantic rates
    overall = (res["retrieval_rate"] + res["completeness_rate"] + res["semantic_rate"]) / 3
    grade = "EXCELLENT" if overall >= 92 else ("GOOD" if overall >= 80 else ("FAIR" if overall >= 65 else "POOR"))
    lines.append(f"  Overall OSM reliability score : {overall:.1f}%  [{grade}]")
    lines.append("")

    recs = []

    if res["none_count"] > 0:
        recs.append(
            f"  [{res['none_count']} failed queries]  Consider pre-cleaning entity names before "
            "querying: fix encoding errors, strip parenthesised county lists from MSA names, "
            "split conjunctive road strings, and correct known typos."
        )

    node_count = res["osm_type_counts"].get("node", 0)
    if node_count > 0:
        recs.append(
            f"  [{node_count} node-type results]  Point results carry no polygon geometry. "
            "For topological reasoning, prefer querying with 'featuretype=settlement' or "
            "restrict osm_type to 'relation'/'way' in the Nominatim search."
        )

    if res["point_like_count"] > 0:
        recs.append(
            f"  [{res['point_like_count']} point-like bboxes]  Entries with bbox < 0.01 km² cannot "
            "support meaningful bounding-box analysis. Flag these in prompts as "
            "'No polygon geometry — treat bbox analysis as unreliable'."
        )

    unreliable_cls = len(res.get("unreliable_class_entries", []))
    if unreliable_cls > 0:
        recs.append(
            f"  [{unreliable_cls} non-standard OSM classes]  Classes like 'amenity', 'tourism', "
            "'building' describe point/area POIs rather than administrative/geographic regions. "
            "These are valid in context (e.g. park within city) but should not be used as "
            "containers for large-scale topological predicates."
        )

    if not recs:
        recs.append("  OSM evidence quality is sufficient across all dimensions — no action required.")

    lines += recs

    lines.append("")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze OSM cache reliability")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help=f"Path to osm_cache.json (default: {DEFAULT_CACHE})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output report path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f"[ERROR] Cache file not found: {args.cache}")
        return

    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print(f"[INFO] Loaded {len(cache)} cache entries from {args.cache}")
    results = analyze(cache)
    report  = generate_report(results, args.cache)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[INFO] Report saved to {args.out}")


if __name__ == "__main__":
    main()
