"""Check generated examples before they enter the dataset.

A language model asked for 150 rows will return 150 rows. Whether they are
balanced, unique, non-leaking, and geocodable is a separate question, and every
one of those has already gone wrong in this project at least once.

Run this on whatever the generator returns, fix what it flags, run it again.
Only merge when it is clean.

    python3 data_generation/validate_new_data.py new_relative.csv --relation relative
    python3 data_generation/validate_new_data.py new_relative.csv --relation relative --geocode

--geocode queries Nominatim (1 request/second, needs internet) and is the check
that matters most: roughly a third of the existing corpus fails it, and an
ungeocodable row is dead weight for every knowledge-graph experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LV = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6"]
HOP_LEVEL = "Level 6"          # multi-hop: relation inferred through via_entity
# Relations whose 2-hop composition is NOT logically forced. A touches C and
# C touches B implies nothing about A and B, so such a chain is unusable.
NON_COMPOSABLE = {"touches", "crosses", "overlaps"}
# Labels that admit a FORCED two-hop composition involving real spatial
# reasoning. Everything else has no determinate answer at Level 6, and 'equals'
# is reachable only by chaining synonyms, which is a naming trick.
HOP_LABELS = {
    "topological": {"contains", "within", "disjoint"},
    "cardinal": {"north_of", "south_of", "east_of", "west_of",
                 "northeast_of", "northwest_of", "southeast_of", "southwest_of"},
    "relative": {"left_of", "right_of", "in_front_of", "behind"},
}
STOPWORDS = {"city", "state", "republic", "federal", "kingdom", "commonwealth",
             "national", "park", "county", "borough", "province", "united",
             "states", "the", "of", "and"}


def content_words(text: str) -> set[str]:
    import re as _re
    return {w for w in _re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}

LABELS = {
    "topological": ["contains", "within", "touches", "crosses",
                    "disjoint", "overlaps", "equals"],
    "cardinal": ["north_of", "south_of", "east_of", "west_of",
                 "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
    "relative": ["left_of", "right_of", "in_front_of", "behind", "next_to"],
}
INVERSE = {
    "north_of": "south_of", "south_of": "north_of",
    "east_of": "west_of", "west_of": "east_of",
    "northeast_of": "southwest_of", "southwest_of": "northeast_of",
    "northwest_of": "southeast_of", "southeast_of": "northwest_of",
    "left_of": "right_of", "right_of": "left_of",
    "in_front_of": "behind", "behind": "in_front_of", "next_to": "next_to",
    "contains": "within", "within": "contains",
    "touches": "touches", "crosses": "crosses",
    "disjoint": "disjoint", "overlaps": "overlaps", "equals": "equals",
}
COLUMNS = ["source_entity", "source_geometry", "target_entity", "target_geometry",
           "corpus", "relation_type", "relation_label", "explanation",
           "ambiguity_level"]
OPTIONAL = ["via_entity"]        # required on Level 6, empty elsewhere

# Words that give the answer away if they appear in the description the model sees.
GIVEAWAY = {
    "north_of": ["north of", "northward of"], "south_of": ["south of"],
    "east_of": ["east of"], "west_of": ["west of"],
    "northeast_of": ["northeast of"], "northwest_of": ["northwest of"],
    "southeast_of": ["southeast of"], "southwest_of": ["southwest of"],
    "left_of": ["left of", "to the left"], "right_of": ["right of", "to the right"],
    "in_front_of": ["in front of"], "behind": ["behind"],
    "next_to": ["next to", "adjacent to"],
    "contains": ["contains"], "within": ["within", "inside of"],
    "touches": ["touches"], "crosses": ["crosses"],
    "disjoint": ["disjoint"], "overlaps": ["overlaps"], "equals": ["equals"],
}

# Constructions that state a RELATIVE bearing, which on cardinal Levels 1-5
# makes the item answerable without any geographic knowledge.
#
# A bare compass word is not enough to flag: "southern California", "South
# Korea" and "West Africa" are place descriptions, not relational claims. Only
# a compass word in relational context leaks the answer. Metaphors for a
# bearing (clock faces, map edges, the sun) are relational by nature and are
# flagged wherever they appear.
import re as _re

_COMPASS = r"(north|south|east|west|northeast|northwest|southeast|southwest)"
RELATIONAL_PATTERNS = [
    _re.compile(rf"\b{_COMPASS}(ern|erly|ward|wards)?\s+(of|from)\b", _re.I),
    _re.compile(rf"\bto the {_COMPASS}(ern)?\b", _re.I),
    _re.compile(rf"\b(further|farther|more|higher|lower)\s+{_COMPASS}", _re.I),
    _re.compile(rf"\b{_COMPASS}(ward|wards)\b", _re.I),
    _re.compile(rf"\blies?\s+{_COMPASS}\b", _re.I),
    _re.compile(rf"\bsits?\s+{_COMPASS}\b", _re.I),
]
# Always relational, whatever the surrounding words.
BEARING_METAPHORS = [
    "o'clock", "oclock",
    "top of the map", "top of the globe", "top of the world",
    "bottom of the map", "bottom of the globe",
    "left of the map", "right of the map",
    "left-hand edge", "right-hand edge", "left edge", "right edge",
    "upward", "downward", "leftward", "rightward",
    "up and to the", "down and to the", "diagonally up", "diagonally down",
    "upper left", "upper right", "lower left", "lower right",
    "closer to the top", "closer to the bottom", "further up", "further down",
    "sunrise", "sunset", "setting sun", "morning sun", "morning light",
    "greets the sun", "toward the dawn", "sunset horizon",
    "closer to the pole", "toward the arctic", "toward the antarctic",
    "nearer the equator", "closer to the equator", "equatorial line",
    "latitudinal grid", "longitudinal grid", "higher latitude", "lower latitude",
    "arctic pole", "icy top",
]


def bearing_cue(text: str) -> str | None:
    """Return the offending phrase, or None. Used only on cardinal Levels 1-5."""
    low = text.lower()
    for m in BEARING_METAPHORS:
        if m in low:
            return m
    for pat in RELATIONAL_PATTERNS:
        hit = pat.search(text)
        if hit:
            return hit.group(0)
    return None


_counts = Counter()


def ok(m):   _counts["pass"] += 1; print(f"  PASS  {m}")
def warn(m): _counts["warn"] += 1; print(f"  WARN  {m}")
def bad(m):  _counts["fail"] += 1; print(f"  FAIL  {m}")


def pair(r):
    return (r["source_entity"].strip().lower(), r["target_entity"].strip().lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--relation", required=True, choices=list(LABELS))
    ap.add_argument("--geocode", action="store_true",
                    help="verify every entity resolves in OpenStreetMap (slow, needs internet)")
    ap.add_argument("--expect-per-cell", type=int,
                    help="required rows per (label, level) cell")
    args = ap.parse_args()

    rel = args.relation
    labels = LABELS[rel]
    path = Path(args.csv_file)
    if not path.exists():
        print(f"not found: {path}")
        return 2

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 78)
    print(f"  VALIDATING {path.name}   relation={rel}   {len(rows)} rows")
    print("=" * 78)

    # --- schema -----------------------------------------------------------
    if not rows:
        bad("file is empty")
        return 1
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        bad(f"missing columns: {missing}")
        return 1
    ok(f"schema: all {len(COLUMNS)} columns present")

    # --- blanks -----------------------------------------------------------
    holes = defaultdict(int)
    for r in rows:
        for c in COLUMNS:                     # via_entity excluded: see the
            if not str(r.get(c, "")).strip():  # multi-hop check below
                holes[c] += 1
    if holes:
        bad("blank cells: " + ", ".join(f"{k}={v}" for k, v in holes.items()))
    else:
        ok("no blank cells")

    # --- vocabulary -------------------------------------------------------
    badlab = Counter(r["relation_label"].strip() for r in rows
                     if r["relation_label"].strip().lower() not in labels)
    if badlab:
        bad(f"unknown labels: {dict(badlab)}")
    else:
        ok(f"labels: all within the {len(labels)} allowed values")

    badlv = Counter(r["ambiguity_level"].strip() for r in rows
                    if r["ambiguity_level"].strip() not in LV)
    if badlv:
        bad(f"unknown ambiguity levels: {dict(badlv)}")
    else:
        ok(f"ambiguity levels: all within Level 1-{len(LV)}")

    # Accept both the plain vocabulary and the GeoJSON type names -- the
    # distinction carries no meaning downstream and rejecting it wastes a
    # regeneration round.
    GEOM_OK = {"point", "line", "polygon", "linestring", "multipolygon",
               "multilinestring", "multipoint"}
    badgeom = Counter(g for r in rows for g in
                      (r["source_geometry"].strip().lower(),
                       r["target_geometry"].strip().lower())
                      if g not in GEOM_OK)
    if badgeom:
        warn(f"unexpected geometry values: {dict(badgeom)}")
    else:
        ok("geometry: point/line/polygon only")

    # --- balance ----------------------------------------------------------
    cells = Counter((r["relation_label"].strip().lower(),
                     r["ambiguity_level"].strip()) for r in rows)
    # Level 6 exists only for labels with a forced composition, so its grid is
    # legitimately smaller. Expecting a full rectangle here would flag correct
    # data as broken.
    grid = {(l, v): cells.get((l, v), 0)
            for l in labels for v in LV
            if v != HOP_LEVEL or l in HOP_LABELS[rel]}
    vals = sorted(set(grid.values()))
    want = args.expect_per_cell
    if len(vals) == 1 and (want is None or vals[0] == want):
        ok(f"balance: every (label x level) cell has exactly {vals[0]} rows")
    else:
        empty = [k for k, v in grid.items() if v == 0]
        bad(f"balance: cell counts vary {vals}"
            + (f"; {len(empty)} EMPTY cells e.g. {empty[:3]}" if empty else ""))

    # --- duplicates within the new file -----------------------------------
    pc = Counter(pair(r) for r in rows)
    dup = [p for p, n in pc.items() if n > 1]
    if dup:
        bad(f"{len(dup)} entity pair(s) used more than once, e.g. {dup[:3]}")
    else:
        ok("no repeated entity pairs within this file")

    # --- mirrors (A,B) alongside (B,A) ------------------------------------
    seen = {pair(r): r["relation_label"].strip().lower() for r in rows}
    mirrors = [(p, seen[p]) for p in seen if (p[1], p[0]) in seen]
    if mirrors:
        bad(f"{len(mirrors)} pair(s) appear together with their mirror, "
            f"e.g. {mirrors[0][0]} — this leaks answers across splits")
    else:
        ok("no mirrored pairs")

    # --- collision with the existing corpus -------------------------------
    corpus_path = REPO / "data" / rel / "corpus.csv"
    if corpus_path.exists():
        with corpus_path.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        old_pairs = {pair(r) for r in old}
        clash = [p for p in pc if p in old_pairs]
        old_mirror = [p for p in pc if (p[1], p[0]) in old_pairs]
        if clash:
            bad(f"{len(clash)} pair(s) already exist in the corpus, e.g. {clash[:3]}")
        else:
            ok(f"no collisions with the {len(old_pairs)} existing pairs")
        if old_mirror:
            warn(f"{len(old_mirror)} pair(s) mirror an existing corpus pair, "
                 f"e.g. {old_mirror[:2]}")

    # --- the answer must not be stated in the text ------------------------
    leaks = []
    for r in rows:
        lab = r["relation_label"].strip().lower()
        text = r["corpus"].lower()
        for phrase in GIVEAWAY.get(lab, []):
            if phrase in text:
                leaks.append((r["source_entity"][:26], lab, phrase))
                break
    if leaks:
        bad(f"{len(leaks)} row(s) state the answer in the description, "
            f"e.g. {leaks[0][0]} says '{leaks[0][2]}' for {leaks[0][1]}")
    else:
        ok("descriptions never contain the label or an obvious synonym")

    # --- multi-hop rows ---------------------------------------------------
    hop = [r for r in rows if r["ambiguity_level"].strip() == HOP_LEVEL]
    flat = [r for r in rows if r["ambiguity_level"].strip() != HOP_LEVEL]
    if "via_entity" not in rows[0]:
        if hop:
            bad(f"{len(hop)} Level 6 row(s) but no via_entity column")
        else:
            warn("no via_entity column and no Level 6 rows — multi-hop absent")
    else:
        no_via = [r for r in hop if not r["via_entity"].strip()]
        if no_via:
            bad(f"{len(no_via)} Level 6 row(s) have an empty via_entity — the "
                f"intermediate place must be named, e.g. "
                f"{no_via[0]['source_entity'][:30]}")
        elif hop:
            ok(f"multi-hop: all {len(hop)} Level 6 rows name an intermediate place")

        stray = [r for r in flat if r["via_entity"].strip()]
        if stray:
            warn(f"{len(stray)} row(s) below Level 6 set via_entity — it is "
                 f"ignored outside multi-hop rows")

        # C must be a third place, not one of the endpoints
        degenerate = [r for r in hop
                      if r["via_entity"].strip().lower() in
                      (r["source_entity"].strip().lower(),
                       r["target_entity"].strip().lower())]
        if degenerate:
            bad(f"{len(degenerate)} Level 6 row(s) use an endpoint as the "
                f"intermediate — that is not a chain")
        elif hop:
            ok("multi-hop: every intermediate is a distinct third place")

    # only labels with a forced composition may appear at Level 6
    if hop:
        allowed = HOP_LABELS[rel]
        unforced = [r for r in hop
                    if r["relation_label"].strip().lower() not in allowed]
        if unforced:
            got = sorted({r["relation_label"].strip().lower() for r in unforced})
            bad(f"{len(unforced)} Level 6 row(s) use a label with no forced "
                f"two-hop composition ({', '.join(got)}). "
                f"Level 6 is valid only for: {', '.join(sorted(allowed))}.")
        else:
            ok(f"multi-hop: every label has a forced composition "
               f"({', '.join(sorted({r['relation_label'].strip().lower() for r in hop}))})")

        # THE decisive check: does the text state TWO links, or only one?
        #
        # The first batch failed this on 35 of 35 rows: "The federal republic
        # fully surrounds the golden state" is a single clause, so the second
        # hop is simply absent and the answer cannot be derived.
        #
        # What matters is the presence of a second linking clause, NOT whether
        # the target is named literally. "...bounded by Santa Clara County, and
        # that jurisdiction sits bounded by the western coastal state" is a
        # perfectly good two-hop description even though it paraphrases the
        # target -- the subject and object are given to the model as separate
        # fields anyway, so a paraphrase in the prose is fine and arguably
        # better, since it stops the row being solvable by name-matching alone.
        LINKERS = ("and that", "and it", "and this", ", and", "which in turn",
                   "; that", "which itself", "and the latter", "in turn")
        one_clause = [r for r in hop
                      if not any(k in r["corpus"].lower() for k in LINKERS)]
        if one_clause:
            bad(f"{len(one_clause)}/{len(hop)} Level 6 row(s) state only ONE link. "
                f"A multi-hop description needs two clauses joined by 'and that', "
                f"'which in turn' or similar. "
                f"e.g. A={one_clause[0]['source_entity']!r} "
                f"via={one_clause[0]['via_entity']!r} "
                f"B={one_clause[0]['target_entity']!r}: "
                f"\"{one_clause[0]['corpus'][:70]}...\"")
        else:
            ok("multi-hop: every description states two linked clauses")

        # softer: the intermediate really should be named, since the whole point
        # is that the reader routes through it
        unnamed = [r for r in hop
                   if content_words(r["via_entity"])
                   and not (content_words(r["via_entity"])
                            & content_words(r["corpus"]))]
        if unnamed:
            warn(f"{len(unnamed)} Level 6 row(s) never name the intermediate place "
                 f"in the description, so the chain is implicit")
        else:
            ok("multi-hop: every description names its intermediate place")

        # C must be a genuinely different place, not a synonym of an endpoint
        def stem(t):
            w = sorted(content_words(t), key=len, reverse=True)
            return w[0][:5] if w else ""
        alias = [r for r in hop
                 if stem(r["via_entity"]) and
                 stem(r["via_entity"]) in (stem(r["source_entity"]),
                                           stem(r["target_entity"]))]
        if alias:
            warn(f"{len(alias)} Level 6 row(s) may route through a synonym of an "
                 f"endpoint rather than a third place, e.g. "
                 f"{alias[0]['source_entity']!r} via {alias[0]['via_entity']!r} — "
                 f"chaining names is not spatial reasoning")
        else:
            ok("multi-hop: no intermediate looks like a synonym of an endpoint")

    # Level 6 should stay plainly worded — otherwise indirection and inference
    # depth are confounded and the level measures neither cleanly.
    if hop:
        ornate = [r for r in hop if any(w in r["corpus"].lower() for w in
                  ("o'clock", "port arm", "starboard", "manga", "arabic script",
                   "wedding ring"))]
        if ornate:
            warn(f"{len(ornate)} Level 6 row(s) also use oblique Level 1-5 "
                 f"phrasing — that confounds wording with inference depth")

    # --- cardinal: the description must not encode the bearing -------------
    # Level 6 is exempt: a chain cannot be stated without directional language,
    # and that level tests composition rather than knowledge.
    if rel == "cardinal":
        flat = [r for r in rows if r["ambiguity_level"].strip() != HOP_LEVEL]
        leaked = []
        for r in flat:
            hit = bearing_cue(r["corpus"])
            if hit:
                leaked.append((r, hit))
        if leaked:
            bad(f"{len(leaked)}/{len(flat)} Level 1-5 rows encode the bearing in "
                f"the description, so the answer needs no geographic knowledge. "
                f"e.g. {leaked[0][0]['source_entity']!r} uses \"{leaked[0][1]}\": "
                f"\"{leaked[0][0]['corpus'][:64]}...\"")
        else:
            ok(f"cardinal: no Level 1-5 description encodes the bearing — the "
               f"answer requires knowing where the places are")

    # --- template reuse ----------------------------------------------------
    # 144 rows built from 48 frames is 48 items shown three times, not 144.
    import re as _re

    def _frame(r):
        t = r["corpus"]
        for n in (r.get("source_entity", ""), r.get("target_entity", ""),
                  r.get("via_entity", "")):
            if n:
                t = t.replace(n, "@")
        return _re.sub(r"\s+", " ", t).strip().lower()

    frames = Counter(_frame(r) for r in rows)
    reused = {f: n for f, n in frames.items() if n > 1}
    ratio = len(frames) / len(rows)
    if ratio < 0.75:
        bad(f"only {len(frames)} distinct sentence frames across {len(rows)} rows "
            f"({ratio:.0%}). Rows built from a shared template are one item shown "
            f"several times, not several items. Worst frame repeats "
            f"{max(frames.values())}x: \"{frames.most_common(1)[0][0][:60]}...\"")
    elif reused:
        warn(f"{len(reused)} sentence frame(s) reused; {len(frames)} distinct "
             f"across {len(rows)} rows ({ratio:.0%})")
    else:
        ok(f"phrasing: {len(frames)} distinct sentence frames for {len(rows)} rows")

    expl = Counter(r.get("explanation", "").strip().lower() for r in rows)
    if len(expl) / len(rows) < 0.6:
        warn(f"only {len(expl)} distinct explanations across {len(rows)} rows — "
             f"they should state the actual reason, not a fixed phrase")

    # --- text sanity ------------------------------------------------------
    short = [r for r in rows if len(r["corpus"].strip()) < 40]
    if short:
        warn(f"{len(short)} description(s) under 40 characters — likely too thin")
    else:
        ok("descriptions are of reasonable length")

    if rel == "relative":
        # 'left' is meaningless without a stated viewpoint
        vp = re.compile(r"facing|standing|looking|from the|approaching|viewed|"
                        r"driving|scanning|as you", re.I)
        noview = [r for r in rows if not vp.search(r["corpus"])]
        if noview:
            bad(f"{len(noview)} row(s) state no observer viewpoint — 'left' is "
                f"undefined without one, e.g. {noview[0]['source_entity'][:30]}")
        else:
            ok("every description establishes an observer viewpoint")

    # --- geocoding --------------------------------------------------------
    if args.geocode:
        print("\n  --- geocoding (Nominatim, 1 req/sec) ---")
        sys.path.insert(0, str(REPO / "src"))
        import urllib.parse
        import urllib.request
        cols = ["source_entity", "target_entity"]
        if "via_entity" in rows[0]:
            cols.append("via_entity")     # part of the reasoning chain
        names = sorted({r[c].strip() for r in rows for c in cols if r.get(c, "").strip()})
        print(f"  checking {len(names)} distinct places "
              f"(~{len(names) * 1.1 / 60:.0f} min)...")
        failed, suspicious = [], []
        for i, name in enumerate(names, 1):
            url = ("https://nominatim.openstreetmap.org/search?"
                   + urllib.parse.urlencode({"q": name, "format": "json",
                                             "addressdetails": 1, "limit": 5}))
            req = urllib.request.Request(
                url, headers={"User-Agent": "spatial-eval-datacheck/1.0"})
            try:
                time.sleep(1.1)
                data = json.loads(urllib.request.urlopen(req, timeout=25).read())
            except Exception as exc:
                warn(f"query failed for {name!r}: {exc}")
                continue
            if not data:
                failed.append(name)
            else:
                pref = ["boundary", "place", "natural", "waterway", "landuse"]
                best = next((d for d in data if d.get("class") in pref), data[0])
                # An administrative-sounding name resolving to a shop or a road
                # is the failure mode that poisoned the existing cache.
                admin = any(w in name.lower() for w in
                            ("city of", "state of", "republic", "county",
                             "province", "kingdom", "district"))
                if admin and best.get("class") in ("amenity", "shop", "building",
                                                   "highway", "tourism"):
                    suspicious.append(
                        f"{name} -> {best.get('class')}/{best.get('type')}")
            if i % 25 == 0:
                print(f"    {i}/{len(names)}")
        if failed:
            bad(f"{len(failed)} place(s) do not resolve at all: {failed[:6]}")
        else:
            ok(f"all {len(names)} places resolve in OpenStreetMap")
        if suspicious:
            bad(f"{len(suspicious)} place(s) resolve to the wrong kind of object: "
                f"{suspicious[:4]}")
        else:
            ok("no administrative name resolved to a point of interest")

    print("\n" + "=" * 78)
    print(f"  {_counts['pass']} passed, {_counts['warn']} warnings, "
          f"{_counts['fail']} failures")
    if _counts["fail"]:
        print("  DO NOT MERGE — send the failures back to the generator and retry.")
    else:
        print("  Clean. Safe to merge into the corpus.")
    print("=" * 78)
    return 1 if _counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
