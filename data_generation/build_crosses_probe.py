"""Build a probe set that asks `crosses` about lines that are not rivers.

In the frozen evaluation every `crosses` item involves a river, so "one of the two places is a
river" finds them all without reading the sentence. This probe asks the same questions about
highways and long-distance trails (fetched by fetch_probe_lines.py) against the catalogue's
polygons, with three labels so that "a road, therefore crosses" is not a winning rule either:

  crosses   the line runs partly inside the area and partly outside it
  within    the whole line lies inside the area
  disjoint  the line passes near the area without entering it

Labels come from the same rules as the corpus (compute_topo_relations.relate). Descriptions are
drawn from the EVALUATION template pool, never the training pool, so a fine-tuned arm sees wording
it was not trained on -- exactly as on the main evaluation. The line is always the subject {A},
which is the role the `crosses` evaluation templates assume.

    <venv with shapely>/bin/python data_generation/build_crosses_probe.py

Writes, next to the main evaluation and without touching it:
  data/topological/probe_crosses.csv            the items, same columns as eval.csv
  data/topological/probe_crosses_manifest.json  the frozen manifest, same format
  data/topological/kg_probe_crosses.json        the knowledge store for these places
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from shapely.geometry import shape

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_generation"))
sys.path.insert(0, str(REPO / "scripts"))
from apply_templates import short                       # noqa: E402
from build_kg import describe                           # noqa: E402
from build_splits import row_hash                       # noqa: E402
from compute_topo_relations import convention_dependent, relate  # noqa: E402

DATA = REPO / "data" / "topological"
GEOM = DATA / "osm" / "geometry.json"
LINES = DATA / "osm" / "probe_lines.json"
SEED = 20261006
PER_LINE = {"crosses": 5, "within": 2, "disjoint": 3}     # at most, per line
NEAR = (0.3, 3.0)                                         # disjoint: degrees from the line
HEADER = ["source_entity", "source_geometry", "target_entity", "target_geometry", "corpus",
          "via_entity", "relation_type", "relation_label", "explanation", "ambiguity_level"]


def polygons():
    raw = json.loads(GEOM.read_text())
    out = {}
    for name, rec in raw.items():
        if not rec or rec["geojson"].get("type") not in ("Polygon", "MultiPolygon"):
            continue
        g = shape(rec["geojson"])
        g = g if g.is_valid else g.buffer(0)
        if not g.is_empty:
            out[name] = (g, rec)
    return out


def related_names(a: str, b: str) -> bool:
    """A name that contains the other states the answer ('Texas' in 'Interstate 10 (Texas)')."""
    x, y = short(a).lower(), short(b).lower()
    return x in y or y in x


def main() -> int:
    polys = polygons()
    lines = json.loads(LINES.read_text())
    pools = json.loads((REPO / "data_generation" / "paraphrases_topological.json").read_text())["eval"]
    used = set()
    for split in ("train", "eval", "corpus"):
        for r in csv.DictReader((DATA / f"{split}.csv").open(encoding="utf-8")):
            used.add(frozenset((r["source_entity"], r["target_entity"])))
    rng = random.Random(SEED)

    cands = defaultdict(lambda: defaultdict(list))         # label -> line -> [(score, poly, info)]
    for lname, lrec in lines.items():
        L = shape(lrec["geojson"])
        for pname, (P, prec) in polys.items():
            if related_names(lname, pname) or frozenset((lname, pname)) in used:
                continue
            d = L.distance(P)
            if d > NEAR[1]:
                continue
            label, info = relate(L, P)
            if label is None or convention_dependent(L, P, label):
                continue
            f = info.get("inside_fraction", 0.0)
            inside_km = f * L.length * 111.32 * 0.75
            if label == "crosses" and 0.03 <= f <= 0.97 and inside_km >= 30:
                cands["crosses"][lname].append((min(f, 1 - f), pname, info))
            elif label == "within":
                cands["within"][lname].append((prec.get("importance") or 0, pname, info))
            elif label == "disjoint" and d >= NEAR[0]:
                cands["disjoint"][lname].append((prec.get("importance") or 0, pname, info | {"gap_deg": d}))

    picked = []
    for label in ("crosses", "within", "disjoint"):
        for lname, items in sorted(cands[label].items()):
            items.sort(key=lambda t: -t[0])                 # clearest crossing, best-known area first
            for score, pname, info in items[:PER_LINE[label]]:
                picked.append((label, lname, pname, info))
    print("  candidates kept:", {lab: sum(1 for p in picked if p[0] == lab) for lab in PER_LINE})

    rows, seen = [], set()
    by_label = defaultdict(list)
    for p in picked:
        by_label[p[0]].append(p)
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        for i, (lab, lname, pname, info) in enumerate(items):
            level = i % 5 + 1
            pool = pools[f"{lab}|{level}"]
            for _ in range(20):
                tpl = pool[rng.randrange(len(pool))]
                text = tpl.replace("{A}", short(lname)).replace("{B}", short(pname))
                if text not in seen:
                    break
            seen.add(text)
            stats = ", ".join(f"{k}={v:.4g}" for k, v in info.items() if isinstance(v, (int, float)))
            rows.append({
                "source_entity": lname, "source_geometry": "LineString",
                "target_entity": pname, "target_geometry": "Polygon",
                "corpus": text, "via_entity": "", "relation_type": "topological",
                "relation_label": lab,
                "explanation": f"Computed from OpenStreetMap geometry: {lname} {lab} {pname} ({stats}).",
                "ambiguity_level": f"Level {level}",
            })

    with (DATA / "probe_crosses.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    entries = []
    for i, r in enumerate(rows):
        entries.append({"row_index": i, "fact_id": f"p{i:04d}", "subject": r["source_entity"],
                        "target": r["target_entity"], "label": r["relation_label"],
                        "ambiguity_level": r["ambiguity_level"], "row_sha256": row_hash(r)})
    man_sha = hashlib.sha256("".join(e["row_sha256"] for e in entries).encode()).hexdigest()
    manifest = {
        "domain": "Topological-Reasoning", "source_csv": "data/topological/probe_crosses.csv",
        "n_rows": len(entries), "n_unique_facts": len(entries), "duplicate_rows": 0,
        "manifest_sha256": man_sha,
        "purpose": "crosses on lines that are not rivers (highways, long-distance trails), with "
                   "within and disjoint items on the same lines as controls",
        "rows": entries,
    }
    (DATA / "probe_crosses_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    nodes = {}
    for r in rows:
        for name, rec in ((r["source_entity"], lines.get(r["source_entity"])),
                          (r["target_entity"], polys.get(r["target_entity"], (None, None))[1])):
            if name not in nodes:
                fact = describe(name, rec, {}, keep_context=False)
                if fact:
                    nodes[name] = fact
    (DATA / "kg_probe_crosses.json").write_text(json.dumps(
        {"relation": "topological", "n_nodes": len(nodes),
         "note": "facts only; the relation under test is never stated here", "nodes": nodes},
        indent=1, ensure_ascii=False))
    lab = defaultdict(int)
    for r in rows:
        lab[(r["relation_label"], r["ambiguity_level"])] += 1
    print(f"  {len(rows)} items written; {len(nodes)} places in the store")
    for k in sorted(lab):
        print("   ", k, lab[k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
