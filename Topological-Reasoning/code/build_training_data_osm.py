"""
build_kg_instruction_dataset_osm.py
================================================================================
Phase 1 — KGs Instruction-Tuning Pipeline

Builds an OSM-grounded instruction-tuning dataset from the 70% training split.

For each training row:
  1. Queries OpenStreetMap (Nominatim) for both geographic entities via the
     existing GeographicKnowledgeGraph class (caching + rate-limit already built in).
  2. Embeds the KG evidence (coordinates, bounding boxes, hierarchy, centroid
     distance) directly into a structured instruction prompt (the INPUT).
  3. Writes a template-based reasoning + the gold label into the OUTPUT section.
  4. Saves every record to JSONL (one JSON object per line).

The key design principle: KG evidence goes in the INPUT; the label stays in
the OUTPUT — this is the instruction-tuning contract that teaches the model
to USE the graph to reason, not to memorise the answer.

A .ckpt.json checkpoint is saved every 10 rows so the script is resumable.

Usage (on the server):
  python build_kg_instruction_dataset_osm.py \
      --dataset  ../dataset/triplet_update_v3_70.csv \
      --cache    results/osm_cache.json \
      --output   ../dataset/osm_kg_train.jsonl
"""

import argparse
import json
import os
import sys
import tempfile

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from strategies_osm import GeographicKnowledgeGraph

VALID_PREDICATES = {"disjoint", "touches", "crosses", "within", "contains", "overlaps", "equals"}

# ---------------------------------------------------------------------------
# Prompt sections (constant across all examples)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are an expert in geospatial topological reasoning using the DE-9IM model."
)

_TASK = (
    "Given a vernacular spatial relation between two geographic entities A and B, "
    "use the OpenStreetMap knowledge graph evidence provided to determine the correct "
    "DE-9IM topological predicate."
)

_VALID_PREDS = "contains, within, touches, crosses, disjoint, overlaps, equals"

_RULES = """\
1. The relation is DIRECTED from entity A to entity B.
2. Use the KG evidence (coordinates, bounding boxes, administrative hierarchy) \
to reason geometrically.
3. Bounding box containment ≠ polygon containment — treat bbox as a clue, not proof.
4. Direction: "contains" means A encloses B; "within" means A is inside B.
5. equals: A and B share the EXACT same geometry (identical extent, same interior and boundary).
6. You must pick EXACTLY ONE predicate from the list above."""

# ---------------------------------------------------------------------------
# Template-based output reasoning (one per predicate)
# The template references KG evidence types so the model learns the connection.
# ---------------------------------------------------------------------------
_REASONING_TEMPLATES: dict[str, str] = {
    "within": (
        "Entity A ({subj_type}, {geom_A}) is spatially contained within Entity B "
        "({obj_type}, {geom_B}). The vernacular '{vernacular}' and the OSM administrative "
        "hierarchy confirm A resides inside B's spatial extent."
    ),
    "contains": (
        "Entity A ({subj_type}, {geom_A}) spatially contains Entity B ({obj_type}, {geom_B}). "
        "The OSM bounding box and hierarchy show B's coordinates fall within A's extent, "
        "and the vernacular '{vernacular}' signals that A is the enclosing entity."
    ),
    "touches": (
        "Entity A ({subj_type}, {geom_A}) shares only a boundary with Entity B "
        "({obj_type}, {geom_B}), with no interior intersection. The OSM bounding boxes "
        "are adjacent or nearly touching, and the vernacular '{vernacular}' indicates "
        "boundary contact only."
    ),
    "crosses": (
        "Entity A ({subj_type}, {geom_A}) is a linear feature that passes through the "
        "interior of Entity B ({obj_type}, {geom_B}), entering and exiting its polygon. "
        "The vernacular '{vernacular}' signals traversal, consistent with a LineString "
        "crossing a Polygon in OSM."
    ),
    "overlaps": (
        "Entity A ({subj_type}, {geom_A}) and Entity B ({obj_type}, {geom_B}) share some "
        "interior area but neither fully contains the other. The OSM bounding boxes show "
        "partial overlap of extents, and the vernacular '{vernacular}' implies partial "
        "intersection."
    ),
    "disjoint": (
        "Entity A ({subj_type}, {geom_A}) and Entity B ({obj_type}, {geom_B}) are spatially "
        "separate. The OSM bounding boxes show no overlap and the centroid distance "
        "confirms they share no boundary or interior. The vernacular '{vernacular}' is "
        "consistent with spatial separation."
    ),
    "equals": (
        "Entity A ({subj_type}, {geom_A}) and Entity B ({obj_type}, {geom_B}) share the "
        "exact same geometry — identical interior, boundary, and exterior. The OSM bounding "
        "boxes are essentially congruent and the centroids coincide. The vernacular "
        "'{vernacular}' explicitly signals geometric identity."
    ),
}


def _fill_reasoning(row: dict, label: str) -> str:
    template = _REASONING_TEMPLATES.get(label, "The spatial relation between A and B is {label}.")
    return template.format(
        subj_type=row.get("placetype_subject", "place"),
        obj_type=row.get("placetype_object", "place"),
        geom_A=row.get("source_geometry", row.get("geometry_type_subject", "unknown")),
        geom_B=row.get("target_geometry",  row.get("geometry_type_object",  "unknown")),
        vernacular=row.get("explanation", row.get("vernacular_relation", row.get("relation_predicate", ""))),
        label=label,
    )


def build_record_text(row: dict, kg_evidence: str, label: str) -> str:
    """Assemble the full instruction-tuning text for one training example."""
    subj = row.get("source_entity", row.get("place_name_subject", "Entity A"))
    obj  = row.get("target_entity", row.get("place_name_object",  "Entity B"))
    vernacular  = row.get("explanation", row.get("vernacular_relation", row.get("relation_predicate", "")))
    subj_type   = row.get("placetype_subject", "place")
    obj_type    = row.get("placetype_object",  "place")
    geom_A      = row.get("source_geometry", row.get("geometry_type_subject", "unknown"))
    geom_B      = row.get("target_geometry",  row.get("geometry_type_object",  "unknown"))
    reasoning   = _fill_reasoning(row, label)

    return (
        f"### SYSTEM PROMPT ###\n{_SYSTEM}\n\n"
        f"### TASK ###\n{_TASK}\n\n"
        f"### VALID PREDICATES ###\n{_VALID_PREDS}\n\n"
        f"### RULES ###\n{_RULES}\n\n"
        f"### KG EVIDENCE (OpenStreetMap) ###\n{kg_evidence}\n\n"
        f"### INPUT ###\n"
        f"Vernacular relation: \"{subj} {vernacular} {obj}\"\n"
        f"Entity A: {subj_type} (geometry: {geom_A})\n"
        f"Entity B: {obj_type} (geometry: {geom_B})\n\n"
        f"### OUTPUT ###\n"
        f"Reasoning: {reasoning}\n"
        f"Answer: A [{label}] B"
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _atomic_write(path: str, data: dict):
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_checkpoint(ckpt_path: str) -> tuple[set, list]:
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed_indices", [])), data.get("records", [])
        except Exception:
            pass
    return set(), []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build OSM-KG instruction-tuning dataset")
    parser.add_argument("--dataset", default="../dataset/topological_balanced_train.csv",
                        help="Path to the balanced training CSV (topological_balanced_train.csv)")
    parser.add_argument("--cache",   default="results/osm_cache.json",
                        help="Path to existing OSM Nominatim cache (reused to avoid re-fetching)")
    parser.add_argument("--output",  default="../dataset/osm_kg_train.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()

    ckpt_path = args.output + ".ckpt.json"
    processed_indices, records = _load_checkpoint(ckpt_path)

    df = pd.read_csv(args.dataset)
    print(f"[DATA]   Loaded {len(df)} training rows from {args.dataset}")
    print(f"[RESUME] {len(processed_indices)} rows already processed; {len(df) - len(processed_indices)} remaining.")

    # Point the KG class at the existing OSM cache so we reuse already-fetched data
    kg = GeographicKnowledgeGraph(kg_path=args.cache)

    skipped = failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OSM-KG dataset"):
        if idx in processed_indices:
            skipped += 1
            continue

        label = str(row.get("relation_label", row.get("spatial_relation", ""))).lower().strip()
        if not label or label not in VALID_PREDICATES:
            failed += 1
            processed_indices.add(idx)
            continue

        place_a  = str(row.get("source_entity", row.get("place_name_subject", "")))
        place_b  = str(row.get("target_entity", row.get("place_name_object",  "")))
        sentence = str(row.get("corpus",        row.get("Sentence", "")))

        try:
            kg_evidence = kg.gather_evidence(
                place_a, place_b,
                sentence=sentence,
                entity=row.to_dict(),
            )
        except Exception as exc:
            print(f"\n[WARN] KG fetch failed at row {idx}: {exc}")
            kg_evidence = "No OSM evidence available for these entities."
            failed += 1

        text = build_record_text(row.to_dict(), kg_evidence, label)

        records.append({
            "text":   text,
            "source": "osm_kg",
            "label":  label,
            "idx":    int(idx),
        })
        processed_indices.add(idx)

        # Checkpoint every 10 rows so a crash loses at most 10 rows of API work
        if len(records) % 10 == 0:
            _atomic_write(ckpt_path, {
                "processed_indices": list(processed_indices),
                "records": records,
            })

    # Final checkpoint + JSONL write
    _atomic_write(ckpt_path, {
        "processed_indices": list(processed_indices),
        "records": records,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE]  {len(records)} instruction examples saved → {args.output}")
    label_counts = {}
    for rec in records:
        label_counts[rec["label"]] = label_counts.get(rec["label"], 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"        {lbl:12s}: {cnt}")
    print(f"        Skipped (already done): {skipped}  |  Invalid/failed: {failed}")


if __name__ == "__main__":
    main()
