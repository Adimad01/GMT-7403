"""
build_kg_instruction_dataset_wikidata.py
================================================================================
Phase 2 — KGs Instruction-Tuning Pipeline

Builds a Wikidata-grounded instruction-tuning dataset from the 70% training split.

For each training row:
  1. Queries Wikidata (MediaWiki Search + SPARQL) for both entities via the
     existing GeographicKnowledgeGraph class (strategies_wikidata.py).
  2. Embeds semantic KG evidence (description, instance-of P31, located-in P131,
     centroid coordinates) directly into the instruction prompt (the INPUT).
  3. Writes a template-based reasoning + the gold label into the OUTPUT section.
  4. Saves to JSONL with a checkpoint for resumability.

Key contrast with the OSM version:
  - OSM provides geometric evidence (bounding boxes, precise coordinates).
  - Wikidata provides semantic/ontological evidence (P131 containment chains,
    P31 entity types, textual descriptions).
  The template-based reasoning in the OUTPUT explicitly references the P131
  hierarchy so the model learns to use ontological structure for reasoning.

Usage (on the server):
  python build_kg_instruction_dataset_wikidata.py \
      --dataset  ../dataset/triplet_update_v3_70.csv \
      --cache    results/wikidata_cache.json \
      --output   ../dataset/wikidata_kg_train.jsonl
"""

import argparse
import json
import os
import sys
import tempfile

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from strategies_wikidata import GeographicKnowledgeGraph

VALID_PREDICATES = {"disjoint", "touches", "crosses", "within", "contains", "overlaps"}

# ---------------------------------------------------------------------------
# Prompt sections
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are an expert in geospatial topological reasoning using the DE-9IM model."
)

_TASK = (
    "Given a vernacular spatial relation between two geographic entities A and B, "
    "use the Wikidata knowledge graph evidence provided (semantic hierarchy, ontological "
    "type, and coordinates) to determine the correct DE-9IM topological predicate."
)

_VALID_PREDS = "contains, within, touches, crosses, disjoint, overlaps"

_RULES = """\
1. The relation is DIRECTED from entity A to entity B.
2. Use the Wikidata KG evidence (P131 located-in chain, P31 instance-of, coordinates).
3. Wikidata provides centroid coordinates only, NOT polygon geometry — do not infer \
containment from coordinates alone.
4. If A appears in B's P131 chain → within.  If B appears in A's P131 chain → contains.
5. Direction: "contains" means A encloses B; "within" means A is inside B.
6. You must pick EXACTLY ONE predicate from the list above."""

# ---------------------------------------------------------------------------
# Template-based output reasoning — references semantic KG facts
# ---------------------------------------------------------------------------
_REASONING_TEMPLATES: dict[str, str] = {
    "within": (
        "Entity A ({subj_type}, {geom_A}) is administratively contained within Entity B "
        "({obj_type}, {geom_B}). The Wikidata P131 (located-in) chain for A lists B as a "
        "parent region, and the vernacular '{vernacular}' is consistent with spatial containment."
    ),
    "contains": (
        "Entity A ({subj_type}, {geom_A}) spatially contains Entity B ({obj_type}, {geom_B}). "
        "The Wikidata P131 chain for B lists A as a parent region, and the vernacular "
        "'{vernacular}' signals that A is the enclosing entity."
    ),
    "touches": (
        "Entity A ({subj_type}, {geom_A}) shares only a boundary with Entity B "
        "({obj_type}, {geom_B}), with no interior intersection. Neither entity appears "
        "in the other's Wikidata P131 chain, and the vernacular '{vernacular}' indicates "
        "adjacency or boundary contact without containment."
    ),
    "crosses": (
        "Entity A ({subj_type}, {geom_A}) is a linear feature that traverses Entity B "
        "({obj_type}, {geom_B}). The Wikidata instance-of (P31) type of A (e.g., river, "
        "road, highway) indicates a linear geometry, and the vernacular '{vernacular}' "
        "confirms traversal through B's polygon."
    ),
    "overlaps": (
        "Entity A ({subj_type}, {geom_A}) and Entity B ({obj_type}, {geom_B}) partially "
        "overlap — they share some interior area but neither fully contains the other. "
        "The Wikidata description or P131 data suggests partial administrative overlap, "
        "consistent with the vernacular '{vernacular}'."
    ),
    "disjoint": (
        "Entity A ({subj_type}, {geom_A}) and Entity B ({obj_type}, {geom_B}) are spatially "
        "separate. Neither appears in the other's Wikidata P131 chain, the centroid "
        "distance is large, and the vernacular '{vernacular}' is consistent with no "
        "shared boundary or interior."
    ),
}


def _fill_reasoning(row: dict, label: str) -> str:
    template = _REASONING_TEMPLATES.get(label, "The spatial relation between A and B is {label}.")
    return template.format(
        subj_type=row.get("placetype_subject", "place"),
        obj_type=row.get("placetype_object", "place"),
        geom_A=row.get("geometry_type_subject", "unknown"),
        geom_B=row.get("geometry_type_object", "unknown"),
        vernacular=row.get("vernacular_relation") or row.get("relation_predicate", ""),
        label=label,
    )


def build_record_text(row: dict, kg_evidence: str, label: str) -> str:
    """Assemble the full instruction-tuning text for one training example."""
    subj       = row.get("place_name_subject", "Entity A")
    obj        = row.get("place_name_object",  "Entity B")
    vernacular = row.get("vernacular_relation") or row.get("relation_predicate", "")
    subj_type  = row.get("placetype_subject", "place")
    obj_type   = row.get("placetype_object",  "place")
    geom_A     = row.get("geometry_type_subject", "unknown")
    geom_B     = row.get("geometry_type_object",  "unknown")
    reasoning  = _fill_reasoning(row, label)

    return (
        f"### SYSTEM PROMPT ###\n{_SYSTEM}\n\n"
        f"### TASK ###\n{_TASK}\n\n"
        f"### VALID PREDICATES ###\n{_VALID_PREDS}\n\n"
        f"### RULES ###\n{_RULES}\n\n"
        f"### KG EVIDENCE (Wikidata) ###\n{kg_evidence}\n\n"
        f"### INPUT ###\n"
        f"Vernacular relation: \"{subj} {vernacular} {obj}\"\n"
        f"Entity A: {subj_type} (geometry: {geom_A})\n"
        f"Entity B: {obj_type} (geometry: {geom_B})\n\n"
        f"### OUTPUT ###\n"
        f"Reasoning: {reasoning}\n"
        f"Answer: A [{label}] B"
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers (identical pattern to OSM builder)
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
    parser = argparse.ArgumentParser(description="Build Wikidata-KG instruction-tuning dataset")
    parser.add_argument("--dataset", default="../dataset/triplet_update_v3_70.csv",
                        help="Path to the 70%% training CSV")
    parser.add_argument("--cache",   default="results/wikidata_cache.json",
                        help="Path to existing Wikidata cache (reused to avoid re-fetching)")
    parser.add_argument("--output",  default="../dataset/wikidata_kg_train.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()

    ckpt_path = args.output + ".ckpt.json"
    processed_indices, records = _load_checkpoint(ckpt_path)

    df = pd.read_csv(args.dataset)
    print(f"[DATA]   Loaded {len(df)} training rows from {args.dataset}")
    print(f"[RESUME] {len(processed_indices)} rows already processed; {len(df) - len(processed_indices)} remaining.")

    kg = GeographicKnowledgeGraph(kg_path=args.cache)

    skipped = failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Wikidata-KG dataset"):
        if idx in processed_indices:
            skipped += 1
            continue

        label = str(row.get("spatial_relation", "")).lower().strip()
        if pd.isna(row.get("spatial_relation")) or label not in VALID_PREDICATES:
            failed += 1
            processed_indices.add(idx)
            continue

        place_a  = str(row.get("place_name_subject", ""))
        place_b  = str(row.get("place_name_object",  ""))
        sentence = str(row.get("Sentence", ""))

        try:
            kg_evidence = kg.gather_evidence(
                place_a, place_b,
                sentence=sentence,
                entity=row.to_dict(),
            )
        except Exception as exc:
            print(f"\n[WARN] Wikidata fetch failed at row {idx}: {exc}")
            kg_evidence = "No Wikidata evidence available for these entities."
            failed += 1

        text = build_record_text(row.to_dict(), kg_evidence, label)

        records.append({
            "text":   text,
            "source": "wikidata_kg",
            "label":  label,
            "idx":    int(idx),
        })
        processed_indices.add(idx)

        if len(records) % 10 == 0:
            _atomic_write(ckpt_path, {
                "processed_indices": list(processed_indices),
                "records": records,
            })

    _atomic_write(ckpt_path, {
        "processed_indices": list(processed_indices),
        "records": records,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE]  {len(records)} instruction examples saved → {args.output}")
    label_counts: dict[str, int] = {}
    for rec in records:
        label_counts[rec["label"]] = label_counts.get(rec["label"], 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"        {lbl:12s}: {cnt}")
    print(f"        Skipped (already done): {skipped}  |  Invalid/failed: {failed}")


if __name__ == "__main__":
    main()
