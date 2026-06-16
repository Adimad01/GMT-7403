"""
reasoning_strategies_neighborhood_details_spatial_relation.py
=================================================================================
Uses dynamic OpenStreetMap (OSM) API to provide spatial evidence (coordinates, 
bounding boxes, hierarchy) without explicitly providing the topological relation.
"""

import re
import json
import time
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from math import radians, sin, cos, sqrt, atan2

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatOllama = None  # only needed for prompting strategies, not KG evidence gathering
    HumanMessage = None


# =====================================================================
# OLLAMA CONFIG
# =====================================================================
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"


# =====================================================================
# CONSTANTS
# =====================================================================
VALID_PREDICATES = {
    "disjoint", "touches", "crosses", "within",
    "contains", "overlaps", "equals",
}

VALID_LIST = "contains, within, touches, crosses, disjoint, overlaps, equals"

VERNACULAR_LEXICON = """Vernacular-to-Topology Reference:
  WITHIN    — "is in", "located in", "part of"              (A is fully inside B)
  CONTAINS  — "is home to", "includes", "hosts"             (A fully encloses B)
  TOUCHES   — "borders", "adjacent to", "enclave of",
              "surrounded by" (partial), "across from" (water boundary)
  CROSSES   — "passes through", "along", "traverses",
              "at the mouth of" / "upstream from" (full LineString)
  OVERLAPS  — "partly in", "extends into", "straddles",
              "part of population in", "county seat of" (may extend into adjacent counties)
  DISJOINT  — "far from", "miles away", "separate from" (confirmed large distance)
  EQUALS    — "is identical to", "same boundary as", "mirrors exactly",
              "exact same extent as", "identical perimeter", "perfect mapping symmetry"
              (A and B share the exact same geometry: same interior, boundary, and exterior)

WARNING — ambiguous vernacular (read carefully):
  "county seat of" → does NOT imply within — the city may extend into adjacent counties → overlaps
  "suburb of"      → does NOT imply touches — suburb may be separated from the main city → disjoint
  "surrounded by"  → may be touches (partial encirclement), not within (full containment)
  "enclave of"     → touches (shares boundary), NOT within (not inside interior)
  "along" + river/road → likely crosses (full LineString traversal), not just touches
  "same as" / "mirrors" → may be equals only if BOTH geometries are truly identical extents
"""

RULES_BLOCK = """Rules:
1. DIRECTION CHECK (mandatory before answering):
   — "contains" means A encloses B; "within" means A is inside B.
   — Explicitly ask: does this predicate apply A→B or B→A?
   — If the data shows B encloses A, the answer is "within", NOT "contains".

2. TEXTUAL QUALIFIER PRIORITY (override geometric signals when contradictory):
   — "most sides", "surrounded by" with incomplete encirclement → touches.
   — "partly in", "extends into", "part of population in" → overlaps.
   — "enclave of" → touches (shared boundary, not interior containment).
   — "across from" over water → touches (water boundary is shared in GIS).
   — "county seat of" alone does NOT imply within.

3. GEOMETRY TYPE CONSTRAINTS:
   — LineString (river, road) traversing a Polygon interior → crosses.
   — LineString touching only Polygon boundary → touches.
   — "At the mouth of" / "upstream from": verify if the full LineString crosses the polygon.
   — Two Polygons partially sharing area → overlaps.

4. BOUNDING BOX CAUTION (never conclude directly from bbox):
   — bbox overlap ≠ polygon overlap. Polygons inside overlapping bboxes may still be disjoint.
   — bbox containment ≠ within/contains. A country bbox contains almost all nearby entities.
   — Treat bbox as a clue, never as proof.

5. BBOX GAP TOLERANCE:
   — A gap of < 0.01 degrees (~1 km) between bboxes is NOT proof of disjoint.
   — Administrative boundaries often share exact edges with 0-meter gap.

6. ENTITY VALIDATION:
   — Verify the retrieved OSM entity matches the expected state/region.
   — If the entity location does not match the context, treat its geometry as unreliable.

7. FULL LINESTRING REASONING:
   — For rivers and roads, reason on the entire geometry, not just the centroid/representative point.
   — A river centroid may fall outside a city even if the river traverses it.

8. EQUALS CHECK (do before all other predicates):
   — equals requires BOTH entities to have the exact same geometry (identical interior, boundary, exterior).
   — Strong equals signals: "identical perimeter", "exact same extent", "perfect mapping symmetry",
     "municipal limits mirror the county boundary", "coextensive", "same footprint".
   — Do NOT use equals unless the text explicitly states geometric identity.

9. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps, equals.
10. End with: Answer: [predicate]
"""

DISAMBIGUATION_TABLE = """
CRITICAL DISTINCTIONS (most confused predicates):

equals vs all others (CHECK THIS FIRST):
  equals   → A and B have the IDENTICAL geometry. Every point of A is a point of B and vice versa.
             Bbox sizes will be nearly identical; centroids will coincide.
             Strong signals: "same boundary", "identical perimeter", "coextensive",
             "municipal limits mirror county boundary", "perfect mapping symmetry".
  contains / within → different size entities — one fully inside the other, NOT identical.
  overlaps → partial overlap — NEVER equals.
  If in doubt: equals is rare. Do not use it unless geometric identity is explicitly stated.

touches vs overlaps:
  touches  → A and B share ONLY their boundary. Interiors do NOT intersect.
             Example: two adjacent counties sharing a border line.
  overlaps → A and B share SOME interior area. Part of A is inside B, part is outside.
             Example: a city that partially crosses a county boundary.

touches vs crosses:
  touches  → boundary contact only, no traversal through interior.
  crosses  → A passes THROUGH B, entering AND exiting its interior.
             Requires different geometry dimensions (LineString through Polygon).
             Example: a highway that enters and exits a city polygon.

contains vs within (DIRECTION IS CRITICAL):
  contains → A is the CONTAINER. B is fully INSIDE A. Ask: "does A surround B?"
  within   → A is INSIDE B. Ask: "is A surrounded by B?"
  KEY RULE: Identify which entity is the enclosing one.
     If A is smaller than B -> "within".  If A is larger and encloses B -> "contains".
     Signals for contains: "home to", "location of", "hosts", "includes".
     Signals for within: "located in", "in", "part of", "county seat of".

disjoint vs touches:
  disjoint -> A and B share NO point at all.
  touches  -> A and B share EXACTLY their boundary edge, nothing more.
  "suburb of" alone does NOT imply touches — suburb may be disjoint from the main city.
"""

DIRECTIONAL_RULE = """
ANTI-BIAS RULE for contains/within:
Before answering, explicitly check: "Is A the LARGER entity that SURROUNDS B?"
  - YES -> contains
  - NO, B surrounds A -> within
Do NOT default to "within" just because A is a sub-entity type or mentioned first.
"""

OVERLAPS_DETECTION = """
4-STEP DISAMBIGUATION (run before concluding within, touches, or overlaps):
  Step 1: Do A and B have the EXACT SAME geometry (identical extent)? -> equals (not within/contains)
  Step 2: Is A completely inside B? -> within (not overlaps)
  Step 3: Is B completely inside A? -> contains (not overlaps)
  Step 4: Do A and B share SOME area but NEITHER fully contains the other? -> overlaps
  Strong overlaps signals: "extends into", "partly in", "straddles",
                           "part of population in", "county seat of" (city crosses county line).
  Strong equals signals: "identical perimeter", "same boundary", "municipal limits mirror county lines",
                         "perfect mapping symmetry", "coextensive".
"""

# =====================================================================
# ENTITY VALIDATION HELPERS (Recommendation 1 & 2 & 3)
# =====================================================================
US_STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
    'MX': 'Mexico',
}

def _extract_state_hint(place_name: str) -> Optional[str]:
    """Extract US state abbreviation from a place name like 'San Jose CA'."""
    parts = place_name.strip().split()
    if parts and parts[-1].upper() in US_STATE_NAMES:
        return parts[-1].upper()
    return None

def _validate_entity_context(place_name: str, osm_data: dict) -> str:
    """Verify the OSM result matches the expected state/region from the place name."""
    if not osm_data:
        return ""
    state_hint = _extract_state_hint(place_name)
    if not state_hint:
        return ""
    state_name = US_STATE_NAMES[state_hint]
    hierarchy = osm_data.get('hierarchy', {})
    hier_str = " ".join(str(v) for v in hierarchy.values()).lower()
    if state_name.lower() in hier_str or state_hint.lower() in hier_str:
        return f"  ✓ Entity confirmed in expected state ({state_name})"
    return (
        f"  ⚠️ ENTITY MISMATCH: Expected state {state_name} ({state_hint}) not found in OSM "
        f"hierarchy — this entity may be from a different region. Treat its geometry as UNRELIABLE."
    )

def _compute_bbox_relationship(bbox_a: list, bbox_b: list) -> str:
    """Analyze the spatial relationship between two OSM bounding boxes."""
    try:
        s_a, n_a, w_a, e_a = float(bbox_a[0]), float(bbox_a[1]), float(bbox_a[2]), float(bbox_a[3])
        s_b, n_b, w_b, e_b = float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3])

        lat_overlap = s_a <= n_b and n_a >= s_b
        lon_overlap = w_a <= e_b and e_a >= w_b
        overlapping = lat_overlap and lon_overlap

        if overlapping:
            a_in_b = s_a >= s_b and n_a <= n_b and w_a >= w_b and e_a <= e_b
            b_in_a = s_b >= s_a and n_b <= n_a and w_b >= w_a and e_b <= e_a
            if a_in_b:
                return ("A's bbox is fully inside B's bbox — "
                        "WARNING: bbox containment ≠ polygon within. Verify with polygon geometry.")
            elif b_in_a:
                return ("B's bbox is fully inside A's bbox — "
                        "WARNING: bbox containment ≠ polygon contains. Verify with polygon geometry.")
            else:
                return ("Bboxes partially overlap — "
                        "WARNING: bbox overlap ≠ polygon overlap. The actual polygons may still be disjoint or only touching.")
        else:
            lat_gap = max(0.0, max(s_a, s_b) - min(n_a, n_b))
            lon_gap = max(0.0, max(w_a, w_b) - min(e_a, e_b))
            gap_deg = max(lat_gap, lon_gap)
            gap_km = gap_deg * 111.0
            if gap_km < 1.0:
                return (f"Bboxes nearly adjacent (gap ≈ {gap_km:.3f} km) — "
                        f"this small gap does NOT confirm disjoint. Administrative boundaries may share exact edges.")
            else:
                return f"Bboxes separated by ≈ {gap_km:.1f} km — entities are likely spatially separate."
    except (ValueError, TypeError, IndexError):
        return "Bbox relationship could not be computed."

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# =====================================================================
# DYNAMIC OSM KNOWLEDGE GRAPH
# =====================================================================
class GeographicKnowledgeGraph:
    """
    Replaces the static JSON graph. Dynamically queries OpenStreetMap (Nominatim)
    to provide coordinates and hierarchies, forcing the LLM to deduce the relation.
    """
    def __init__(self, kg_path: str = "results/osm_cache.json"):
        self.cache_file = kg_path 
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _fetch_osm_data(self, place_name: str) -> Optional[dict]:
        if not place_name or place_name.lower() == "nan":
            return None
        if place_name in self.cache:
            return self.cache[place_name]

        state_hint = _extract_state_hint(place_name)
        query = f"{place_name}, {US_STATE_NAMES[state_hint]}" if state_hint else place_name

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": 5,
            "countrycodes": "us,mx",
        }
        headers = {"User-Agent": "LavalUniversity-GeomaticsPhDEval/1.0"}

        try:
            time.sleep(1.1)
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data:
                preferred_classes = ['boundary', 'place', 'highway', 'waterway', 'natural']
                result = next((r for r in data if r.get('class') in preferred_classes), data[0])
                extracted = {
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "boundingbox": result.get("boundingbox"),
                    "osm_type": result.get("osm_type"),
                    "class": result.get("class"),
                    "type": result.get("type"),
                    "hierarchy": result.get("address", {})
                }
                self.cache[place_name] = extracted
                self._save_cache()
                return extracted
            else:
                self.cache[place_name] = None
                self._save_cache()
                return None
        except Exception as e:
            print(f"⚠️ OSM API Error for '{place_name}': {e}")
            return None

    def fetch(self, place_name: str) -> Optional[dict]:
        """Single-place lookup used by the per-step RAG loop (rag_loop.py)."""
        return self._fetch_osm_data(place_name)

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "", entity: dict = None, log_fn=None) -> str:
        evidence_lines = [f'Sentence: "{sentence}"\n']
        evidence_lines.append("--- OpenStreetMap (OSM) Evidence ---")
        evidence_lines.append("Use the coordinates, bounding boxes, and hierarchy below to deduce the spatial relation. Do not guess.\n")

        data_a = self._fetch_osm_data(place_a)
        data_b = self._fetch_osm_data(place_b)

        def format_entity_evidence(name, data, entity_label):
            if not data:
                return f"{entity_label} ({name}): No OSM data available."

            lines = [f"{entity_label}: {name}"]
            lines.append(f"  • Coordinates: Lat {data.get('lat')}, Lon {data.get('lon')}")

            bbox = data.get('boundingbox')
            if bbox and len(bbox) == 4:
                lines.append(f"  • Bounding Box: [South: {bbox[0]}, North: {bbox[1]}, West: {bbox[2]}, East: {bbox[3]}]")

            lines.append(f"  • Feature Category: {data.get('class')} / {data.get('type')}")

            hierarchy = data.get('hierarchy')
            if hierarchy:
                hier_str = " > ".join([f"{k}: {v}" for k, v in hierarchy.items()])
                lines.append(f"  • Administrative Hierarchy: {hier_str}")

            validation_msg = _validate_entity_context(name, data)
            if validation_msg:
                lines.append(validation_msg)

            return "\n".join(lines)

        evidence_lines.append(format_entity_evidence(place_a, data_a, "Entity A"))
        evidence_lines.append("")
        evidence_lines.append(format_entity_evidence(place_b, data_b, "Entity B"))

        bbox_a = data_a.get('boundingbox') if data_a else None
        bbox_b = data_b.get('boundingbox') if data_b else None
        if bbox_a and bbox_b and len(bbox_a) == 4 and len(bbox_b) == 4:
            bbox_rel = _compute_bbox_relationship(bbox_a, bbox_b)
            evidence_lines.append("\n--- Bounding Box Analysis ---")
            evidence_lines.append(f"  {bbox_rel}")

        # Distance between centroids
        if data_a and data_b:
            try:
                dist = _haversine_km(float(data_a['lat']), float(data_a['lon']),
                                     float(data_b['lat']), float(data_b['lon']))
                if dist < 1.0:
                    dist_hint = f"Distance between centroids: {dist:.3f} km — very close, likely touching or overlapping"
                elif dist < 50.0:
                    dist_hint = f"Distance between centroids: {dist:.1f} km — moderate distance"
                else:
                    dist_hint = f"Distance between centroids: {dist:.1f} km — large distance, likely disjoint"
                evidence_lines.append(f"  {dist_hint}")
            except (TypeError, ValueError):
                pass

        evidence_text = "\n".join(evidence_lines)

        if log_fn:
            log_fn(evidence_text)

        print(f"\n[OSM EVIDENCE Fetched for {place_a} & {place_b}]")
        return evidence_text


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def _build_context_block(entity: Dict[str, Any]) -> str:
    # UPDATED: Safely grab the relationship phrase, prioritizing 'vernacular_relation'
    rel_phrase = entity.get("vernacular_relation") or entity.get("relation_predicate") or entity.get("sentence", "")
    rel_phrase = rel_phrase.strip()

    return (
        f"Entity A: {entity['place_name_subject']} "
        f"(type: {entity.get('placetype_subject', 'place')}, "
        f"geometry: {entity.get('geometry_type_subject', 'unknown')})\n"
        f"Entity B: {entity['place_name_object']} "
        f"(type: {entity.get('placetype_object', 'place')}, "
        f"geometry: {entity.get('geometry_type_object', 'unknown')})\n"
        f"Vernacular: \"{entity['place_name_subject']} {rel_phrase} {entity['place_name_object']}\"\n"
        f"Valid predicates: {VALID_LIST}\n"
    )

def _gather_kg_evidence(kg: GeographicKnowledgeGraph, place_a: str, place_b: str, sentence: str = "", entity: dict = None, log_fn=None) -> str:
    return kg.gather_evidence(place_a, place_b, sentence, entity, log_fn)

def _geometric_weight(text: str) -> float:
    """Score a branch/thought by presence of geometric evidence. Higher = more reliable."""
    geo_keywords = [
        'bbox', 'bounding box', 'coordinates', 'lat', 'lon', 'polygon',
        'linestring', 'geometry', 'degrees', 'km', 'kilometer',
        'south', 'north', 'west', 'east', 'extends into',
        'boundary', 'border', 'intersect', 'interior', 'traverse',
    ]
    text_lower = text.lower()
    hits = sum(1 for kw in geo_keywords if kw in text_lower)
    return 1.0 + 0.3 * min(hits, 5)


def extract_predicate(text: str) -> Optional[str]:
    """Robust extraction of topological predicates using Regex."""
    if not text:
        return None
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)
    text_lower = text_clean.lower()

    patterns = [
        r"answer\s*[:=]\s*\[?(\w+)\]?",
        r"predicate\s*[:=]\s*\[?(\w+)\]?",
        r"conclusion\s*[:=]\s*\[?(\w+)\]?",
        r"suggested\s+predicate\s*[:=]\s*\[?(\w+)\]?",
        r"the\s+(?:relation|predicate|answer)\s+is\s+[:\[]?(\w+)",
        r"therefore[,\s]+(?:the\s+)?(?:answer|predicate|relation)\s+is\s+[:\[]?(\w+)",
        r"final\s+(?:answer|predicate)\s*[:=]\s*\[?(\w+)\]?",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text_lower))
        if matches:
            pred = matches[-1].group(1).strip()
            if pred in VALID_PREDICATES:
                return pred

    found_preds = []
    for p in VALID_PREDICATES:
        idx = text_lower.rfind(p)
        if idx != -1:
            found_preds.append((idx, p))
    if found_preds:
        found_preds.sort(key=lambda x: x[0])
        return found_preds[-1][1]
    return None

def llm_generate(prompt: str, llm: ChatOllama, timeout_seconds: int = 90):
    import threading
    result = {"text": ""}
    def run():
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result["text"] = response.content
        except Exception as e:
            print(f"\n[LLM CRITICAL ERROR]: {str(e)}")
            result["text"] = ""
            
    t = threading.Thread(target=run)
    t.start()
    t.join(timeout_seconds)
    return result["text"]


# =====================================================================
# BASE CLASS
# =====================================================================
class ReasoningStrategy(ABC):
    def __init__(self, kg: GeographicKnowledgeGraph, temperature: float = 0.2, max_new_tokens: int = 1024,
                 base_url: str = BASE_URL, model_name: str = MODEL_NAME, model_fn=None):
        self.kg = kg
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.model_name = model_name
        self.model_fn = model_fn  # optional GPU callable: (prompt: str) -> str
        self.llm = None if model_fn is not None else ChatOllama(
            model=model_name, temperature=temperature, base_url=base_url
        )

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]: ...

    def _generate(self, prompt: str, **kwargs) -> str:
        if self.model_fn is not None:
            return self.model_fn(prompt)
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)
        if max_tokens != self.max_new_tokens or temp != self.temperature:
            llm = ChatOllama(model=self.model_name, temperature=temp, base_url=self.base_url)
            return llm_generate(prompt, llm=llm, timeout_seconds=90)
        return llm_generate(prompt, llm=self.llm, timeout_seconds=90)


# =====================================================================
# 1. CHAIN-OF-THOUGHT (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "CoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a = entity["place_name_subject"]
        place_b = entity["place_name_object"]
        
        # Grab vernacular safely for logging
        rel_phrase = entity.get("vernacular_relation") or entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "CoT", "mode": "dynamic_osm", "steps": []}

        def _log(step_name: str, content: str):
            trace["steps"].append({"step": step_name, "content": content})
            if log_fn:
                log_fn(f"\n  [CoT] ── {step_name} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_phrase}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        _log("OSM_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

{kg_evidence}

Think step-by-step:
1. Check entity validation warnings — if a mismatch is flagged, treat that geometry as unreliable.
2. Check for EQUALS first: does the vernacular explicitly describe identical extents or coextensive boundaries?
3. Read the vernacular carefully for qualifiers like "partly", "extends into", "surrounded by", "enclave".
4. Analyze the Bounding Box Analysis section — note any warnings about bbox vs. polygon.
5. Verify the direction: does your predicate apply A→B or B→A?
6. State your final predicate.

Reasoning:"""

        response = self._generate(prompt)
        _log("LLM_REASONING", response)

        predicate = extract_predicate(response)
        if predicate is None:
            fallback_prompt = f"{context}\n{kg_evidence}\nThe topological relation is:\nAnswer: ["
            fallback_resp = self._generate(fallback_prompt)
            predicate = extract_predicate(fallback_resp)

        trace["prediction"] = predicate
        if log_fn:
            log_fn(f"\n  [CoT] ✅ FINAL PREDICTION: {predicate}")

        return predicate, trace


# =====================================================================
# 2. TREE-OF-THOUGHT (ToT)
# =====================================================================
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "ToT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a = entity["place_name_subject"]
        place_b = entity["place_name_object"]
        rel_phrase = entity.get("vernacular_relation") or entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "ToT", "mode": "dynamic_osm", "branches": [], "vote": None}

        def _log(step: str, content: str):
            if log_fn: log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_phrase}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["osm_evidence"] = kg_evidence
        _log("OSM_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        branch_prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

{kg_evidence}

Explore THREE different reasoning branches for "{place_a} {rel_phrase} {place_b}" using the OSM evidence.
Each branch must:
  - Check for EQUALS first: does the text describe geometrically identical / coextensive entities?
  - Check entity validation warnings before using geometry.
  - Prioritize textual qualifiers ("partly", "extends into", "enclave", "identical perimeter", etc.) over bbox signals.
  - Explicitly verify the A→B direction of the predicate.
  - Note whether reasoning is based on polygon geometry (reliable) or bbox only (use with caution).

Format exactly as:
BRANCH 1: [approach]
[reasoning]
Suggested predicate: [predicate]

BRANCH 2: [approach]
[reasoning]
Suggested predicate: [predicate]

BRANCH 3: [approach]
[reasoning]
Suggested predicate: [predicate]

Begin:"""

        branch_response = self._generate(branch_prompt)
        _log("BRANCHES_RAW", branch_response)

        branch_pattern = r"BRANCH\s+\d+\s*:\s*(.*?)(?=BRANCH\s+\d+|$)"
        branches = re.findall(branch_pattern, branch_response, re.DOTALL | re.IGNORECASE)

        votes = []
        branch_texts = []
        for i, b_text in enumerate(branches):
            pred = extract_predicate(b_text)
            votes.append(pred)
            branch_texts.append(b_text)
            trace["branches"].append({"index": i+1, "predicate": pred, "content": b_text.strip()[:400]})
            _log(f"BRANCH_{i+1}", f"Predicted: {pred}\n{b_text.strip()}")

        if not votes or all(v is None for v in votes):
            final_pred = extract_predicate(self._generate(f"{context}\nAnswer: ["))
        else:
            weighted_scores: Dict[str, float] = {}
            for pred, b_text in zip(votes, branch_texts):
                if pred is None:
                    continue
                weighted_scores[pred] = weighted_scores.get(pred, 0.0) + _geometric_weight(b_text)
            final_pred = max(weighted_scores, key=weighted_scores.get) if weighted_scores else None

        trace["prediction"] = final_pred
        if log_fn: log_fn(f"\n  [ToT] ✅ FINAL PREDICTION: {final_pred}")

        return final_pred, trace


# =====================================================================
# 3. GRAPH-OF-THOUGHT (GoT)
# =====================================================================
@dataclass
class ThoughtNode:
    id: int
    content: str
    predicate: Optional[str] = None
    confidence: float = 0.0
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    node_type: str = "thought"

class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "GoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a = entity["place_name_subject"]
        place_b = entity["place_name_object"]
        rel_phrase = entity.get("vernacular_relation") or entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "mode": "dynamic_osm", "nodes": [], "aggregation": None}

        def _log(step: str, content: str):
            if log_fn: log_fn(f"\n  [GoT] ── {step} ──\n{content}")

        def _add_node(**kwargs) -> ThoughtNode:
            nonlocal next_id
            node = ThoughtNode(id=next_id, **kwargs)
            thought_graph.append(node)
            next_id += 1
            return node

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_phrase}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["osm_evidence"] = kg_evidence
        _log("OSM_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        phase1_prompt = f"""Generate FOUR distinct thought nodes analyzing "{place_a} {rel_phrase} {place_b}".
Analyze the provided OSM coordinates, bounding boxes, and administrative hierarchy.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

{kg_evidence}

Each thought must:
  - Check for EQUALS first: does the text describe geometrically identical / coextensive entities?
  - Check entity validation warnings before using any geometry.
  - Explicitly note if reasoning is based on polygon geometry (reliable) or bbox only (caution).
  - Prioritize textual qualifiers over geometric signals when they conflict.
  - Verify the A→B direction before concluding.

Format:
THOUGHT 1: [angle]
[reasoning]
Predicate: [predicate]

... (up to THOUGHT 4)

Begin:"""

        phase1_resp = self._generate(phase1_prompt)
        _log("PHASE1_RAW", phase1_resp)

        thought_pattern = r"THOUGHT\s+\d+\s*:\s*(.*?)(?=THOUGHT\s+\d+|$)"
        thoughts = re.findall(thought_pattern, phase1_resp, re.DOTALL | re.IGNORECASE)

        for t_text in thoughts:
            pred = extract_predicate(t_text)
            weight = _geometric_weight(t_text)
            _add_node(content=t_text.strip()[:500], predicate=pred, confidence=weight)

        weighted_scores: Dict[str, float] = {}
        for node in thought_graph:
            if node.predicate:
                weighted_scores[node.predicate] = weighted_scores.get(node.predicate, 0.0) + node.confidence
        final_pred = max(weighted_scores, key=weighted_scores.get) if weighted_scores else None

        # Fallback 1: model didn't emit THOUGHT N: headers — scan full phase1 response.
        if final_pred is None:
            final_pred = extract_predicate(phase1_resp)
            _log("FALLBACK1", f"thought-scan failed, extracted from full response: {final_pred}")

        # Fallback 2: still None — ask directly for the answer.
        if final_pred is None:
            synth_prompt = (
                f"{context}\n{kg_evidence}\n"
                "Based on the spatial evidence above, the DE-9IM topological predicate is:\n"
                "Answer: ["
            )
            synth_resp = self._generate(synth_prompt)
            final_pred = extract_predicate(synth_resp)
            _log("FALLBACK2", f"synthesis response: {synth_resp[:200]} → {final_pred}")

        trace["prediction"] = final_pred
        if log_fn: log_fn(f"\n  [GoT] ✅ FINAL PREDICTION: {final_pred}")

        return final_pred, trace


# =====================================================================
# STRATEGY FACTORY
# =====================================================================
STRATEGY_MAP = {
    "cot": ChainOfThought,
    "tot": TreeOfThought,
    "got": GraphOfThought,
}

def get_strategy(name: str, kg: GeographicKnowledgeGraph, model_fn=None, **kwargs) -> ReasoningStrategy:
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name_lower](kg, model_fn=model_fn, **kwargs)