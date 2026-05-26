"""
reasoning_strategies_neighborhood_details_spatial_relation_Wikidata.py
=================================================================================
Uses dynamic Wikidata APIs (MediaWiki Search + SPARQL) to provide semantic 
and spatial evidence without explicitly providing the topological relation.
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
    "contains", "overlaps",
}

VALID_LIST = "contains, within, touches, crosses, disjoint, overlaps"

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

WARNING — ambiguous vernacular (read carefully):
  "county seat of" → does NOT imply within — the city may extend into adjacent counties → overlaps
  "suburb of"      → does NOT imply touches — suburb may be separated from the main city → disjoint
  "surrounded by"  → may be touches (partial encirclement), not within (full containment)
  "enclave of"     → touches (shares boundary), NOT within (not inside interior)
  "along" + river/road → likely crosses (full LineString traversal), not just touches
"""

RULES_BLOCK = """Rules:
1. DIRECTION CHECK (mandatory before answering):
   — "contains" means A encloses B; "within" means A is inside B.
   — Explicitly ask: does this predicate apply A→B or B→A?
   — If the data shows B encloses A, the answer is "within", NOT "contains".

2. TEXTUAL QUALIFIER PRIORITY (override semantic signals when contradictory):
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

4. WIKIDATA COORDINATE CAUTION:
   — Wikidata provides only point coordinates (centroid), not polygon geometry.
   — Do NOT infer overlaps, within, or touches from coordinates alone.
   — If no polygon geometry is available, rely on the vernacular and description text.

5. ENTITY VALIDATION:
   — Verify the retrieved Wikidata entity matches the expected state/region.
   — If the description or "Located in" field does not match the context, treat its data as unreliable.

6. FULL LINESTRING REASONING:
   — For rivers and roads, reason on the entire geometry, not just the centroid.
   — A river centroid may fall outside a city even if the river traverses it.

7. PARTIAL COUNT ≠ CONTAINMENT:
   — "X inhabitants in county Y" means part of the city is in county Y → overlaps, not within.
   — A partial population count in a region implies the rest is elsewhere.

8. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps.
9. End with: Answer: [predicate]
"""

DISAMBIGUATION_TABLE = """
CRITICAL DISTINCTIONS (most confused predicates):

touches vs overlaps:
  touches  -> A and B share ONLY their boundary. Interiors do NOT intersect.
             Example: two adjacent counties sharing a border line.
  overlaps -> A and B share SOME interior area. Part of A is inside B, part is outside.
             Example: a city that partially crosses a county boundary.

touches vs crosses:
  touches  -> boundary contact only, no traversal through interior.
  crosses  -> A passes THROUGH B, entering AND exiting its interior.
             Requires different geometry dimensions (LineString through Polygon).
             Example: a highway that enters and exits a city polygon.

contains vs within (DIRECTION IS CRITICAL):
  contains -> A is the CONTAINER. B is fully INSIDE A. Ask: "does A surround B?"
  within   -> A is INSIDE B. Ask: "is A surrounded by B?"
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
3-STEP CHECK to detect OVERLAPS (do this before concluding within or touches):
  Step 1: Is A completely inside B? -> within (not overlaps)
  Step 2: Is B completely inside A? -> contains (not overlaps)
  Step 3: Do A and B share SOME area but NEITHER fully contains the other? -> overlaps
  Strong overlaps signals: "extends into", "partly in", "straddles",
                           "part of population in", "county seat of" (city crosses county line).
"""

# =====================================================================
# ENTITY VALIDATION HELPERS (Recommendation 1 & 6)
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
    parts = place_name.strip().split()
    if parts and parts[-1].upper() in US_STATE_NAMES:
        return parts[-1].upper()
    return None

def _validate_wikidata_entity_context(place_name: str, wd_data: dict) -> str:
    """Verify the Wikidata result matches the expected state/region from the place name."""
    if not wd_data:
        return ""
    state_hint = _extract_state_hint(place_name)
    if not state_hint:
        return ""
    state_name = US_STATE_NAMES[state_hint]
    full_text = " ".join([
        " ".join(wd_data.get('located_in', [])),
        wd_data.get('description', ''),
        wd_data.get('label', ''),
    ]).lower()
    if state_name.lower() in full_text or state_hint.lower() in full_text:
        return f"  ✓ Entity confirmed in expected region ({state_name})"
    return (
        f"  ⚠️ ENTITY MISMATCH: Expected {state_name} ({state_hint}) not found in Wikidata data "
        f"— this entity may be from a different region. Treat its data as UNRELIABLE."
    )

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# =====================================================================
# DYNAMIC WIKIDATA KNOWLEDGE GRAPH
# =====================================================================
class GeographicKnowledgeGraph:
    """
    Dynamically queries Wikidata to provide semantic context (descriptions, 
    instance of, located in) and coordinates to force LLM deduction.
    """
    def __init__(self, kg_path: str = "results/wikidata_cache.json"):
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

    def _fetch_wikidata_data(self, place_name: str) -> Optional[dict]:
        if not place_name or place_name.lower() == "nan":
            return None
            
        if place_name in self.cache:
            return self.cache[place_name]

        headers = {"User-Agent": "LavalUniversity-GeomaticsPhDEval/1.0"}

        try:
            # STEP 1: Search Entity by name
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                "action": "wbsearchentities",
                "search": place_name,
                "language": "en",
                "format": "json",
                "limit": 5
            }
            time.sleep(1.1)
            search_res = requests.get(search_url, params=search_params, headers=headers)
            search_res.raise_for_status()
            search_data = search_res.json()

            if not search_data.get("search"):
                self.cache[place_name] = None
                self._save_cache()
                return None

            geo_keywords = ["city", "town", "village", "county", "river", "lake", "highway",
                            "road", "state", "municipality", "settlement", "census", "parish"]
            search_results = search_data["search"]
            entity = next(
                (r for r in search_results if any(kw in r.get("description", "").lower() for kw in geo_keywords)),
                search_results[0]
            )
            q_id = entity.get("id")
            description = entity.get("description", "No description available")
            label = entity.get("label", place_name)

            # STEP 2: Fetch precise properties via SPARQL
            sparql_url = "https://query.wikidata.org/sparql"
            query = f"""
            SELECT ?lat ?lon ?instanceLabel ?locatedInLabel WHERE {{
              OPTIONAL {{
                wd:{q_id} p:P625 ?statement .
                ?statement psv:P625 ?coordinate_node .
                ?coordinate_node wikibase:geoLatitude ?lat .
                ?coordinate_node wikibase:geoLongitude ?lon .
              }}
              OPTIONAL {{ wd:{q_id} wdt:P31 ?instance . ?instance rdfs:label ?instanceLabel . FILTER(LANG(?instanceLabel) = "en") }}
              OPTIONAL {{ wd:{q_id} wdt:P131 ?locatedIn . ?locatedIn rdfs:label ?locatedInLabel . FILTER(LANG(?locatedInLabel) = "en") }}
            }} LIMIT 15
            """
            sparql_res = requests.get(sparql_url, params={"query": query, "format": "json"}, headers=headers)
            sparql_res.raise_for_status()
            sparql_data = sparql_res.json()

            bindings = sparql_data.get("results", {}).get("bindings", [])

            # Aggregate SPARQL rows (since entities can have multiple instances/locations)
            instances = set()
            locations = set()
            lat, lon = None, None

            for b in bindings:
                if "lat" in b and lat is None:
                    lat = b["lat"]["value"]
                if "lon" in b and lon is None:
                    lon = b["lon"]["value"]
                if "instanceLabel" in b:
                    instances.add(b["instanceLabel"]["value"])
                if "locatedInLabel" in b:
                    locations.add(b["locatedInLabel"]["value"])

            extracted = {
                "q_id": q_id,
                "label": label,
                "description": description,
                "lat": lat,
                "lon": lon,
                "instances": list(instances),
                "located_in": list(locations)
            }

            self.cache[place_name] = extracted
            self._save_cache()
            return extracted
                
        except Exception as e:
            print(f"⚠️ Wikidata API Error for '{place_name}': {e}")
            return None

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "", entity: dict = None, log_fn=None) -> str:
        evidence_lines = [f'Sentence: "{sentence}"\n']
        evidence_lines.append("--- Wikidata Evidence ---")
        evidence_lines.append("Use the descriptions, coordinates, and hierarchical categories below to deduce the spatial relation. Do not guess.\n")

        data_a = self._fetch_wikidata_data(place_a)
        data_b = self._fetch_wikidata_data(place_b)

        def format_entity_evidence(name, data, entity_label):
            if not data:
                return f"{entity_label} ({name}): No Wikidata information available."

            lines = [f"{entity_label}: {name}"]
            lines.append(f"  • Entity: {data.get('label')} ({data.get('q_id')})")
            lines.append(f"  • Description: {data.get('description')}")

            if data.get('lat') and data.get('lon'):
                lines.append(f"  • Coordinates (centroid only — not polygon): Lat {data.get('lat')}, Lon {data.get('lon')}")

            if data.get('instances'):
                lines.append(f"  • Instance of: {', '.join(data.get('instances'))}")

            if data.get('located_in'):
                lines.append(f"  • Administrative hierarchy (P131): {' -> '.join(data.get('located_in'))}")
                lines.append(f"    (If A appears in B's P131 chain -> within; if B appears in A's chain -> contains)")

            validation_msg = _validate_wikidata_entity_context(name, data)
            if validation_msg:
                lines.append(validation_msg)

            return "\n".join(lines)

        evidence_lines.append(format_entity_evidence(place_a, data_a, "Entity A"))
        evidence_lines.append("")
        evidence_lines.append(format_entity_evidence(place_b, data_b, "Entity B"))

        # Distance between centroids
        if data_a and data_b:
            try:
                dist = _haversine_km(float(data_a.get('lat')), float(data_a.get('lon')),
                                     float(data_b.get('lat')), float(data_b.get('lon')))
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

        print(f"\n[WIKIDATA EVIDENCE Fetched for {place_a} & {place_b}]")
        return evidence_text

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def _build_context_block(entity: Dict[str, Any]) -> str:
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
        'coordinates', 'lat', 'lon', 'polygon', 'linestring', 'geometry',
        'degrees', 'km', 'kilometer', 'extends into', 'boundary', 'border',
        'intersect', 'interior', 'traverse', 'located in', 'description',
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
    def __init__(self, kg: GeographicKnowledgeGraph, temperature: float = 0.2, max_new_tokens: int = 1024, base_url: str = BASE_URL, model_name: str = MODEL_NAME):
        self.kg = kg
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]: ...

    def _generate(self, prompt: str, **kwargs) -> str:
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
        rel_phrase = entity.get("vernacular_relation") or entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "CoT", "mode": "dynamic_wikidata", "steps": []}

        def _log(step_name: str, content: str):
            trace["steps"].append({"step": step_name, "content": content})
            if log_fn: log_fn(f"\n  [CoT] ── {step_name} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_phrase}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        _log("WIKIDATA_EVIDENCE", kg_evidence)
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
1. Check entity validation warnings — if a mismatch is flagged, treat that data as unreliable.
2. Read the vernacular carefully for qualifiers: "partly", "extends into", "enclave", "surrounded by".
3. Remember: Wikidata provides only centroid coordinates, NOT polygon geometry — do not infer containment from coordinates alone.
4. If a partial population count appears in the description, interpret it as overlaps, not within.
5. Verify the A→B direction before concluding.

Reasoning:"""

        response = self._generate(prompt)
        _log("LLM_REASONING", response)

        predicate = extract_predicate(response)
        if predicate is None:
            fallback_prompt = f"{context}\n{kg_evidence}\nThe topological relation is:\nAnswer: ["
            fallback_resp = self._generate(fallback_prompt)
            predicate = extract_predicate(fallback_resp)

        trace["prediction"] = predicate
        if log_fn: log_fn(f"\n  [CoT] ✅ FINAL PREDICTION: {predicate}")

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

        trace = {"strategy": "ToT", "mode": "dynamic_wikidata", "branches": [], "vote": None}

        def _log(step: str, content: str):
            if log_fn: log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_phrase}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["wikidata_evidence"] = kg_evidence
        _log("WIKIDATA_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        branch_prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

{kg_evidence}

Explore THREE different reasoning branches for "{place_a} {rel_phrase} {place_b}" using the Wikidata evidence.
Each branch must:
  - Check entity validation warnings before using any data.
  - Prioritize textual qualifiers ("partly", "extends into", "enclave", etc.) over coordinate signals.
  - Remember that Wikidata provides only centroid coordinates — do not infer containment from proximity alone.
  - Explicitly verify the A→B direction of the predicate.

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
        trace = {"strategy": "GoT", "mode": "dynamic_wikidata", "nodes": [], "aggregation": None}

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
        trace["wikidata_evidence"] = kg_evidence
        _log("WIKIDATA_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        phase1_prompt = f"""Generate FOUR distinct thought nodes analyzing "{place_a} {rel_phrase} {place_b}".
Analyze the provided Wikidata semantic hierarchies and descriptions.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

{kg_evidence}

Each thought must:
  - Check entity validation warnings before using any data.
  - Explicitly note if reasoning relies on description text (preferred) or coordinates only (caution).
  - Prioritize textual qualifiers over proximity-based inference.
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

def get_strategy(name: str, kg: GeographicKnowledgeGraph, **kwargs) -> ReasoningStrategy:
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name_lower](kg, **kwargs)