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
from collections import Counter

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


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

VERNACULAR_LEXICON = """Vernacular-to-Topology Reference (one example each):
  WITHIN    — e.g. "is in"         (A is inside B)
  CONTAINS  — e.g. "is home to"    (A encloses B)
  TOUCHES   — e.g. "borders"       (A and B share a boundary, no overlap)
  CROSSES   — e.g. "passes through"(A traverses B)
  OVERLAPS  — e.g. "overlaps with" (A and B partially share area)
  DISJOINT  — e.g. "is far from"   (A and B are completely separate)
  EQUALS    — e.g. "is the same as"(A and B occupy the same space)

Note: geometry types constrain possible relations.
"""

RULES_BLOCK = """Rules:
1. The relation is DIRECTED: A [predicate] B.
2. Consider geometry types (Point, LineString, Polygon, MultiPolygon).
3. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps, equals.
4. Carefully interpret the vernacular expression.
5. Use the provided OpenStreetMap coordinates, bounding boxes, and administrative hierarchy to deduce the spatial relationship. 
6. End with: Answer: [predicate]
"""

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

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place_name,
            "format": "json",
            "addressdetails": 1,
            "limit": 1
        }
        headers = {"User-Agent": "LavalUniversity-GeomaticsPhDEval/1.0"}

        try:
            time.sleep(1.1) 
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data:
                result = data[0]
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
                
            return "\n".join(lines)

        evidence_lines.append(format_entity_evidence(place_a, data_a, "Entity A"))
        evidence_lines.append("")
        evidence_lines.append(format_entity_evidence(place_b, data_b, "Entity B"))

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
        r"conclusion\s*[:=]\s*(\w+)",
        r"suggested\s+predicate\s*[:=]\s*(\w+)"
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

{context}

{kg_evidence}

Think step-by-step. Analyze the coordinates, bounding boxes, and hierarchical relationship to determine the best topological predicate.

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

{context}

{kg_evidence}

Explore THREE different reasoning branches for "{place_a} {rel_phrase} {place_b}" using the OSM coordinates and hierarchy.

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
        for i, b_text in enumerate(branches):
            pred = extract_predicate(b_text)
            votes.append(pred)
            trace["branches"].append({"index": i+1, "predicate": pred, "content": b_text.strip()[:400]})
            _log(f"BRANCH_{i+1}", f"Predicted: {pred}\n{b_text.strip()}")

        if not votes or all(v is None for v in votes):
            final_pred = extract_predicate(self._generate(f"{context}\nAnswer: ["))
        else:
            counter = Counter([v for v in votes if v])
            final_pred = counter.most_common(1)[0][0] if counter else None

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

{context}

{kg_evidence}

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
            _add_node(content=t_text.strip()[:500], predicate=pred, confidence=1.0 if pred else 0.0)

        all_preds = [n.predicate for n in thought_graph if n.predicate]
        final_pred = Counter(all_preds).most_common(1)[0][0] if all_preds else None

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