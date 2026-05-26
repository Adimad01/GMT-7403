"""
reasoning_strategies_neighborhood_details_spatial_relation.py
=================================================================================
Uses the static Geographic Knowledge Graph (knowledge_graph.json)
"""

import re
import json
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

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

2. TEXTUAL QUALIFIER PRIORITY (override graph evidence when contradictory):
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

4. FULL LINESTRING REASONING:
   — For rivers and roads, reason on the entire geometry, not just the centroid.
   — A river centroid may fall outside a city even if the river traverses it.

5. PARTIAL COUNT ≠ CONTAINMENT:
   — "X inhabitants in county Y" means part of the city is in county Y → overlaps, not within.

6. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps.
7. Use the knowledge graph evidence to support your reasoning.
8. End with: Answer: [predicate]
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
# STATIC KNOWLEDGE GRAPH
# =====================================================================
class GeographicKnowledgeGraph:

    def __init__(self, kg_path: str = "results/knowledge_graph.json"):
        self.kg_path = kg_path
        self.nodes: Dict = {}
        self.links: List[Dict] = []
        self._load_kg()

    def _load_kg(self):
        if not os.path.exists(self.kg_path):
            print("⚠️ KG file not found")
            return

        with open(self.kg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {n["id"]: n for n in data.get("nodes", [])}
        self.links = data.get("links", [])

        print(f"✅ Static KG loaded: {len(self.nodes)} nodes, {len(self.links)} links")

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "", entity: dict = None, log_fn=None) -> str:
        evidence_lines = [f'Sentence: "{sentence}"']

        direct_links = [
            link for link in self.links
            if (link["source"] == place_a and link["target"] == place_b)
            or (link["source"] == place_b and link["target"] == place_a)
        ]

        if direct_links:
            evidence_lines.append("Direct relations found:")
            for link in sorted(direct_links, key=lambda x: (x["source"], x["target"]))[:8]:
                pred = link.get("predicate") or link.get("spatial_logic") or link.get("vernacular", "")
                evidence_lines.append(f"{link['source']} →[{pred}]→ {link['target']}")
        else:
            evidence_lines.append("No direct relation found.")

        def build_neighborhood(center):
            neighbors = []
            for link in self.links:
                if link["source"] == center:
                    neighbor = link["target"]
                    vernacular = link.get("vernacular", "")
                    spatial_logic = link.get("spatial_logic", "unknown")
                    predicate = link.get("predicate", "")
                elif link["target"] == center:
                    neighbor = link["source"]
                    vernacular = link.get("vernacular", "")
                    spatial_logic = link.get("spatial_logic", "unknown")
                    predicate = link.get("predicate", "")
                else:
                    continue

                node = self.nodes.get(neighbor, {})
                neighbors.append((
                    neighbor,
                    node.get("placetype", "unknown"),
                    node.get("geometry", "unknown"),
                    vernacular,
                    spatial_logic,
                    predicate
                ))

            neighbors = sorted(neighbors, key=lambda x: x[0])[:10]
            return [
                f"- {name} | type={ptype} | geometry={geom} | vernacular={vern} | logic={logic}"
                for name, ptype, geom, vern, logic, pred in neighbors
            ]

        neigh_a = build_neighborhood(place_a)
        neigh_b = build_neighborhood(place_b)

        evidence_lines.append(f"\nNeighborhood of A ({place_a}):")
        evidence_lines.extend(neigh_a if neigh_a else ["none"])

        evidence_lines.append(f"\nNeighborhood of B ({place_b}):")
        evidence_lines.extend(neigh_b if neigh_b else ["none"])

        node_a = self.nodes.get(place_a, {})
        node_b = self.nodes.get(place_b, {})

        if node_a:
            evidence_lines.append(f"\nA ({place_a}) | type={node_a.get('placetype')} | geometry={node_a.get('geometry')}")
        if node_b:
            evidence_lines.append(f"B ({place_b}) | type={node_b.get('placetype')} | geometry={node_b.get('geometry')}")

        evidence_text = "\n".join(evidence_lines)

        if log_fn:
            log_fn(evidence_text)

        print(f"\n[KG EVIDENCE for {place_a} & {place_b}]:\n{evidence_text}{'...' if len(evidence_text) > 1000 else ''}")
        return evidence_text


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def _build_context_block(entity: Dict[str, Any]) -> str:
    return (
        f"Entity A: {entity['place_name_subject']} "
        f"(type: {entity.get('placetype_subject', 'place')}, "
        f"geometry: {entity.get('geometry_type_subject', 'unknown')})\n"
        f"Entity B: {entity['place_name_object']} "
        f"(type: {entity.get('placetype_object', 'place')}, "
        f"geometry: {entity.get('geometry_type_object', 'unknown')})\n"
        f"Vernacular: \"{entity['place_name_subject']} "
        f"{entity.get('relation_predicate', '')} "
        f"{entity['place_name_object']}\"\n"
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
            # We now print the error to terminal so you know if the LLM server failed
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
        
        # ✅ FIX: Removed the invalid keyword arguments here
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
        rel_pred = entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "CoT", "mode": "static_kg", "steps": []}

        def _log(step_name: str, content: str):
            trace["steps"].append({"step": step_name, "content": content})
            if log_fn:
                log_fn(f"\n  [CoT] ── {step_name} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_pred}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        _log("KG_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
{kg_evidence}

Think step-by-step:
1. Read the vernacular carefully for qualifiers: "partly", "extends into", "enclave", "surrounded by".
2. Use neighborhood links as context, but do not let graph hierarchy alone override textual signals.
3. For LineString entities (rivers, roads), reason on the full geometry, not just the centroid.
4. Verify the A→B direction before concluding.

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
        rel_pred = entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "ToT", "mode": "static_kg", "branches": [], "vote": None}

        def _log(step: str, content: str):
            if log_fn: log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_pred}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["kg_evidence"] = kg_evidence
        _log("KG_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        branch_prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
{kg_evidence}

Explore THREE different reasoning branches for "{place_a} {rel_pred} {place_b}".
Each branch must:
  - Prioritize textual qualifiers ("partly", "extends into", "enclave", etc.) over graph hierarchy.
  - For LineString entities, reason on the full geometry, not the centroid.
  - Explicitly verify the A→B direction of the predicate.
  - Note whether reasoning is based on explicit geometry (reliable) or administrative hierarchy only (use with caution).

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
        rel_pred = entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "mode": "static_kg", "nodes": [], "aggregation": None}

        def _log(step: str, content: str):
            if log_fn: log_fn(f"\n  [GoT] ── {step} ──\n{content}")

        def _add_node(**kwargs) -> ThoughtNode:
            nonlocal next_id
            node = ThoughtNode(id=next_id, **kwargs)
            thought_graph.append(node)
            next_id += 1
            return node

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_pred}\"")
        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["kg_evidence"] = kg_evidence
        _log("KG_EVIDENCE", kg_evidence)
        context = _build_context_block(entity)

        phase1_prompt = f"""Generate FOUR distinct thought nodes analyzing "{place_a} {rel_pred} {place_b}".

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{DISAMBIGUATION_TABLE}

{DIRECTIONAL_RULE}

{OVERLAPS_DETECTION}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
{kg_evidence}

Each thought must:
  - Prioritize textual qualifiers ("partly", "extends into", "enclave", etc.) over graph hierarchy.
  - For LineString entities, reason on the full geometry, not the centroid.
  - Explicitly verify the A→B direction before concluding.
  - Note whether reasoning is based on explicit geometry (reliable) or administrative hierarchy only (use with caution).

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