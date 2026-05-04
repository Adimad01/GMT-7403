"""
reasoning_strategies.py — Static KG Reasoning Strategies
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
from collections import Counter

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# =====================================================================
# OLLAMA CONFIG
# =====================================================================
BASE_URL = os.getenv("OLLAMA_URL", "http://ollama.apps.crdig.ulaval.ca")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-oss")


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
5. Consider what makes sense given the place types and geometry types involved.
6. Use the knowledge graph evidence to support your reasoning.
7. End with: Answer: [predicate]
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

        print(
            f"✅ Static KG loaded: {len(self.nodes)} nodes, {len(self.links)} links"
        )

    # =================================================================
    # ⭐⭐⭐ ENRICHED GEOAI EVIDENCE ⭐⭐⭐
    # =================================================================
    def gather_evidence(
        self,
        place_a: str,
        place_b: str,
        sentence: str = "",
        entity: dict = None,
        log_fn=None,
    ) -> str:

        evidence_lines = [f'Sentence: "{sentence}"']

        # ---------------- DIRECT LINKS ----------------
        direct_links = [
            link for link in self.links
            if (link["source"] == place_a and link["target"] == place_b)
            or (link["source"] == place_b and link["target"] == place_a)
        ]

        if direct_links:
            evidence_lines.append("Direct relations found:")
            for link in sorted(direct_links, key=lambda x: (x["source"], x["target"]))[:8]:
                pred = (
                    link.get("predicate")
                    or link.get("spatial_logic")
                    or link.get("vernacular", "")
                )
                evidence_lines.append(
                    f"{link['source']} →[{pred}]→ {link['target']}"
                )
        else:
            evidence_lines.append("No direct relation found.")

        # ---------------- NEIGHBORHOOD BUILDER ----------------
        def build_neighborhood(center):

            neighbors = []

            for link in self.links:

                if link["source"] == center:
                    neighbor = link["target"]
                    relation = link.get("predicate") or link.get("spatial_logic", "")

                elif link["target"] == center:
                    neighbor = link["source"]
                    relation = link.get("predicate") or link.get("spatial_logic", "")

                else:
                    continue

                node = self.nodes.get(neighbor, {})

                neighbors.append(
                    (
                        neighbor,
                        node.get("placetype", "unknown"),
                        node.get("geometry", "unknown"),
                        relation,
                    )
                )

            # deterministic ordering
            neighbors = sorted(neighbors, key=lambda x: x[0])[:10]

            return [
                f"- {n} | type={t} | geometry={g} | relation={r}"
                for n, t, g, r in neighbors
            ]

        # ---------------- NEIGHBORHOODS ----------------
        neigh_a = build_neighborhood(place_a)
        neigh_b = build_neighborhood(place_b)

        evidence_lines.append(f"\nNeighborhood of A ({place_a}):")
        evidence_lines.extend(neigh_a if neigh_a else ["none"])

        evidence_lines.append(f"\nNeighborhood of B ({place_b}):")
        evidence_lines.extend(neigh_b if neigh_b else ["none"])

        # ---------------- NODE INFO ----------------
        node_a = self.nodes.get(place_a, {})
        node_b = self.nodes.get(place_b, {})

        if node_a:
            evidence_lines.append(
                f"\nA ({place_a}) | type={node_a.get('placetype')} | geometry={node_a.get('geometry')}"
            )

        if node_b:
            evidence_lines.append(
                f"B ({place_b}) | type={node_b.get('placetype')} | geometry={node_b.get('geometry')}"
            )

        evidence_text = "\n".join(evidence_lines)

        if log_fn:
            log_fn(evidence_text)

        return evidence_text


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def extract_predicate(text: str) -> Optional[str]:
    if not text:
        return None
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)
    text_lower = text_clean.lower()

    patterns = [
        r"answer\s*[:=]\s*\[?(\w+)\]?",
        r"predicate\s*[:=]\s*\[?(\w+)\]?",
        r"conclusion\s*[:=]\s*(\w+)",
        r"suggested\s+predicate\s*[:=]\s*(\w+)",
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


def llm_generate(prompt: str,
                 llm: ChatOllama,
                 timeout_seconds: int = 90):

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


def _gather_kg_evidence(
    kg: "GeographicKnowledgeGraph",
    place_a: str,
    place_b: str,
    sentence: str = "",
    entity: dict = None,
    log_fn=None,
) -> str:
    return kg.gather_evidence(place_a, place_b, sentence, entity, log_fn)


# =====================================================================
# BASE CLASS
# =====================================================================
class ReasoningStrategy(ABC):
    """Base class for all reasoning strategies using static KG."""

    def __init__(self, kg: GeographicKnowledgeGraph,
                 temperature: float = 0.2,
                 max_new_tokens: int = 1024,
                 base_url: str = BASE_URL,
                 model_name: str = MODEL_NAME):
        self.kg = kg
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.model_name = model_name

        # Reusable LLM instance
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        ...

    def _generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)

        if max_tokens != self.max_new_tokens or temp != self.temperature:
            llm = ChatOllama(
                model=self.model_name,
                temperature=temp,
                base_url=self.base_url,
            )
            return llm_generate(prompt, llm=llm, timeout_seconds=90)
        return llm_generate(prompt, llm=self.llm, timeout_seconds=90)


# =====================================================================
# 1. CHAIN-OF-THOUGHT (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "CoT"

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
        print(f"  [CoT] KG Evidence:\n{kg_evidence}")
        _log("KG_EVIDENCE", kg_evidence)

        context = _build_context_block(entity)

        prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
{kg_evidence}

Think step-by-step and determine the best topological predicate.

Reasoning:"""

        response = self._generate(prompt)
        _log("LLM_REASONING", response)

        predicate = extract_predicate(response)

        # Fallback if no predicate found
        if predicate is None:
            fallback_prompt = f"{context}\n{kg_evidence}\nThe topological relation is:\nAnswer: ["
            fallback_resp = self._generate(fallback_prompt, max_new_tokens=150)
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
    def name(self) -> str:
        return "ToT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a = entity["place_name_subject"]
        place_b = entity["place_name_object"]
        rel_pred = entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        trace = {"strategy": "ToT", "mode": "static_kg", "branches": [], "vote": None}

        def _log(step: str, content: str):
            if log_fn:
                log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        _log("INPUT", f"A: {place_a} | B: {place_b} | Vernacular: \"{rel_pred}\"")

        kg_evidence = _gather_kg_evidence(self.kg, place_a, place_b, sentence, entity, log_fn)
        trace["kg_evidence"] = kg_evidence
        _log("KG_EVIDENCE", kg_evidence)

        context = _build_context_block(entity)

        branch_prompt = f"""You are an expert in geospatial topological reasoning.

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
{kg_evidence}

Explore THREE different reasoning branches for "{place_a} {rel_pred} {place_b}".

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

        # Parse branches
        branch_pattern = r"BRANCH\s+\d+\s*:\s*(.*?)(?=BRANCH\s+\d+|$)"
        branches = re.findall(branch_pattern, branch_response, re.DOTALL | re.IGNORECASE)

        votes = []
        for i, b_text in enumerate(branches):
            pred = extract_predicate(b_text)
            votes.append(pred)
            trace["branches"].append({"index": i+1, "predicate": pred, "content": b_text.strip()[:400]})
            _log(f"BRANCH_{i+1}", f"Predicted: {pred}\n{b_text.strip()}")

        # Voting
        if not votes or all(v is None for v in votes):
            final_pred = extract_predicate(self._generate(f"{context}\nAnswer: [", max_new_tokens=100))
        else:
            counter = Counter([v for v in votes if v])
            final_pred = counter.most_common(1)[0][0] if counter else None

        trace["prediction"] = final_pred
        if log_fn:
            log_fn(f"\n  [ToT] ✅ FINAL PREDICTION: {final_pred}")

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
    def name(self) -> str:
        return "GoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a = entity["place_name_subject"]
        place_b = entity["place_name_object"]
        rel_pred = entity.get("relation_predicate", "")
        sentence = entity.get("sentence", "")

        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "mode": "static_kg", "nodes": [], "aggregation": None}

        def _log(step: str, content: str):
            if log_fn:
                log_fn(f"\n  [GoT] ── {step} ──\n{content}")

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

        # Phase 1: Generate initial thoughts
        phase1_prompt = f"""Generate FOUR distinct thought nodes analyzing "{place_a} {rel_pred} {place_b}".

{VERNACULAR_LEXICON}

{RULES_BLOCK}

{context}

--- KNOWLEDGE GRAPH EVIDENCE ---
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

        # Simplified final aggregation (for stability)
        all_preds = [n.predicate for n in thought_graph if n.predicate]
        final_pred = Counter(all_preds).most_common(1)[0][0] if all_preds else None

        trace["prediction"] = final_pred
        if log_fn:
            log_fn(f"\n  [GoT] ✅ FINAL PREDICTION: {final_pred}")

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
    """Factory function to create reasoning strategy."""
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name_lower](kg, **kwargs)