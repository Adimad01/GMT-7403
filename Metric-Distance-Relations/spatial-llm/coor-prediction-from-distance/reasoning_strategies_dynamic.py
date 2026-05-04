"""
reasoning_strategies_dynamic.py — Volet C Reasoning Strategies 
=========================================================================
Implements CoT, ToT, and GoT strategies for estimating metric distance 
using pure coordinate extraction from the Dynamic OSM Knowledge Graph.
"""

import re
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from geographic_kg_dynamic import DynamicMetricKnowledgeGraph

BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"
MAX_EARTH_DISTANCE_KM = 20_100

def extract_distance(text: str) -> Optional[float]:
    if not text or not text.strip(): return None
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)

    def _clean(raw: str) -> Optional[float]:
        cleaned = re.sub(r"[,\s\u00a0\u202f]", "", raw)
        try:
            val = float(cleaned)
            if 1.0 <= val <= MAX_EARTH_DISTANCE_KM: return val
        except ValueError: pass
        return None

    patterns = [
        r"(?:final\s+)?(?:estimate|answer|distance)\s*[:=]\s*~?\s*([\d][\d,\s]*\.?\d*)\s*(?:km|kilometers)?",
        r"(?:approximately|about|roughly|≈|~)\s*([\d][\d,\s]*\.?\d*)\s*km",
        r"([\d][\d,\s]*\.?\d*)\s*(?:km|kilometers)"
    ]
    
    for pat in patterns:
        matches = list(re.finditer(pat, text_clean, re.IGNORECASE))
        if matches:
            val = _clean(matches[-1].group(1))
            if val is not None: return round(val, 1)

    all_nums = list(re.finditer(r"([\d][\d,\s]*\.?\d*)", text_clean))
    for m in reversed(all_nums):
        val = _clean(m.group(1))
        if val is not None: return round(val, 1)
    return None

def llm_generate(prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
    url = f"{BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_new_tokens}}
    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            time.sleep(3 * attempt)
    return ""

TASK_DESCRIPTION = """You are an expert geographer estimating distance.
Task: Estimate the exact distance in km between City A and City B.
You must output a single number.
End your response EXACTLY with: Final estimate: <number> km"""

RULES_BLOCK = """Rules:
1. Use the provided latitude and longitude coordinates to calculate or estimate the great-circle (Haversine) distance.
2. Consider the geographic state or region to sanity-check your mathematical result.
3. Provide your estimates purely as numbers in km."""

class ReasoningStrategy(ABC):
    def __init__(self, kg: DynamicMetricKnowledgeGraph, temp: float = 0.1, max_tokens: int = 2048):
        self.kg, self.temp, self.max_tokens = kg, temp, max_tokens
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def reason(self, city_a: str, city_b: str, evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]: ...
    def _generate(self, prompt: str) -> str: return llm_generate(prompt, self.max_tokens, self.temp)


# =====================================================================
# 1 — CHAIN-OF-THOUGHT (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "CoT"
    def reason(self, city_a: str, city_b: str, evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:
        trace = {"strategy": "CoT", "steps": []}
        def _log(step_name, content):
            trace["steps"].append({"step": step_name, "content": content[:1000]})
            if log_fn: log_fn(f"\n  [CoT] ── {step_name} ──\n{content[:300]}")

        _log("INPUT", f"A: {city_a} | B: {city_b}")
        
        prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n--- EVIDENCE ---\n{evidence}\n\nThink step-by-step. Extract the coordinates, perform the math, and synthesize your final distance.\n\nReasoning:"""
        
        response = self._generate(prompt)
        _log("LLM_REASONING", response)
        distance = extract_distance(response)
        
        if distance is None: 
            fallback = self._generate(prompt + "\n" + response + "\nAnswer with just the number.\nFinal estimate:")
            distance = extract_distance(fallback)
            
        trace["prediction"] = distance
        if log_fn: log_fn(f"\n  [CoT] ✅ FINAL PREDICTION: {distance} km")
        return distance, trace


# =====================================================================
# 2 — TREE-OF-THOUGHT (ToT)
# =====================================================================
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "ToT"
    def reason(self, city_a: str, city_b: str, evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:
        trace = {"strategy": "ToT", "branches": [], "vote": None}
        def _log(step, content):
            if log_fn: log_fn(f"\n  [ToT] ── {step} ──\n{content[:500]}")

        branch_prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n--- EVIDENCE ---\n{evidence}\n
Explore THREE different reasoning branches for estimating the distance between {city_a} and {city_b}.

Format exactly as:
BRANCH 1: [approach]
[reasoning]
Suggested estimate: [estimate in km]

BRANCH 2: [approach]
[reasoning]
Suggested estimate: [estimate in km]

BRANCH 3: [approach]
[reasoning]
Suggested estimate: [estimate in km]

Begin:"""

        resp = self._generate(branch_prompt)
        _log("BRANCHES_RAW", resp)

        branch_pattern = r"BRANCH\s+\d+\s*:\s*(.*?)(?=BRANCH\s+\d+|$)"
        branches = re.findall(branch_pattern, resp, re.DOTALL | re.IGNORECASE)

        valid_estimates = []
        for i, b_text in enumerate(branches):
            est = extract_distance(b_text)
            trace["branches"].append({"index": i+1, "content": b_text.strip()[:500], "estimate": est})
            if est is not None: valid_estimates.append(est)
            _log(f"BRANCH_{i+1}", f"Predicted: {est}\n{b_text.strip()}")

        if not valid_estimates:
            fallback = self._generate(f"--- EVIDENCE ---\n{evidence}\nWhat is the distance in km?\nFinal estimate:")
            final_est = extract_distance(fallback)
        else:
            # Use median of the generated branches to ensure stability
            valid_estimates.sort()
            final_est = valid_estimates[len(valid_estimates)//2]

        trace["prediction"] = final_est
        if log_fn: log_fn(f"\n  [ToT] ✅ FINAL PREDICTION: {final_est} km")
        return final_est, trace


# =====================================================================
# 3 — GRAPH-OF-THOUGHT (GoT)
# =====================================================================
@dataclass
class ThoughtNode:
    id: int
    content: str
    estimate: Optional[float] = None
    confidence: float = 0.0
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    node_type: str = "thought"

class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "GoT"
    def reason(self, city_a: str, city_b: str, evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:
        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "nodes": [], "aggregation": None}

        def _log(step, content):
            if log_fn: log_fn(f"\n  [GoT] ── {step} ──\n{content[:500]}")

        def _add_node(**kwargs) -> ThoughtNode:
            nonlocal next_id
            node = ThoughtNode(id=next_id, **kwargs)
            thought_graph.append(node)
            next_id += 1
            return node

        phase1_prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n--- EVIDENCE ---\n{evidence}\n
Generate FOUR distinct thought nodes analyzing the distance between {city_a} and {city_b}.

Format:
THOUGHT 1: [angle]
[reasoning]
Estimate: [estimate in km]

... (up to THOUGHT 4)

Begin:"""

        phase1_resp = self._generate(phase1_prompt)
        _log("PHASE1_RAW", phase1_resp)

        thought_pattern = r"THOUGHT\s+\d+\s*:\s*(.*?)(?=THOUGHT\s+\d+|$)"
        thoughts = re.findall(thought_pattern, phase1_resp, re.DOTALL | re.IGNORECASE)

        for t_text in thoughts:
            est = extract_distance(t_text)
            _add_node(content=t_text.strip()[:500], estimate=est, confidence=1.0 if est else 0.0, node_type="thought")

        thought_nodes = [n for n in thought_graph if n.node_type == "thought"]
        merge_nodes = []
        
        # Phase 2: Merge the generated thoughts
        for pair_idx in range(0, len(thought_nodes) - 1, 2):
            t_a, t_b = thought_nodes[pair_idx], thought_nodes[pair_idx + 1]
            merge_prompt = f"Two paths analyzed the distance:\nPath A: {t_a.content[:300]}\nPath B: {t_b.content[:300]}\nSynthesize these into ONE final estimate.\nFinal estimate: "
            merge_resp = self._generate(merge_prompt)
            merge_est = extract_distance(merge_resp)
            m_node = _add_node(content=merge_resp.strip()[:500], estimate=merge_est, confidence=1.5 if merge_est else 0.0, parents=[t_a.id, t_b.id], node_type="merge")
            merge_nodes.append(m_node)

        valid_merge_ests = [mn for mn in merge_nodes if mn.estimate is not None]
        
        # Phase 3: Final Selection
        if len(valid_merge_ests) >= 2:
            summaries = "\n".join([f"Merged Path {j+1}: {mn.estimate} km" for j, mn in enumerate(merge_nodes) if mn.estimate])
            final_prompt = f"Merged paths:\n{summaries}\nWhat is the SINGLE BEST final estimate? Best estimate: "
            final_est = extract_distance(self._generate(final_prompt))
        elif len(valid_merge_ests) == 1:
            final_est = valid_merge_ests[0].estimate
        else:
            all_ests = sorted([n.estimate for n in thought_graph if n.estimate is not None])
            final_est = all_ests[len(all_ests)//2] if all_ests else None

        trace["prediction"] = final_est
        if log_fn: log_fn(f"\n  [GoT] ✅ FINAL PREDICTION: {final_est} km")
        return final_est, trace


STRATEGY_MAP = {"cot": ChainOfThought, "tot": TreeOfThought, "got": GraphOfThought}
def get_strategy(name: str, kg: DynamicMetricKnowledgeGraph, **kwargs) -> ReasoningStrategy:
    return STRATEGY_MAP[name.lower()](kg, **kwargs)