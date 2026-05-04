"""
reasoning_strategies.py — Volet C Reasoning Strategies (Metric Distance)
=========================================================================
Implements three LLM-driven reasoning strategies for estimating the
geographic distance between two US cities, grounded on a data-driven
Knowledge Graph.

  1. Chain-of-Thought  (CoT) — Linear step-by-step reasoning
  2. Tree-of-Thought   (ToT) — LLM generates its own reasoning branches, then votes
  3. Graph-of-Thought  (GoT) — LLM builds a thought graph with merging

Design principles:
  - The KG provides grounding evidence (coordinates, anchor distances,
    corpus signals) gathered ONCE per pair, then injected as context.
  - The LLM dynamically creates its own reasoning branches / thought paths.
  - For ToT and GoT, the LLM decides HOW to decompose the problem
    (no hardcoded perspectives).
  - The LLM must produce a numeric km estimate — we parse it from output.
  - The direct distance d(A, B) is NEVER given — the LLM must reason from
    coordinates, anchor triangulation, and corpus signals.

Uses an Ollama-compatible HTTP API at BASE_URL for inference (gpt-oss).
"""

import re
import time
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from geographic_kg import MetricKnowledgeGraph


# =====================================================================
# OLLAMA API CONFIGURATION
# =====================================================================
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"


# =====================================================================
# DISTANCE EXTRACTION
# =====================================================================
MAX_EARTH_DISTANCE_KM = 20_100

# Known bad values the LLM may produce (Haversine R, 2*R, etc.)
REJECT_VALUES = {6371, 6371.0, 6370, 6372, 12742, 12742.0, 3959, 3956}


def extract_distance(text: str) -> Optional[float]:
    """
    Extract a numeric distance (in km) from LLM output.

    Strategy:
      1. Look for "Final estimate: <number>" or "Answer: <number>"
      2. Look for a number followed by "km"
      3. Fallback: last plausible number in range [1, 20100]

    Rejects known bad values (e.g., Earth's radius 6371 from Haversine).
    Returns float km or None.
    """
    if not text or not text.strip():
        return None

    # Strip <think>...</think> tags (reasoning models)
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)

    def _clean(raw: str) -> Optional[float]:
        cleaned = re.sub(r"[,\s\u00a0\u202f]", "", raw)
        try:
            val = float(cleaned)
            if 1.0 <= val <= MAX_EARTH_DISTANCE_KM:
                # Reject known bad values (Earth's radius, etc.)
                if round(val) in REJECT_VALUES:
                    return None
                return val
        except ValueError:
            pass
        return None

    # Strategy 1: structured answer patterns
    answer_patterns = [
        r"(?:final\s+)?(?:estimate|answer|distance)\s*[:=]\s*~?\s*"
        r"([\d][\d,\s]*\.?\d*)\s*(?:km|kilometers)?",
        r"(?:approximately|about|roughly|≈|~)\s*([\d][\d,\s]*\.?\d*)\s*km",
    ]
    for pat in answer_patterns:
        matches = list(re.finditer(pat, text_clean, re.IGNORECASE))
        if matches:
            val = _clean(matches[-1].group(1))
            if val is not None:
                return round(val, 1)

    # Strategy 2: number + km
    km_pattern = re.compile(
        r"([\d][\d,\s]*\.?\d*)\s*(?:km|kilometers)", re.IGNORECASE
    )
    matches = list(km_pattern.finditer(text_clean))
    if matches:
        val = _clean(matches[-1].group(1))
        if val is not None:
            return round(val, 1)

    # Strategy 3: last plausible number
    num_pattern = re.compile(r"([\d][\d,\s]*\.?\d*)")
    all_nums = list(num_pattern.finditer(text_clean))
    for m in reversed(all_nums):
        val = _clean(m.group(1))
        if val is not None:
            return round(val, 1)

    return None


# =====================================================================
# HELPERS
# =====================================================================
def llm_generate(prompt: str,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.1,
                 base_url: str = BASE_URL,
                 model_name: str = MODEL_NAME,
                 llm=None,
                 max_retries: int = 5) -> str:
    """Call the Ollama API directly via HTTP POST (non-streaming, reliable)."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            if text:
                return text
            print(f"  ⚠️  Empty LLM response (attempt {attempt}/{max_retries}), retrying...")
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"  ⚠️  Ollama API error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(3 * attempt)
    return ""


# =====================================================================
# TASK DESCRIPTION (shared across all strategies)
# =====================================================================
TASK_DESCRIPTION = """You are an expert geographer and spatial analyst.

Your task: Estimate the distance in kilometers between City A and City B.

You are given:
  - The coordinates (latitude, longitude) of both cities
  - A table of anchor cities with their coordinates and KNOWN distances to both A and B
  - Corpus co-occurrence signals (how often the two cities are mentioned together)
  - Whether both cities are in the same US state

You may use any reasoning approach: Haversine formula on coordinates, triangulation
from anchor distances, pattern recognition from corpus signals, or any combination.

IMPORTANT: You must produce a SINGLE numeric distance estimate in km.
End your response with: Final estimate: <number> km
"""

RULES_BLOCK = """Rules:
1. Use the coordinates and anchor distance evidence to triangulate.
2. The Haversine formula gives great-circle distance from lat/lon.
3. Triangle inequality: for any anchor C, |d(A,C) - d(B,C)| ≤ d(A,B) ≤ d(A,C) + d(B,C).
4. Combine multiple evidence sources for the best estimate.
5. Provide your final answer as a single number in km.
6. End with: Final estimate: <number> km
"""


# =====================================================================
# BASE CLASS
# =====================================================================
class ReasoningStrategy(ABC):
    """Abstract base for a reasoning strategy."""

    def __init__(self, kg: MetricKnowledgeGraph,
                 temperature: float = 0.1, max_new_tokens: int = 2048,
                 base_url: str = BASE_URL, model_name: str = MODEL_NAME):
        self.kg = kg
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.model_name = model_name

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, city_a: str, city_b: str,
               evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:
        """
        Perform reasoning and return (estimated_km, trace_dict).

        Args:
            city_a:   Name of city A.
            city_b:   Name of city B.
            evidence: Pre-built KG evidence block (from kg.build_evidence_block).
            log_fn:   Optional logging callback.

        Returns:
            (distance_km, trace_dict)
        """
        ...

    def _generate(self, prompt: str, **kwargs) -> str:
        """Generate response via direct HTTP API call."""
        return llm_generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            base_url=self.base_url,
            model_name=self.model_name,
        )


# =====================================================================
# 1 — CHAIN-OF-THOUGHT  (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    """
    Linear step-by-step reasoning grounded on KG evidence.
    The LLM is guided through: coordinates → haversine → anchors →
    corpus signals → synthesis → final estimate.
    """

    @property
    def name(self) -> str:
        return "CoT"

    def reason(self, city_a: str, city_b: str,
               evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:

        trace = {"strategy": "CoT", "steps": []}

        def _log(step_name, content):
            trace["steps"].append({"step": step_name, "content": content[:1000]})
            if log_fn:
                log_fn(f"[CoT] {step_name}: {content[:300]}")

        _log("KG_Evidence", evidence[:500])

        prompt = f"""{TASK_DESCRIPTION}

{RULES_BLOCK}

--- GEOGRAPHIC KNOWLEDGE GRAPH EVIDENCE ---
{evidence}

Now estimate the distance between City A and City B step by step.

Step 1 — COORDINATE ANALYSIS:
Look at the latitude and longitude of both cities.
Estimate the approximate distance using the Haversine formula or your geographic knowledge.
Note the latitude difference and longitude difference.

Step 2 — ANCHOR TRIANGULATION:
Examine the anchor cities table. For each anchor city C:
  - The distance d(A,B) must satisfy: |d(A,C) - d(B,C)| ≤ d(A,B) ≤ d(A,C) + d(B,C)
  - Compute the lower bound max(|d(A,C) - d(B,C)|) across anchors
  - Compute the upper bound min(d(A,C) + d(B,C)) across anchors
  - The true distance lies within this interval

Step 3 — CORPUS SIGNAL ANALYSIS:
What do the co-occurrence counts suggest about proximity?
High co-occurrence with "near" tokens → cities are close.
High "far" counts → cities are distant.

Step 4 — SYNTHESIS:
Combine your coordinate-based estimate with the anchor triangulation bounds
and corpus signals. Pick the best single estimate.

Step 5 — FINAL ANSWER:
State your final distance estimate.

Reasoning:"""

        response = self._generate(prompt)
        _log("Reasoning", response)

        distance = extract_distance(response)

        # Fallback: re-ask for just the number
        if distance is None:
            fallback_prompt = (
                f"Based on your analysis above, what is your single best "
                f"estimate for the distance between {city_a} and {city_b} in km?\n"
                f"Answer with just the number.\n"
                f"Final estimate:"
            )
            fallback_resp = self._generate(
                prompt + "\n" + response + "\n\n" + fallback_prompt,
                max_new_tokens=100,
            )
            _log("Fallback", fallback_resp)
            distance = extract_distance(fallback_resp)

        trace["prediction"] = distance
        return distance, trace


# =====================================================================
# 2 — TREE-OF-THOUGHT  (ToT) — Sequential branching
# =====================================================================

# Branch perspective prompts — each is a SEPARATE, short LLM call
BRANCH_PERSPECTIVES = [
    (
        "Coordinate-Based Estimation",
        "Estimate the distance using ONLY the coordinates of City A and City B.\n"
        "Apply the Haversine formula or your geographic intuition from the\n"
        "latitude/longitude values. Show your reasoning step by step.",
    ),
    (
        "Anchor Triangulation",
        "Estimate the distance using ONLY the anchor city distances.\n"
        "For each anchor C, the triangle inequality gives:\n"
        "  |d(A,C) - d(B,C)| ≤ d(A,B) ≤ d(A,C) + d(B,C)\n"
        "Find the tightest lower and upper bounds from the anchor table.\n"
        "Your estimate should lie within these bounds.",
    ),
    (
        "Geographic Reasoning & Cross-Validation",
        "Combine geographic knowledge with the evidence.\n"
        "Consider: Are the cities in the same region? What do the corpus\n"
        "signals suggest? Cross-validate a coordinate estimate against\n"
        "the anchor bounds. Produce your best refined estimate.",
    ),
]


class TreeOfThought(ReasoningStrategy):
    """
    Three separate LLM calls, each from a different perspective.
    Then a voting/evaluation call picks the best estimate.
    The LLM reasons freely within each perspective.
    """

    @property
    def name(self) -> str:
        return "ToT"

    def reason(self, city_a: str, city_b: str,
               evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:

        trace = {"strategy": "ToT", "branches": [], "vote": None}

        def _log(step, content):
            if log_fn:
                log_fn(f"[ToT] {step}: {content[:300]}")

        _log("KG_Evidence", evidence[:300])

        # ── Step 1: Run 3 separate branch calls ──────────────────────
        branch_estimates = []
        for i, (title, instruction) in enumerate(BRANCH_PERSPECTIVES):
            branch_prompt = f"""You are an expert geographer estimating the distance between two US cities.

--- GEOGRAPHIC KNOWLEDGE GRAPH EVIDENCE ---
{evidence}

APPROACH: {title}
{instruction}

Produce a single numeric distance estimate in km.
End with: Final estimate: <number> km

Reasoning:"""

            resp = self._generate(branch_prompt)
            est = extract_distance(resp)
            branch_estimates.append(est)
            trace["branches"].append({
                "index": i + 1,
                "title": title,
                "content": resp.strip()[:500],
                "estimate": est,
            })
            _log(f"Branch{i+1}_{title}", f"est={est}")

        # ── Step 2: Evaluate and select ───────────────────────────────
        valid_estimates = [e for e in branch_estimates if e is not None]

        if not valid_estimates:
            # All branches failed — direct fallback
            fallback_prompt = f"""You are an expert geographer.

--- EVIDENCE ---
{evidence}

What is the distance between City A and City B in km?
Final estimate:"""
            fallback_resp = self._generate(fallback_prompt, max_new_tokens=300)
            final_est = extract_distance(fallback_resp)
            trace["vote"] = {"method": "fallback", "result": final_est}
            _log("Vote_Fallback", f"est={final_est}")
        elif len(valid_estimates) == 1:
            final_est = valid_estimates[0]
            trace["vote"] = {"method": "single", "result": final_est}
        else:
            # Voting call: LLM picks the best — NO full evidence to avoid R=6371 leakage
            summaries = []
            for b in trace["branches"]:
                if b["estimate"] is not None:
                    summaries.append(f"- {b['title']}: {b['estimate']:.1f} km")

            eval_prompt = f"""Three independent reasoning branches estimated the distance between two US cities:

{chr(10).join(summaries)}

Which estimate is most reliable? If they mostly agree, pick the consensus value.
If they disagree, the coordinate-based Haversine calculation is usually most accurate.
Respond with ONLY the best distance number in km.

Best estimate: """

            eval_resp = self._generate(eval_prompt)
            eval_est = extract_distance(eval_resp)

            if eval_est is not None:
                final_est = eval_est
                trace["vote"] = {
                    "method": "evaluation",
                    "candidates": valid_estimates,
                    "eval_response": eval_resp[:300],
                    "result": final_est,
                }
            else:
                # Fallback: median
                valid_sorted = sorted(valid_estimates)
                mid = len(valid_sorted) // 2
                final_est = valid_sorted[mid]
                trace["vote"] = {
                    "method": "median_fallback",
                    "candidates": valid_estimates,
                    "result": final_est,
                }

            _log("Vote", f"Final={final_est}, candidates={valid_estimates}")

        trace["prediction"] = final_est
        return final_est, trace


# =====================================================================
# 3 — GRAPH-OF-THOUGHT  (GoT) — Sequential graph with merging
# =====================================================================

# Four thought perspectives — each is a SEPARATE LLM call
THOUGHT_PERSPECTIVES = [
    (
        "Haversine Computation",
        "Compute the Haversine (great-circle) distance from the coordinates.\n"
        "Show the math: convert lat/lon to radians, apply the formula.\n"
        "The Haversine formula: d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)))\n"
        "where R = 6371 km.",
    ),
    (
        "Tight Triangulation Bounds",
        "From the anchor table, find:\n"
        "  Lower bound = max of |d(A,C) - d(B,C)| across all anchors\n"
        "  Upper bound = min of d(A,C) + d(B,C) across all anchors\n"
        "Report the bounds and their midpoint as your estimate.",
    ),
    (
        "Nearest-Anchor Interpolation",
        "Find the anchor cities that are closest to A and closest to B.\n"
        "Use their known distances to interpolate an estimate for d(A,B).\n"
        "Consider the geographic layout.",
    ),
    (
        "Regional & Corpus Analysis",
        "Consider what region/area each city is in (state, part of US).\n"
        "Use the corpus signals (co-occurrence, near/far counts) to\n"
        "judge proximity. Cross-check with the coordinate positions.",
    ),
]


@dataclass
class ThoughtNode:
    """A node in the Graph-of-Thought."""
    id: int
    content: str
    estimate: Optional[float] = None
    confidence: float = 0.0
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    node_type: str = "thought"  # thought | merge | final


class GraphOfThought(ReasoningStrategy):
    """
    Graph-structured reasoning with KG grounding.
    Each phase uses SEPARATE short LLM calls (not one giant prompt).

    Phase 1: 4 separate thought calls (different perspectives)
    Phase 2: 2 merge calls (pairwise)
    Phase 3: 1 final aggregation call
    """

    @property
    def name(self) -> str:
        return "GoT"

    def reason(self, city_a: str, city_b: str,
               evidence: str, log_fn=None) -> Tuple[Optional[float], Dict]:

        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "nodes": [], "aggregation": None}

        def _log(step, content):
            if log_fn:
                log_fn(f"[GoT] {step}: {content[:300]}")

        def _add_node(**kwargs) -> ThoughtNode:
            nonlocal next_id
            node = ThoughtNode(id=next_id, **kwargs)
            thought_graph.append(node)
            next_id += 1
            return node

        _log("KG_Evidence", evidence[:300])

        # ── Phase 1: 4 separate thought calls ─────────────────────────
        for i, (title, instruction) in enumerate(THOUGHT_PERSPECTIVES):
            thought_prompt = f"""You are an expert geographer estimating the distance between two US cities.

--- GEOGRAPHIC KNOWLEDGE GRAPH EVIDENCE ---
{evidence}

APPROACH: {title}
{instruction}

Produce a single numeric distance estimate in km.
End with: Final estimate: <number> km

Reasoning:"""

            resp = self._generate(thought_prompt)
            est = extract_distance(resp)
            node = _add_node(
                content=resp.strip()[:500],
                estimate=est,
                confidence=1.0 if est else 0.0,
                node_type="thought",
            )
            _log(f"Phase1_{title}", f"est={est}")

        # ── Phase 2: Merge thoughts pairwise ──────────────────────────
        thought_nodes = [n for n in thought_graph if n.node_type == "thought"]

        merge_nodes = []
        for pair_idx in range(0, len(thought_nodes) - 1, 2):
            t_a = thought_nodes[pair_idx]
            t_b = thought_nodes[pair_idx + 1]

            est_a_str = f"{t_a.estimate:.1f} km" if t_a.estimate else "failed"
            est_b_str = f"{t_b.estimate:.1f} km" if t_b.estimate else "failed"

            merge_prompt = f"""Two reasoning paths estimated the distance between two US cities:

Path A ({est_a_str}):
{t_a.content[:300]}

Path B ({est_b_str}):
{t_b.content[:300]}

Synthesize these into a SINGLE refined estimate. If they agree, use the consensus.
Respond with your best estimate in km.
Final estimate: """

            merge_resp = self._generate(merge_prompt, max_new_tokens=1024)
            merge_est = extract_distance(merge_resp)

            m_node = _add_node(
                content=merge_resp.strip()[:500],
                estimate=merge_est,
                confidence=1.5 if merge_est else 0.0,
                parents=[t_a.id, t_b.id],
                node_type="merge",
            )
            t_a.children.append(m_node.id)
            t_b.children.append(m_node.id)
            merge_nodes.append(m_node)
            _log(f"Phase2_Merge{m_node.id}", f"est={merge_est}")

        # Carry forward unpaired thought
        if len(thought_nodes) % 2 == 1:
            merge_nodes.append(thought_nodes[-1])

        # ── Phase 3: Final aggregation ────────────────────────────────
        valid_merge_ests = [mn for mn in merge_nodes if mn.estimate is not None]

        if len(valid_merge_ests) >= 2:
            summaries = []
            for j, mn in enumerate(merge_nodes):
                est_str = f"{mn.estimate:.1f} km" if mn.estimate else "failed"
                summaries.append(f"Path {j+1} ({est_str}): {mn.content[:200]}")

            final_prompt = f"""Multiple merged reasoning paths estimated the distance between two US cities:

{chr(10).join(summaries)}

What is the SINGLE BEST final estimate? If they agree, use the consensus value.
Respond with ONLY the best distance number in km.
Best estimate: """

            final_resp = self._generate(final_prompt, max_new_tokens=512)
            final_est = extract_distance(final_resp)

            _add_node(
                content=final_resp.strip()[:500],
                estimate=final_est,
                confidence=2.0 if final_est else 0.0,
                parents=[mn.id for mn in merge_nodes],
                node_type="final",
            )
            _log("Phase3_Final", f"est={final_est}")
        elif len(valid_merge_ests) == 1:
            final_est = valid_merge_ests[0].estimate
        elif merge_nodes:
            # Try any merge node estimate
            final_est = next(
                (mn.estimate for mn in merge_nodes if mn.estimate), None
            )
        else:
            final_est = None

        # Fallback: median of all non-None estimates across the graph
        if final_est is None:
            all_ests = [n.estimate for n in thought_graph if n.estimate is not None]
            if all_ests:
                all_ests_sorted = sorted(all_ests)
                mid = len(all_ests_sorted) // 2
                final_est = all_ests_sorted[mid]
                _log("Fallback_Median", f"est={final_est}, from={all_ests}")

        # ── Build trace ───────────────────────────────────────────────
        for node in thought_graph:
            trace["nodes"].append({
                "id": node.id,
                "type": node.node_type,
                "estimate": node.estimate,
                "confidence": node.confidence,
                "parents": node.parents,
                "content_preview": node.content[:200],
            })

        candidates = [(n.estimate, n.confidence)
                       for n in thought_graph if n.estimate is not None]
        trace["aggregation"] = {"candidates": candidates, "final": final_est}
        trace["prediction"] = final_est

        _log("Result", f"prediction={final_est}")

        return final_est, trace


# =====================================================================
# STRATEGY FACTORY
# =====================================================================
STRATEGY_MAP = {
    "cot": ChainOfThought,
    "tot": TreeOfThought,
    "got": GraphOfThought,
}


def get_strategy(name: str, kg: MetricKnowledgeGraph, **kwargs) -> ReasoningStrategy:
    """Factory to create a reasoning strategy by name."""
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name_lower](kg, **kwargs)
