"""
strategies_cardinal.py
================================================================================
CoT / ToT / GoT reasoning strategies for cardinal direction inference.

Task: given (source_entity, target_entity, corpus), predict the cardinal
direction label: north_of, south_of, east_of, west_of, northeast_of,
northwest_of, southeast_of, or southwest_of.

Dataset: cardinal_direction_relations.csv from Topological-Reasoning/dataset/
  Columns: source_entity, target_entity, corpus, relation_label, ...
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


# =====================================================================
# CONSTANTS
# =====================================================================
VALID_DIRECTIONS = {
    "north_of", "south_of", "east_of", "west_of",
    "northeast_of", "northwest_of", "southeast_of", "southwest_of",
}

VALID_LIST = (
    "north_of, south_of, east_of, west_of, "
    "northeast_of, northwest_of, southeast_of, southwest_of"
)

_LABEL_ALIASES = {
    "north":      "north_of",
    "south":      "south_of",
    "east":       "east_of",
    "west":       "west_of",
    "northeast":  "northeast_of",
    "northwest":  "northwest_of",
    "southeast":  "southeast_of",
    "southwest":  "southwest_of",
    "north-east": "northeast_of",
    "north-west": "northwest_of",
    "south-east": "southeast_of",
    "south-west": "southwest_of",
    "ne":         "northeast_of",
    "nw":         "northwest_of",
    "se":         "southeast_of",
    "sw":         "southwest_of",
}


# =====================================================================
# HELPERS
# =====================================================================
def extract_direction(text: str) -> Optional[str]:
    """Extract a valid cardinal direction label from model output."""
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"[*_`]", "", clean)
    lower = clean.lower()

    # Explicit answer patterns (prefer later occurrences)
    patterns = [
        r"answer\s*[:=]\s*\[?([a-z_\-]+)\]?",
        r"final\s+(?:answer|direction|relation)\s*[:=]\s*\[?([a-z_\-]+)\]?",
        r"direction\s*(?:is|:)\s*\[?([a-z_\-]+)\]?",
        r"relation\s*(?:is|:)\s*\[?([a-z_\-]+)\]?",
        r"therefore[,\s]+(?:it is|the direction is|the relation is)\s*\[?([a-z_\-]+)\]?",
    ]
    for pat in patterns:
        for m in re.finditer(pat, lower):
            raw = m.group(1).strip().rstrip(".,;:")
            if raw in VALID_DIRECTIONS:
                return raw
            if raw in _LABEL_ALIASES:
                return _LABEL_ALIASES[raw]

    # Scan for valid labels (take last occurrence)
    found = []
    for lbl in VALID_DIRECTIONS:
        idx = lower.rfind(lbl)
        if idx != -1:
            found.append((idx, lbl))
    for alias, canonical in _LABEL_ALIASES.items():
        idx = lower.rfind(alias)
        if idx != -1:
            found.append((idx, canonical))
    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][1]
    return None


def _spatial_weight(text: str) -> float:
    """Score a reasoning branch by spatial signal words."""
    keywords = [
        "north", "south", "east", "west",
        "above", "below", "up", "down",
        "cardinal", "direction", "compass",
        "geographic", "latitude", "longitude",
    ]
    hits = sum(1 for kw in keywords if kw in text.lower())
    return 1.0 + 0.3 * min(hits, 5)


# =====================================================================
# BASE CLASS
# =====================================================================
class ReasoningStrategy(ABC):
    def __init__(self, model_fn, max_new_tokens: int = 512, temperature: float = 0.1):
        self._generate = model_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]: ...


# =====================================================================
# 1. CHAIN-OF-THOUGHT (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "CoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        src   = entity["source_entity"]
        tgt   = entity["target_entity"]
        corpus = entity["corpus"]
        trace = {"strategy": "CoT"}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [CoT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")

        prompt = (
            "You are an expert in spatial geography and cardinal directions.\n\n"
            f"Given the following description of the spatial relationship between "
            f"'{src}' and '{tgt}', determine the cardinal direction of '{src}' "
            f"relative to '{tgt}'.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"The possible cardinal directions are:\n  {VALID_LIST}\n\n"
            "Think step by step:\n"
            "1. Identify the spatial language in the corpus (words like 'above', "
            "'north of', 'up', 'below', 'to the east', etc.).\n"
            "2. Map those cues to a cardinal direction.\n"
            f"3. State your conclusion: '{src}' is [direction] '{tgt}'.\n\n"
            "End with: Answer: [direction]\n\n"
            "Reasoning:"
        )

        response = self._generate(prompt)
        _log("LLM_RESPONSE", response)

        direction = extract_direction(response)

        if direction is None:
            fallback = (
                f"Corpus: \"{corpus}\"\n"
                f"The cardinal direction of '{src}' relative to '{tgt}' is:\n"
                "Answer: ["
            )
            fb_resp = self._generate(fallback)
            direction = extract_direction("Answer: [" + fb_resp)
            _log("FALLBACK", fb_resp[:200] + f" -> {direction}")

        trace["prediction"] = direction
        if log_fn:
            log_fn(f"\n  [CoT] FINAL: {direction}")
        return direction, trace


# =====================================================================
# 2. TREE-OF-THOUGHT (ToT)
# =====================================================================
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "ToT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        src    = entity["source_entity"]
        tgt    = entity["target_entity"]
        corpus = entity["corpus"]
        trace  = {"strategy": "ToT", "branches": []}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [ToT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")

        prompt = (
            "You are an expert in spatial geography and cardinal directions.\n\n"
            f"Determine the cardinal direction of '{src}' with respect to '{tgt}' "
            f"using the corpus below. Explore THREE independent reasoning paths. "
            f"For each branch, choose your own focus and reasoning approach.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n\n"
            "Label each branch as 'BRANCH N: <your chosen focus>', reason through it, "
            "then end it with 'Answer: [direction]'.\n\n"
            "Begin:"
        )

        branch_resp = self._generate(prompt)
        _log("BRANCHES_RAW", branch_resp)

        branch_pattern = r"BRANCH\s+\d+\s*:(.*?)(?=BRANCH\s+\d+|$)"
        branches = re.findall(branch_pattern, branch_resp, re.DOTALL | re.IGNORECASE)

        weighted: Dict[str, float] = {}
        for i, b_text in enumerate(branches):
            pred = extract_direction(b_text)
            trace["branches"].append({"index": i + 1, "direction": pred,
                                      "content": b_text.strip()[:300]})
            _log(f"BRANCH_{i+1}", f"-> {pred}")
            if pred:
                weighted[pred] = weighted.get(pred, 0.0) + _spatial_weight(b_text)

        final_dir = max(weighted, key=weighted.get) if weighted else None

        if final_dir is None:
            final_dir = extract_direction(branch_resp)
        if final_dir is None:
            fb = (
                f"Corpus: \"{corpus}\"\n"
                f"The cardinal direction of '{src}' relative to '{tgt}' is:\n"
                "Answer: ["
            )
            final_dir = extract_direction("Answer: [" + self._generate(fb))
            _log("FALLBACK", f"-> {final_dir}")

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [ToT] FINAL: {final_dir}")
        return final_dir, trace


# =====================================================================
# 3. GRAPH-OF-THOUGHT (GoT)
# =====================================================================
@dataclass
class ThoughtNode:
    id: int
    content: str
    direction: Optional[str] = None
    confidence: float = 0.0


class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "GoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        src    = entity["source_entity"]
        tgt    = entity["target_entity"]
        corpus = entity["corpus"]
        trace  = {"strategy": "GoT", "nodes": []}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [GoT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")

        phase1_prompt = (
            "You are an expert in spatial geography and cardinal directions.\n\n"
            f"Determine the cardinal direction of '{src}' with respect to '{tgt}' "
            f"using the corpus below. Build a reasoning graph with FOUR thought nodes. "
            f"For each thought, choose your own focus and reasoning angle.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n\n"
            "Label each node as 'THOUGHT N: <your chosen focus>', reason through it, "
            "then end it with 'Direction: [direction]'.\n\n"
            "Begin:"
        )

        phase1_resp = self._generate(phase1_prompt)
        _log("PHASE1_RAW", phase1_resp)

        thought_pattern = r"THOUGHT\s+\d+\s*:(.*?)(?=THOUGHT\s+\d+|$)"
        thoughts = re.findall(thought_pattern, phase1_resp, re.DOTALL | re.IGNORECASE)

        nodes: List[ThoughtNode] = []
        weighted: Dict[str, float] = {}
        for i, t_text in enumerate(thoughts):
            pred   = extract_direction(t_text)
            weight = _spatial_weight(t_text)
            nodes.append(ThoughtNode(id=i, content=t_text.strip()[:400],
                                     direction=pred, confidence=weight))
            if pred:
                weighted[pred] = weighted.get(pred, 0.0) + weight

        trace["nodes"] = [{"id": n.id, "direction": n.direction,
                           "confidence": n.confidence} for n in nodes]
        final_dir = max(weighted, key=weighted.get) if weighted else None

        # Fallback 1: scan full response
        if final_dir is None:
            final_dir = extract_direction(phase1_resp)
            _log("FALLBACK1", f"scan full response -> {final_dir}")

        # Fallback 2: direct synthesis prompt
        if final_dir is None:
            synth = (
                f"Corpus: \"{corpus}\"\n"
                f"Based on the spatial evidence above, the cardinal direction of "
                f"'{src}' relative to '{tgt}' is:\nAnswer: ["
            )
            final_dir = extract_direction("Answer: [" + self._generate(synth))
            _log("FALLBACK2", f"synthesis -> {final_dir}")

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [GoT] FINAL: {final_dir}")
        return final_dir, trace


# =====================================================================
# STRATEGY FACTORY
# =====================================================================

# Stub for backward-compat (eval_engine imports this even when --use-kg is off)
class CardinalKnowledgeGraph:
    pass


STRATEGY_MAP = {
    "cot": ChainOfThought,
    "tot": TreeOfThought,
    "got": GraphOfThought,
}


def get_strategy(name: str, kg=None, model_fn=None,
                 max_new_tokens: int = 512, temperature: float = 0.1,
                 **kwargs) -> ReasoningStrategy:
    cls = STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGY_MAP)}")
    return cls(model_fn=model_fn, max_new_tokens=max_new_tokens,
               temperature=temperature)
