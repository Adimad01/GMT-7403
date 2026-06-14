"""
strategies_relative.py
================================================================================
CoT / ToT / GoT reasoning strategies for relative direction inference.

Task: given (source_entity, target_entity, corpus), predict the relative
direction label: behind, in_front_of, left_of, next_to, or right_of.

Dataset: relative_direction_relations.csv from Topological-Reasoning/dataset/
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
    "behind", "in_front_of", "left_of", "next_to", "right_of",
}

VALID_LIST = "behind, in_front_of, left_of, next_to, right_of"

_LABEL_ALIASES = {
    "in front of":  "in_front_of",
    "in front":     "in_front_of",
    "front":        "in_front_of",
    "ahead":        "in_front_of",
    "in_front":     "in_front_of",
    "left":         "left_of",
    "to the left":  "left_of",
    "right":        "right_of",
    "to the right": "right_of",
    "next to":      "next_to",
    "beside":       "next_to",
    "adjacent":     "next_to",
    "behind":       "behind",
    "at the back":  "behind",
    "in back":      "behind",
}


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    return s.lower().strip().rstrip(".").rstrip(",").strip()


# =====================================================================
# HELPERS
# =====================================================================
def extract_direction(text: str) -> Optional[str]:
    """Extract a valid relative direction label from model output."""
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"[*`]", "", clean)  # keep underscores — labels use them (left_of, next_to)
    lower = clean.lower()

    # Explicit answer patterns
    patterns = [
        r"answer\s*[:=]\s*\[?([a-z_\s]+?)\]?(?:\s|$|[.,;:])",
        r"final\s+(?:answer|direction|relation)\s*[:=]\s*\[?([a-z_\s]+?)\]?(?:\s|$|[.,;:])",
        r"direction\s*(?:is|:)\s*\[?([a-z_\s]+?)\]?(?:\s|$|[.,;:])",
        r"relation\s*(?:is|:)\s*\[?([a-z_\s]+?)\]?(?:\s|$|[.,;:])",
    ]
    for pat in patterns:
        for m in re.finditer(pat, lower):
            raw = m.group(1).strip().rstrip(".,;:")
            if raw in VALID_DIRECTIONS:
                return raw
            canon = _LABEL_ALIASES.get(raw)
            if canon:
                return canon

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
    keywords = [
        "left", "right", "front", "behind", "beside", "next",
        "perspective", "observer", "facing", "port", "starboard",
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
        src    = entity["source_entity"]
        tgt    = entity["target_entity"]
        corpus = entity["corpus"]
        trace  = {"strategy": "CoT"}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [CoT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")

        prompt = (
            "You are an expert in spatial and relative directions.\n\n"
            f"Given the following description of the spatial relationship between "
            f"'{src}' and '{tgt}', determine the relative direction of '{src}' "
            f"relative to '{tgt}' from an observer's perspective.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"The possible relative directions are:\n  {VALID_LIST}\n\n"
            "Think step by step:\n"
            "1. Identify the observer's viewpoint and orientation described in the corpus.\n"
            "2. Map the spatial language to a relative direction "
            "(left, right, in front, behind, next to).\n"
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
                f"The relative direction of '{src}' relative to '{tgt}' is:\n"
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
            "You are an expert in spatial and relative directions.\n\n"
            f"Given the corpus describing the spatial relationship between "
            f"'{src}' and '{tgt}', explore THREE independent reasoning paths.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n\n"
            "BRANCH 1: Literal language analysis\n"
            "  - Identify explicit relative direction words (left, right, front, behind, beside).\n"
            "Answer: [direction]\n\n"
            "BRANCH 2: Observer perspective reasoning\n"
            "  - Determine the observer's viewpoint and apply the left/right/front/behind logic.\n"
            "Answer: [direction]\n\n"
            "BRANCH 3: Synthesis\n"
            "  - Combine both approaches and state the most likely direction.\n"
            "Answer: [direction]\n\n"
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
                f"The relative direction of '{src}' relative to '{tgt}' is:\n"
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
            "You are an expert in spatial and relative directions.\n\n"
            f"Analyze the spatial relationship between '{src}' and '{tgt}' "
            f"using the corpus below. Build a reasoning graph with FOUR thought nodes.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n\n"
            "THOUGHT 1: Relative language extraction\n"
            "  - List every relative directional word or phrase in the corpus.\n"
            "Direction: [direction]\n\n"
            "THOUGHT 2: Observer perspective\n"
            "  - Establish the observer's viewpoint and apply relative direction logic.\n"
            "Direction: [direction]\n\n"
            "THOUGHT 3: Consistency check\n"
            "  - Do the language cues and perspective reasoning agree?\n"
            "Direction: [direction]\n\n"
            "THOUGHT 4: Final aggregation\n"
            "  - State the definitive relative direction.\n"
            "Direction: [direction]\n\n"
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

        if final_dir is None:
            final_dir = extract_direction(phase1_resp)
            _log("FALLBACK1", f"scan full response -> {final_dir}")

        if final_dir is None:
            synth = (
                f"Corpus: \"{corpus}\"\n"
                f"The relative direction of '{src}' relative to '{tgt}' is:\nAnswer: ["
            )
            final_dir = extract_direction("Answer: [" + self._generate(synth))
            _log("FALLBACK2", f"synthesis -> {final_dir}")

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [GoT] FINAL: {final_dir}")
        return final_dir, trace


# =====================================================================
# FACTORY
# =====================================================================
STRATEGY_MAP = {
    "cot": ChainOfThought,
    "tot": TreeOfThought,
    "got": GraphOfThought,
}


def get_strategy(name: str, model_fn=None, max_new_tokens: int = 512,
                 temperature: float = 0.1, **kwargs) -> ReasoningStrategy:
    cls = STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGY_MAP)}")
    return cls(model_fn=model_fn, max_new_tokens=max_new_tokens,
               temperature=temperature)
