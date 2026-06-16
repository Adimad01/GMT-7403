"""
strategies_osm_relative.py
================================================================================
OSM-evidence-aware CoT / ToT / GoT strategies for relative direction inference
(behind / in_front_of / left_of / next_to / right_of).

Relative direction depends on an OBSERVER REFERENCE FRAME that the dataset only
expresses in free text — OSM coordinates alone cannot fix left vs. right.  So the
OSM evidence here is purely informational: it surfaces every fact we can pull
(coordinates, bounding boxes, hierarchy, centroid distance, absolute bearing /
offset) and the prompt instructs the model to combine it with the corpus's
viewpoint cues.  The LLM decides the label.

Used by eval_engine_relative.py for kg-mode in {none, input}:
  • kg-mode none  → NullKG (empty evidence → base behavior)
  • kg-mode input → OSMEvidenceKG (evidence prepended once)
Per-step RAG (Exp 6) is handled by rag_loop.RAGStrategy.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re

from strategies_relative import (
    VALID_DIRECTIONS,
    VALID_LIST,
    extract_direction,
    _spatial_weight,
)


def _evidence(kg, src: str, tgt: str, corpus: str, log_fn=None) -> str:
    if kg is None:
        return ""
    try:
        return kg.gather_evidence(src, tgt, sentence=corpus, log_fn=None) or ""
    except Exception as exc:
        if log_fn:
            log_fn(f"[KG WARN] {exc}")
        return ""


def _evidence_clause(evidence: str) -> str:
    if not evidence.strip():
        return ""
    return (
        f"\n{evidence}\n\n"
        "The OSM facts above describe ABSOLUTE map geometry only. Relative direction "
        "(left/right/front/behind) depends on the observer's viewpoint and facing "
        "described in the corpus — use the corpus to fix the reference frame, and the "
        "OSM facts only as secondary support.\n"
    )


# =====================================================================
# BASE
# =====================================================================
class ReasoningStrategy(ABC):
    def __init__(self, kg=None, model_fn=None, max_new_tokens: int = 1024,
                 temperature: float = 0.1):
        self.kg = kg
        self._generate = model_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]: ...


# =====================================================================
# 1. CHAIN-OF-THOUGHT
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "CoT"

    def reason(self, entity, log_fn=None):
        src, tgt, corpus = entity["source_entity"], entity["target_entity"], entity["corpus"]
        trace = {"strategy": "CoT", "mode": "osm"}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [CoT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")
        evidence = _evidence(self.kg, src, tgt, corpus, log_fn)
        if evidence:
            _log("OSM_EVIDENCE", evidence)

        prompt = (
            "You are an expert in spatial and relative directions.\n\n"
            f"Determine the relative direction of '{src}' relative to '{tgt}' from an "
            "observer's perspective.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"The possible relative directions are:\n  {VALID_LIST}\n"
            f"{_evidence_clause(evidence)}\n"
            "Think step by step:\n"
            "1. Identify the observer's viewpoint and orientation in the corpus.\n"
            "2. Map the spatial language to a relative direction.\n"
            f"3. State your conclusion: '{src}' is [direction] '{tgt}'.\n\n"
            "End with: Answer: [direction]\n\n"
            "Reasoning:"
        )
        response = self._generate(prompt)
        _log("LLM_RESPONSE", response)

        direction = extract_direction(response)
        if direction is None:
            fb = (f"Corpus: \"{corpus}\"\n{evidence}\n"
                  f"The relative direction of '{src}' relative to '{tgt}' is:\nAnswer: [")
            direction = extract_direction("Answer: [" + self._generate(fb))
            _log("FALLBACK", f"-> {direction}")

        trace["prediction"] = direction
        if log_fn:
            log_fn(f"\n  [CoT] FINAL: {direction}")
        return direction, trace


# =====================================================================
# 2. TREE-OF-THOUGHT
# =====================================================================
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "ToT"

    def reason(self, entity, log_fn=None):
        src, tgt, corpus = entity["source_entity"], entity["target_entity"], entity["corpus"]
        trace = {"strategy": "ToT", "mode": "osm", "branches": []}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [ToT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")
        evidence = _evidence(self.kg, src, tgt, corpus, log_fn)
        if evidence:
            _log("OSM_EVIDENCE", evidence)

        prompt = (
            "You are an expert in spatial and relative directions.\n\n"
            f"Determine the relative direction of '{src}' with respect to '{tgt}'. "
            "Explore THREE independent reasoning paths.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n"
            f"{_evidence_clause(evidence)}\n"
            "Label each branch as 'BRANCH N: <focus>', reason through it, then end it "
            "with 'Answer: [direction]'.\n\nBegin:"
        )
        branch_resp = self._generate(prompt)
        _log("BRANCHES_RAW", branch_resp)

        branches = re.findall(r"BRANCH\s+\d+\s*:(.*?)(?=BRANCH\s+\d+|$)",
                              branch_resp, re.DOTALL | re.IGNORECASE)
        weighted: Dict[str, float] = {}
        for i, b in enumerate(branches):
            pred = extract_direction(b)
            trace["branches"].append({"index": i + 1, "direction": pred})
            if pred:
                weighted[pred] = weighted.get(pred, 0.0) + _spatial_weight(b)

        final_dir = max(weighted, key=weighted.get) if weighted else None
        if final_dir is None:
            final_dir = extract_direction(branch_resp)
        if final_dir is None:
            fb = (f"Corpus: \"{corpus}\"\n{evidence}\n"
                  f"The relative direction of '{src}' relative to '{tgt}' is:\nAnswer: [")
            final_dir = extract_direction("Answer: [" + self._generate(fb))

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [ToT] FINAL: {final_dir}")
        return final_dir, trace


# =====================================================================
# 3. GRAPH-OF-THOUGHT
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

    def reason(self, entity, log_fn=None):
        src, tgt, corpus = entity["source_entity"], entity["target_entity"], entity["corpus"]
        trace = {"strategy": "GoT", "mode": "osm", "nodes": []}

        def _log(step, content):
            if log_fn:
                log_fn(f"\n  [GoT] -- {step} --\n{content}")

        _log("INPUT", f"{src} ? {tgt} | corpus: {corpus[:120]}")
        evidence = _evidence(self.kg, src, tgt, corpus, log_fn)
        if evidence:
            _log("OSM_EVIDENCE", evidence)

        prompt = (
            "You are an expert in spatial and relative directions.\n\n"
            f"Determine the relative direction of '{src}' with respect to '{tgt}'. "
            "Build a reasoning graph with FOUR thought nodes.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Possible directions: {VALID_LIST}\n"
            f"{_evidence_clause(evidence)}\n"
            "Label each node as 'THOUGHT N: <focus>', reason through it, then end it "
            "with 'Direction: [direction]'.\n\nBegin:"
        )
        phase1 = self._generate(prompt)
        _log("PHASE1_RAW", phase1)

        thoughts = re.findall(r"THOUGHT\s+\d+\s*:(.*?)(?=THOUGHT\s+\d+|$)",
                              phase1, re.DOTALL | re.IGNORECASE)
        nodes: List[ThoughtNode] = []
        weighted: Dict[str, float] = {}
        for i, t in enumerate(thoughts):
            pred = extract_direction(t)
            w = _spatial_weight(t)
            nodes.append(ThoughtNode(i, t.strip()[:400], pred, w))
            if pred:
                weighted[pred] = weighted.get(pred, 0.0) + w

        trace["nodes"] = [{"id": n.id, "direction": n.direction, "confidence": n.confidence}
                          for n in nodes]
        final_dir = max(weighted, key=weighted.get) if weighted else None
        if final_dir is None:
            final_dir = extract_direction(phase1)
        if final_dir is None:
            fb = (f"Corpus: \"{corpus}\"\n{evidence}\n"
                  f"The relative direction of '{src}' relative to '{tgt}' is:\nAnswer: [")
            final_dir = extract_direction("Answer: [" + self._generate(fb))

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [GoT] FINAL: {final_dir}")
        return final_dir, trace


# =====================================================================
# FACTORY
# =====================================================================
STRATEGY_MAP = {"cot": ChainOfThought, "tot": TreeOfThought, "got": GraphOfThought}


def get_strategy(name: str, kg=None, model_fn=None, max_new_tokens: int = 1024,
                 temperature: float = 0.1, **kwargs) -> ReasoningStrategy:
    cls = STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGY_MAP)}")
    return cls(kg=kg, model_fn=model_fn, max_new_tokens=max_new_tokens,
               temperature=temperature)
