"""
strategies_relative.py
================================================================================
CoT / ToT / GoT reasoning strategies for the SpatialEvalLLM relative reasoning
task (ring / square / tree navigation).

No external KG is used — the task is pure multi-step positional reasoning.
"""
import re
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# ANSWER EXTRACTION
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> Optional[str]:
    """Extract the predicted object from model output."""
    if not text:
        return None
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Look for explicit answer markers (case-insensitive)
    patterns = [
        r"answer\s*[:=]\s*\[?([^\]\n]+)\]?",
        r"final\s+(?:answer|position|item)\s*[:=]\s*\[?([^\]\n]+)\]?",
        r"you\s+(?:will\s+)?find\s*[:=]?\s*(?:a\s+|an\s+)?([^\.\n,]+)",
        r"(?:result|conclusion)\s*[:=]\s*\[?([^\]\n]+)\]?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".").rstrip(",").strip()
            if val:
                return val.lower()

    # Last non-empty line as fallback
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1].rstrip(".").rstrip(",").strip()
        return last.lower() if last else None
    return None


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    return s.lower().strip().rstrip(".").rstrip(",").strip()


# ---------------------------------------------------------------------------
# BASE STRATEGY
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert at spatial navigation and positional reasoning.
You will be given a question that describes a path (ring, square grid, or tree) with
named objects at positions. You need to carefully track each move step by step and
determine which object you find at the final position.

Rules:
- Track your position precisely after EACH movement.
- For a ring/circle: moves wrap around.
- For a square grid: moves follow the grid adjacency.
- For a tree: moves follow parent-child edges.
- Your final answer must be the EXACT object name from the question.
- End your response with: Answer: [object name]
"""


# ---------------------------------------------------------------------------
# 1. CHAIN-OF-THOUGHT (CoT)
# ---------------------------------------------------------------------------
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "CoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question = entity["question"]
        trace = {"strategy": "CoT"}

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n\n"
            "Work through this step by step, tracking your exact position after each move.\n"
            "Answer: ["
        )

        response = self._generate(prompt)
        if log_fn:
            log_fn(f"\n  [CoT] Response:\n{response}")

        predicted = extract_answer("Answer: [" + response)

        # Fallback: ask directly
        if predicted is None:
            fallback = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nThe object you will find is:\nAnswer: ["
            fb_resp = self._generate(fallback)
            predicted = extract_answer("Answer: [" + fb_resp)
            if log_fn:
                log_fn(f"\n  [CoT] Fallback: {fb_resp[:200]} -> {predicted}")

        trace["prediction"] = predicted
        return predicted, trace


# ---------------------------------------------------------------------------
# 2. TREE-OF-THOUGHT (ToT)
# ---------------------------------------------------------------------------
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "ToT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question = entity["question"]
        trace = {"strategy": "ToT", "branches": []}

        branch_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n\n"
            "Generate THREE independent step-by-step solutions. Each solution must track "
            "every move carefully. Format:\n\n"
            "SOLUTION 1:\n[step-by-step tracking]\nAnswer: [object]\n\n"
            "SOLUTION 2:\n[step-by-step tracking]\nAnswer: [object]\n\n"
            "SOLUTION 3:\n[step-by-step tracking]\nAnswer: [object]\n\n"
            "Begin:"
        )

        branch_resp = self._generate(branch_prompt)
        if log_fn:
            log_fn(f"\n  [ToT] Branches:\n{branch_resp}")

        # Extract answers from each solution
        solution_pattern = r"SOLUTION\s+\d+\s*:\s*(.*?)(?=SOLUTION\s+\d+|$)"
        solutions = re.findall(solution_pattern, branch_resp, re.DOTALL | re.IGNORECASE)
        if not solutions:
            # Try splitting on "Answer:" occurrences
            solutions = re.split(r"(?=Answer\s*:)", branch_resp, flags=re.IGNORECASE)

        votes: Dict[str, int] = {}
        for sol in solutions:
            pred = extract_answer(sol)
            if pred:
                norm = normalize(pred)
                votes[norm] = votes.get(norm, 0) + 1

        final_pred = max(votes, key=votes.get) if votes else None

        # Fallback
        if final_pred is None:
            final_pred = extract_answer(branch_resp)
        if final_pred is None:
            synth = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer: ["
            final_pred = extract_answer("Answer: [" + self._generate(synth))

        trace["votes"] = votes
        trace["prediction"] = final_pred
        if log_fn:
            log_fn(f"\n  [ToT] Votes: {votes}  -> {final_pred}")

        return final_pred, trace


# ---------------------------------------------------------------------------
# 3. GRAPH-OF-THOUGHT (GoT)
# ---------------------------------------------------------------------------
class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "GoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question = entity["question"]
        trace = {"strategy": "GoT", "nodes": []}

        phase1_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n\n"
            "Build a position-tracking graph. For each move, create a NODE:\n\n"
            "NODE 0: Starting position = [object]\n"
            "NODE 1: After move 1 -- [direction/steps] -> [object]\n"
            "NODE 2: After move 2 -- [direction/steps] -> [object]\n"
            "...\n"
            "NODE N: Final position = [object]\n\n"
            "Begin building the graph:"
        )

        phase1_resp = self._generate(phase1_prompt)
        if log_fn:
            log_fn(f"\n  [GoT] Phase1:\n{phase1_resp}")

        # Extract from the last NODE entry
        node_pattern = r"NODE\s+\d+\s*:.*?(?:=|->|--)\s*\[?([^\]\n]+)\]?"
        nodes = re.findall(node_pattern, phase1_resp, re.IGNORECASE | re.DOTALL)

        final_pred = None
        if nodes:
            # Take the last node's value
            final_pred = normalize(nodes[-1].strip().rstrip("."))
            if log_fn:
                log_fn(f"\n  [GoT] Nodes found: {nodes[-3:]}  -> {final_pred}")

        # Fallback 1: extract from full response
        if not final_pred:
            final_pred = extract_answer(phase1_resp)
            if log_fn:
                log_fn(f"\n  [GoT] Fallback1 from full response: {final_pred}")

        # Fallback 2: direct synthesis
        if not final_pred:
            synth = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer: ["
            final_pred = extract_answer("Answer: [" + self._generate(synth))
            if log_fn:
                log_fn(f"\n  [GoT] Fallback2 synthesis: {final_pred}")

        trace["prediction"] = final_pred
        return final_pred, trace


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------
STRATEGY_MAP = {"cot": ChainOfThought, "tot": TreeOfThought, "got": GraphOfThought}

def get_strategy(name: str, model_fn=None, max_new_tokens: int = 512,
                 temperature: float = 0.1, **kwargs) -> ReasoningStrategy:
    cls = STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGY_MAP)}")
    return cls(model_fn=model_fn, max_new_tokens=max_new_tokens, temperature=temperature)
