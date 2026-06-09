"""
strategies_cardinal.py
================================================================================
CoT / ToT / GoT reasoning strategies for cardinal direction inference.

Each example is a natural-language question like:
  "I am running east along the north shore of the lake. In which direction is the lake?"
  Answer: south

The three strategies share a common `extract_direction()` function and wrap the
GPT-OSS-20B model (either via Ollama or a GPU callable).

For KG-enhanced configs (Config 3 / Config 4) pass a `kg` instance of
`CardinalKnowledgeGraph`; for base/LoRA-only configs pass `kg=None`.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatOllama = None
    HumanMessage = None


# =====================================================================
# OLLAMA CONFIG
# =====================================================================
BASE_URL   = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"


# =====================================================================
# CONSTANTS
# =====================================================================
VALID_DIRECTIONS = {
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
}

VALID_LIST = "north, south, east, west, north-east, north-west, south-east, south-west"

# Canonical aliases used in model output
_DIR_ALIASES = {
    "northeast":  "north-east",
    "northwest":  "north-west",
    "southeast":  "south-east",
    "southwest":  "south-west",
    "ne":         "north-east",
    "nw":         "north-west",
    "se":         "south-east",
    "sw":         "south-west",
    "n":          "north",
    "s":          "south",
    "e":          "east",
    "w":          "west",
}

COMPASS_RULES = """Compass / Cardinal Direction Rules:
  OPPOSITE PAIRS:
    north ↔ south,  east ↔ west
    north-east ↔ south-west,  north-west ↔ south-east

  PERPENDICULAR PAIRS (when facing a direction):
    Facing north  → right is east,  left is west,  behind is south
    Facing south  → right is west,  left is east,  behind is north
    Facing east   → right is south, left is north,  behind is west
    Facing west   → right is north, left is south,  behind is east
    Facing north-east → right is south-east, left is north-west, behind is south-west
    Facing north-west → right is north-east, left is south-west, behind is south-east
    Facing south-east → right is south-west, left is north-east, behind is north-west
    Facing south-west → right is north-west, left is south-east, behind is north-east

  SHORE-TO-BODY RULES (critical):
    Walking along the NORTH shore  → water body is to the SOUTH
    Walking along the SOUTH shore  → water body is to the NORTH
    Walking along the EAST shore   → water body is to the WEST
    Walking along the WEST shore   → water body is to the EAST
    Walking along the NORTH-EAST shore → water body is to the SOUTH-WEST
    Walking along the NORTH-WEST shore → water body is to the SOUTH-EAST
    Walking along the SOUTH-EAST shore → water body is to the NORTH-WEST
    Walking along the SOUTH-WEST shore → water body is to the NORTH-EAST

  TURN-AROUND RULE:
    "turns around" / "heads back in the direction they came from"
    → direction of travel REVERSES (opposite direction), but the SHORE stays the same.
    → The shore-to-body relationship is unchanged by the reversal.
    → Example: walking west along south shore, turns around → now walking east along south shore
      → water is still to the NORTH (south shore → north).
"""

RULES_BLOCK = """Reasoning Rules:
1. IDENTIFY SHORE: extract which shore of the water body the person is on
   (north shore, south shore, east shore, west shore, etc.)

2. APPLY SHORE-TO-BODY RULE: the water body lies on the OPPOSITE side of the shore name.
   (e.g. standing on the north shore → water is to the south)

3. HANDLE TURN-AROUND: if the question says the person "turns around" or "heads back",
   the direction of travel reverses, but the shore does NOT change.
   The answer depends on the shore, not the direction of travel.

4. HANDLE "SEA" vs "LAKE" vs "ISLAND": the reasoning is identical regardless of the
   water body type. Focus on which shore the person is on.

5. IGNORE WALKING DIRECTION FOR THE FINAL ANSWER: the direction of travel (running east,
   walking west, etc.) is context; the SHORE determines where the water is.

6. PICK EXACTLY ONE direction from: north, south, east, west,
   north-east, north-west, south-east, south-west

7. End with: Answer: [direction]
"""


# =====================================================================
# CARDINAL KNOWLEDGE GRAPH (rule-based, no API calls)
# =====================================================================
class CardinalKnowledgeGraph:
    """
    Rule-based knowledge graph for cardinal direction reasoning.
    Provides structured directional rules as text evidence.
    No external API calls — fully deterministic.
    """

    def gather_evidence(self, question: str, log_fn=None) -> str:
        lines = [
            f'Question: "{question}"\n',
            "--- Cardinal Direction Evidence (Rule-Based KG) ---",
            COMPASS_RULES,
        ]
        evidence = "\n".join(lines)
        if log_fn:
            log_fn(evidence)
        return evidence


# =====================================================================
# HELPERS
# =====================================================================
def extract_direction(text: str) -> Optional[str]:
    """Extract a valid cardinal direction from model output."""
    if not text:
        return None
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)
    text_lower = text_clean.lower()

    patterns = [
        r"answer\s*[:=]\s*\[?([a-z\-]+)\]?",
        r"direction\s*[:=]\s*\[?([a-z\-]+)\]?",
        r"conclusion\s*[:=]\s*\[?([a-z\-]+)\]?",
        r"the\s+(?:answer|direction)\s+is\s+[:\[]?([a-z\-]+)",
        r"therefore[,\s]+(?:the\s+)?(?:answer|direction)\s+is\s+[:\[]?([a-z\-]+)",
        r"final\s+(?:answer|direction)\s*[:=]\s*\[?([a-z\-]+)\]?",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text_lower))
        if matches:
            raw = matches[-1].group(1).strip().rstrip(".,;:")
            if raw in VALID_DIRECTIONS:
                return raw
            if raw in _DIR_ALIASES:
                return _DIR_ALIASES[raw]

    found = []
    for d in VALID_DIRECTIONS:
        idx = text_lower.rfind(d)
        if idx != -1:
            found.append((idx, d))
    for alias, canonical in _DIR_ALIASES.items():
        if len(alias) <= 2:
            # single/double-letter abbreviations: require word boundary to avoid
            # matching "s" inside "west" or "e" inside "east"
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text_lower):
                found.append((m.start(), canonical))
        else:
            idx = text_lower.rfind(alias)
            if idx != -1:
                found.append((idx, canonical))
    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][1]
    return None


def _linguistic_weight(text: str) -> float:
    """Score a reasoning branch by spatial reasoning signal words."""
    keywords = [
        "shore", "north shore", "south shore", "east shore", "west shore",
        "opposite", "perpendicular", "compass", "bearing", "direction",
        "turns around", "reverses", "water body", "faces", "behind",
    ]
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return 1.0 + 0.3 * min(hits, 5)


def llm_generate(prompt: str, llm, timeout_seconds: int = 90) -> str:
    import threading
    result = {"text": ""}

    def run():
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result["text"] = response.content
        except Exception as e:
            print(f"\n[LLM CRITICAL ERROR]: {e}")

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout_seconds)
    return result["text"]


# =====================================================================
# BASE CLASS
# =====================================================================
class ReasoningStrategy(ABC):
    def __init__(self, kg: Optional[CardinalKnowledgeGraph] = None,
                 temperature: float = 0.1, max_new_tokens: int = 512,
                 base_url: str = BASE_URL, model_name: str = MODEL_NAME,
                 model_fn=None):
        self.kg = kg
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.model_name = model_name
        self.model_fn = model_fn
        self.llm = None if model_fn is not None else ChatOllama(
            model=model_name, temperature=temperature, base_url=base_url
        )

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]: ...

    def _generate(self, prompt: str) -> str:
        if self.model_fn is not None:
            return self.model_fn(prompt)
        return llm_generate(prompt, llm=self.llm, timeout_seconds=90)

    def _kg_evidence(self, question: str, log_fn=None) -> str:
        if self.kg is None:
            return ""
        return self.kg.gather_evidence(question, log_fn=log_fn)


# =====================================================================
# 1. CHAIN-OF-THOUGHT (CoT)
# =====================================================================
class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "CoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question  = entity["question"]
        ground_truth = entity.get("direction", "?")
        trace = {"strategy": "CoT", "steps": []}

        def _log(step: str, content: str):
            trace["steps"].append({"step": step, "content": content})
            if log_fn:
                log_fn(f"\n  [CoT] ── {step} ──\n{content}")

        _log("INPUT", f"Q: {question}  |  GT: {ground_truth}")

        kg_evidence = self._kg_evidence(question, log_fn=log_fn)
        kg_block    = f"\n{kg_evidence}\n" if kg_evidence else ""

        prompt = f"""You are an expert in spatial reasoning and cardinal directions.

{COMPASS_RULES}

{RULES_BLOCK}
{kg_block}
Question: {question}

Think step-by-step:
1. Identify which shore of the water body the person is on (north, south, east, etc.)
2. Apply the shore-to-body rule: water lies OPPOSITE the shore name.
3. Check if the person turns around — if so, the shore stays the same, only the travel direction reverses.
4. State the direction from the person to the water body.

Valid answers: {VALID_LIST}

Reasoning:"""

        response = self._generate(prompt)
        _log("LLM_REASONING", response)

        direction = extract_direction(response)
        if direction is None:
            fallback = self._generate(
                f"Question: {question}\nThe direction to the water body is:\nAnswer: ["
            )
            direction = extract_direction(fallback)

        trace["prediction"] = direction
        if log_fn:
            log_fn(f"\n  [CoT] ✅ FINAL PREDICTION: {direction}")
        return direction, trace


# =====================================================================
# 2. TREE-OF-THOUGHT (ToT)
# =====================================================================
class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "ToT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question     = entity["question"]
        ground_truth = entity.get("direction", "?")
        trace        = {"strategy": "ToT", "branches": [], "vote": None}

        def _log(step: str, content: str):
            if log_fn:
                log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        _log("INPUT", f"Q: {question}  |  GT: {ground_truth}")

        kg_evidence = self._kg_evidence(question, log_fn=log_fn)
        kg_block    = f"\n{kg_evidence}\n" if kg_evidence else ""

        branch_prompt = f"""You are an expert in spatial reasoning and cardinal directions.

{COMPASS_RULES}

{RULES_BLOCK}
{kg_block}
Question: {question}

Explore THREE different reasoning branches.

BRANCH 1: Shore-based geometric approach
  - Identify the shore name and apply the opposite-direction rule directly.
Suggested direction: [direction]

BRANCH 2: Embodied / perspective approach
  - Imagine standing on that shore facing the travel direction.
  - Determine which side the water is on relative to the body.
Suggested direction: [direction]

BRANCH 3: Turn-around / reversal analysis
  - Check whether the person turns around or reverses direction.
  - Re-derive which direction the water is from the person after any reversal.
Suggested direction: [direction]

Valid answers: {VALID_LIST}

Begin:"""

        branch_response = self._generate(branch_prompt)
        _log("BRANCHES_RAW", branch_response)

        branch_pattern = r"BRANCH\s+\d+\s*:(.*?)(?=BRANCH\s+\d+|$)"
        branches = re.findall(branch_pattern, branch_response, re.DOTALL | re.IGNORECASE)

        votes = []
        branch_texts = []
        for i, b_text in enumerate(branches):
            pred = extract_direction(b_text)
            votes.append(pred)
            branch_texts.append(b_text)
            trace["branches"].append({"index": i + 1, "direction": pred,
                                       "content": b_text.strip()[:400]})
            _log(f"BRANCH_{i + 1}", f"Direction: {pred}\n{b_text.strip()}")

        if not votes or all(v is None for v in votes):
            fallback = self._generate(
                f"Question: {question}\nThe direction to the water body is:\nAnswer: ["
            )
            final_dir = extract_direction(fallback)
        else:
            weighted: Dict[str, float] = {}
            for pred, b_text in zip(votes, branch_texts):
                if pred is None:
                    continue
                weighted[pred] = weighted.get(pred, 0.0) + _linguistic_weight(b_text)
            final_dir = max(weighted, key=weighted.get) if weighted else None

        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [ToT] ✅ FINAL PREDICTION: {final_dir}")
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
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    node_type: str = "thought"


class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "GoT"

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        question     = entity["question"]
        ground_truth = entity.get("direction", "?")

        thought_graph: List[ThoughtNode] = []
        next_id = 0
        trace = {"strategy": "GoT", "nodes": [], "aggregation": None}

        def _log(step: str, content: str):
            if log_fn:
                log_fn(f"\n  [GoT] ── {step} ──\n{content}")

        def _add_node(**kwargs) -> ThoughtNode:
            nonlocal next_id
            node = ThoughtNode(id=next_id, **kwargs)
            thought_graph.append(node)
            next_id += 1
            return node

        _log("INPUT", f"Q: {question}  |  GT: {ground_truth}")

        kg_evidence = self._kg_evidence(question, log_fn=log_fn)
        kg_block    = f"\n{kg_evidence}\n" if kg_evidence else ""

        phase1_prompt = f"""You are an expert in spatial reasoning and cardinal directions.

{COMPASS_RULES}

{RULES_BLOCK}
{kg_block}
Question: {question}

Generate FOUR distinct thought nodes to analyze this direction question.

THOUGHT 1: Shore identification
  - Extract which shore of the water body the person is on.
  - State: "The person is on the [X] shore."
Direction: [direction]

THOUGHT 2: Shore-opposite rule
  - Apply: water body lies OPPOSITE the shore name.
  - State: "Opposite of [shore] is [direction]."
Direction: [direction]

THOUGHT 3: Turn-around check
  - Does the question mention "turns around" or "heads back"?
  - If yes, re-verify the shore stays the same and travel direction reverses.
  - If no turn-around, confirm the shore-based answer.
Direction: [direction]

THOUGHT 4: Aggregation
  - Reconcile THOUGHT 1, 2, 3.
  - State the final direction to the water body.
Direction: [direction]

Valid answers: {VALID_LIST}

Begin:"""

        phase1_resp = self._generate(phase1_prompt)
        _log("PHASE1_RAW", phase1_resp)

        thought_pattern = r"THOUGHT\s+\d+\s*:(.*?)(?=THOUGHT\s+\d+|$)"
        thoughts = re.findall(thought_pattern, phase1_resp, re.DOTALL | re.IGNORECASE)

        for t_text in thoughts:
            pred   = extract_direction(t_text)
            weight = _linguistic_weight(t_text)
            _add_node(content=t_text.strip()[:500], direction=pred, confidence=weight)

        weighted: Dict[str, float] = {}
        for node in thought_graph:
            if node.direction:
                weighted[node.direction] = weighted.get(node.direction, 0.0) + node.confidence
        final_dir = max(weighted, key=weighted.get) if weighted else None

        trace["nodes"] = [
            {"id": n.id, "direction": n.direction, "confidence": n.confidence}
            for n in thought_graph
        ]
        trace["prediction"] = final_dir
        if log_fn:
            log_fn(f"\n  [GoT] ✅ FINAL PREDICTION: {final_dir}")
        return final_dir, trace


# =====================================================================
# STRATEGY FACTORY
# =====================================================================
STRATEGY_MAP = {
    "cot": ChainOfThought,
    "tot": TreeOfThought,
    "got": GraphOfThought,
}


def get_strategy(name: str, kg: Optional[CardinalKnowledgeGraph] = None,
                 model_fn=None, **kwargs) -> ReasoningStrategy:
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name_lower](kg=kg, model_fn=model_fn, **kwargs)
