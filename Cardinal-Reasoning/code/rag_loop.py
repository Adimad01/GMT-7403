"""
rag_loop.py — bounded per-step Retrieval-Augmented reasoning (Exp 6).
================================================================================
Implements the "KG @ inference" mechanism (Pan et al. §4.2): the KG is kept
separate from the model and queried *during* reasoning.  The model may emit a
line `NEXT_QUERY: <place name>` to request more OSM facts; the loop resolves it
through the KG, appends `RETRIEVED (...): ...`, and lets the model continue.
Bounded to `max_rounds` follow-up retrievals so cost stays predictable.

This single implementation serves all three domains (Topological, Cardinal,
Relative) and all three strategy kinds (CoT / ToT / GoT).  Each domain supplies
a DomainSpec describing how to parse an entity, the valid-label list, the task
noun phrase, and the label-extraction function.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, List


_NEXT_QUERY_RE = re.compile(r"NEXT_QUERY\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class DomainSpec:
    """Per-domain configuration for the generic RAG loop."""
    task_noun: str                       # e.g. "DE-9IM topological predicate"
    valid_list: str                      # human-readable valid label list
    extract_fn: Callable[[str], Optional[str]]   # text -> canonical label | None
    parse_entity: Callable[[Dict[str, Any]], Tuple[str, str, str]]
    #   parse_entity(entity) -> (place_a, place_b, corpus_or_sentence)


# kind -> how many reasoning units to ask for + the framing verb
_KIND_INSTRUCTION = {
    "cot": "Reason step by step.",
    "tot": "Explore THREE independent reasoning branches, then converge on one answer.",
    "got": "Build FOUR reasoning thought-nodes, then aggregate them into one answer.",
}


def _format_place(name: str, data: Optional[dict]) -> str:
    if not data:
        return f"{name}: No OSM data available."
    parts = [f"{name}: Lat {data.get('lat')}, Lon {data.get('lon')}"]
    bbox = data.get("boundingbox")
    if bbox and len(bbox) == 4:
        parts.append(f"bbox[S,N,W,E]=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
    cat = f"{data.get('class')}/{data.get('type')}"
    parts.append(f"category={cat}")
    hierarchy = data.get("hierarchy")
    if hierarchy:
        parts.append("hierarchy=" + ", ".join(f"{k}:{v}" for k, v in hierarchy.items()))
    return "  ".join(parts)


class RAGStrategy:
    """Generic per-step RAG strategy. Quacks like the domain strategy classes:
    exposes `.name` and `.reason(entity, log_fn) -> (label, trace)`.
    """

    def __init__(self, kind: str, kg, model_fn: Callable[[str], str],
                 spec: DomainSpec, max_rounds: int = 2):
        self.kind = kind.lower()
        self.kg = kg
        self.model_fn = model_fn
        self.spec = spec
        self.max_rounds = max_rounds

    @property
    def name(self) -> str:
        return self.kind.upper()

    def _build_initial_prompt(self, place_a: str, place_b: str, corpus: str,
                              base_evidence: str) -> str:
        instr = _KIND_INSTRUCTION.get(self.kind, _KIND_INSTRUCTION["cot"])
        return (
            "You are an expert in geospatial reasoning.\n\n"
            f"Task: determine the {self.spec.task_noun} for "
            f"'{place_a}' relative to '{place_b}'.\n\n"
            f"Corpus: \"{corpus}\"\n\n"
            f"Valid answers: {self.spec.valid_list}\n\n"
            f"{base_evidence}\n\n"
            f"{instr}\n"
            "If you need additional geographic facts about a specific place to decide, "
            "request them by writing a line exactly:\n"
            "  NEXT_QUERY: <place name>\n"
            "and then stop. Otherwise, finish with:\n"
            "  Answer: [<one valid answer>]\n\n"
            "Begin:"
        )

    def reason(self, entity: Dict[str, Any], log_fn=None) -> Tuple[Optional[str], Dict]:
        place_a, place_b, corpus = self.spec.parse_entity(entity)
        trace: Dict[str, Any] = {"strategy": self.name, "mode": "rag_inference",
                                 "rounds": [], "queries": []}

        def _log(step: str, content: str):
            if log_fn:
                log_fn(f"\n  [RAG-{self.name}] ── {step} ──\n{content}")

        base_evidence = self.kg.gather_evidence(place_a, place_b, sentence=corpus,
                                                entity=entity, log_fn=None)
        _log("BASE_EVIDENCE", base_evidence or "(none)")

        prompt = self._build_initial_prompt(place_a, place_b, corpus, base_evidence)
        transcript: List[str] = []
        answer: Optional[str] = None

        for round_i in range(self.max_rounds + 1):
            response = self.model_fn(prompt)
            transcript.append(response)
            trace["rounds"].append({"round": round_i, "response": response[:600]})
            _log(f"ROUND_{round_i}", response)

            answer = self.spec.extract_fn(response)
            if answer is not None:
                break

            if round_i == self.max_rounds:
                break  # out of retrieval budget

            m = _NEXT_QUERY_RE.search(response)
            if not m:
                break  # model neither answered nor asked — stop and fall back

            query_place = m.group(1).strip().strip("[]\"'").rstrip(".")
            trace["queries"].append(query_place)
            data = self.kg.fetch(query_place) if hasattr(self.kg, "fetch") else None
            retrieved = _format_place(query_place, data)
            _log(f"RETRIEVED_{round_i}", retrieved)

            prompt = (
                f"{prompt}\n{response}\n\n"
                f"RETRIEVED ({query_place}): {retrieved}\n\n"
                "Use this new evidence. If you still need another place, write another "
                "NEXT_QUERY line; otherwise finish now with: Answer: [<one valid answer>]\n"
            )

        # Final fallback: scan the whole transcript, then ask directly.
        if answer is None:
            answer = self.spec.extract_fn("\n".join(transcript))
        if answer is None:
            direct = self.model_fn(
                f"Corpus: \"{corpus}\"\n{base_evidence}\n"
                f"The {self.spec.task_noun} for '{place_a}' relative to '{place_b}' is:\n"
                "Answer: ["
            )
            answer = self.spec.extract_fn("Answer: [" + direct)
            _log("FALLBACK_DIRECT", f"{direct[:200]} → {answer}")

        trace["prediction"] = answer
        if log_fn:
            log_fn(f"\n  [RAG-{self.name}] ✅ FINAL: {answer}")
        return answer, trace
