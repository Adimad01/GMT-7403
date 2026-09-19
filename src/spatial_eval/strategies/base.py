"""Strategy interface and registry.

A strategy owns *how the model is prompted*, nothing else. It never chooses
which examples to use and never touches the dataset -- that is fixed by the
manifests, so the only thing that varies between arms is the prompting.

Add a strategy by subclassing `Strategy`, giving it a `name`, and decorating it
with `@register`. It becomes runnable immediately; no other file changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..data import Demo, Example
from ..parsing import parse_label

_REGISTRY: dict[str, type["Strategy"]] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type["Strategy"]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. available: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class StrategyResult:
    prediction: str | None          # parsed label, or None if unparseable
    parse_rule: str                 # which parsing rule fired
    raw: str                        # final raw completion
    trace: list[dict] = field(default_factory=list)   # every call made
    n_calls: int = 0


@dataclass
class Context:
    relation: str
    labels: list[str]
    seed: int
    generate: callable              # (prompt, seed) -> str
    demos: dict[str, list[Demo]] | None = None
    kg: dict[str, dict] | None = None   # stored facts, or None for no store


class Strategy(ABC):
    name: str = ""
    description: str = ""

    # ---- shared prompt scaffolding -------------------------------------
    @staticmethod
    def task_header(relation: str, labels: list[str]) -> str:
        noun = {"topological": "topological relation",
                "cardinal": "cardinal direction",
                "relative": "relative direction"}[relation]
        return (f"You are given a description of the spatial arrangement of two "
                f"places. Identify the {noun} that holds between them.\n\n"
                f"Allowed answers: {', '.join(labels)}\n")

    @staticmethod
    def facts(ex: Example, ctx: "Context") -> str:
        """Stored facts about the places this question names, or nothing.

        Only those places. The store holds every evaluation entity, and
        pasting all of them would bury the three that matter in several
        hundred that do not, besides costing the context window.

        The relation under test is never among these facts: the store is
        built without it, and checked by scripts/check_kg.py.
        """
        if not ctx.kg:
            return ""
        names = [n for n in (ex.observer, ex.subject, ex.target) if n]
        lines = []
        for name in dict.fromkeys(names):          # keep order, drop repeats
            f = ctx.kg.get(name)
            if not f:
                continue
            bits = [f.get("kind", "place"),
                    f"centre {f['lat']:.2f}, {f['lon']:.2f}"]
            if f.get("extent_km"):
                bits.append(f"extent {f['extent_km'][0]}x{f['extent_km'][1]} km")
            if f.get("bbox"):
                b = f["bbox"]
                bits.append(f"bounds S{b[0]:.2f} W{b[1]:.2f} N{b[2]:.2f} E{b[3]:.2f}")
            if f.get("context"):
                bits.append("in " + ", ".join(f["context"]))
            lines.append(f"- {name}: " + "; ".join(bits))
        if not lines:
            return ""
        return "Known facts about these places:\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def question(ex: Example, ctx: "Context | None" = None) -> str:
        head = Strategy.facts(ex, ctx) if ctx is not None else ""
        return (head
                + f"Description: {ex.text}\n"
                f"Subject: {ex.subject}\n"
                f"Object: {ex.target}\n"
                f"Question: what is the relation of the subject with respect to "
                f"the object?\n")

    @staticmethod
    def answer_instruction() -> str:
        return ("End your reply with a single line in exactly this form:\n"
                "ANSWER: <label>\n")

    # ---- interface ------------------------------------------------------
    @abstractmethod
    def build_prompt(self, ex: Example, ctx: Context) -> str:
        """The first (often only) prompt."""

    def run(self, ex: Example, ctx: Context) -> StrategyResult:
        """Default single-call execution. Multi-step strategies override this."""
        prompt = self.build_prompt(ex, ctx)
        raw = ctx.generate(prompt, ctx.seed)
        label, rule = parse_label(raw, ctx.labels, ctx.relation)
        return StrategyResult(prediction=label, parse_rule=rule, raw=raw,
                              trace=[{"step": "answer", "prompt": prompt, "output": raw}],
                              n_calls=1)
