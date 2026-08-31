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
    def question(ex: Example) -> str:
        return (f"Description: {ex.text}\n"
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
