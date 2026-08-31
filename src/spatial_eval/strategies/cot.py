"""Chain-of-Thought: one linear reasoning pass before the answer."""
from __future__ import annotations

from ..data import Example
from .base import Context, Strategy, register


@register
class ChainOfThought(Strategy):
    name = "cot"
    description = "Single linear reasoning chain, then the answer."

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        return (self.task_header(ctx.relation, ctx.labels)
                + "\n" + self.question(ex)
                + "\nThink step by step. State what the description says about "
                  "the arrangement, rule out the labels that cannot apply, then "
                  "commit to one.\n"
                + self.answer_instruction())
