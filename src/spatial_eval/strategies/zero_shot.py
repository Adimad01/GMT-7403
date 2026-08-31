"""Zero-shot: ask directly, no demonstrations, no reasoning scaffold."""
from __future__ import annotations

from ..data import Example
from .base import Context, Strategy, register


@register
class ZeroShot(Strategy):
    name = "zero_shot"
    description = "Direct question, no demonstrations and no reasoning scaffold."

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        return (self.task_header(ctx.relation, ctx.labels)
                + "\n" + self.question(ex)
                + "\nAnswer with the label only.\n"
                + self.answer_instruction())
