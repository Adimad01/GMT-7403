"""Graph-of-Thought: generate thoughts, cross-link them, then synthesise.

Where Tree-of-Thought keeps branches independent until a final vote, this
strategy explicitly asks the model to relate the partial thoughts to each other
-- to note where they agree, where they conflict, and what a conflict implies --
before committing. The distinguishing feature is the edges between thoughts,
not the number of calls.
"""
from __future__ import annotations

from ..data import Example
from ..parsing import parse_label
from .base import Context, Strategy, StrategyResult, register

N_THOUGHTS = 3


@register
class GraphOfThought(Strategy):
    name = "got"
    description = ("Several partial thoughts, explicitly cross-linked, then a "
                   "synthesis over the resulting graph.")

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        return self._thought_prompt(ex, ctx, 0)

    def _thought_prompt(self, ex: Example, ctx: Context, i: int) -> str:
        facets = [
            "What does the description assert about the two places, in plain terms? "
            "Do not name a label yet.",
            "What kind of spatial configuration do these two places have, given "
            "what they are? Do not name a label yet.",
            "Which labels are clearly impossible here, and why? Do not commit to "
            "a final one.",
        ]
        return (self.task_header(ctx.relation, ctx.labels)
                + "\n" + self.question(ex)
                + f"\nPartial analysis {i + 1}. {facets[i % len(facets)]}\n"
                  "Answer in at most four sentences.\n")

    def run(self, ex: Example, ctx: Context) -> StrategyResult:
        trace, thoughts = [], []
        for i in range(N_THOUGHTS):
            prompt = self._thought_prompt(ex, ctx, i)
            raw = ctx.generate(prompt, ctx.seed + 2000 * (i + 1))
            thoughts.append(raw.strip())
            trace.append({"step": f"thought_{i + 1}", "prompt": prompt, "output": raw})

        nodes = "\n\n".join(f"Thought {i + 1}: {t[:600]}" for i, t in enumerate(thoughts))
        synth = (self.task_header(ctx.relation, ctx.labels)
                 + "\n" + self.question(ex)
                 + "\nThree partial analyses were produced. None is a final answer.\n\n"
                 + nodes
                 + "\n\nRelate them to one another: state where they agree, where "
                   "they conflict, and what each conflict implies about the "
                   "correct label. Then resolve the graph into one answer.\n"
                 + self.answer_instruction())
        raw = ctx.generate(synth, ctx.seed)
        lab, rule = parse_label(raw, ctx.labels, ctx.relation)
        trace.append({"step": "synthesise", "prompt": synth, "output": raw,
                      "parsed": lab, "rule": rule})
        return StrategyResult(prediction=lab, parse_rule=rule, raw=raw,
                              trace=trace, n_calls=N_THOUGHTS + 1)
