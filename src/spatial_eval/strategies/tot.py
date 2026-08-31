"""Tree-of-Thought: explore independent branches, then adjudicate.

Three branches are generated separately -- each committing to a candidate label
with its justification -- and a final call weighs them against each other. The
branches are independent by construction: each is generated from the same
prompt with a different branch index folded into the seed, so they explore
different continuations rather than repeating one another.
"""
from __future__ import annotations

from ..data import Example
from ..parsing import parse_label
from .base import Context, Strategy, StrategyResult, register

N_BRANCHES = 3


@register
class TreeOfThought(Strategy):
    name = "tot"
    description = ("Three independent reasoning branches, then an adjudication "
                   "step that selects among them.")

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        return self._branch_prompt(ex, ctx, 0)

    def _branch_prompt(self, ex: Example, ctx: Context, i: int) -> str:
        angles = [
            "Reason from what the description states literally.",
            "Reason from the geometry of the two places and how they can be arranged.",
            "Reason by elimination: rule out every label that cannot hold, and see "
            "what survives.",
        ]
        return (self.task_header(ctx.relation, ctx.labels)
                + "\n" + self.question(ex)
                + f"\nApproach {i + 1}: {angles[i % len(angles)]}\n"
                  "Give a short justification, then commit to one label.\n"
                + self.answer_instruction())

    def run(self, ex: Example, ctx: Context) -> StrategyResult:
        trace, candidates = [], []
        for i in range(N_BRANCHES):
            prompt = self._branch_prompt(ex, ctx, i)
            # Offset the seed per branch, otherwise identical prompts would give
            # identical branches and the tree would collapse to a single path.
            raw = ctx.generate(prompt, ctx.seed + 1000 * (i + 1))
            lab, rule = parse_label(raw, ctx.labels, ctx.relation)
            candidates.append((lab, raw))
            trace.append({"step": f"branch_{i + 1}", "prompt": prompt,
                          "output": raw, "parsed": lab, "rule": rule})

        summary = "\n\n".join(
            f"Branch {i + 1} concluded: {lab or 'no clear answer'}\n"
            f"Its reasoning: {raw.strip()[:600]}"
            for i, (lab, raw) in enumerate(candidates))

        adjudicate = (self.task_header(ctx.relation, ctx.labels)
                      + "\n" + self.question(ex)
                      + "\nThree independent analyses were produced:\n\n"
                      + summary
                      + "\n\nWeigh these against the description and decide which "
                        "conclusion is best supported. You may pick a label none "
                        "of them chose if all three are wrong.\n"
                      + self.answer_instruction())
        raw = ctx.generate(adjudicate, ctx.seed)
        lab, rule = parse_label(raw, ctx.labels, ctx.relation)
        trace.append({"step": "adjudicate", "prompt": adjudicate,
                      "output": raw, "parsed": lab, "rule": rule})

        # If adjudication produced nothing usable, fall back to a branch
        # majority rather than discarding the work.
        if lab is None:
            votes = [c for c, _ in candidates if c]
            if votes:
                lab = max(set(votes), key=votes.count)
                rule = "branch_majority"

        return StrategyResult(prediction=lab, parse_rule=rule, raw=raw,
                              trace=trace, n_calls=N_BRANCHES + 1)
