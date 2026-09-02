"""Graph-of-Thought: generate thoughts, cross-link them, then synthesise.

Where Tree-of-Thought keeps branches independent until a final vote, this
strategy explicitly asks the model to relate the partial thoughts to each other
-- where they agree, where they conflict, and what a conflict implies -- before
committing. The distinguishing feature is the edges between thoughts, not the
number of calls.

Answer extraction is deliberately defensive. The first version of this strategy
lost 21 completions to unparseable output across a full run, while
Tree-of-Thought lost none -- not because the reasoning was worse, but because
ToT had a fallback and this did not. Measured on the same rows, excluding the
unparseable ones moved cardinal accuracy from 91.7% to 99.1%, i.e. almost the
entire apparent deficit was a formatting failure. Two changes address it:

  * the synthesis step is given a tight length budget, so it cannot spend its
    whole token allowance on prose and get truncated before the answer line;
  * if synthesis still yields nothing parseable, an extraction call asks only
    for the label -- with enough tokens to actually reach it. The first version
    capped that call at 24 tokens and it failed on 7 of the 7 rows it fired on,
    truncated before it could answer.
"""
from __future__ import annotations

from ..data import Example
from ..parsing import parse_label
from .base import Context, Strategy, StrategyResult, register

N_THOUGHTS = 3
THOUGHT_CHARS = 500          # how much of each thought is carried into synthesis
# The extraction call needs room to REACH its answer, not just to state it.
# This model opens with reasoning prose, so a tight budget truncates it before
# any ANSWER: line appears. Set to 24 in the first attempt, the rescue failed on
# 7 of 7 rows it fired on; the call was being cut off mid-thought.
EXTRACT_TOKENS = 256


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
                  "Answer in at most three sentences.\n")

    def _synthesis_prompt(self, ex: Example, ctx: Context, thoughts: list[str]) -> str:
        nodes = "\n\n".join(f"Thought {i + 1}: {t[:THOUGHT_CHARS]}"
                            for i, t in enumerate(thoughts))
        return (self.task_header(ctx.relation, ctx.labels)
                + "\n" + self.question(ex)
                + "\nThree partial analyses were produced. None is a final answer.\n\n"
                + nodes
                + "\n\nIn at most four sentences, say where these analyses agree, "
                  "where they conflict, and which reading the description supports. "
                  "Be brief -- the final line matters more than the discussion.\n\n"
                + f"Then, on its own final line, write exactly:\nANSWER: <one of "
                  f"{', '.join(ctx.labels)}>\n")

    def _extract_prompt(self, ex: Example, ctx: Context, synthesis: str) -> str:
        """Last resort: ask only for the label.

        Given enough tokens to reach an answer, but told plainly not to spend
        them explaining. Constraining the budget instead of the instruction is
        what broke the first version.
        """
        return (f"An analysis of a spatial relation concluded:\n\n"
                f"{synthesis.strip()[:800]}\n\n"
                f"Which single label does that conclusion support?\n"
                f"Allowed answers: {', '.join(ctx.labels)}\n\n"
                f"Do not explain. Answer with one line, in exactly this form:\n"
                f"ANSWER: <label>\n")

    def run(self, ex: Example, ctx: Context) -> StrategyResult:
        trace, thoughts = [], []
        for i in range(N_THOUGHTS):
            prompt = self._thought_prompt(ex, ctx, i)
            raw = ctx.generate(prompt, ctx.seed + 2000 * (i + 1))
            thoughts.append(raw.strip())
            trace.append({"step": f"thought_{i + 1}", "prompt": prompt, "output": raw})

        synth = self._synthesis_prompt(ex, ctx, thoughts)
        raw = ctx.generate(synth, ctx.seed)
        lab, rule = parse_label(raw, ctx.labels, ctx.relation)
        trace.append({"step": "synthesise", "prompt": synth, "output": raw,
                      "parsed": lab, "rule": rule})
        n_calls = N_THOUGHTS + 1

        if lab is None:
            # Recover the answer rather than scoring the row wrong for a
            # formatting slip. This mirrors the fallback Tree-of-Thought has had
            # from the start, whose absence here cost 21 rows in the first run.
            extract = self._extract_prompt(ex, ctx, raw)
            raw2 = ctx.generate(extract, ctx.seed + 7919,
                                max_new_tokens=EXTRACT_TOKENS)
            lab, rule = parse_label(raw2, ctx.labels, ctx.relation)
            n_calls += 1
            trace.append({"step": "extract", "prompt": extract, "output": raw2,
                          "parsed": lab, "rule": rule})
            if lab is not None:
                rule = f"recovered_{rule}"

        return StrategyResult(prediction=lab, parse_rule=rule, raw=raw,
                              trace=trace, n_calls=n_calls)
