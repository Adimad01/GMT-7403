"""Few-shot: the same question preceded by pinned demonstrations.

The demonstrations come from `fewshot_manifest.json`, never sampled at run
time, so every arm that uses few-shot sees byte-identical demos for a given
evaluation row.

The demonstrations are deliberately NOT label-conditioned. An earlier version
gave every demonstration the target row's gold label, which put the answer in
the prompt and held few-shot at 89-100% with no response to the ambiguity level
at all. The manifest now pins three demonstrations spanning at least two
labels, at most one of them carrying the target's label.

What remains to report is the pool. `few_shot` draws from the train split,
which is also the fine-tuning set, so the number is optimistic for a fine-tuned
model; `few_shot_eval` draws from the evaluation split, which no adapter has
seen. The two are separate cells and must not be averaged together.
"""
from __future__ import annotations

from ..data import Example
from .base import Context, Strategy, register


@register
class FewShot(Strategy):
    name = "few_shot"
    description = "Pinned demonstrations, then the question. No reasoning scaffold."
    # Which pool the pinned demonstrations come from. The subclass overrides it;
    # the runner reads it to decide which manifest to load.
    demo_source = "train"

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        if ctx.demos is None:
            raise RuntimeError("few_shot requires the few-shot manifest to be loaded")
        demos = ctx.demos.get(ex.key)
        if not demos:
            raise RuntimeError(
                f"no pinned demonstrations for row {ex.row_index}. The few-shot "
                "manifest is out of sync with the eval manifest.")

        blocks = []
        for d in demos:
            blocks.append(
                f"Description: {d.text}\n"
                f"Subject: {d.subject}\n"
                f"Object: {d.target}\n"
                f"ANSWER: {d.label}\n")
        return (self.task_header(ctx.relation, ctx.labels)
                + "\nWorked examples:\n\n" + "\n".join(blocks)
                + "\nNow the new case.\n\n" + self.question(ex, ctx)
                + self.answer_instruction())
