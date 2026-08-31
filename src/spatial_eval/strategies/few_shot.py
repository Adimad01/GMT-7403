"""Few-shot: the same question preceded by pinned demonstrations.

The demonstrations come from `fewshot_manifest.json`, never sampled at run
time, so every arm that uses few-shot sees byte-identical demos for a given
evaluation row.

Note the demos are label-conditioned by construction: all five share the target
row's gold label, one per ambiguity level. That reveals the answer class, so
few-shot numbers are a leakage-aware probe to be compared against zero-shot --
not a clean baseline, and not comparable across labels.
"""
from __future__ import annotations

from ..data import Example
from .base import Context, Strategy, register


@register
class FewShot(Strategy):
    name = "few_shot"
    description = "Pinned demonstrations, then the question. No reasoning scaffold."

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
                + "\nNow the new case.\n\n" + self.question(ex)
                + self.answer_instruction())
