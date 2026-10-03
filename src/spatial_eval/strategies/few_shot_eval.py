"""Few-shot with demonstrations drawn from the evaluation split.

Identical prompt to `few_shot`. The only difference is the pool the three
demonstrations come from, and that difference is the point: the train split is
what the LoRA adapters were fitted on, so train-sourced demonstrations put text
the adapter has memorised into its own prompt. That is why the adapter arms
carried no few-shot cell. Evaluation rows are unseen by every adapter, so this
strategy can be run on all four arms and compared.

The trade is that a demonstration is itself a scored item. A row never
demonstrates itself -- the builder excludes it and the loader refuses a
manifest where that was lost -- but it does appear in other rows' prompts.

Report it under its own name. Merged with the train-sourced numbers it would
average two different conditions.
"""
from __future__ import annotations

from .base import register
from .few_shot import FewShot


@register
class FewShotEval(FewShot):
    name = "few_shot_eval"
    description = "Pinned demonstrations drawn from the eval split, then the question."
    demo_source = "eval"
