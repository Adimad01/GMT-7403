"""
fewshot.py — label-conditioned few-shot demonstration builder.
================================================================================
Builds a 5-shot demonstration prefix from the TRAIN split: one example per
ambiguity level (L1–L5), ALL sharing the target example's gold label.

⚠ LABEL LEAKAGE (intentional, per the experiment spec): because every demo
carries the target's correct label, the demonstrations reveal the answer class.
Few-shot runs are therefore tagged "_fs5" and must be read as a label-conditioned
probe, NOT a clean accuracy baseline. Compare them to zero-shot accordingly.

Demos are plain `Corpus → Answer:[label]` text (no per-demo OSM evidence), so
few-shot is orthogonal to the KG mode; the target example still receives its own
OSM evidence in kg-mode input/rag.

Auto-resolves columns for both dataset schemas:
  label  : relation_label | spatial_relation
  corpus : corpus | relation_predicate
  level  : ambiguity_level
"""
import csv
import random
import re

_LABEL_COLS = ["relation_label", "spatial_relation"]
_CORPUS_COLS = ["corpus", "relation_predicate"]
_LEVEL_COL = "ambiguity_level"


def _pick(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _level_num(s):
    m = re.search(r"(\d+)", s or "")
    return int(m.group(1)) if m else None


class FewShotSelector:
    def __init__(self, train_csv: str, seed: int = 42, levels=(1, 2, 3, 4, 5)):
        with open(train_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"[fewshot] empty train data: {train_csv}")
        self.label_col = _pick(rows[0].keys(), _LABEL_COLS)
        self.corpus_col = _pick(rows[0].keys(), _CORPUS_COLS)
        if not self.label_col or not self.corpus_col:
            raise SystemExit(f"[fewshot] could not resolve label/corpus columns in {train_csv}")
        self.levels = levels
        self.rng = random.Random(seed)
        # index: label -> level -> [rows]
        self.by: dict = {}
        for r in rows:
            lab = (r.get(self.label_col) or "").strip().lower()
            if not lab:
                continue
            lvl = _level_num(r.get(_LEVEL_COL))
            self.by.setdefault(lab, {}).setdefault(lvl, []).append(r)

    def build_block(self, label: str) -> str:
        """Return a 5-demo prefix for `label` (one per level, same label), or ''."""
        label = (label or "").strip().lower()
        pools = self.by.get(label)
        if not pools:
            return ""

        used, demos = set(), []
        # one demo per ambiguity level
        for lvl in self.levels:
            cands = [r for r in pools.get(lvl, []) if id(r) not in used]
            if cands:
                r = self.rng.choice(cands)
                used.add(id(r))
                demos.append(r)
        # backfill to len(levels) from any level of the same label
        if len(demos) < len(self.levels):
            rest = [r for lst in pools.values() for r in lst if id(r) not in used]
            self.rng.shuffle(rest)
            for r in rest:
                if len(demos) >= len(self.levels):
                    break
                used.add(id(r))
                demos.append(r)
        if not demos:
            return ""

        lines = ["Here are labeled examples of this task. Study them, then answer "
                 "the new case in the same format.\n"]
        for i, r in enumerate(demos, 1):
            corpus = (r.get(self.corpus_col) or "").strip()
            lab = (r.get(self.label_col) or "").strip().lower()
            lines.append(f"Example {i}:\nCorpus: \"{corpus}\"\nAnswer: [{lab}]\n")
        lines.append("Now answer this new case:\n")
        return "\n".join(lines)
