"""Accuracy by ambiguity level, from whatever results exist so far.

Safe to run while a job is going: it only reads. Rows still being written are
skipped rather than waited for.

The question it answers is whether the levels separate difficulty. If Level 1
and Level 5 come out within a few points of each other then the ladder is not
doing anything, every strategy will sit near the same ceiling, and no
comparison between them can be measured however many rows are added.

    python3 scripts/level_report.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    if not RESULTS.exists():
        print("  no results yet")
        return 0

    by_lvl: dict[str, list[int]] = defaultdict(list)
    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for pred in sorted(RESULTS.glob("*/*/seed*/predictions.jsonl")):
        rel, strat = pred.parts[-4], pred.parts[-3]
        for line in pred.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok":
                continue
            lvl = r.get("ambiguity_level", "?")
            by_lvl[lvl].append(bool(r["correct"]))
            by_cell[(rel, strat, lvl)].append(bool(r["correct"]))

    if not by_lvl:
        print("  no completed rows yet")
        return 0

    print("  ACCURACY BY AMBIGUITY LEVEL (all relations, all strategies)\n")
    print("  " + "level".ljust(10) + "rows".rjust(7) + "correct".rjust(9)
          + "accuracy".rjust(10))
    for lvl in sorted(by_lvl):
        v = by_lvl[lvl]
        print(f"  {lvl.ljust(10)}{len(v):>7}{sum(v):>9}{sum(v)/len(v)*100:>9.0f}%")

    lv = [lvl for lvl in sorted(by_lvl) if lvl != "Level 6"]
    if len(lv) >= 2:
        hi = sum(by_lvl[lv[0]]) / len(by_lvl[lv[0]])
        lo = sum(by_lvl[lv[-1]]) / len(by_lvl[lv[-1]])
        print(f"\n  {lv[0]} to {lv[-1]}: {hi*100:.0f}% -> {lo*100:.0f}%  "
              f"(spread {abs(hi-lo)*100:.0f} points)")
        if abs(hi - lo) < 0.10:
            print("  A spread this small means the levels are not separating")
            print("  difficulty, and strategy comparisons will have no room to")
            print("  show a difference.")

    print("\n  BY RELATION AND STRATEGY\n")
    print("  " + "relation".ljust(13) + "strategy".ljust(11)
          + "".join(f"L{i}".rjust(7) for i in range(1, 7)))
    seen = sorted({(r, s) for r, s, _ in by_cell})
    for rel, strat in seen:
        cells = []
        for i in range(1, 7):
            v = by_cell.get((rel, strat, f"Level {i}"), [])
            cells.append(f"{sum(v)/len(v)*100:.0f}%" if v else "-")
        print("  " + rel.ljust(13) + strat.ljust(11)
              + "".join(c.rjust(7) for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
