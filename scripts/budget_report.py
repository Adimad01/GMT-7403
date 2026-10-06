"""Compare each cell with its cut answers rerun under a larger token limit.

For every cell in results_budget<N>/, the original predictions are taken from results/ and the
rerun rows substituted, which gives the cell as it would have been with the larger limit (see
rerun_truncated.py for why that substitution is exact). Levels 1 to 5, as in the audit.

    python3 scripts/budget_report.py --tokens 4096
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["row_index"])] = r
    return out


def ok(r: dict | None) -> bool:
    return bool(r) and r.get("status") == "ok" and str(r.get("correct")) == "True"


def tagged(r: dict | None) -> bool:
    return bool(r) and r.get("parse_rule") in ("answer_tag", "answer_tag_short")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    root = REPO / f"results_budget{a.tokens}"
    cells = sorted(root.glob("*/*/seed1*/predictions.jsonl"))
    if not cells:
        print(f"  nothing in {root}; run scripts/rerun_truncated.py first")
        return 1
    print(f"  limit {a.tokens} tokens, levels 1 to 5\n")
    for p in cells:
        rel, strat, cell = p.parts[-4], p.parts[-3], p.parts[-2]
        orig = {i: r for i, r in load(REPO / "results" / rel / strat / cell / "predictions.jsonl").items()
                if r.get("ambiguity_level") != "Level 6"}
        new = {i: r for i, r in load(p).items() if i in orig}
        merged = dict(orig) | new
        n = len(orig)
        acc0 = 100 * sum(ok(r) for r in orig.values()) / n
        acc1 = 100 * sum(ok(r) for r in merged.values()) / n
        still = sum(not tagged(r) for r in new.values())
        won = sum(ok(new[i]) and not ok(orig[i]) for i in new)
        lost = sum(ok(orig[i]) and not ok(new[i]) for i in new)
        print(f"  {rel}/{strat}/{cell}")
        print(f"    rerun {len(new)} cut answers; {len(new) - still} now finish, {still} still cut")
        print(f"    accuracy {acc0:.1f} % -> {acc1:.1f} %  (+{won} / -{lost} answers, McNemar p = {mcnemar(won, lost):.3g})")
        tuned = REPO / "results" / rel / strat / "seed1_lora" / "predictions.jsonl"
        if tuned.exists():
            t = {i: r for i, r in load(tuned).items() if i in orig}
            acct = 100 * sum(ok(r) for r in t.values()) / len(t)
            print(f"    fine-tuned adapter, same strategy: {acct:.1f} %")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
