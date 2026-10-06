"""Rerun the answers that were cut off at the token limit, with a larger limit.

Each answer of the grid was capped at 1,024 tokens. A base-model answer still reasoning at that
point stops mid-sentence, before its ANSWER line, and the recorded label is the last predicate the
text happened to mention. In relative CoT that is 42 % of the base model's answers and 61 % with
the knowledge graph -- so those scores partly measure reasoning length.

Only the cut answers need rerunning. Generation is seeded from the seed and the prompt, so with the
same seed the first 1,024 tokens come out identical: an answer that finished under the old limit
finishes the same way under the new one. Rerunning the cut rows and substituting them gives the
cell as it would have been with the larger limit.

The reruns are written to results_budget<N>/ (SPATIAL_RESULTS_DIR), never to results/.
budget_report.py then compares each original cell with the substituted one.

    python3 scripts/rerun_truncated.py --list      # how many answers each cell would rerun
    python3 scripts/rerun_truncated.py             # CoT, base and kg, all three families (the default)
    python3 scripts/rerun_truncated.py --cells relative:zero_shot:base relative:zero_shot:kg

Safe to relaunch after an interruption: each cell resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAR = {"base": "", "kg": "_kg"}
STRATS = ("zero_shot", "cot", "few_shot_eval", "tot", "got")


def cut_rows(rel: str, strat: str, arm: str, with_l6: bool) -> list[int]:
    path = REPO / "results" / rel / strat / f"seed1{VAR[arm]}" / "predictions.jsonl"
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[int(r["row_index"])] = r
    return sorted(i for i, r in latest.items()
                  if r.get("status") == "ok"
                  and r.get("parse_rule") not in ("answer_tag", "answer_tag_short")
                  and (with_l6 or r.get("ambiguity_level") != "Level 6"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["relative:cot:base", "relative:cot:kg", "topological:cot:base",
                             "topological:cot:kg", "cardinal:cot:base", "cardinal:cot:kg"],
                    help="relation:strategy:arm, arm being base or kg")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--include-level-6", action="store_true")
    ap.add_argument("--list", action="store_true", help="only count the answers to rerun")
    a = ap.parse_args()

    out = f"results_budget{a.tokens}"
    env = dict(os.environ, SPATIAL_RESULTS_DIR=out)
    env.pop("SPATIAL_EVAL_SET", None)
    total = 0
    for c in a.cells:
        rel, strat, arm = c.split(":")
        if arm not in VAR or strat not in STRATS:
            sys.exit(f"bad cell {c}: use relation:strategy:arm with arm base or kg")
        rows = cut_rows(rel, strat, arm, a.include_level_6)
        calls = len(rows) * (4 if strat in ("tot", "got") else 1)
        hours = calls * a.tokens / 29 / 3600
        total += hours
        print(f"  {rel}/{strat}/{arm}: {len(rows)} cut answers, at most {hours:.1f} h at {a.tokens} tokens")
        if a.list or not rows:
            continue
        cmd = [sys.executable, "-m", "spatial_eval.cli", "run", "-r", rel, "-s", strat, "--seeds", "1",
               "--rows", *map(str, rows), "--save-traces", "--max-new-tokens", str(a.tokens)]
        if arm == "kg":
            cmd += ["--kg-mode", "input"]
        print(f"=== {rel}/{strat}/{arm} -> {out}/")
        if subprocess.call(cmd, cwd=REPO, env=env) != 0:
            print(f"    FAILED {rel}/{strat}/{arm}; relaunch the same command to resume")
            continue
        cell = REPO / out / rel / strat / f"seed1{VAR[arm]}"
        (cell / "rows.json").write_text(json.dumps({"rows": rows, "tokens": a.tokens}))
    print(f"\n  at most {total:.1f} h in all")
    if not a.list:
        print(f"  compare with: python3 scripts/budget_report.py --tokens {a.tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
