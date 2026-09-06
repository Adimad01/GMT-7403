"""Show the history of every experiment cell that has finished.

results/ holds only the current state, and has been deleted several times in
the course of this work. The ledger is appended to as each cell completes and
is committed, so it answers questions the results directory cannot: what was
run, when, against which version of the data, at which revision of the code,
and what came out.

    python3 scripts/experiment_log.py                 # everything
    python3 scripts/experiment_log.py --relation cardinal
    python3 scripts/experiment_log.py --latest        # newest run per cell
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "experiments" / "ledger.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation")
    ap.add_argument("--strategy")
    ap.add_argument("--latest", action="store_true",
                    help="only the most recent entry for each cell")
    ap.add_argument("--full", action="store_true",
                    help="include per-level accuracy")
    args = ap.parse_args()

    if not LEDGER.exists():
        print("  no experiments recorded yet")
        return 0

    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.relation:
        rows = [r for r in rows if r["relation"] == args.relation]
    if args.strategy:
        rows = [r for r in rows if r["strategy"] == args.strategy]
    if args.latest:
        keep = OrderedDict()
        for r in rows:
            keep[r["run_id"]] = r          # later entries overwrite earlier
        rows = list(keep.values())

    if not rows:
        print("  nothing matches")
        return 0

    print(f"  {len(rows)} run(s)\n")
    print("  " + "finished".ljust(18) + "cell".ljust(30) + "rows".rjust(6)
          + "acc".rjust(8) + "unpars".rjust(8) + "mins".rjust(7)
          + "  data      code")
    for r in rows:
        acc = f"{r['accuracy']*100:.1f}%" if r.get("accuracy") is not None else "-"
        print("  " + r["finished_at"][:16].replace("T", " ").ljust(18)
              + r["run_id"].ljust(30)
              + str(r["n_completed"]).rjust(6)
              + acc.rjust(8)
              + str(r["n_unparsed"]).rjust(8)
              + f"{r['elapsed_seconds']/60:.0f}".rjust(7)
              + "  " + (r["eval_manifest_sha256"] or "")[:8].ljust(10)
              + (r.get("git_commit") or "?"))
        if args.full and r.get("accuracy_by_level"):
            lv = "  ".join(f"{k.replace('Level ', 'L')}={v*100:.0f}%"
                           for k, v in r["accuracy_by_level"].items())
            print(" " * 20 + lv)

    shas = {r["eval_manifest_sha256"] for r in rows}
    if len(shas) > 1:
        print(f"\n  NOTE: these runs span {len(shas)} different versions of the")
        print("  evaluation data. Only rows sharing a data hash are comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
