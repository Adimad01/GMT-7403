"""Remove the mock-backend rows a smoke test left inside real results.

Running the pipeline once with ``--backend mock --limit 3`` writes three
fabricated predictions into every cell it touches, marked ``status="ok"``.
Resume then treats them as finished, so a later real run skips them and the
cell ends up with three fake rows among genuine ones. Nothing in a prediction
record says which backend produced it, so they cannot be spotted after the
fact -- but ``--limit N`` always takes the first N evaluation rows, which
makes them identifiable by position.

This deletes those rows. Re-running the ordinary command afterwards recomputes
them properly, because resume only skips what is still recorded.

    python3 scripts/purge_smoke_rows.py --limit 3 --seeds 1          # report
    python3 scripts/purge_smoke_rows.py --limit 3 --seeds 1 --apply  # delete
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
BUSY_SECONDS = 300          # a file touched this recently may be mid-write


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, required=True,
                    help="the --limit the smoke test used")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="the seeds the smoke test wrote (default: 1)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the files; otherwise just report")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"  no results directory at {RESULTS}")
        return 0

    victims = set(range(args.limit))
    seeds = {f"seed{s}" for s in args.seeds}
    total_removed = busy = 0
    touched = []

    for pred in sorted(RESULTS.glob("*/*/seed*/predictions.jsonl")):
        if pred.parent.name not in seeds:
            continue
        age = time.time() - pred.stat().st_mtime
        if age < BUSY_SECONDS:
            # A cell still being written must not be rewritten underneath the
            # runner; it is reported instead so it can be handled once idle.
            busy += 1
            print(f"  SKIPPED (written {age:.0f}s ago, may be running): "
                  f"{pred.relative_to(REPO)}")
            continue

        kept, dropped = [], 0
        for line in pred.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("row_index") in victims:
                dropped += 1
            else:
                kept.append(line)
        if dropped:
            touched.append((pred, dropped, len(kept)))
            total_removed += dropped
            if args.apply:
                pred.write_text("\n".join(kept) + ("\n" if kept else ""),
                                encoding="utf-8")

    for pred, dropped, left in touched:
        print(f"  {'removed' if args.apply else 'would remove'} {dropped} row(s), "
              f"{left} remain  {pred.relative_to(REPO)}")

    print(f"\n  {len(touched)} cell(s) affected, {total_removed} row(s) "
          f"{'removed' if args.apply else 'to remove'}")
    if busy:
        print(f"  {busy} cell(s) skipped as possibly in progress — rerun this "
              f"once the job has finished")
    if not args.apply and total_removed:
        print("  rerun with --apply to delete them, then rerun the normal "
              "run command; resume will recompute exactly these rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
