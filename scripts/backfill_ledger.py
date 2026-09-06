"""Import completed cells into the ledger from the run.json files they left.

The ledger only starts recording once a process has loaded the code that writes
it, so cells that finished under an earlier build are absent. Everything needed
is already in results/<relation>/<strategy>/seed<n>/run.json, alongside the
predictions, so those entries can be reconstructed rather than lost.

Existing ledger entries are left alone: a run is identified by its id and the
data hash it ran against, and one already recorded is not written twice.

    python3 scripts/backfill_ledger.py            # report
    python3 scripts/backfill_ledger.py --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LEDGER = REPO / "experiments" / "ledger.jsonl"


def existing() -> set[tuple]:
    if not LEDGER.exists():
        return set()
    out = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                out.add((r["run_id"], r.get("eval_manifest_sha256")))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not RESULTS.exists():
        print("  no results directory")
        return 0

    have = existing()
    add = []
    for run_json in sorted(RESULTS.glob("*/*/seed*/run.json")):
        try:
            s = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (s.get("run_id"), s.get("eval_manifest_sha256"))
        if key in have or not s.get("run_id"):
            continue
        preds = run_json.parent / "predictions.jsonl"
        ok, by_level = [], {}
        if preds.exists():
            for line in preds.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    ok.append(r)
                    by_level.setdefault(r.get("ambiguity_level", "?"), []).append(
                        bool(r.get("correct")))
        add.append({
            "finished_at": s.get("finished_at"),
            "git_commit": None,               # unknown for a past run
            "run_id": s["run_id"],
            "relation": s.get("relation"),
            "strategy": s.get("strategy"),
            "seed": s.get("seed"),
            "model_id": (s.get("model") or {}).get("model_id"),
            "backend": (s.get("model") or {}).get("backend"),
            "eval_manifest_sha256": s.get("eval_manifest_sha256"),
            "fewshot_manifest_sha256": s.get("fewshot_manifest_sha256"),
            "n_examples": s.get("n_examples"),
            "n_completed": s.get("n_completed"),
            "n_failed": s.get("n_failed"),
            "n_unparsed": s.get("n_unparsed"),
            "accuracy": (round(sum(r.get("correct", False) for r in ok) / len(ok), 4)
                         if ok else None),
            "accuracy_by_level": {k: round(sum(v) / len(v), 4)
                                  for k, v in sorted(by_level.items())},
            "elapsed_seconds": s.get("elapsed_seconds"),
            "backfilled": True,
        })

    for r in add:
        acc = f"{r['accuracy']*100:.1f}%" if r["accuracy"] is not None else "-"
        print(f"  {'importing' if args.apply else 'would import'}  "
              f"{r['run_id']:<32} {r['n_completed']:>4} rows  {acc}")
    if not add:
        print("  nothing to import; the ledger already has every finished cell")
        return 0
    if args.apply:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            for r in add:
                fh.write(json.dumps(r) + "\n")
        print(f"\n  {len(add)} entry(ies) appended to {LEDGER.name}")
    else:
        print(f"\n  {len(add)} to import; rerun with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
