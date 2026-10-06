"""Read the crosses probe: does `crosses` survive when the line is not a river?

For each arm and strategy: accuracy on the probe's crosses / within / disjoint items, the share of
probe items answered `crosses`, and, beside it, the same arm's accuracy on the river `crosses` of
the main evaluation. A model that reads the sentence should do about as well on roads and trails
as on rivers; a model that leans on "a river, therefore crosses" should drop.

    python3 scripts/probe_report.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = {"base": "seed1", "kg": "seed1_kg", "lora": "seed1_lora", "lora_kg": "seed1_lora_kg",
        "lorakg_kg": "seed1_lorakg_kg"}


def preds(path: Path) -> dict[int, str | None]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["row_index"])] = r.get("predicted") if r.get("status") == "ok" else None
    return out


def main() -> int:
    probe = list(csv.DictReader((REPO / "data/topological/probe_crosses.csv").open(encoding="utf-8")))
    gold = {i: r["relation_label"] for i, r in enumerate(probe)}
    ev = list(csv.DictReader((REPO / "data/topological/eval.csv").open(encoding="utf-8")))
    river = [i for i, r in enumerate(ev) if r["relation_label"] == "crosses" and r["ambiguity_level"] != "Level 6"]
    print(f"  probe: {len(probe)} items {dict(Counter(gold.values()))}; main eval river crosses: {len(river)}\n")
    print(f"  {'arm':10s} {'strategy':10s} {'crosses':>9s} {'within':>8s} {'disjoint':>9s} {'all':>6s}"
          f" {'said crosses':>13s} {'river crosses':>14s}")
    for strat in ("zero_shot", "cot"):
        for arm, d in ARMS.items():
            p = preds(REPO / "results_probe_crosses/topological" / strat / d / "predictions.jsonl")
            if not p:
                continue
            acc = lambda lab: 100 * sum(p.get(i) == lab for i, g in gold.items() if g == lab) / max(1, sum(g == lab for g in gold.values()))
            allacc = 100 * sum(p.get(i) == g for i, g in gold.items()) / len(gold)
            said = 100 * sum(v == "crosses" for v in p.values()) / len(gold)
            m = preds(REPO / "results/topological" / strat / d / "predictions.jsonl")
            riv = 100 * sum(m.get(i) == "crosses" for i in river) / len(river)
            print(f"  {arm:10s} {strat:10s} {acc('crosses'):8.1f}% {acc('within'):7.1f}% {acc('disjoint'):8.1f}%"
                  f" {allacc:5.1f}% {said:12.1f}% {riv:13.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
