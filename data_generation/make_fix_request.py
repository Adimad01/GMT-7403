"""Turn validation failures into a targeted replacement request.

Regenerating a whole batch to fix twenty rows wastes the good ones and invites
new mistakes in rows that were already fine. This reads a generated file,
works out exactly which rows are unusable and why, and writes a prompt asking
only for those replacements — at the same label and ambiguity level, so the
grid stays balanced.

    python3 data_generation/make_fix_request.py new_topological.csv --relation topological

Writes data_generation/fix_request_<relation>.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LV = [f"Level {i}" for i in range(1, 7)]
HOP_LEVEL = "Level 6"
HOP_LABELS = {
    "topological": {"contains", "within", "disjoint"},
    "cardinal": {"north_of", "south_of", "east_of", "west_of",
                 "northeast_of", "northwest_of", "southeast_of", "southwest_of"},
    "relative": {"left_of", "right_of", "in_front_of", "behind"},
}
LABELS = {
    "topological": ["contains", "within", "touches", "crosses",
                    "disjoint", "overlaps", "equals"],
    "cardinal": ["north_of", "south_of", "east_of", "west_of",
                 "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
    "relative": ["left_of", "right_of", "in_front_of", "behind", "next_to"],
}
GIVEAWAY = {
    "north_of": ["north of"], "south_of": ["south of"], "east_of": ["east of"],
    "west_of": ["west of"], "northeast_of": ["northeast of"],
    "northwest_of": ["northwest of"], "southeast_of": ["southeast of"],
    "southwest_of": ["southwest of"],
    "left_of": ["left of", "to the left"], "right_of": ["right of", "to the right"],
    "in_front_of": ["in front of"], "behind": ["behind"],
    "next_to": ["next to", "adjacent to"],
    "contains": ["contains"], "within": ["within", "inside of"],
    "touches": ["touches"], "crosses": ["crosses"], "disjoint": ["disjoint"],
    "overlaps": ["overlaps"], "equals": ["equals"],
}


def rd(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pair(r):
    return (r["source_entity"].strip().lower(), r["target_entity"].strip().lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--relation", required=True, choices=list(LABELS))
    ap.add_argument("--per-cell", type=int, default=5)
    args = ap.parse_args()

    rel = args.relation
    rows = rd(Path(args.csv_file))
    corpus = rd(REPO / "data" / rel / "corpus.csv")
    old_pairs = {pair(r) for r in corpus}
    all_old = sorted({f'{r["source_entity"]} | {r["target_entity"]}' for r in corpus})

    # reasons a row must go, keyed by line number in the file (header = line 1)
    doomed: dict[int, list[str]] = defaultdict(list)

    seen_here = Counter(pair(r) for r in rows)
    mirror_here = {pair(r) for r in rows}
    for i, r in enumerate(rows):
        ln = i + 2
        p = pair(r)
        if p in old_pairs:
            doomed[ln].append("this entity pair already exists in the corpus")
        if (p[1], p[0]) in mirror_here:
            doomed[ln].append("this pair also appears mirrored in your own output")
        if seen_here[p] > 1:
            doomed[ln].append("this pair is used more than once in your output")
        lab = r["relation_label"].strip().lower()
        for phrase in GIVEAWAY.get(lab, []):
            if phrase in r["corpus"].lower():
                doomed[ln].append(f'the description contains the word "{phrase}", '
                                  f'which gives the answer away')
                break
        if r["ambiguity_level"].strip() == HOP_LEVEL and lab not in HOP_LABELS[rel]:
            doomed[ln].append(f"'{lab}' has no forced two-hop composition, so it "
                              f"cannot appear at Level 6")

    # cells left short once the doomed rows are removed
    kept = Counter((r["relation_label"].strip().lower(), r["ambiguity_level"].strip())
                   for i, r in enumerate(rows) if (i + 2) not in doomed)
    need: list[tuple[str, str, int]] = []
    for lab in LABELS[rel]:
        for lv in LV:
            if lv == HOP_LEVEL and lab not in HOP_LABELS[rel]:
                continue
            short = args.per_cell - kept.get((lab, lv), 0)
            if short > 0:
                need.append((lab, lv, short))

    total_new = sum(n for _, _, n in need)
    if not doomed and not need:
        print("  nothing to fix — this file is clean")
        return 0

    lines = [f"# Replacement request ({rel})",
             "",
             f"I previously asked you for a batch of {rel} spatial-relation rows.",
             f"Most were good. **{len(doomed)} rows must be replaced**, and I need",
             f"**{total_new} new rows** to restore the balance.",
             "",
             "Keep everything else exactly as it was — do not resend the good rows.",
             "",
             "## Rows to replace, and why",
             ""]
    for ln in sorted(doomed):
        r = rows[ln - 2]
        lines.append(f"- **line {ln}** — `{r['source_entity']}` → "
                     f"`{r['target_entity']}` ({r['relation_label']}, "
                     f"{r['ambiguity_level']})")
        for why in doomed[ln]:
            lines.append(f"  - {why}")
    lines += ["", "## What to send back", "",
              f"Exactly {total_new} rows, distributed like this:", ""]
    for lab, lv, n in need:
        lines.append(f"- `{lab}` at **{lv}** — {n} row{'s' if n > 1 else ''}")

    lines += ["", "## Rules for the replacements", "",
              "1. Same label and same ambiguity level as the row being replaced —",
              "   the grid must stay balanced.",
              "2. A DIFFERENT pair of places. Never reuse a pair from the list at the",
              "   bottom, and never reuse one already in your previous batch.",
              "3. Never send a pair together with its mirror. If you write",
              "   \"A contains B\", do not also write \"B within A\" — those two rows",
              "   become each other's answer key.",
              "4. The description must NOT contain the label word or an obvious",
              "   synonym. Say \"sits entirely inside\", never \"is within\".",
              "5. Every place must be findable in OpenStreetMap: use full official",
              "   names (\"City of Seattle\", \"State of Colorado\") or named natural",
              "   features. No generic descriptions, abstractions, or interior rooms.",
              "6. Every row must be factually TRUE. Verify before writing.",
              ""]
    if any(lv == HOP_LEVEL for _, lv, _ in need):
        lines += ["7. For Level 6 rows: the description must state BOTH links through",
                  "   the intermediate place named in `via_entity`, and must mention",
                  "   all three places. The intermediate must be a real third place,",
                  "   never a synonym of an endpoint.",
                  ""]
    lines += ["## Output format", "",
              "Return ONLY CSV rows — no header, no prose, no markdown fences.",
              "Same column order as before:", "",
              "source_entity,source_geometry,target_entity,target_geometry,corpus,"
              "via_entity,relation_type,relation_label,explanation,ambiguity_level",
              "",
              "## Entity pairs already used — never reuse any of these", "",
              *all_old,
              "",
              "## Also do not reuse any pair from your previous batch", ""]
    lines += [f'{r["source_entity"]} | {r["target_entity"]}' for r in rows]

    out = REPO / "data_generation" / f"fix_request_{rel}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {len(doomed)} rows to drop, {total_new} replacements needed")
    for lab, lv, n in need:
        print(f"    {lab:<10} {lv}: {n}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
