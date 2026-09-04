"""Build the relative-direction corpus from the selected triples.

Every row is re-derived from coordinates before it is written: the label must
come out of the observer's frame, and the ambiguity level must match how far
that frame is rotated from north. Nothing is taken from the selection file on
trust.

A note on the descriptions. For cardinal the sentence carried the item's
content -- identity facts about two places -- so rows built from one template
were the same item with the names swapped. Here the sentence is scaffolding:
it fixes the viewpoint and names three places, and every bit of the reasoning
lives in which three. Two rows sharing a frame are still different items. The
frames are varied anyway, but that is for readability, not to manufacture
apparent diversity.

    python3 data_generation/build_relative_corpus.py --out data/relative/corpus.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_relative_truth import (check, classify, frame,            # noqa: E402
                                  rotation_level)

REPO = Path(__file__).resolve().parents[1]
HEADER = ["source_entity", "source_geometry", "target_entity", "target_geometry",
          "observer_entity", "corpus", "via_entity", "relation_type",
          "relation_label", "explanation", "ambiguity_level"]

# Descriptions are composed from two grammatical families so that every
# combination reads correctly. Family A pairs a participial opening with a
# clause whose subject is the observer; Family B pairs a prepositional opening
# with a clause whose subject is the place being located. Mixing the two would
# dangle the participle.
OPEN_A = [
 "Standing in {v} and facing {b} squarely",
 "Working in {v} with the instrument facing {b}",
 "Posted in {v} and looking steadily at {b}",
 "Sitting in the tower at {v} and facing {b}",
 "Camped outside {v} and facing {b} across the plain",
 "Looking out from {v} with the gaze locked on {b}",
 "Flying over {v} and facing {b} on the present heading",
 "Moored off {v} and facing {b} over the water",
 "Waiting on the platform at {v}, facing {b}",
 "Set up above {v} and looking dead at {b}",
 "Halted on the ridge road above {v}, facing {b}",
 "Anchored off {v} with the bow facing {b}",
 "On the roof of the tallest block in {v}, looking at {b}",
 "Stationed at the airfield outside {v} and facing {b}",
]
CLOSE_A = [
 ", an observer also has {a} in view.",
 ", a surveyor picks up {a} in the same sweep.",
 ", the watch officer records {a} within the same arc.",
 ", a navigator notes {a} on the same segment of chart.",
 ", the operator catches {a} in the identical sector.",
 ", a geographer takes in {a} at the same moment.",
 ", the pilot keeps {a} in the same window.",
 ", a walker takes in {a} without turning.",
 ", the controller sees {a} on the same display.",
 ", an astronomer sweeps past {a} on the same pass.",
 ", a cartographer marks {a} in the same quadrant.",
 ", the signaller has {a} inside the same spread.",
 ", the surveyor's assistant logs {a} in the same arc.",
 ", a passenger notices {a} through the same window.",
]
OPEN_B = [
 "From the waterfront at {v}, with the line of sight running to {b}",
 "From the observatory at {v}, facing {b}",
 "From the ridge above {v}, looking steadily at {b}",
 "From the radar station at {v}, its antenna facing {b}",
 "As you scan outward from {v} along the axis to {b}",
 "Viewed from {v} by someone facing {b} without shifting stance",
 "From the terrace at {v}, the gaze held on {b}",
 "Approaching {v} from seaward and facing {b}",
 "Driving out of {v} on the road that runs at {b}",
 "From the lighthouse at {v}, its beam facing {b}",
 "From the pier at {v}, the sight line held on {b}",
 "From the signal tower at {v}, looking toward {b}",
 "As you look out from {v} down the axis to {b}",
 "From the escarpment above {v}, facing {b}",
]
CLOSE_B = [
 ", {a} lies in the same field of view.",
 ", {a} falls inside the same angular window.",
 ", {a} shares that stretch of horizon.",
 ", {a} occupies part of the same vista.",
 ", {a} enters the same panorama.",
 ", {a} sits within the same bearing spread.",
 ", {a} appears in the same outlook.",
 ", {a} shows up in the same direction band.",
 ", {a} is caught in the same scene.",
 ", {a} stands within the same sweep.",
 ", {a} turns up in the same forward view.",
 ", {a} is held in the same frame.",
 ", {a} lies inside the same sweep of view.",
 ", {a} appears within the same span of horizon.",
]

PATTERNS = ([a + b for a in OPEN_A for b in CLOSE_A]
            + [a + b for a in OPEN_B for b in CLOSE_B])

HOP_PATTERNS = {
 "left_of": [
  "Standing in {v} and sweeping the view clockwise, {a} is met first, then {c}, and finally {b}.",
  "Looking out from {v}, a clockwise turn of the head reaches {a}, then {c}, and last of all {b}.",
  "As you pan clockwise from {v}, the order in which the three come up is {a}, then {c}, and then {b}.",
  "Viewed from {v} with the eye travelling clockwise, {a} appears before {c}, and {c} before {b}.",
  "From the vantage at {v}, a clockwise sweep passes {a}, goes on to {c}, and ends at {b}.",
  "From the parapet at {v}, turning clockwise, {a} comes up first, then {c}, and {b} last.",
  "Standing in {v}, a clockwise scan of the skyline meets {a}, afterwards {c}, and finally {b}.",
  "As you rotate clockwise on the spot in {v}, {a} appears before {c}, and {c} before {b}.",
  "Viewed from {v}, moving the eye clockwise picks out {a}, then {c}, and only then {b}.",
  "From the lookout at {v}, a clockwise arc reaches {a} before {c}, and {c} before {b}.",
  "Looking out from {v} and swinging clockwise, {a} comes before {c}, and {c} comes before {b}.",
 ],
 "right_of": [
  "Standing in {v} and sweeping the view clockwise, {b} is met first, then {c}, and finally {a}.",
  "Looking out from {v}, a clockwise turn of the head reaches {b}, then {c}, and last of all {a}.",
  "As you pan clockwise from {v}, the three come up in the order {b}, then {c}, and then {a}.",
  "Viewed from {v} with the eye travelling clockwise, {b} shows up before {c}, and {c} before {a}.",
  "From the vantage at {v}, a clockwise sweep takes in {b}, continues to {c}, and finishes at {a}.",
  "From the parapet at {v}, turning clockwise, {b} comes up first, then {c}, and {a} last.",
  "Standing in {v}, a clockwise scan of the skyline meets {b}, afterwards {c}, and finally {a}.",
  "As you rotate clockwise on the spot in {v}, {b} appears before {c}, and {c} before {a}.",
  "Viewed from {v}, moving the eye clockwise picks out {b}, then {c}, and only then {a}.",
  "From the lookout at {v}, a clockwise arc reaches {b} before {c}, and {c} before {a}.",
  "Looking out from {v} and swinging clockwise, {b} comes before {c}, and {c} comes before {a}.",
 ],
 "in_front_of": [
  "Standing in {v} and looking down one line of sight, {a} is reached first, then {c}, and finally {b}.",
  "Looking out from {v} along a single bearing, the nearest is {a}, next {c}, and furthest {b}.",
  "As you sight from {v} down one axis, distance grows from {a} to {c}, and on to {b}.",
  "Viewed from {v} along one bearing, {a} stands closest, {c} lies past it, and {b} further still.",
  "From the tower at {v}, all three sit on one line: {a} nearest, then {c}, and {b} last.",
  "From the parapet at {v}, along one bearing, {a} is nearest, {c} beyond it, and {b} beyond that.",
  "Standing in {v} and sighting down a single line, {a} lies nearer than {c}, and {c} nearer than {b}.",
  "As you look from {v} along one axis, {a} is met first, {c} next, and {b} furthest out.",
  "Viewed from {v} on one bearing, {a} holds the near ground, and {c} stands between it and {b}.",
  "From the lookout at {v}, {a} is closer than {c}, and {c} is closer than {b}.",
  "Looking out from {v} down a single sight line, {a} precedes {c}, and {c} precedes {b}.",
 ],
 "behind": [
  "Standing in {v} and looking down one line of sight, {b} is reached first, then {c}, and finally {a}.",
  "Looking out from {v} along a single bearing, the nearest is {b}, next {c}, and furthest {a}.",
  "As you sight from {v} down one axis, distance grows from {b} to {c}, and on to {a}.",
  "Viewed from {v} along one bearing, {b} stands closest, {c} lies past it, and {a} further still.",
  "From the tower at {v}, all three sit on one line: {b} nearest, then {c}, and {a} last.",
  "From the parapet at {v}, along one bearing, {b} is nearest, {c} beyond it, and {a} beyond that.",
  "Standing in {v} and sighting down a single line, {b} lies nearer than {c}, and {c} nearer than {a}.",
  "As you look from {v} along one axis, {b} is met first, {c} next, and {a} furthest out.",
  "Viewed from {v} on one bearing, {b} holds the near ground, and {c} stands between it and {a}.",
  "From the lookout at {v}, {b} is closer than {c}, and {c} is closer than {a}.",
  "Looking out from {v} down a single sight line, {b} precedes {c}, and {c} precedes {a}.",
 ],
}


def fmt(n: str) -> str:
    return f"City of {n.title()}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="data_generation/relative_picks.json")
    ap.add_argument("--out")
    args = ap.parse_args()

    picks = json.loads(Path(args.picks).read_text())
    rows, failures, used_text = [], [], set()
    hop_n = {k: 0 for k in HOP_PATTERNS}
    flat_n = 0

    for p in picks:
        v, a, b, lab, lvl = (p["observer"], p["subject"], p["target"],
                             p["label"], p["level"])
        probs = check(v, a, b, lab)
        if lvl != 6 and rotation_level(v, b) != lvl:
            probs.append(f"frame rotation puts this at Level {rotation_level(v, b)}, "
                         f"not {lvl}")
        if lvl == 6:
            c = p["via"]
            for x, y, which in ((a, c, "first"), (c, b, "second")):
                for pr in check(v, x, y, lab):
                    probs.append(f"{which} step ({x} | {y}): {pr}")
        if probs:
            failures.append((lvl, lab, v, a, b, probs))
            continue

        if lvl == 6:
            i = hop_n[lab]
            hop_n[lab] += 1
            text = HOP_PATTERNS[lab][i % len(HOP_PATTERNS[lab])].format(
                v=v.title(), a=a.title(), b=b.title(), c=p["via"].title())
        else:
            if flat_n >= len(PATTERNS):
                failures.append((lvl, lab, v, a, b, ["ran out of distinct frames"]))
                continue
            text = PATTERNS[flat_n].format(
                v=v.title(), a=a.title(), b=b.title())
            flat_n += 1
        if text in used_text:
            failures.append((lvl, lab, v, a, b, ["duplicate description text"]))
            continue
        used_text.add(text)

        alpha, depth, spread, rot = frame(v, a, b)
        if lvl == 6:
            expl = (f"With the observer in {v.title()}, the same ordering holds "
                    f"for both steps through {p['via'].title()}, so it holds "
                    f"end to end. Sight line turned {rot:.0f} degrees from north.")
        else:
            expl = (f"From {v.title()} the sight line to {b.title()} runs "
                    f"{rot:.0f} degrees off north; {a.title()} sits {alpha:+.0f} "
                    f"degrees from that line at {depth:.2f} times the distance.")

        rows.append({
            "source_entity": fmt(a), "source_geometry": "Polygon",
            "target_entity": fmt(b), "target_geometry": "Polygon",
            "observer_entity": fmt(v), "corpus": text,
            "via_entity": fmt(p["via"]) if lvl == 6 else "",
            "relation_type": "relative_direction", "relation_label": lab,
            "explanation": expl, "ambiguity_level": f"Level {lvl}",
        })

    if failures:
        print(f"  {len(failures)} item(s) failed verification:\n")
        for lvl, lab, v, a, b, probs in failures[:20]:
            print(f"    L{lvl} {lab:<12} from {v}: {a} | {b}")
            for pr in probs:
                print(f"         - {pr}")
        print()
    print(f"  {len(rows)}/{len(picks)} rows verified")

    if args.out and not failures:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {dest}")
    elif args.out:
        print("  nothing written — fix the failures first")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
