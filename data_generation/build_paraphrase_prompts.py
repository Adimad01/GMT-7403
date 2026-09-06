"""Write the prompts that ask for vernacular paraphrase templates.

The corpus already holds verified triplets: subject, predicate, object, with
the predicate computed from OpenStreetMap geometry rather than asserted. What
it needs is the wording -- a natural-language description that expresses each
relation indirectly, so the task is to interpret a spatial expression rather
than to match a keyword.

Templates rather than finished sentences. One template with {A} and {B}
placeholders serves many rows, so a few thousand lines of output cover the
whole corpus, and the same template can be kept out of the evaluation split if
it was used in training.

    python3 data_generation/build_paraphrase_prompts.py
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PER_CELL = 24

LADDER = """
## The five levels

The level is how INDIRECT the wording is. It is not about how hard the
geography is -- the places are decided elsewhere, and the same pair may appear
at any level.

  Level 1  A near-synonym. Plain, everyday phrasing that a reader maps to the
           relation immediately, without the predicate word itself.
  Level 2  Ordinary description. Still direct, but phrased as a fact about the
           places rather than as a spatial term.
  Level 3  Indirect. The relation follows from something described -- a journey,
           a boundary, a shared or unshared feature.
  Level 4  Oblique. The reader has to work slightly to see which relation is
           meant; the wording describes a consequence rather than the relation.
  Level 5  Highly oblique. Metaphor, imagery, or an inference from an
           unrelated-sounding fact. Still unambiguous on reflection.

A worked ladder for north_of:

  1  {A} sits directly above {B} on the map.
  2  {A} occupies a higher position on the chart than {B}.
  3  Leaving {B}, you drive steadily upward on the map to reach {A}.
  4  Of the two, {A} is the one that sees the shorter winter day.
  5  A compass needle at {B} points along the line that leads to {A}.
"""

RULES = """
## Hard rules

1. NEVER write the predicate itself, or a phrase that contains it. For
   north_of, the strings "north of" and "northward of" are banned. The reader
   must interpret, not match.

2. NEVER name a place. Use only the placeholders {A} and {B}. A template that
   mentions a city cannot be reused.

3. The relation must run from {A} to {B} in that order. "{A} sits above {B}"
   is north_of. "{B} sits above {A}" is the opposite relation and is wrong.

4. Each template must be unambiguous. A reader who knows the convention should
   recover exactly one relation from it. If it could equally mean two, drop it.

5. No two templates may paraphrase each other. Twenty rewordings of "above" is
   one template, not twenty.

6. Keep them short: one sentence, under about twenty-five words.

## Output format

One template per line, exactly:

    PREDICATE | LEVEL | template text

For example:

    north_of | 1 | {A} sits directly above {B} on the map.
    north_of | 3 | Leaving {B}, you travel steadily upward to reach {A}.

No numbering, no bullets, no headers, no commentary.
"""


def write(path: Path, title: str, labels: list[str], meaning: dict[str, str],
          extra: str = "") -> None:
    n = len(labels) * 5 * PER_CELL
    lines = [f"# {title}", "",
             f"I am building a dataset that tests whether a language model can "
             f"read a spatial description written in ordinary language and "
             f"recover the formal relation it expresses.",
             "",
             f"I need **{PER_CELL} templates for each predicate at each of five "
             f"levels** — {len(labels)} predicates x 5 levels x {PER_CELL} = "
             f"**{n} templates**.",
             "", "## The predicates", ""]
    for lab in labels:
        lines.append(f"  `{lab}` — {meaning[lab]}")
    lines += [LADDER, extra, RULES]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path.name}   ({n} templates requested)")


def main() -> int:
    out = REPO / "data_generation"

    write(out / "prompt_paraphrase_cardinal.md",
          "Vernacular templates for cardinal directions",
          ["north_of", "south_of", "east_of", "west_of", "northeast_of",
           "northwest_of", "southeast_of", "southwest_of"],
          {"north_of": "A lies toward the top of the map from B",
           "south_of": "A lies toward the bottom of the map from B",
           "east_of": "A lies toward the right of the map from B",
           "west_of": "A lies toward the left of the map from B",
           "northeast_of": "A lies diagonally up and to the right of B",
           "northwest_of": "A lies diagonally up and to the left of B",
           "southeast_of": "A lies diagonally down and to the right of B",
           "southwest_of": "A lies diagonally down and to the left of B"},
          "\n## Note\n\nMap metaphors, clock bearings, references to the sun,"
          " the poles or the\nequator are all welcome here — they are exactly"
          " the vernacular this\nexperiment is about. Only the literal"
          " predicate is banned.\n")

    write(out / "prompt_paraphrase_topological.md",
          "Vernacular templates for topological relations",
          ["contains", "within", "touches", "crosses", "overlaps",
           "disjoint", "equals"],
          {"contains": "A completely encloses B; all of B is inside A",
           "within": "A is completely enclosed by B; all of A is inside B",
           "touches": "A and B share a boundary but no interior",
           "crosses": "A passes through B, entering and leaving it",
           "overlaps": "A and B share part of their area, neither containing the other",
           "disjoint": "A and B share no ground at all",
           "equals": "A and B occupy exactly the same extent under two names"},
          "\n## Note\n\nThese are areas on a map — countries, regions, lakes,"
          " parks, rivers.\nDescribe the geometry, not the politics: 'governs'"
          " and 'administers' are\nnot spatial relations, whereas 'wholly"
          " encircles' and 'shares a rim with'\nare.\n")

    write(out / "prompt_paraphrase_relative.md",
          "Vernacular templates for viewpoint-relative directions",
          ["left_of", "right_of", "in_front_of", "behind", "next_to"],
          {"left_of": "seen by an observer facing B, A appears to the left",
           "right_of": "seen by an observer facing B, A appears to the right",
           "in_front_of": "A stands between the observer and B",
           "behind": "B stands between the observer and A",
           "next_to": "A and B appear side by side, at much the same distance"},
          "\n## Note — this one differs\n\nThese relations only mean anything"
          " from a viewpoint, so every template\nmust establish one. A third"
          " placeholder {V} is available for the observer's\nposition, and each"
          " template must use all three.\n\n    left_of | 1 | Standing in {V}"
          " and facing {B}, {A} appears off to one side, on the hand that"
          " writes.\n    behind | 2 | From {V}, {B} has to be passed before"
          " {A} comes into reach.\n\nDo not use the words left, right, front or"
          " behind. Everything else is fair.\n")

    (out / "prompt_paraphrase_README.md").write_text(
        "# The loop\n\n"
        "1. Paste one prompt into Gemini. Save what it returns.\n"
        "2. Send me the file.\n"
        "3. I check every template: that it never names its own predicate,\n"
        "   that it uses the right placeholders, that no two are duplicates,\n"
        "   and that each cell has enough.\n"
        "4. Templates are split so that the ones used in training never appear\n"
        "   in evaluation. Without that a fine-tuned model memorises the\n"
        "   phrase rather than reading it, and few-shot demonstrations hand\n"
        "   the mapping over outright.\n"
        "5. Anything short or rejected comes back to you as a follow-up prompt\n"
        "   naming exactly which predicate and level need more.\n",
        encoding="utf-8")
    print("  wrote prompt_paraphrase_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
