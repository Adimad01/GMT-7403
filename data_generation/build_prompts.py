"""Generate the data-request prompt for each spatial relation.

Rerun this whenever the corpus changes so the prompts carry current examples
and an up-to-date list of pairs already in use.

    python3 data_generation/build_prompts.py
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

random.seed(11)
REPO = Path(__file__).resolve().parents[1]
LV = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6"]


def rd(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SPEC = {
    "relative": dict(
        n_per_cell=6,
        # next_to does NOT compose: A beside C and C beside B leaves A and B
        # possibly far apart.
        hop_labels=["left_of", "right_of", "in_front_of", "behind"],
        labels=["left_of", "right_of", "in_front_of", "behind", "next_to"],
        what="RELATIVE DIRECTION — where one place sits from a stated viewpoint",
        levels=[
            "Plain but non-literal wording. Nautical or aviation terms: 'port arm', "
            "'starboard side', 'off the bow'.",
            "Clock-face bearings from a stated facing direction: 'towards the 9 o'clock "
            "mark', 'at 3 o'clock'.",
            "Cultural or bodily reference the reader must decode: 'your traditional "
            "wedding ring hand' (left), 'the hand you salute with' (right).",
            "Writing-system reference: 'where a line of Arabic script terminates' "
            "(Arabic reads right-to-left, so its end is on the LEFT).",
            "Obscure cultural convention needing two inference steps: 'the margin where "
            "a traditional Japanese manga volume concludes' (manga reads right-to-left, "
            "so it concludes on the LEFT).",
        ],
        hop_rule=(
            "From a SINGLE fixed viewpoint, left/right ordering is transitive. State "
            "two links and let the reader compose them:\n"
            "  A is left of C  +  C is left of B   =>  A is left of B\n"
            "The same holds for right_of, in_front_of and behind along one axis. For "
            "next_to, use adjacency in a stated row: 'A sits beside C, and C beside B, "
            "with nothing between them' => A is near B — only claim next_to when the "
            "three genuinely form a compact row."),
        hop_example=(
            "Standing on the National Mall facing the Capitol, the Washington Monument "
            "sits to the port side of the National Museum of American History, and that "
            "museum in turn sits to the port side of the National Gallery of Art. "
            "(A=Washington Monument, C=Museum of American History, B=National Gallery — "
            "all three named, both links stated.)"),
        note="Every description MUST state the observer's viewpoint or facing "
             "direction explicitly — 'left' is meaningless without one."),

    "cardinal": dict(
        n_per_cell=3,
        # every direction composes with itself along its own axis
        hop_labels=["north_of", "south_of", "east_of", "west_of",
                    "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
        labels=["north_of", "south_of", "east_of", "west_of",
                "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
        what="CARDINAL DIRECTION — the compass bearing from the second place to the first",
        levels=[
            "Straightforward, matching intuition: 'to reach Seattle from Portland you "
            "drive straight up Interstate 5'.",
            "Clock-face or map-axis phrasing instead of a compass word: 'head towards "
            "the 12 o'clock position on your map'.",
            "Mildly surprising, where a national stereotype misleads: Detroit is NORTH "
            "of Windsor, Canada.",
            "Clearly counter-intuitive: the true bearing contradicts what most people "
            "assume from climate, culture or rough mental maps.",
            "Strongly counter-intuitive, needing real geographic knowledge: 'Venice sits "
            "closer to the icy top of the world than Halifax' (Venice IS north of "
            "Halifax).",
        ],
        hop_rule=(
            "North/south and east/west compose along their own axis. State two links "
            "and let the reader chain them:\n"
            "  A is north of C  +  C is north of B   =>  A is north of B\n"
            "Diagonals compose only when both steps share the diagonal (northeast + "
            "northeast => northeast). Do NOT chain a north step with an east step and "
            "claim northeast — that is not forced unless the distances make it so, and "
            "the reader cannot know them."),
        hop_example=(
            "Kampala sits further up the 12 o'clock axis than Khartoum, and Khartoum in "
            "turn sits further up that same axis than Cairo. "
            "(A=Kampala, C=Khartoum, B=Cairo — all three named, both links stated.)"),
        note="CRITICAL: cardinal is currently SATURATED — the model already answers "
             "97-100% correctly, so easy items are worthless. Every new item must be "
             "genuinely HARD: the honest bearing must surprise an educated reader. If a "
             "well-informed person would answer instantly, do not include it."),

    "topological": dict(
        n_per_cell=5,
        # Only these three are reachable by a forced composition that involves
        # real spatial reasoning. touches/crosses/overlaps compose to nothing,
        # and 'equals' is only reachable by chaining synonyms, which is a naming
        # trick rather than an inference.
        hop_labels=["contains", "within", "disjoint"],
        labels=["contains", "within", "touches", "crosses",
                "disjoint", "overlaps", "equals"],
        what="TOPOLOGICAL RELATION — how the areas of two places relate (DE-9IM style)",
        levels=[
            "Very well-known places, relation stated plainly: 'California fully envelops "
            "Los Angeles'.",
            "Well-known but needing a moment: a country completely encircling a microstate.",
            "Requires specific knowledge: municipal limits versus an enclave's borders.",
            "Unusual geopolitical cases: enclaves, exclaves, condominiums, disputed zones.",
            "Large natural or geomorphic features rather than administrative ones: oceans, "
            "trenches, deserts, mountain ranges, river basins.",
        ],
        hop_rule=(
            "Only some compositions are logically FORCED. Use these, and no others:\n"
            "  A within C   + C within B    => A within B\n"
            "  A contains C + C contains B  => A contains B\n"
            "  A within C   + C disjoint B  => A disjoint from B\n"
            "NEVER chain 'touches' with 'touches' — A touches C and C touches B implies "
            "NOTHING about A and B. The same applies to crosses and overlaps. If the "
            "composition is not forced, the item is unusable."),
        hop_example=(
            "The Vatican Museums sit entirely inside the walls of Vatican City, and "
            "Vatican City in turn lies wholly inside the municipal boundary of Rome. "
            "(A=Vatican Museums, C=Vatican City, B=Rome — all three named, both links "
            "stated, and C is a real third place rather than a synonym.)"),
        note="Label meanings: contains = A fully encloses B. within = A is fully inside "
             "B. touches = share a boundary but interiors do not overlap. crosses = they "
             "cut through each other. disjoint = no contact at all. overlaps = partial "
             "overlap, neither contains the other. equals = the same area under two names."),
}

MULTIHOP_BLOCK = """- **Level 6 — MULTI-HOP** — the relation between A and B is NOT stated. The
  description states two links through an intermediate place C, and the reader
  must compose them.

  Keep the WORDING PLAIN at this level. Levels 1-5 make the phrasing harder;
  Level 6 makes the *inference* harder. If a row is both obscurely worded and
  multi-hop, we cannot tell which caused the difficulty, and the row is wasted.

  **Level 6 exists ONLY for these labels: {hop_labels}.**
  The other labels have no forced two-hop composition, so do not produce Level 6
  rows for them at all. The grid is deliberately ragged here.

{hop_rule}

  THREE RULES THAT DECIDE WHETHER THE ROW IS USABLE:

  1. The description MUST state BOTH links. It must mention A, C and B. A
     description that only says "A relates to C" is unusable, because nothing
     connects C to B and the answer cannot be derived from the text.

       BAD  — mentions only the first link:
         A=United States, C=California, B=San Francisco
         "The federal republic fully surrounds the golden state."
         (San Francisco never appears; the reader cannot answer.)

       GOOD — both links present:
         "The federal republic fully surrounds the golden state, and that state
          in turn completely encloses the bay city."

  2. C must be a genuinely DIFFERENT PLACE, not another name for A or B.
     Chaining synonyms ("United Mexican States equals Mexico, which touches the
     United States") satisfies the letter of the composition rule but involves
     no spatial reasoning at all. Never use 'equals' or a naming alias as a hop.

  3. A reader who knows only the sentence — not world geography — must be able
     to reach the answer. If the row can only be solved by already knowing where
     things are, it tests memory rather than composition.

  Name the intermediate place in the `via_entity` column. It must satisfy the
  same OpenStreetMap requirements as A and B — it is part of the reasoning
  chain and gets geocoded too.

  Example of the style wanted:
    "{hop_example}"
"""


def build(rel: str, spec: dict) -> str:
    rows = rd(REPO / "data" / rel / "corpus.csv")
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["relation_label"].strip().lower(),
                 r["ambiguity_level"].strip())].append(r)

    existing = sorted({f'{r["source_entity"]} | {r["target_entity"]}' for r in rows})
    flat_cells = len(spec["labels"]) * 5           # Levels 1-5, every label
    hop_cells = len(spec["hop_labels"])            # Level 6, composable labels only
    n_cells = flat_cells + hop_cells
    total = n_cells * spec["n_per_cell"]
    n_hop = hop_cells * spec["n_per_cell"]

    examples = []
    for lab in spec["labels"]:
        for lv in LV[:5]:                      # no Level 6 exists yet to show
            got = by_cell.get((lab, lv), [])
            if got:
                r = random.choice(got)
                examples.append(
                    f'{r["source_entity"]},{r["source_geometry"]},{r["target_entity"]},'
                    f'{r["target_geometry"]},"{r["corpus"]}",,{r["relation_type"]},'
                    f'{r["relation_label"]},"{r["explanation"]}",{r["ambiguity_level"]}')

    # Show ALL of them. A sample cannot be avoided: the last batch collided with
    # 48 existing pairs, nearly all of them outside the 120 that were shown.
    shown = existing
    levels_md = "\n".join(f"- **{LV[i]}** — {d}" for i, d in enumerate(spec["levels"]))
    levels_md += "\n" + MULTIHOP_BLOCK.format(
        hop_rule=spec["hop_rule"], hop_example=spec["hop_example"],
        hop_labels=", ".join(spec["hop_labels"]))

    return f"""# TASK: generate {total} new spatial-relation examples ({rel})

You are extending a research dataset used to test how well language models
reason about space. I need {total} NEW rows, {spec['n_per_cell']} per cell:

  Levels 1-5: all {len(spec['labels'])} labels x 5 levels = {flat_cells} cells
  Level 6   : only {hop_cells} labels ({', '.join(spec['hop_labels'])}) = {hop_cells} cells

  total {n_cells} cells x {spec['n_per_cell']} = {total} rows, of which {n_hop} are multi-hop.

The Level 6 grid is deliberately smaller: the remaining labels have no forced
two-hop composition, so multi-hop rows for them would have no determinate
answer.

## What the data captures

{spec['what']}

Allowed labels (use these exact strings): {', '.join(spec['labels'])}

{spec['note']}

## The six ambiguity levels

Levels 1-5 describe HOW HARD THE WORDING is, not how uncertain the geography
is: the correct answer is always unambiguous, only the phrasing gets harder.
Level 6 is different in kind — it adds an inference step instead.

{levels_md}

## HARD REQUIREMENT: every place must be findable in OpenStreetMap

Each row is geocoded automatically through Nominatim. A place that does not
resolve, or resolves to the wrong thing, makes the row useless. Roughly a third
of the current dataset fails this, so it matters. This applies to `via_entity`
on Level 6 rows as well.

USE:
- Administrative units with their full official style: "City of Seattle",
  "State of Colorado", "Republic of Italy", "Cook County"
- Named natural features that exist as OSM objects: "Lake Michigan",
  "Sonoran Desert", "Danube River", "Mount Kilimanjaro"
- Internationally known landmarks with a fixed footprint: "Eiffel Tower",
  "Vatican City", "Golden Gate Bridge"

DO NOT USE:
- Generic descriptions: "the main square", "Administration Offices",
  "Theater Audience", "the parking lot"
- Abstract or notional entities: "the Prime Meridian", "the Tropic of Cancer",
  "the observer"
- Interior spaces or rooms: prefer the whole building or campus
- Anything needing disambiguation: bare "Springfield", bare "Georgia"
- Businesses, events, or anything temporary

Rule of thumb: if searching the name alone on openstreetmap.org would not land
on the right object, do not use it.

## Output format

Return ONLY valid CSV. No prose, no markdown fences, no commentary.
Header row exactly as below, then {total} data rows.

Columns:
  source_entity     the subject place (A)
  source_geometry   one of: point, line, polygon
  target_entity     the object place (B)
  target_geometry   one of: point, line, polygon
  corpus            the natural-language description (the model sees ONLY this
                    plus the two names — the answer must be derivable from it)
  via_entity        Level 6 ONLY: the intermediate place C. Leave EMPTY for
                    Levels 1-5.
  relation_type     always: {rows[0]['relation_type']}
  relation_label    one of the allowed labels above
  explanation       one sentence saying why the label holds; for Level 6, spell
                    out the two-step chain
  ambiguity_level   Level 1 .. Level 6

source_entity,source_geometry,target_entity,target_geometry,corpus,via_entity,relation_type,relation_label,explanation,ambiguity_level

## Additional rules

1. The label describes A with respect to B, in that order.
2. Do not reuse any (source_entity, target_entity) pair listed at the bottom.
3. Do not use the same pair twice in your own output.

4. NEVER produce a pair together with its mirror. This is the rule most often
   broken: the previous batch did it 21 times. Filling `contains` and `within`
   from the same fact is the path of least resistance, and it is exactly what
   ruins the data — the two rows become each other's answer key, and once they
   land in different splits the model has seen the test answer during training.

       FORBIDDEN, as a pair:
         South Africa , Lesotho      , contains
         Lesotho      , South Africa , within

       CORRECT — different facts for each label:
         South Africa , Lesotho      , contains
         Vatican City , Italy        , within

   Every `contains` row and every `within` row must use a DIFFERENT pair of
   places. The same applies to any other label and its inverse.
5. The `corpus` text must NOT contain the label word or an obvious synonym.
   Write "sits at the 12 o'clock mark", not "is north of".
6. On Level 6 the text must state the two links and NOT the A-B relation. If a
   reader can answer without composing both steps, it is not multi-hop.
7. `explanation` is never shown to the model — do not rely on it to make a row
   solvable.
8. Vary geography: do not draw every example from the United States.
9. Every row must be factually TRUE. Verify the geography before writing it.

## Existing examples, one per label x level (match this style)

Note these predate the `via_entity` column, so it is empty in all of them.

{chr(10).join(examples)}

## Entity pairs already used — do not repeat these

{chr(10).join(shown)}
"""


def main() -> None:
    out_dir = REPO / "data_generation"
    for rel, spec in SPEC.items():
        path = out_dir / f"prompt_{rel}.md"
        text = build(rel, spec)
        path.write_text(text, encoding="utf-8")
        n_cells = len(spec["labels"]) * 5 + len(spec["hop_labels"])
        n_hop = len(spec["hop_labels"]) * spec["n_per_cell"]
        print(f"  {path.name:<28} {n_cells * spec['n_per_cell']:>4} rows "
              f"({spec['n_per_cell']}/cell x {n_cells} cells, "
              f"{n_hop} multi-hop across {len(spec['hop_labels'])} labels)")


if __name__ == "__main__":
    main()
