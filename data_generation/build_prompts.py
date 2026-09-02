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
        hop_labels=["north_of", "south_of", "east_of", "west_of",
                    "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
        labels=["north_of", "south_of", "east_of", "west_of",
                "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
        what="CARDINAL DIRECTION — the compass bearing from the second place to the first",
        # Levels here grade the GEOGRAPHY, not the phrasing. Once the direction
        # cue is removed from the text (see headline rule), difficulty can only
        # come from how counter-intuitive the true bearing is.
        levels=[
            "Both places are globally famous and the bearing matches everyone's "
            "mental map. Oslo and Rome; Cairo and Johannesburg.",
            "Still uncontroversial, but needs the reader to place the two "
            "countries correctly. Warsaw and Athens; Lima and Caracas.",
            "A stereotype misleads. Detroit and Windsor: Canada is 'above' the "
            "USA, yet Detroit is the northern one. Reno and Los Angeles: Reno is "
            "further west.",
            "Clearly counter-intuitive. Most educated readers would guess wrong: "
            "the bearing contradicts what climate, culture or a rough mental map "
            "suggests.",
            "Strongly counter-intuitive, needing real knowledge of latitudes or "
            "longitudes. Venice is north of Halifax. Nairobi is east of Rio de "
            "Janeiro. Most people are confident and wrong.",
        ],
        hop_rule=(
            "North/south and east/west compose along their own axis:\n"
            "  A is north of C  +  C is north of B   =>  A is north of B\n"
            "Diagonals compose only when BOTH steps share the same diagonal "
            "(northeast + northeast => northeast). Never chain a north step with "
            "an east step and claim northeast: that is not forced unless the "
            "distances happen to make it so, and the reader cannot know them."),
        hop_example=(
            "Kampala sits further from the North Pole than Khartoum, and Khartoum "
            "in turn sits further from the North Pole than Cairo. "
            "(A=Kampala, C=Khartoum, B=Cairo — all three named, both links "
            "stated. Level 6 is the ONE place a directional phrase is allowed, "
            "because without it there is no chain to compose.)"),
        note=(
            "READ THIS FIRST — it is the reason this dataset is being rebuilt.\n"
            "\n"
            "A model currently answers 97-100% of the existing cardinal items "
            "correctly. Not because it knows geography, but because the previous "
            "descriptions gave the answer away:\n"
            "\n"
            "    \"Despite assumptions about climate, Calgary actually sits closer\n"
            "     to the top of the globe than Yangon.\"\n"
            "\n"
            "\"Closer to the top of the globe\" means north. The reader never needs "
            "to know where Calgary or Yangon are — swap in any two names and the "
            "answer is unchanged. The item measures paraphrase decoding, not "
            "spatial knowledge, and that is why the task is saturated.\n"
            "\n"
            "THE HEADLINE RULE: on Levels 1-5, the description must NOT encode "
            "the direction in any form. It introduces the two places and stops. "
            "The answer must come from knowing where they are.\n"
            "\n"
            "FORBIDDEN anywhere in a Level 1-5 description — this list is not "
            "exhaustive, the rule is the intent behind it:\n"
            "  compass words        north, south, east, west, and compounds\n"
            "  map metaphors        top/bottom/left/right of the map, upward,\n"
            "                       downward, leftward, rightward, above, below\n"
            "  clock bearings       12 o'clock, 3 o'clock, any dial reference\n"
            "  sun references       sunrise, sunset, morning sun, setting sun,\n"
            "                       greets the sun earlier\n"
            "  pole/equator refs    closer to the pole, toward the Arctic, nearer\n"
            "                       the equator, higher latitude\n"
            "  travel directions    head up, drive down, travel toward the left\n"
            "  quadrant language    upper left, lower right, diagonally up\n"
            "\n"
            "    BAD  \"Reykjavik sits closer to the top of the globe than\n"
            "          Vientiane.\"\n"
            "          (states the answer; no geography needed)\n"
            "\n"
            "    GOOD \"Reykjavik is Iceland's coastal capital. Vientiane sits on\n"
            "          the Mekong in Laos.\"\n"
            "          (identifies both places; the reader must know the\n"
            "           latitudes)\n"
            "\n"
            "What the description IS for: disambiguating the two places and "
            "giving them enough context to be well-posed — which country, which "
            "river, which region. Never their relative position.\n"
            "\n"
            "Level 6 is the single exception, explained under that level."),
        exemplars=[
            'City of Oslo,Polygon,City of Rome,Polygon,"Oslo sits at the head of a '
            'long fjord and serves as Norway\'s seat of government. Rome straddles '
            'the Tiber in central Italy.",,cardinal_direction,north_of,"Oslo lies '
            'near 60N, Rome near 42N. Nothing here is surprising; both are '
            'well-placed in most mental maps.",Level 1',

            'City of Lima,Polygon,City of Caracas,Polygon,"Lima is Peru\'s capital '
            'and holds close to a third of the country\'s population. Caracas is '
            'Venezuela\'s capital and largest city.",,cardinal_direction,west_of,'
            '"Lima sits near 77W against Caracas near 67W. A reader who can place '
            'Peru and Venezuela on the continent gets this right.",Level 2',

            'City of Detroit,Polygon,City of Windsor,Polygon,"Detroit grew around '
            'the American car industry on the river of the same name. Windsor faces '
            'it from Ontario across that water.",,cardinal_direction,north_of,'
            '"Detroit is at 42.33N and Windsor at 42.31N. Because Canada is drawn '
            'above the United States, almost everyone guesses the reverse.",Level 3',

            'City of Reno,Polygon,City of Los Angeles,Polygon,"Reno lies in the '
            'Nevada high desert beside the Truckee River. Los Angeles is '
            'California\'s largest coastal metropolis.",,cardinal_direction,west_of,'
            '"Reno sits at 119.8W and Los Angeles at 118.2W. The California '
            'coastline bends far enough that an inland Nevada city is the more '
            'westerly of the two.",Level 4',

            'City of Venice,Polygon,City of Halifax,Polygon,"Venice is built across '
            'lagoon islands in the Italian Veneto. Halifax is the Atlantic port that '
            'anchors Nova Scotia.",,cardinal_direction,north_of,"Venice is at 45.4N '
            'and Halifax at 44.6N. A Mediterranean city and a cold Canadian port '
            'invite the opposite guess.",Level 5',

            'City of Kampala,Polygon,City of Cairo,Polygon,"Kampala sits further '
            'from the North Pole than Khartoum, and Khartoum in turn sits further '
            'from the North Pole than Cairo.",City of Khartoum,cardinal_direction,'
            'south_of,"Two south-of steps compose. Kampala 0.3N, Khartoum 15.5N, '
            'Cairo 30.0N.",Level 6',
        ],
        extra_rules=[
            "NO TEMPLATES. The previous batch produced 144 rows from 48 sentence "
            "frames, each reused three times with only the names swapped, and 16 "
            "explanations reused nine times each. That inflates the row count "
            "without adding information. Every description must be a distinct "
            "sentence, and no two may share a recognisable frame. If you find "
            "yourself writing \"Traveling from X, ... to reach Y\" a second time, "
            "rewrite it.",
            "Vary sentence length and shape: some one clause, some two; some "
            "leading with the subject, some with the object; some naming a "
            "river, a coastline, an economic role, a founding date.",
            "Every `explanation` must also be distinct and must state the actual "
            "reason — which latitude or longitude relation holds, and why a "
            "reader might get it wrong. Not \"the phrase indicates a northern "
            "trajectory\".",
            "Prefer well-known cities over obscure ones. The task is to test "
            "whether the model KNOWS the geography, so the places must be ones a "
            "knowledgeable person could reasonably be expected to place.",
            "THE DIRECTION MUST BE UNAMBIGUOUS. Each of the eight labels covers a "
            "45-degree sector, so a pair whose true bearing lands near a sector "
            "edge has no single defensible answer — a careful geographer could "
            "call it either way, and grading a model on it measures nothing. "
            "Pick pairs that sit well inside their sector. Concretely: for "
            "`north_of`/`south_of`/`east_of`/`west_of` the offset along the other "
            "axis must be small relative to the offset along the named axis; for "
            "the four diagonals the two offsets must be roughly comparable. "
            "Chicago and Taipei are almost exactly on the north/northeast line — "
            "never choose a pair like that.",
            "NO NEAR-ANTIPODAL PAIRS. Two places on opposite sides of the globe "
            "have no well-defined compass direction between them: the bearing "
            "flips depending on which way you travel. Keep both places within "
            "roughly a third of the globe of each other — same continent, "
            "neighbouring continents, or across one ocean, never across the "
            "Pacific AND the pole. Perth to Caracas, Asuncion to Manila, and "
            "Hanoi to Lima are all unusable for this reason.",
        ]),

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

    # Where a relation supplies hand-written exemplars, use those. Sampling the
    # corpus would be actively harmful for cardinal: its existing rows use the
    # cue-based design this prompt forbids, and a demonstration outweighs an
    # instruction every time.
    if spec.get("exemplars"):
        examples = list(spec["exemplars"])
        examples_note = ("These are hand-written in the style wanted, one per level.\n"
                         "The rows already in the corpus use an older design that this\n"
                         "prompt forbids, so they are deliberately not shown.")
    else:
        examples = []
        examples_note = ("Note these predate the `via_entity` column, so it is empty "
                         "in all of them.")
        for lab in spec["labels"]:
            for lv in LV[:5]:                  # no Level 6 exists yet to show
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

    extra = spec.get("extra_rules") or []
    extra_block = ""
    if extra:
        extra_block = "\n" + "\n\n".join(
            f"{chr(65 + i)}. {r}" for i, r in enumerate(extra)) + "\n"

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
  source_geometry   one of: Point, LineString, Polygon, MultiPolygon
  target_entity     the object place (B)
  target_geometry   one of: Point, LineString, Polygon, MultiPolygon
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
{extra_block}
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

## Worked examples — match this style

{examples_note}

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
