# Generating more examples

The evaluation sets are too small to detect the effects being studied: at 105
rows with 10 pairwise comparisons, the chance of detecting a real 6-point
difference is about 18%. The fix is more data, and the corpora cap what is
possible — 616 unique entity pairs across all three relations, which cannot
fund both a well-powered evaluation and a usable fine-tuning split.

## How much is needed

| relation | have | add | of which multi-hop | why |
|---|---:|---:|---:|---|
| relative | 65 pairs | **+180** | 30 | eval is 20 facts, power ~0.04; the corpus is the hard cap |
| cardinal | 96 pairs | **+144 redesigned** | 24 | see below — the task as built does not test geography |
| topological | 455 pairs | **+210** | 35 | enough for eval already; this funds a training split |

## Level 6 — multi-hop

Levels 1–5 vary how obliquely the relation is *worded*. **Level 6 varies
inferential depth instead**: the A–B relation is never stated, only two links
through an intermediate place `C`, named in the new `via_entity` column.

Its wording stays plain deliberately. A row that is both obscurely phrased and
multi-hop cannot tell you which caused the difficulty.

Level 6 matters more than the others for this project: it is the case where a
knowledge graph should demonstrably help, because the KG supplies the
intermediate link the model would otherwise have to know.

Only logically forced compositions are allowed, so **the Level 6 grid is
deliberately smaller than Levels 1–5**:

| relation | labels with a valid two-hop composition |
|---|---|
| topological | `contains`, `within`, `disjoint` — 3 of 7 |
| cardinal | all 8 (each direction composes along its own axis) |
| relative | `left_of`, `right_of`, `in_front_of`, `behind` — 4 of 5 |

`A touches C` + `C touches B` implies nothing about A and B, and `next_to`
likewise fails to compose. `equals` is excluded because it is reachable only by
chaining synonyms, which is a naming trick rather than spatial reasoning.

### The rule that matters most

**The description must state BOTH links.** The first generated batch failed this
on 34 of 35 multi-hop rows: each stated only "A relates to C" and never
mentioned B, so the answer could not be derived from the sentence at all — only
recalled from world knowledge. That inverts the purpose of the level.

    BAD   A=United States, C=California, B=San Francisco
          "The federal republic fully surrounds the golden state."

    GOOD  "The federal republic fully surrounds the golden state, and that
           state in turn completely encloses the bay city." 

## Workflow

1. Paste `prompt_<relation>.md` into a capable model (Gemini, Claude, GPT).
   Each prompt carries the schema, the level definitions, the OpenStreetMap
   constraints, one worked example per label × level, and the list of entity
   pairs already used.

2. Save the returned CSV, e.g. `new_relative.csv`.

3. Validate — never merge unchecked output:

   ```bash
   python3 data_generation/validate_new_data.py new_relative.csv --relation relative
   ```

   Then, with internet, the check that matters most:

   ```bash
   python3 data_generation/validate_new_data.py new_relative.csv --relation relative --geocode
   ```

   Geocoding runs at 1 request/second. About a third of the *existing* corpus
   fails it, which is why every new row is checked before it is accepted.

4. If it reports failures, generate a targeted replacement request rather than
   regenerating the whole batch:

   ```bash
   python3 data_generation/make_fix_request.py new_topological.csv --relation topological
   ```

   That writes `fix_request_<relation>.md` naming each unusable row and why,
   listing exactly how many replacements each (label, level) cell needs to stay
   balanced, and carrying the full avoid-list. Paste it to the generator, append
   the returned rows to your file after deleting the named ones, and validate
   again.

   Regenerating everything to fix twenty rows discards the good ones and invites
   fresh mistakes in rows that were already fine.

5. Merge only when the validator is clean.

## What the validator checks

Schema and blanks · label and level vocabulary · exact balance across
label × level · duplicate entity pairs within the file · mirrored pairs
(`A contains B` alongside `B within A`, which leaks answers between splits) ·
collisions with the existing corpus · whether the description gives the answer
away · description length · an explicit observer viewpoint for `relative`
(where "left" is undefined without one) · and, with `--geocode`, whether every
place resolves in OpenStreetMap and resolves to the *right kind* of object.

That last check exists because "City of Detroit" currently resolves to a road
in South Africa in the shipped cache, and roughly 27% of cardinal pairs have
coordinates two or more compass sectors from the truth.


## Why cardinal was redesigned

The model answers 97–100% of the existing cardinal items correctly, and the
reason is not that it knows geography. The descriptions state the bearing:

    "Despite assumptions about climate, Calgary actually sits closer to the
     top of the globe than Yangon."

"Closer to the top of the globe" means north. Swap in any two names and the
answer is unchanged — the item measures paraphrase decoding, not spatial
knowledge. That is the whole explanation for the saturation, and it also
explains why a knowledge graph could never help on this task: coordinates add
nothing when the answer is already in the sentence.

The rebuilt prompt forbids every form of directional cue on Levels 1–5 —
compass words, map metaphors, clock bearings, sun references, pole and equator
references, quadrant language. The description introduces the two places and
stops:

    "Reykjavik is Iceland's coastal capital. Vientiane sits on the Mekong
     in Laos."

Now the answer requires knowing where they are, which is exactly the knowledge
a coordinate graph supplies.

**Level 6 is the one exception.** A two-hop chain cannot be stated without
directional language, and that level tests composition rather than recall. The
validator exempts it.

**Consequence to be aware of:** rows built this way are not directly comparable
with the 96 rows already in the corpus, which use the old cue-based design.
Either replace the old cardinal rows, or keep both and report the contrast —
a collapse in accuracy when the cue is removed is itself a finding about what
the original task measured.

### Two checks now enforce this

- **Direction cues** — any of ~50 forbidden phrases in a Level 1–5 cardinal
  description is a failure.
- **Template reuse** — the previous batch built 144 rows from 48 sentence
  frames, each reused three times, and 16 explanations reused nine times each.
  That is 48 items shown three times, not 144 items. Below 75% distinct frames
  is a failure.
