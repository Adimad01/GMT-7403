# Generating more examples

The evaluation sets are too small to detect the effects being studied: at 105
rows with 10 pairwise comparisons, the chance of detecting a real 6-point
difference is about 18%. The fix is more data, and the corpora cap what is
possible — 616 unique entity pairs across all three relations, which cannot
fund both a well-powered evaluation and a usable fine-tuning split.

## How much is needed

| relation | have | add | why |
|---|---:|---:|---|
| relative | 65 pairs | **+150** | eval is 20 facts, power ~0.04; the corpus is the hard cap |
| cardinal | 96 pairs | **+120 harder** | 97–100% accuracy — saturated, so easy items add nothing |
| topological | 455 pairs | **+175** | enough for eval already; this funds a training split |

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

4. Send failures back to the generator and repeat. Merge only when clean.

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
