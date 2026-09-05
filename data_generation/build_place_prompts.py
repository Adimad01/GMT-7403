"""Write the prompts that ask Gemini for new PLACES rather than finished rows.

An earlier round asked for complete rows. Three quarters of the checkable ones
were right, but the labels it got wrong were the subtle cases -- Indianapolis
'equals' Marion County, when Speedway and Lawrence are excluded from the
consolidated city -- and a sixth of its entities were things OpenStreetMap does
not hold at all: Mainland France, the Chicago Metropolitan Area, the Pacific
Plate.

Asking for places instead moves the work to where a language model is reliable
(knowing which places exist and what is notable about them) and keeps the part
it gets wrong (which relation holds, and how hard the item is) on this side,
where it is computed from geometry. One new place joins a few hundred pairs, so
the yield per row of output is far higher too.

    python3 data_generation/build_place_prompts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS            # noqa: E402
from topo_identity import IDENTITY                 # noqa: E402

RULES = """
## The one rule that matters

Your clause must NOT say where the place is in relation to anything else.

Banned, because each hands over an answer I am trying to test:

  parentage    "the French capital", "a city in Colorado", "Italy's largest port"
  neighbours   "on the German border", "across the strait from Spain"
  direction    "in the north of", "the westernmost", "at a high latitude"
  size or area "covers 270,000 square kilometres", "the largest state"

Write history, culture, economy, or physical character instead. A reader who
knows geography should recognise the place; a reader who does not should learn
nothing about where it sits.

    BAD   Lyon is a city in south-eastern France on the Rhone.
    GOOD  Lyon was the Roman capital of Gaul.

    BAD   Denver is the capital of Colorado, a mile above sea level.
    GOOD  Denver's airport is the largest by land area in the country.

## Output format

One place per line, exactly:

    NAME | clause

NAME must be the name OpenStreetMap uses, precise enough to be unambiguous.
Qualify anything that repeats: "Fayette County, Kentucky", not "Fayette County".
No numbering, no bullets, no headers, no commentary.
"""


def write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.name}  ({len(body.splitlines())} lines)")


def main() -> int:
    out = REPO / "data_generation"

    have_topo = sorted(IDENTITY)
    have_city = sorted(c.title() for c in COORDS)

    write(out / "prompt_places_topological.md", f"""# Places for the topological dataset

I need **300 new places** that OpenStreetMap holds as a polygon, for a dataset
about containment, adjacency and overlap between real areas.

## What to send

A spread across these kinds, roughly evenly:

  - national and sub-national administrative units (countries, states,
    provinces, regions, departments, cantons, prefectures)
  - counties, districts and boroughs — qualified by their state or country
  - cities and towns
  - national parks and protected areas
  - islands and island groups
  - lakes, seas, gulfs and bays
  - rivers and major watercourses
  - deserts, forests and mountain ranges that OSM maps as an area

## What will be rejected

Anything OpenStreetMap does not hold as a single mapped area:

  - vernacular regions — the Rust Belt, the Midwest, the Sahel, the Levant
  - statistical constructs — metropolitan areas, built-up areas, urban
    agglomerations, commuter belts
  - "Mainland X" — OSM's France includes Corsica; there is no mainland object
  - geological features — tectonic plates, ridges, faults, trenches
  - numbered routes — Interstate 10, Route 66
  - anything invented to sound official: "German Sovereign Territory",
    "Physical River Thames Current"

If you are unsure whether OSM holds it as an area, leave it out. A place I
cannot geocode is worth nothing to me; a boring one I can geocode is worth a
few hundred rows.
{RULES}
## Places I already have — do not send these again

{chr(10).join(have_topo)}
""")

    write(out / "prompt_places_cities.md", f"""# Cities for the cardinal and relative datasets

I need **300 new cities** for two datasets: one about compass direction between
places, one about what a viewer standing in one city sees when facing another.

## What to send

Cities and large towns from every inhabited region, weighted away from western
Europe and North America — those are already well covered. I especially want:

  - Central Asia, the Caucasus, Siberia
  - West, Central and East Africa
  - South and South-East Asia beyond the obvious capitals
  - the Pacific, the Caribbean, Central America
  - interior South America
  - second and third cities, not only capitals

A mix of prominence is wanted, not only famous ones. Obscure cities make the
harder items, and the dataset needs both ends.

## What will be rejected

  - anything that is not a settlement: regions, provinces, islands, mountains
  - former or renamed cities whose current OSM name differs — send the name
    OSM uses today
  - ambiguous bare names. There are dozens of Springfields and several
    Cambridges. Qualify anything that repeats: "Cambridge, Massachusetts"
{RULES}
## Cities I already have — do not send these again

{chr(10).join(have_city)}
""")

    write(out / "prompt_places_README.md", """# The loop

1. Paste one prompt into Gemini. Save what it returns as a text file.
2. Send me the file.
3. I geocode every place against OpenStreetMap, discard whatever does not
   resolve or resolves to the wrong kind of thing, and report the counts.
4. I compute the relations, the labels and the difficulty levels from the
   geometry, and generate the rows.
5. If a cell is short, I hand you a follow-up prompt naming exactly what is
   missing.

Gemini is never asked which relation holds or how hard an item is. Both are
computed here, which is why its errors last time -- Indianapolis 'equals'
Marion County, rivers 'crossing' states they lie entirely within -- cannot
recur through this route.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
