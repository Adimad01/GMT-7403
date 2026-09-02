# TASK: generate 144 new spatial-relation examples (cardinal)

You are extending a research dataset used to test how well language models
reason about space. I need 144 NEW rows, 3 per cell:

  Levels 1-5: all 8 labels x 5 levels = 40 cells
  Level 6   : only 8 labels (north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of) = 8 cells

  total 48 cells x 3 = 144 rows, of which 24 are multi-hop.

The Level 6 grid is deliberately smaller: the remaining labels have no forced
two-hop composition, so multi-hop rows for them would have no determinate
answer.

## What the data captures

CARDINAL DIRECTION — the compass bearing from the second place to the first

Allowed labels (use these exact strings): north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of

READ THIS FIRST — it is the reason this dataset is being rebuilt.

A model currently answers 97-100% of the existing cardinal items correctly. Not because it knows geography, but because the previous descriptions gave the answer away:

    "Despite assumptions about climate, Calgary actually sits closer
     to the top of the globe than Yangon."

"Closer to the top of the globe" means north. The reader never needs to know where Calgary or Yangon are — swap in any two names and the answer is unchanged. The item measures paraphrase decoding, not spatial knowledge, and that is why the task is saturated.

THE HEADLINE RULE: on Levels 1-5, the description must NOT encode the direction in any form. It introduces the two places and stops. The answer must come from knowing where they are.

FORBIDDEN anywhere in a Level 1-5 description — this list is not exhaustive, the rule is the intent behind it:
  compass words        north, south, east, west, and compounds
  map metaphors        top/bottom/left/right of the map, upward,
                       downward, leftward, rightward, above, below
  clock bearings       12 o'clock, 3 o'clock, any dial reference
  sun references       sunrise, sunset, morning sun, setting sun,
                       greets the sun earlier
  pole/equator refs    closer to the pole, toward the Arctic, nearer
                       the equator, higher latitude
  travel directions    head up, drive down, travel toward the left
  quadrant language    upper left, lower right, diagonally up

    BAD  "Reykjavik sits closer to the top of the globe than
          Vientiane."
          (states the answer; no geography needed)

    GOOD "Reykjavik is Iceland's coastal capital. Vientiane sits on
          the Mekong in Laos."
          (identifies both places; the reader must know the
           latitudes)

What the description IS for: disambiguating the two places and giving them enough context to be well-posed — which country, which river, which region. Never their relative position.

Level 6 is the single exception, explained under that level.

## The six ambiguity levels

Levels 1-5 describe HOW HARD THE WORDING is, not how uncertain the geography
is: the correct answer is always unambiguous, only the phrasing gets harder.
Level 6 is different in kind — it adds an inference step instead.

- **Level 1** — Both places are globally famous and the bearing matches everyone's mental map. Oslo and Rome; Cairo and Johannesburg.
- **Level 2** — Still uncontroversial, but needs the reader to place the two countries correctly. Warsaw and Athens; Lima and Caracas.
- **Level 3** — A stereotype misleads. Detroit and Windsor: Canada is 'above' the USA, yet Detroit is the northern one. Reno and Los Angeles: Reno is further west.
- **Level 4** — Clearly counter-intuitive. Most educated readers would guess wrong: the bearing contradicts what climate, culture or a rough mental map suggests.
- **Level 5** — Strongly counter-intuitive, needing real knowledge of latitudes or longitudes. Venice is north of Halifax. Nairobi is east of Rio de Janeiro. Most people are confident and wrong.
- **Level 6 — MULTI-HOP** — the relation between A and B is NOT stated. The
  description states two links through an intermediate place C, and the reader
  must compose them.

  Keep the WORDING PLAIN at this level. Levels 1-5 make the phrasing harder;
  Level 6 makes the *inference* harder. If a row is both obscurely worded and
  multi-hop, we cannot tell which caused the difficulty, and the row is wasted.

  **Level 6 exists ONLY for these labels: north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of.**
  The other labels have no forced two-hop composition, so do not produce Level 6
  rows for them at all. The grid is deliberately ragged here.

North/south and east/west compose along their own axis:
  A is north of C  +  C is north of B   =>  A is north of B
Diagonals compose only when BOTH steps share the same diagonal (northeast + northeast => northeast). Never chain a north step with an east step and claim northeast: that is not forced unless the distances happen to make it so, and the reader cannot know them.

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
    "Kampala sits further from the North Pole than Khartoum, and Khartoum in turn sits further from the North Pole than Cairo. (A=Kampala, C=Khartoum, B=Cairo — all three named, both links stated. Level 6 is the ONE place a directional phrase is allowed, because without it there is no chain to compose.)"


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
Header row exactly as below, then 144 data rows.

Columns:
  source_entity     the subject place (A)
  source_geometry   one of: Point, LineString, Polygon, MultiPolygon
  target_entity     the object place (B)
  target_geometry   one of: Point, LineString, Polygon, MultiPolygon
  corpus            the natural-language description (the model sees ONLY this
                    plus the two names — the answer must be derivable from it)
  via_entity        Level 6 ONLY: the intermediate place C. Leave EMPTY for
                    Levels 1-5.
  relation_type     always: cardinal_direction
  relation_label    one of the allowed labels above
  explanation       one sentence saying why the label holds; for Level 6, spell
                    out the two-step chain
  ambiguity_level   Level 1 .. Level 6

source_entity,source_geometry,target_entity,target_geometry,corpus,via_entity,relation_type,relation_label,explanation,ambiguity_level

## Additional rules

A. NO TEMPLATES. The previous batch produced 144 rows from 48 sentence frames, each reused three times with only the names swapped, and 16 explanations reused nine times each. That inflates the row count without adding information. Every description must be a distinct sentence, and no two may share a recognisable frame. If you find yourself writing "Traveling from X, ... to reach Y" a second time, rewrite it.

B. Vary sentence length and shape: some one clause, some two; some leading with the subject, some with the object; some naming a river, a coastline, an economic role, a founding date.

C. Every `explanation` must also be distinct and must state the actual reason — which latitude or longitude relation holds, and why a reader might get it wrong. Not "the phrase indicates a northern trajectory".

D. Prefer well-known cities over obscure ones. The task is to test whether the model KNOWS the geography, so the places must be ones a knowledgeable person could reasonably be expected to place.

E. THE DIRECTION MUST BE UNAMBIGUOUS. Each of the eight labels covers a 45-degree sector, so a pair whose true bearing lands near a sector edge has no single defensible answer — a careful geographer could call it either way, and grading a model on it measures nothing. Pick pairs that sit well inside their sector. Concretely: for `north_of`/`south_of`/`east_of`/`west_of` the offset along the other axis must be small relative to the offset along the named axis; for the four diagonals the two offsets must be roughly comparable. Chicago and Taipei are almost exactly on the north/northeast line — never choose a pair like that.

F. NO NEAR-ANTIPODAL PAIRS. Two places on opposite sides of the globe have no well-defined compass direction between them: the bearing flips depending on which way you travel. Keep both places within roughly a third of the globe of each other — same continent, neighbouring continents, or across one ocean, never across the Pacific AND the pole. Perth to Caracas, Asuncion to Manila, and Hanoi to Lima are all unusable for this reason.

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

These are hand-written in the style wanted, one per level.
The rows already in the corpus use an older design that this
prompt forbids, so they are deliberately not shown.

City of Oslo,Polygon,City of Rome,Polygon,"Oslo sits at the head of a long fjord and serves as Norway's seat of government. Rome straddles the Tiber in central Italy.",,cardinal_direction,north_of,"Oslo lies near 60N, Rome near 42N. Nothing here is surprising; both are well-placed in most mental maps.",Level 1
City of Lima,Polygon,City of Caracas,Polygon,"Lima is Peru's capital and holds close to a third of the country's population. Caracas is Venezuela's capital and largest city.",,cardinal_direction,west_of,"Lima sits near 77W against Caracas near 67W. A reader who can place Peru and Venezuela on the continent gets this right.",Level 2
City of Detroit,Polygon,City of Windsor,Polygon,"Detroit grew around the American car industry on the river of the same name. Windsor faces it from Ontario across that water.",,cardinal_direction,north_of,"Detroit is at 42.33N and Windsor at 42.31N. Because Canada is drawn above the United States, almost everyone guesses the reverse.",Level 3
City of Reno,Polygon,City of Los Angeles,Polygon,"Reno lies in the Nevada high desert beside the Truckee River. Los Angeles is California's largest coastal metropolis.",,cardinal_direction,west_of,"Reno sits at 119.8W and Los Angeles at 118.2W. The California coastline bends far enough that an inland Nevada city is the more westerly of the two.",Level 4
City of Venice,Polygon,City of Halifax,Polygon,"Venice is built across lagoon islands in the Italian Veneto. Halifax is the Atlantic port that anchors Nova Scotia.",,cardinal_direction,north_of,"Venice is at 45.4N and Halifax at 44.6N. A Mediterranean city and a cold Canadian port invite the opposite guess.",Level 5
City of Kampala,Polygon,City of Cairo,Polygon,"Kampala sits further from the North Pole than Khartoum, and Khartoum in turn sits further from the North Pole than Cairo.",City of Khartoum,cardinal_direction,south_of,"Two south-of steps compose. Kampala 0.3N, Khartoum 15.5N, Cairo 30.0N.",Level 6

## Entity pairs already used — do not repeat these

African Continent | European Mainland
Aleutian Islands | Hawaiian Archipelago
Atlantic Ocean | United States
Australia | India
Australia | Indonesia
Big Diomede Island | Little Diomede Island
Canada | United States
Canary Islands | Spain
Chile | Argentina
City of Boston | City of Philadelphia
City of Boston | City of Washington D.C.
City of Detroit | City of Windsor
City of Edinburgh | City of Dublin
City of Edmonton | City of Calgary
City of Lima | City of Atlanta
City of London | City of Toronto
City of Los Angeles | City of Phoenix
City of Los Angeles | City of Reno
City of Miami | City of Atlanta
City of Miami | City of New Orleans
City of Miami | City of Orlando
City of Miami | City of Seattle
City of Moscow | City of Paris
City of New York | City of Chicago
City of New York | City of Rome
City of New York | City of Santiago
City of Paris | City of Montreal
City of Perth | City of Darwin
City of Reno | City of Los Angeles
City of Reykjavik | City of London
City of Rome | City of New York
City of San Antonio | City of Austin
City of San Diego | City of Chicago
City of San Diego | City of Denver
City of San Diego | City of Las Vegas
City of San Diego | City of New York
City of San Diego | City of Salt Lake City
City of Seattle | City of Denver
City of Seattle | City of Las Vegas
City of Seattle | City of Miami
City of Seattle | City of Portland
City of Tokyo | City of Seoul
City of Venice | City of Boston
City of Venice | City of Halifax
City of Vladivostok | City of Beijing
Cuba | State of Florida
Cuba | Yucatan Peninsula
Easter Island | Galapagos Islands
Falkland Islands | Argentine Mainland
Falkland Islands | City of Cape Town
Falkland Islands | City of Lima
Finland | Germany
Florida Keys | City of Brownsville
Florida Keys | State of Louisiana
Iceland | Republic of Ireland
Iceland | United Kingdom
Italy | Switzerland
Japan | China
Japan | Philippines
Japan | Taiwan
Madagascar | African Mainland
Madagascar | Mainland Africa
Mexico | United States
New Zealand | Australia
New Zealand | City of Tokyo
New Zealand | Indonesia
Norway | Denmark
Pacific Ocean | United States
Panama City | City of Colon
Point Roberts | City of Vancouver
Portugal | France
Portugal | Spain
Scotland | England
South America | Africa
Spain | France
State of Alaska | Contiguous United States
State of Alaska | State of Colorado
State of Alaska | State of Hawaii
State of Alaska | State of Texas
State of Arizona | State of Utah
State of California | State of Nevada
State of Florida | State of Alabama
State of Florida | State of Mississippi
State of Hawaii | State of California
State of Hawaii | State of Oregon
State of Kansas | State of Missouri
State of Maine | State of New York
State of Maine | State of Pennsylvania
State of Minnesota | State of Iowa
State of Nevada | State of California
State of Ohio | State of Indiana
State of Texas | State of Oklahoma
State of Washington | State of Colorado
Svalbard Archipelago | City of Utqiagvik
Svalbard | Iceland
Tierra del Fuego | Cape of Good Hope
