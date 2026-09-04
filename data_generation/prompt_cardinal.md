# TASK: generate 336 new spatial-relation examples (cardinal)

You are extending a research dataset used to test how well language models
reason about space. I need 336 NEW rows, 7 per cell:

  Levels 1-5: all 8 labels x 5 levels = 40 cells
  Level 6   : only 8 labels (north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of) = 8 cells

  total 48 cells x 7 = 336 rows, of which 56 are multi-hop.

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

Levels 1-5 grade HOW HARD THE GEOGRAPHY is. The correct answer is always
unambiguous; what changes is how much a reader must know, and how far the
configuration sits from the obvious case. The wording never carries the
answer — see the headline rule below. Level 6 is different in kind: it adds
an inference step instead.

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
Header row exactly as below, then 336 data rows.

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

City of Abu Dhabi | City of Dushanbe
City of Abu Dhabi | City of Kathmandu
City of Abu Dhabi | City of Minsk
City of Accra | City of Mumbai
City of Accra | City of Paris
City of Accra | City of Pune
City of Addis Ababa | City of Yangon
City of Amman | City of Algiers
City of Amsterdam | City of Lagos
City of Ankara | City of Marseille
City of Antananarivo | City of London
City of Antananarivo | City of Mogadishu
City of Ashgabat | City of Muscat
City of Asuncion | City of Dublin
City of Asuncion | City of Faro
City of Athens | City of Salvador
City of Atlanta | City of Fortaleza
City of Atlanta | City of Recife
City of Beirut | City of Malmo
City of Beirut | City of Tunis
City of Belgrade | City of Abu Dhabi
City of Berlin | City of Luanda
City of Berlin | City of Madrid
City of Bishkek | City of Melbourne
City of Bogota | City of Harare
City of Bogota | City of Khartoum
City of Bratislava | City of Windhoek
City of Brisbane | City of Hanoi
City of Bucharest | City of Buenos Aires
City of Budapest | City of Beirut
City of Buenos Aires | City of San Diego
City of Cape Town | City of Almaty
City of Cape Town | City of Cairo
City of Cape Town | City of Oslo
City of Cape Town | City of Pune
City of Casablanca | City of Athens
City of Casablanca | City of Stockholm
City of Casablanca | City of Turin
City of Chennai | City of Karachi
City of Colombo | City of Addis Ababa
City of Curitiba | City of Paris
City of Dakar | City of Milan
City of Damascus | City of Valencia
City of Dar es Salaam | City of Edinburgh
City of Dar es Salaam | City of Islamabad
City of Delhi | City of Baghdad
City of Faro | City of La Paz
City of Fortaleza | City of New York
City of Genoa | City of Casablanca
City of Gothenburg | City of Maputo
City of Guayaquil | City of Caracas
City of Guayaquil | City of Harare
City of Halifax | City of Windhoek
City of Hanoi | City of Sydney
City of Harare | City of Almaty
City of Harare | City of Khartoum
City of Helsinki | City of Athens
City of Helsinki | City of Cape Town
City of Hong Kong | City of Beijing
City of Hong Kong | City of Port Moresby
City of Islamabad | City of Tehran
City of Islamabad | City of Windhoek
City of Jakarta | City of Accra
City of Jakarta | City of Suva
City of Johannesburg | City of Helsinki
City of Kampala | City of Dubai
City of Kampala | City of Kingston
City of Kampala | City of Manila
City of Kampala | City of Singapore
City of Karachi | City of Cairo
City of Karachi | City of Cape Town
City of Khartoum | City of Helsinki
City of La Paz | City of Salt Lake City
City of Lagos | City of London
City of Lagos | City of Pune
City of Lagos | City of San Juan
City of London | City of Accra
City of London | City of Minsk
City of Luanda | City of Tunis
City of Lyon | City of Cairo
City of Lyon | City of Chisinau
City of Madrid | City of Santiago
City of Malmo | City of Windhoek
City of Managua | City of Windhoek
City of Maputo | City of Riga
City of Maputo | City of Tallinn
City of Mexico City | City of Quito
City of Miami | City of Lima
City of Minsk | City of Edinburgh
City of Mogadishu | City of Kuala Lumpur
City of Montevideo | City of Algiers
City of Montevideo | City of Calgary
City of Montevideo | City of Paris
City of Moscow | City of Perth
City of Mumbai | City of Maputo
City of Mumbai | City of Perth
City of Munich | City of Valparaiso
City of Nairobi | City of Edinburgh
City of Nairobi | City of Fortaleza
City of Naples | City of Dublin
City of Naples | City of Recife
City of New York | City of Bogota
City of Nice | City of Bergen
City of Nicosia | City of Ashgabat
City of Oslo | City of Odesa
City of Oslo | City of Rome
City of Perth | City of Colombo
City of Porto | City of Lima
City of Prague | City of Tripoli
City of Quito | City of Salt Lake City
City of Quito | City of Santiago
City of Quito | City of Toronto
City of Reno | City of Auckland
City of Riyadh | City of Kinshasa
City of Riyadh | City of Warsaw
City of Salt Lake City | City of Rio de Janeiro
City of Salvador | City of Lyon
City of Salvador | City of Windhoek
City of San Francisco | City of Wellington
City of Santiago | City of Caracas
City of Sao Paulo | City of Winnipeg
City of Sarajevo | City of Asuncion
City of Sarajevo | City of Harare
City of Singapore | City of Mogadishu
City of Suva | City of Pune
City of Sydney | City of Tashkent
City of Tbilisi | City of Riyadh
City of Tokyo | City of Beijing
City of Tokyo | City of Ho Chi Minh City
City of Ulaanbaatar | City of Wellington
City of Valparaiso | City of La Paz
City of Venice | City of St Petersburg
City of Vientiane | City of Dubai
City of Warsaw | City of Cape Town
City of Warsaw | City of Montevideo
City of Warsaw | City of Yerevan
City of Washington | City of Lima
City of Wellington | City of Ho Chi Minh City
City of Windhoek | City of Bergen
City of Windhoek | City of Kinshasa
City of Windhoek | City of Muscat
City of Yangon | City of Riyadh
City of Yerevan | City of Jakarta
City of Zurich | City of Jerusalem
