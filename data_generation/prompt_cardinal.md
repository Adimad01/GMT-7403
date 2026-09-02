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

CRITICAL: cardinal is currently SATURATED — the model already answers 97-100% correctly, so easy items are worthless. Every new item must be genuinely HARD: the honest bearing must surprise an educated reader. If a well-informed person would answer instantly, do not include it.

## The six ambiguity levels

Levels 1-5 describe HOW HARD THE WORDING is, not how uncertain the geography
is: the correct answer is always unambiguous, only the phrasing gets harder.
Level 6 is different in kind — it adds an inference step instead.

- **Level 1** — Straightforward, matching intuition: 'to reach Seattle from Portland you drive straight up Interstate 5'.
- **Level 2** — Clock-face or map-axis phrasing instead of a compass word: 'head towards the 12 o'clock position on your map'.
- **Level 3** — Mildly surprising, where a national stereotype misleads: Detroit is NORTH of Windsor, Canada.
- **Level 4** — Clearly counter-intuitive: the true bearing contradicts what most people assume from climate, culture or rough mental maps.
- **Level 5** — Strongly counter-intuitive, needing real geographic knowledge: 'Venice sits closer to the icy top of the world than Halifax' (Venice IS north of Halifax).
- **Level 6 — MULTI-HOP** — the relation between A and B is NOT stated. The
  description states two links through an intermediate place C, and the reader
  must compose them.

  Keep the WORDING PLAIN at this level. Levels 1-5 make the phrasing harder;
  Level 6 makes the *inference* harder. If a row is both obscurely worded and
  multi-hop, we cannot tell which caused the difficulty, and the row is wasted.

  **Level 6 exists ONLY for these labels: north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of.**
  The other labels have no forced two-hop composition, so do not produce Level 6
  rows for them at all. The grid is deliberately ragged here.

North/south and east/west compose along their own axis. State two links and let the reader chain them:
  A is north of C  +  C is north of B   =>  A is north of B
Diagonals compose only when both steps share the diagonal (northeast + northeast => northeast). Do NOT chain a north step with an east step and claim northeast — that is not forced unless the distances make it so, and the reader cannot know them.

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
    "Kampala sits further up the 12 o'clock axis than Khartoum, and Khartoum in turn sits further up that same axis than Cairo. (A=Kampala, C=Khartoum, B=Cairo — all three named, both links stated.)"


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
  source_geometry   one of: point, line, polygon
  target_entity     the object place (B)
  target_geometry   one of: point, line, polygon
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

Scotland,Polygon,England,Polygon,"Scotland is located at the top of the island, sitting directly up from England.",,cardinal_direction,north_of,"The phrases 'at the top' and 'directly up' provide clear vernacular hints for north.",Level 1
State of Minnesota,Polygon,State of Iowa,Polygon,"If you leave Iowa, keep moving vertically toward the top of the map to cross the border into Minnesota.",,cardinal_direction,north_of,"The terms 'vertically toward the top' give a clear 2D mapping instruction for a northern trajectory.",Level 2
Svalbard,MultiPolygon,Iceland,Polygon,"The archipelago of Svalbard is positioned much further up the longitudinal lines, sitting drastically closer to the absolute top of the world than Iceland.",,cardinal_direction,north_of,"The phrase 'closer to the absolute top' provides a strict upward latitudinal mapping.",Level 3
Svalbard,MultiPolygon,Iceland,Polygon,"The archipelago of Svalbard is positioned much further up the longitudinal lines, sitting drastically closer to the absolute top of the world than Iceland.",,cardinal_direction,north_of,"The phrase pointing to the absolute top provides a strict upward latitudinal mapping.",Level 4
City of Venice,Polygon,City of Halifax,Polygon,"Despite its famous Mediterranean-style climate and gondolas, the Italian city of Venice actually sits closer to the icy top of the world than the freezing Canadian maritime hub of Halifax.",,cardinal_direction,north_of,"A paradox of climate vs. latitude; heading towards the top of the globe maps to an upward direction.",Level 5
Italy,Polygon,Switzerland,Polygon,"Italy extends downwards from Switzerland, pointing toward the bottom of the European map.",,cardinal_direction,south_of,"The terms 'downwards' and 'bottom of the map' clearly indicate a southern direction.",Level 1
State of Texas,Polygon,State of Oklahoma,Polygon,"Texas is situated directly underneath Oklahoma on the map, pointing straight toward the equator.",,cardinal_direction,south_of,"The expressions 'directly underneath' and 'toward the equator' map to a southern cardinal direction.",Level 2
City of New York,Polygon,City of Rome,Polygon,"Despite the harsh winters, New York City actually sits closer to the tropics, positioned downwards on the latitudinal grid from Rome.",,cardinal_direction,south_of,"A counter-intuitive global relationship where the 'downwards' hint overrides weather assumptions.",Level 3
African Continent,Polygon,European Mainland,Polygon,"The vast landmass of the African continent extends deeply downward from the Mediterranean, occupying the lower latitudes underneath the European mainland.",,cardinal_direction,south_of,"A continental relationship where moving downward and underneath maps to a descending vector.",Level 4
City of New York,Polygon,City of Rome,Polygon,"In a classic cartographic illusion, the snow-prone metropolis of New York is actually positioned surprisingly closer to the tropical belt and downwards on the latitudinal plane relative to the sunny historical center of Rome.",,cardinal_direction,south_of,"A counter-intuitive global spatial relationship where a US city is situated downward relative to a European capital.",Level 5
Atlantic Ocean,Polygon,United States,Polygon,"On a standard map, the Atlantic Ocean sits immediately to the right of the United States.",,cardinal_direction,east_of,"The vernacular 'to the right' on a map corresponds to the eastern direction.",Level 1
State of Ohio,Polygon,State of Indiana,Polygon,"Ohio is positioned immediately rightward of Indiana, sliding along the exact same horizontal parallel.",,cardinal_direction,east_of,"The words 'rightward' and 'horizontal parallel' provide a clear vernacular mapping for east.",Level 2
City of Los Angeles,Polygon,City of Reno,Polygon,"Because of the deep concave shape of the coastline, the Pacific beaches of Los Angeles are surprisingly positioned further towards the 3 o'clock mark than the inland desert of Reno.",,cardinal_direction,east_of,"A cartographic anomaly explained by the coastline; '3 o'clock mark' strictly dictates the rightward direction.",Level 3
New Zealand,Polygon,Australia,Polygon,"Sitting further along the grid, New Zealand welcomes the morning light a full two hours ahead, placing it completely rightward of the Australian continent.",,cardinal_direction,east_of,"Time zone chronologies and morning light combined with the rightward orientation establish this spatial vector.",Level 4
City of Los Angeles,Polygon,City of Reno,Polygon,"Due to the extreme concave sweep of the California coastline, the Pacific beaches of Los Angeles are actually located further towards the Atlantic morning than the inland Nevada desert city of Reno.",,cardinal_direction,east_of,"A counter-intuitive anomaly where a coastal city sits further rightward than an inland city due to the curve of the continent.",Level 5
Portugal,Polygon,Spain,Polygon,"Portugal occupies the left-hand side of the Iberian Peninsula, sitting right next to Spain.",,cardinal_direction,west_of,"The phrase 'left-hand side' provides a direct linguistic cue for west.",Level 1
City of Los Angeles,Polygon,City of Phoenix,Polygon,"Leaving Phoenix, you drive straight towards the 9 o'clock position to hit the beaches of Los Angeles.",,cardinal_direction,west_of,"The vernacular '9 o'clock position' is a direct analog for the western direction.",Level 2
Big Diomede Island,Polygon,Little Diomede Island,Polygon,"Despite being a full calendar day ahead, the Russian-owned Big Diomede Island sits physically closer to the 9 o'clock mark and the evening horizon than its American twin.",,cardinal_direction,west_of,"A relationship across the Date Line mapping to a leftward vector despite the temporal paradox.",Level 3
Chile,Polygon,Argentina,Polygon,"Hemmed in by the Andes mountains, the slender nation of Chile occupies the extreme Pacific-facing edge of the continent, positioned completely towards the sunset from Argentina.",,cardinal_direction,west_of,"A macro-hemispheric layout where the source occupies a position distinctly leftward of the target.",Level 4
City of Reno,Polygon,City of Los Angeles,Polygon,"Defying all logical continental intuition, the landlocked Nevada city of Reno is physically positioned further towards the Pacific sunset than the coastal metropolis of Los Angeles.",,cardinal_direction,west_of,"The inverse of the California concavity paradox, pointing strictly to a leftward vector.",Level 5
Japan,Polygon,Taiwan,Polygon,"Looking at the map, Japan is situated diagonally up and towards the sunrise from Taiwan.",,cardinal_direction,northeast_of,"Combining 'diagonally up' and 'towards the sunrise' yields northeast.",Level 1
City of Boston,Polygon,City of Washington D.C.,Polygon,"To travel from Washington D.C. to Boston, you angle your trip up and toward the 3 o'clock mark.",,cardinal_direction,northeast_of,"Combining 'up' and the '3 o'clock mark' yields a vernacular expression for northeast.",Level 2
City of Venice,Polygon,City of Boston,Polygon,"Venice's canals are actually located on a diagonal trajectory pointing up and towards the 2 o'clock position relative to the Massachusetts bay of Boston.",,cardinal_direction,northeast_of,"A combined 'up' and '2 o'clock' geographical vector across the Atlantic.",Level 3
Japan,Polygon,Philippines,Polygon,"To sail from the Philippines to Japan, a vessel must chart a diagonal course aiming for both higher latitudes and the morning dawn.",,cardinal_direction,northeast_of,"A combined upward and rightward geographical vector across the ocean.",Level 4
City of Edinburgh,Polygon,City of Dublin,Polygon,"The Scottish capital of Edinburgh is located on a geographic diagonal from the Republic of Ireland, pointing concurrently towards the top of the map and the prime meridian.",,cardinal_direction,northeast_of,"An island-to-island relationship forming an upward and rightward diagonal vector.",Level 5
State of Washington,Polygon,State of Colorado,Polygon,"The State of Washington sits in the extreme upper left quadrant of the country compared to Colorado.",,cardinal_direction,northwest_of,"The vernacular 'upper left quadrant' indicates a combined north and west direction.",Level 1
Iceland,Polygon,Republic of Ireland,Polygon,"Iceland sits diagonally upwards and towards the fading daylight from the Republic of Ireland.",,cardinal_direction,northwest_of,"Combining 'upwards' and 'fading daylight' creates a vernacular mapping for northwest.",Level 2
State of Alaska,Polygon,State of Hawaii,Polygon,"Alaska sits dramatically closer to the icy top of the globe and further towards the 10 o'clock mark than the tropical islands of Hawaii.",,cardinal_direction,northwest_of,"A diagonal spatial vector pointing upward ('top of the globe') and leftward ('10 o'clock').",Level 3
Iceland,Polygon,United Kingdom,Polygon,"Iceland sits isolated in the frigid waters, positioned diagonally upward and towards the open Atlantic evening from the British Isles.",,cardinal_direction,northwest_of,"An island relationship where the source is mapped diagonally upward and to the left.",Level 4
State of Alaska,Polygon,State of Texas,Polygon,"Severed from the contiguous landmass, the colossal Alaskan frontier rests in the extreme upper left quadrant of the continent, positioned diagonally towards the upper tundra and the Bering sunset from the Lone Star State.",,cardinal_direction,northwest_of,"A combined geographical relationship pointing upward and toward the evening sun.",Level 5
City of Miami,Polygon,City of Atlanta,Polygon,"To drive from Atlanta to Miami, you must travel diagonally down and towards the morning sun.",,cardinal_direction,southeast_of,"Combining 'down' (south) and 'morning sun' (east) equates to southeast.",Level 1
State of Florida,Polygon,State of Mississippi,Polygon,"Florida hangs off the continent, pointing diagonally towards the equator and the dawn relative to Mississippi.",,cardinal_direction,southeast_of,"Combining 'toward the equator' (down) and 'the dawn' (right) results in southeast.",Level 2
City of Miami,Polygon,City of Seattle,Polygon,"Flying from Seattle, a traveler must head on a massive diagonal trajectory aiming for the equator and the 5 o'clock mark to reach the beaches of Miami.",,cardinal_direction,southeast_of,"Combining 'equator' (down) and '5 o'clock mark' equates directly to this diagonal direction.",Level 3
Madagascar,Polygon,African Mainland,Polygon,"The island of Madagascar is located on a diagonal slope dropping downwards and pointing towards the 4 o'clock mark from the bulk of the African mainland.",,cardinal_direction,southeast_of,"A diagonal spatial vector pointing downward and toward the 4 o'clock position.",Level 4
New Zealand,MultiPolygon,City of Tokyo,Polygon,"To navigate from the dense metropolis of Tokyo to the New Zealand landmass, one must sail diagonally across the ocean, crossing the equator while steering steadily towards the 5 o'clock dawn.",,cardinal_direction,southeast_of,"An extreme cross-oceanic vector pointing downward and to the right.",Level 5
City of San Diego,Polygon,City of Denver,Polygon,"When flying from Denver to San Diego, you head diagonally down towards the evening sun.",,cardinal_direction,southwest_of,"Combining 'down' (south) and 'evening sun' (west) equates to southwest.",Level 1
Portugal,Polygon,France,Polygon,"Portugal is situated diagonally towards the warmer climates and the setting sun from France.",,cardinal_direction,southwest_of,"Combining 'warmer climates' (down/equator-bound) and 'setting sun' (left) yields southwest.",Level 2
Canary Islands,MultiPolygon,Spain,Polygon,"The Canary Islands sit diagonally closer to the tropical line and further towards the open ocean twilight relative to the Spanish mainland.",,cardinal_direction,southwest_of,"A regional layout where the source occupies a position that is lower ('tropical line') and to the left ('twilight').",Level 3
Falkland Islands,MultiPolygon,City of Cape Town,Polygon,"The contested Falkland Islands sit diagonally deeper into the freezing lower latitudes and further towards the 8 o'clock setting sun than the African city of Cape Town.",,cardinal_direction,southwest_of,"An extreme cross-oceanic vector pointing downward and to the left.",Level 4
Easter Island,Polygon,Galapagos Islands,MultiPolygon,"The monolithic statues of Easter Island sit isolated in the deep ocean, positioned on a severe diagonal vector pointing towards the absolute bottom of the map and the ultimate oceanic sunset from the Galapagos.",,cardinal_direction,southwest_of,"An island-to-island regional layout mapped diagonally downward and to the left.",Level 5

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
