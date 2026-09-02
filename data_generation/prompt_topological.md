# TASK: generate 190 new spatial-relation examples (topological)

You are extending a research dataset used to test how well language models
reason about space. I need 190 NEW rows, 5 per cell:

  Levels 1-5: all 7 labels x 5 levels = 35 cells
  Level 6   : only 3 labels (contains, within, disjoint) = 3 cells

  total 38 cells x 5 = 190 rows, of which 15 are multi-hop.

The Level 6 grid is deliberately smaller: the remaining labels have no forced
two-hop composition, so multi-hop rows for them would have no determinate
answer.

## What the data captures

TOPOLOGICAL RELATION — how the areas of two places relate (DE-9IM style)

Allowed labels (use these exact strings): contains, within, touches, crosses, disjoint, overlaps, equals

Label meanings: contains = A fully encloses B. within = A is fully inside B. touches = share a boundary but interiors do not overlap. crosses = they cut through each other. disjoint = no contact at all. overlaps = partial overlap, neither contains the other. equals = the same area under two names.

## The six ambiguity levels

Levels 1-5 describe HOW HARD THE WORDING is, not how uncertain the geography
is: the correct answer is always unambiguous, only the phrasing gets harder.
Level 6 is different in kind — it adds an inference step instead.

- **Level 1** — Very well-known places, relation stated plainly: 'California fully envelops Los Angeles'.
- **Level 2** — Well-known but needing a moment: a country completely encircling a microstate.
- **Level 3** — Requires specific knowledge: municipal limits versus an enclave's borders.
- **Level 4** — Unusual geopolitical cases: enclaves, exclaves, condominiums, disputed zones.
- **Level 5** — Large natural or geomorphic features rather than administrative ones: oceans, trenches, deserts, mountain ranges, river basins.
- **Level 6 — MULTI-HOP** — the relation between A and B is NOT stated. The
  description states two links through an intermediate place C, and the reader
  must compose them.

  Keep the WORDING PLAIN at this level. Levels 1-5 make the phrasing harder;
  Level 6 makes the *inference* harder. If a row is both obscurely worded and
  multi-hop, we cannot tell which caused the difficulty, and the row is wasted.

  **Level 6 exists ONLY for these labels: contains, within, disjoint.**
  The other labels have no forced two-hop composition, so do not produce Level 6
  rows for them at all. The grid is deliberately ragged here.

Only some compositions are logically FORCED. Use these, and no others:
  A within C   + C within B    => A within B
  A contains C + C contains B  => A contains B
  A within C   + C disjoint B  => A disjoint from B
NEVER chain 'touches' with 'touches' — A touches C and C touches B implies NOTHING about A and B. The same applies to crosses and overlaps. If the composition is not forced, the item is unusable.

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
    "The Vatican Museums sit entirely inside the walls of Vatican City, and Vatican City in turn lies wholly inside the municipal boundary of Rome. (A=Vatican Museums, C=Vatican City, B=Rome — all three named, both links stated, and C is a real third place rather than a synonym.)"


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
Header row exactly as below, then 190 data rows.

Columns:
  source_entity     the subject place (A)
  source_geometry   one of: point, line, polygon
  target_entity     the object place (B)
  target_geometry   one of: point, line, polygon
  corpus            the natural-language description (the model sees ONLY this
                    plus the two names — the answer must be derivable from it)
  via_entity        Level 6 ONLY: the intermediate place C. Leave EMPTY for
                    Levels 1-5.
  relation_type     always: topological
  relation_label    one of the allowed labels above
  explanation       one sentence saying why the label holds; for Level 6, spell
                    out the two-step chain
  ambiguity_level   Level 1 .. Level 6

source_entity,source_geometry,target_entity,target_geometry,corpus,via_entity,relation_type,relation_label,explanation,ambiguity_level

## Additional rules

1. The label describes A with respect to B, in that order.
2. Do not reuse any (source_entity, target_entity) pair listed at the bottom.
3. Do not use the same pair twice in your own output, and never produce a pair
   together with its mirror (if you write "A contains B", do not also write
   "B within A" — that leaks answers between our train and test splits).
4. The `corpus` text must NOT contain the label word or an obvious synonym.
   Write "sits at the 12 o'clock mark", not "is north of".
5. On Level 6 the text must state the two links and NOT the A-B relation. If a
   reader can answer without composing both steps, it is not multi-hop.
6. `explanation` is never shown to the model — do not rely on it to make a row
   solvable.
7. Vary geography: do not draw every example from the United States.
8. Every row must be factually TRUE. Verify the geography before writing it.

## Existing examples, one per label x level (match this style)

Note these predate the `via_entity` column, so it is empty in all of them.

State of California,Polygon,City of Los Angeles,Polygon,"The State of California fully envelops the sprawling jurisdiction of the City of Los Angeles in its southern half.",,topological,contains,"A state polygon fully encompasses a city administrative polygon.",Level 1
Lake Nicaragua,Polygon,Ometepe Island,Polygon,"The massive Central American freshwater basin completely wraps around the twin-volcano island landmass.",,topological,contains,"A lake polygon serving as the complete boundary for a volcanic island polygon.",Level 2
State of Texas,Polygon,The Alamo Mission Footprint,Polygon,"The sprawling state lines completely serve as the absolute boundary box for the historical mission's real estate.",,topological,contains,"A massive state polygon acting as the absolute container for a historical architectural polygon.",Level 3
Australia,Polygon,Australian Capital Territory,Polygon,"The massive continental bounds completely serve as the absolute container for the inland capital territory governing the nation.",,topological,contains,"A massive continental polygon completely enclosing a specific administrative territory.",Level 4
Switzerland,Polygon,Campione d'Italia,Polygon,"The Swiss cantons completely encircle the tiny Italian exclave situated near the waters of Lake Lugano, isolating it from the Italian mainland.",,topological,contains,"A sovereign nation acting as the geographic container for a tiny foreign exclave.",Level 5
City of Paris,Polygon,France,Polygon,"The densely populated capital city is strictly bounded by the national frontiers of the European nation.",,topological,within,"A city polygon entirely trapped by a nation polygon.",Level 1
Badlands National Park,Polygon,State of South Dakota,Polygon,"The eroded buttes and pinnacles of the federal preserve are completely surrounded by the midwestern state boundaries.",,topological,within,"A park trapped by a complete geometric host.",Level 2
Abyei Demilitarized Box,Polygon,Republic of Sudan,Polygon,"The highly contested, jointly administered zone is strictly bound by the sovereign national borders.",,topological,within,"A specialized demilitarized administrative zone completely encircled by a national polygon.",Level 3
Busingen am Hochrhein,Polygon,Swiss Canton of Schaffhausen,Polygon,"The German municipal exclave operates entirely locked inside the Swiss regional borders, utilizing Swiss public services and currency.",,topological,within,"A foreign municipal exclave completely trapped by a host nation's regional canton.",Level 4
United Nations Headquarters,Polygon,Borough of Manhattan,Polygon,"Despite its international diplomatic status, the physical real estate sits entirely confined inside the municipal borders of the massive urban grid.",,topological,within,"An international legal polygon physically completely confined inside a city borough polygon.",Level 5
State of Colorado,Polygon,State of New Mexico,Polygon,"The southern edge of the State of Colorado meets the northern edge of the State of New Mexico forming a perfectly straight latitudinal line.",,topological,touches,"Two state administrative polygons meeting at a shared edge with no internal intersection.",Level 1
Vatican City,Polygon,City of Rome,Polygon,"The ancient masonry walls that define the Vatican stand right at the municipal edge of the Italian capital.",,topological,touches,"A sovereign microstate polygon meeting a city municipal polygon at a physical wall.",Level 2
City of Detroit,Polygon,City of Windsor,Polygon,"The American and Canadian municipalities sit flush against the international boundary drawn invisibly down the center of the Detroit River.",,topological,touches,"Two city polygons from different nations that share a continuous aquatic border.",Level 3
Republic of Botswana,Polygon,Republic of Zambia,Polygon,"Near the Kazungula Bridge, the sovereign territories converge at a highly specific 150-meter micro-boundary precisely in the middle of the Zambezi River.",,topological,touches,"Two massive national polygons that meet at an incredibly tiny, almost microscopic linear boundary.",Level 4
Vennbahn Railway Right-of-Way,Polygon,German Micro-Enclaves,MultiPolygon,"The narrow transit corridor slices through the landscape, abutting the fragmented sovereign territories it creates without entering them.",,topological,touches,"A narrow transit polygon abutting highly fractured micro-enclave polygons.",Level 5
Suez Canal,LineString,Egypt,Polygon,"The engineered waterway cuts a path straight through the desert sands of the nation.",,topological,crosses,"An artificial canal traversing a national polygon.",Level 1
Tropic of Capricorn,LineString,Australia,Polygon,"The global latitudinal line traverses the arid red center of the massive continent.",,topological,crosses,"An abstract planetary boundary line traversing a massive landmass polygon.",Level 2
Eurotunnel Subterranean Tube,LineString,English Channel Maritime Zone,Polygon,"The massive transit architecture bores deep in the bedrock, threading directly under the national maritime jurisdiction.",,topological,crosses,"A subterranean line traversing a maritime polygon.",Level 3
Appalachian Trail,LineString,Mason-Dixon Line,LineString,"The legendary hiking path weaves a route directly over the historical survey boundary severing the northern and southern states.",,topological,crosses,"A physical hiking trail line intersecting a historical boundary line.",Level 4
Trans-Canada Highway Ribbon,LineString,Precambrian Canadian Shield,Polygon,"The massive continental road network threads its asphalt trajectory entirely through the exposed, ancient geological bedrock of the north.",,topological,crosses,"A highway passing completely through a massive geological polygon.",Level 5
United Kingdom,Polygon,New Zealand,Polygon,"The British Isles and the Kiwi islands sit at nearly perfectly opposite ends of the planetary sphere.",,topological,disjoint,"Two island nations geographically completely severed from one another.",Level 1
Nile River,LineString,Amazon River,LineString,"The longest waterway in Africa flows completely independently from the massive river network dominating South America.",,topological,disjoint,"Two massive river line geometries on different continents.",Level 2
Brooklyn Bridge,LineString,East River,Polygon,"The suspended roadway of the Brooklyn Bridge spans high in the air above the East River without its physical structure ever making contact with the water's surface.",,topological,disjoint,"A bridge line geometry and a river polygon geometry that overlap in a 2D map but are strictly separated in 3D topological space.",Level 3
Northwest Angle,Polygon,Contiguous United States,MultiPolygon,"Although legally part of Minnesota, the Northwest Angle is completely severed from the lower 48 states by the waters of the Lake of the Woods, requiring travelers to drive through Canada to reach it.",,topological,disjoint,"An exclave polygon and its parent MultiPolygon that share absolutely zero intersecting spatial coordinates due to an aquatic barrier.",Level 4
London Underground Central Line Subterranean Track,LineString,River Thames Surface Water,Polygon,"The commuter rail dives deep into the clay earth beneath the capital, maintaining perfect vertical isolation from the historic waters above.",,topological,disjoint,"A subterranean transit line perfectly isolated in 3D space from a surface water body.",Level 5
Navajo Nation,Polygon,State of Arizona,Polygon,"The tribal lands of the Navajo Nation cover a large portion of northeastern Arizona while holding lands elsewhere.",,topological,overlaps,"A tribal polygon and a state polygon sharing a massive interior area.",Level 1
Andes Mountains,Polygon,Chile,Polygon,"The massive Andes mountain range blankets the entire eastern border region of Chile while also sprawling deep into Argentina.",,topological,overlaps,"A physical geological polygon and a political national polygon partially intersect.",Level 2
The Alps,Polygon,Switzerland,Polygon,"The massive multinational geological footprint heavily shares space with the sovereign national territory of the Swiss.",,topological,overlaps,"A massive multinational geological polygon sharing interior space with a nation.",Level 3
Tornado Alley,Polygon,State of Oklahoma,Polygon,"The dangerous meteorological zone heavily encompasses the entire state while bleeding outward into the surrounding plains.",,topological,overlaps,"A meteorological zone polygon intersecting heavily with a state polygon.",Level 4
Kuril Islands,MultiPolygon,Japanese Territorial Claim,Polygon,"The competing diplomatic maps mutually project over the exact same frigid island chain in the northern Pacific.",,topological,overlaps,"A physical multi-polygon sharing identical coordinates with a diplomatic claim.",Level 5
Principality of Monaco,Polygon,City of Monaco,Polygon,"The sovereign microstate and its single integrated municipality are entirely synonymous in their territorial extent.",,topological,equals,"A national polygon and a city polygon that are perfectly identical.",Level 1
Vatican City,Polygon,Holy See Territory,Polygon,"The physical masonry walls that define Vatican City map perfectly to the sovereign territorial limits of the Holy See.",,topological,equals,"Physical architectural polygon maps exactly 1:1 with international legal polygon.",Level 2
State of Rhode Island,Polygon,Providence Plantations,Polygon,"The modern geometric borders of the State of Rhode Island map flawlessly to the historical colonial land-grant lines of the Providence Plantations.",,topological,equals,"A modern administrative polygon and a historical legal polygon that resolve to the exact same geometric coordinates.",Level 3
City of London,Polygon,The Square Mile,Polygon,"The ancient, independently governed municipal enclave officially known as the City of London shares a rigidly exact and identical topological footprint with the financial district colloquially defined as the Square Mile.",,topological,equals,"Two completely different semantic named entities (one historical/administrative, one colloquial/financial) that resolve to the exact same geospatial polygon.",Level 4
Demilitarized Zone (DMZ),Polygon,1953 Korean Armistice Ceasefire Buffer,Polygon,"The heavily fortified physical perimeter of the DMZ seamlessly mirrors the theoretical boundaries drafted in the 1953 Korean Armistice Ceasefire Buffer agreement, creating a perfect mapping symmetry.",,topological,equals,"A physically constructed military buffer zone perfectly matching a legally codified treaty polygon down to the coordinate level.",Level 5

## Entity pairs already used — do not repeat these

Brooklyn Bridge | East River
Abyei Area | Republic of South Sudan
Australian Continent | Lake Eyre Basin
Manitoulin Island | Lake Manitou
The Pentagon Grounds | State of Virginia
Mongolia | China
City of Juneau | Juneau Borough
Spratly Islands Maritime Claims | Vietnamese Exclusive Economic Zone
Pampas Grassland | Argentina
City of Miami | City of Seattle
Lake Vostok Subglacial Volume | Antarctic Ice Sheet
Navajo Nation | Hopi Reservation
City of Minneapolis | City of St. Paul
State of New Mexico | White Sands National Park
Sub-Seabed Chunnel Transit Line | French Maritime Legal Zone
Central Park | State of New York
Kaliningrad Oblast | Mainland Russia
Rust Belt | State of Ohio
Inner Temple Legal Precinct | City of London Corporation
Camp David Restricted Perimeter | Catoctin Mountain Park
Interstate 80 | Continental Divide
Kashmir Region | Indian Territorial Claim
City of Sacramento | State of California
Abstract Geodetic Equator | Physical Amazon River Basin
Equator | Amazon River
Trafalgar Square Base | Nelson's Column
Province of Alberta | Canada
Geographic South Pole | 90 Degrees South Latitude
Amazon Rainforest | Brazil
City of Rome | Vatican City
City of Madrid | Spain
Interstate 70 | State of Colorado
Republic of Cyprus | UN Buffer Zone
Rust Belt | American Midwest
Prime Meridian | English Channel
Zion National Park | State of Utah
Argentina | Norway
City of Anchorage | Anchorage Municipality
City of San Diego | Mexico
Lake Superior | Isle Royale
Dutch Baarle-Nassau Enclave N8 | Belgian Baarle-Hertog Enclave H22
Abyei Administrative Area | Republic of Sudan
Suez Canal | Panama Canal
Republic of Italy | San Marino
Grand Teton National Park | State of Wyoming
France | Spain
International Date Line (Conceptual) | 180 Degrees Longitude
Baja California Peninsula | Mainland Mexico
Bermuda | Atlantic Ocean
Borough of The Bronx | Bronx County
City of Rome | Italy
Prime Meridian | Spain
Route 66 | State of Florida
Dahlak Archipelago | Red Sea
Abyei Demilitarized Box | Republic of Sudan
Republic of Chile | Easter Island Landmass
Iceland | Antarctica
Independent City of Baltimore | Baltimore City Government Territory
Rocky Mountains | Andes Mountains
Lombard Street | Broadway (New York)
Catacombs of Paris Subterranean Network | City of Paris
Prince Edward Islands | Republic of South Africa
Mojave Desert | State of California
Prime Meridian (Greenwich) | 0 Degrees Longitude
Channel Islands | United Kingdom
Yellowstone National Park | Yellowstone Caldera
City of London | United Kingdom
Rocky Mountains | The Alps
City of Carson City | Consolidated Municipality of Carson City
Gobi Desert | Mongolia
City of Sydney | City of Cape Town
Uluru | Australia
Demilitarized Zone (DMZ) | 1953 Ceasefire Buffer Zone
State of Kansas | State of Nebraska
Vesuvius Main Crater | Mount Vesuvius Volcanic Cone
City of Dallas | City of Fort Worth
State of Ohio | City of Columbus
Route 66 | Colorado River
Pacific Ocean | Mariana Trench
City of Orlando | State of Florida
Ssese Islands | Lake Victoria
International Space Station Docking Ring | SpaceX Crew Dragon Adapter
Statue of Liberty | Eiffel Tower
Republic of Cyprus | UN Buffer Zone in Cyprus
Black Forest | Baden-Württemberg
United Nations Headquarters | Borough of Manhattan
Artificial Suez Canal Trench | Sinai Desert Landmass
Busingen am Hochrhein | Federal Republic of Germany
Busingen am Hochrhein | Swiss Canton of Schaffhausen
Appalachian Trail | Rocky Mountains
Navigational International Date Line | Chukchi Sea Legal Boundary
Diego Garcia | Indian Ocean
Constantinople Walled City (1453) | Fatih District of Istanbul
State of California | Yosemite National Park
Interstate 40 | State of Arizona
Diomede Islands Maritime Border | International Date Line
State of Idaho | State of Montana
Red Sea | Dahlak Archipelago
Mount Fuji | Japan
France | City of Paris
Niagara Falls Horseshoe | Canada-US Border
City of Jacksonville | Duval County
Baden-Württemberg | Black Forest
Canada | United States
State of Texas | City of Austin
Mediterranean Sea | Island of Malta
White Sands National Park | State of New Mexico
San Marino | Italian Peninsula
Rio Grande | State of New Mexico
Ring of Fire | Pacific Plate
The Gambia | Senegal
Eurotunnel Terminal Boundary | French Maritime Customs Zone
United Kingdom | New Zealand
City of Philadelphia | Philadelphia County
Amazon Basin | Andes Mountains Foothills
St. Peter's Basilica Footprint | Vatican City Jurisdiction
English Channel | French Coast
Italy | City of Rome
Lake Victoria | Ssese Islands
State of South Dakota | Badlands National Park
