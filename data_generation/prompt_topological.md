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
  source_geometry   one of: Point, LineString, Polygon, MultiPolygon
  target_entity     the object place (B)
  target_geometry   one of: Point, LineString, Polygon, MultiPolygon
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

Abstract Geodetic Equator | Physical Amazon River Basin
Abstract Geodetic Prime Meridian | Physical River Thames Current
Abstract Tropic of Cancer | Physical Nile River Flow
Abstract Tropic of Capricorn | Physical Great Dividing Range
Abyei Administrative Area | Republic of Sudan
Abyei Area | Republic of South Sudan
Abyei Area | Republic of Sudan
Abyei Demilitarized Box | Republic of Sudan
Amazon Basin | Andes Mountains Foothills
Amazon Basin | Peru
Amazon Rainforest | Brazil
Amazon River | Brazil
Andes Mountains Foothills | Amazon Basin
Andes Mountains | Chile
Appalachian Footpath | Mason-Dixon Survey Line
Appalachian Mountains | State of Pennsylvania
Appalachian Trail Southern Terminus | Springer Mountain Summit
Appalachian Trail | Mason-Dixon Line
Appalachian Trail | Potomac River
Appalachian Trail | Rocky Mountains
Arctic Circle | Canada
Argentina | Norway
Argentine Antarctica Claim | British Antarctic Territory
Artificial Panama Canal Trench | Continental Divide Geological Ridge
Artificial Suez Canal Trench | Sinai Desert Landmass
Ashmore and Cartier Islands | Commonwealth of Australia Maritime Bounds
Australia | Australian Capital Territory
Australia | Canada
Australia | Uluru
Australian Capital Territory | New South Wales
Australian Continent | Lake Eyre Basin
Baarle-Hertog Cadastral Plots | Baarle-Nassau Cadastral Plots
Baarle-Hertog Municipality | Baarle-Nassau Municipality
Baarle-Nassau Border Monument 214 | Belgian-Dutch Sovereignty Line
Baarle-Nassau Border Monument | Belgian-Dutch Sovereignty Line
Baden-Württemberg | Black Forest
Badlands National Park | State of South Dakota
Baja California Peninsula | Mainland Mexico
Bermuda | Atlantic Ocean
Bir Tawil | Hala'ib Triangle
Black Forest | Baden-Württemberg
Black Forest | State of Baden-Württemberg
Borough of Brooklyn | Kings County
Borough of Manhattan | New York County
Borough of Queens | Queens County
Borough of Staten Island | Richmond County
Borough of The Bronx | Bronx County
Bosphorus Bridge Roadway | Bosphorus Strait
Bosphorus Bridge | Bosphorus Strait
Brazil | Argentina
Brooklyn Bridge Roadway | East River Surface
Brooklyn Bridge | East River
Buckingham Palace | City of London
Busingen am Hochrhein | Federal Republic of Germany
Busingen am Hochrhein | Mainland Germany
Busingen am Hochrhein | Swiss Canton of Schaffhausen
CERN Large Hadron Collider | City of Geneva
CERN Large Hadron Collider | City of Geneva Surface
Cabinda Province | Mainland Angola
Camp David Restricted Perimeter | Catoctin Mountain Park
Camp David | Catoctin Mountain Park
Camp Zeist Legal Compound (1999) | Kingdom of the Netherlands
Campione d'Italia | Mainland Italy
Canada | Province of Saskatchewan
Canada | United States
Catacombs of Paris Subterranean Network | City of Paris
Central Park Reservoir | New York City Municipal Grid
Central Park | State of New York
Challenger Deep | Mariana Trench Geomorphic Bounds
Channel Islands | United Kingdom
Channel Islands | United Kingdom Mainland
Channel Tunnel | English Channel
Channel Tunnel | Prime Meridian
City of Alexandria | Independent City of Alexandria
City of Anchorage | Anchorage Municipality
City of Baltimore | Independent City of Baltimore
City of Beverly Hills | City of Los Angeles
City of Broomfield | Broomfield County
City of Carson City | Consolidated Municipality of Carson City
City of Chicago | City of Houston
City of Columbus | State of Ohio
City of Dallas | City of Fort Worth
City of Denver | Denver County
City of Denver | State of Colorado
City of Detroit | City of Hamtramck
City of Detroit | City of Windsor
City of El Paso | Ciudad Juarez
City of Hamtramck | City of Detroit
City of Honolulu | Mainland United States
City of Jacksonville | Duval County
City of Juneau | Juneau Borough
City of London Corporation | Greater London Authority
City of London | The Square Mile
City of London | United Kingdom
City of Los Angeles | City of Beverly Hills
City of Madrid | Spain
City of Miami | City of Seattle
City of Minneapolis | City of St. Paul
City of Nashville | Davidson County
City of Norwood | City of Cincinnati
City of Orlando | State of Florida
City of Paris | Catacombs of Paris Subterranean Network
City of Paris | City of Tokyo
City of Paris | France
City of Philadelphia | Philadelphia County
City of Phoenix | State of Arizona
City of Rome | Italy
City of Rome | Vatican City
City of Sacramento | State of California
City of San Diego | Mexico
City of San Francisco | San Francisco County
City of Seattle | City of Bellevue
City of Seattle | State of Washington
City of Sitka | Sitka Borough
City of Springfield | State of Illinois
City of St. Louis | Independent City of St. Louis
City of Sydney | City of Cape Town
City of Washington | District of Columbia
Colorado River | Grand Canyon National Park
Colorado River | State of Utah
Commonwealth of Australia | Ashmore and Cartier Islands
Congo Basin | Democratic Republic of the Congo
Constantinople Walled City (1453) | Fatih District of Istanbul
Dahala Khagrabari | Cooch Behar District
Dahlak Archipelago | Red Sea
Danube River | Hungary
Death Valley Below-Sea-Level Basin | State of California
Demilitarized Zone (DMZ) | 1953 Ceasefire Buffer Zone
Demilitarized Zone (DMZ) | 1953 Korean Armistice Ceasefire Buffer
Denali Wilderness Area | State of Alaska
Diego Garcia | Indian Ocean
Diomede Islands Maritime Border | International Date Line
Dutch Baarle-Nassau Enclave N8 | Belgian Baarle-Hertog Enclave H22
Easter Island Landmass | Republic of Chile Maritime Claims
Egypt | Sudan
Eiffel Tower Spire Tip | Highest Architectural Node of Paris
Eiffel Tower | City of Paris
English Channel | French Coast
Equator | Amazon River
Equator | Lake Victoria
Eurostar Railway | France
Eurotunnel Subterranean Tube | English Channel Maritime Zone
Eurotunnel Subterranean Tube | English Channel Surface Waters
Eurotunnel Terminal Boundary | French Maritime Customs Zone
Eurotunnel Terminal Entrance | French Customs Zone
Falkland Islands EEZ | Argentine Sea
Falkland Islands | United Kingdom
Falkland Islands | United Kingdom Mainland
Four Corners Monument Plaque | State of Arizona Boundary
Four Corners Monument | State of Utah
France | City of Paris
France | Spain
French Guiana | Metropolitan France
Gaza Strip | Egypt
Geographic North Pole | Convergence of All Longitude Lines
Geographic South Pole | 90 Degrees South Latitude
Germany | Poland
Glacier National Park | State of Montana
Gobi Desert | China
Gobi Desert | Mongolia
Golden Gate Bridge Northern Anchor | Marin Headlands
Golden Gate Bridge | Marin County
Golden Gate Park | City of San Francisco
Gotthard Base Rail Tube | Swiss-Italian Geopolitical Plane
Gotthard Base Tunnel North Portal | Swiss Alps Geomorphic Base
Gotthard Base Tunnel Portal | Swiss Alps Base
Gotthard Base Tunnel | Rhine River Watershed
Gotthard Base Tunnel | Swiss Alps Surface
Grand Teton National Park | State of Wyoming
Great Barrier Reef | Coral Sea
Great Pyramid Perimeter | King's Chamber Subterranean Volume
Great Rift Valley | Kenya
Great Victoria Desert | South Australia
Great Victoria Desert | Western Australia
Great Wall of China | Japan
Greater London Authority | City of London Corporation
Guantanamo Bay Naval Base | United States Mainland
Gulf Stream | North Atlantic
Hala'ib Triangle | Egyptian Territorial Claim
Himalaya Mountains | Nepal
Historic Route 66 Asphalt | Mississippi River Hydrological Flow
Historical Mount McKinley Summit | Modern Denali Summit
Historical Ottoman Empire Footprint (1683) | Modern European Union Territory
Historical Roman Empire (AD 117) | Modern European Union
Hoover Dam Concrete Face | Lake Mead
Hoover Dam Concrete Face | Lake Mead Water Volume
Hopi Reservation | Navajo Nation
Iberian Peninsula | Spain
Iceland | Antarctica
Iceland | Madagascar
Independent City of Baltimore | Baltimore City Government Territory
Independent City of Baltimore | City of Baltimore
Independent City of Baltimore | City of Baltimore Municipal Limits
Independent City of St. Louis | St. Louis County Municipal Boundary
India | Nepal
Indian Ocean | Diego Garcia
Inner Temple Legal Precinct | City of London Corporation
International Date Line (Conceptual) | 180 Degrees Longitude
International Space Station Docking Ring | SpaceX Crew Dragon Adapter
Interstate 15 | Mojave Desert
Interstate 40 | State of Arizona
Interstate 70 | State of Colorado
Interstate 80 | Continental Divide
Interstate 90 | State of South Dakota
Interstate 95 Asphalt | Mason-Dixon Historical Boundary
Interstate 95 | Mason-Dixon Line
Isla del Sol | Lake Titicaca
Island of Malta | Mediterranean Sea
Island of Oahu | Pacific Ocean
Isle of Wight | Mainland England
Italy | City of Rome
Italy | San Marino
Japan | Brazil
Kalahari Desert | Botswana
Kaliningrad Oblast | Mainland Russia
Kashmir Region | Indian Territorial Claim
King's Chamber Subterranean Volume | Great Pyramid Perimeter
Kingdom of Lesotho | Indian Ocean Maritime Boundary
Kingdom of Lesotho | South African Borders
Kingdom of Saudi Arabia | Rub' al Khali Desert
Kurdish Inhabited Region | Republic of Turkey
Kurdish Inhabited Region | Turkey
Kuril Islands | Japanese Territorial Claim
Lake Baikal | Caspian Sea
Lake Eyre Basin | Australian Continent
Lake Nicaragua | Ometepe Island
Lake Superior | Isle Royale
Lake Victoria | Lake Superior
Lake Victoria | Ssese Islands
Lake Vostok Subglacial Volume | Antarctic Ice Sheet
Little Diomede Island | Big Diomede Island
Loch Ness Water Volume | Great Glen Fault
Lombard Street | Broadway (New York)
London Underground Central Line Subterranean Track | River Thames Surface Water
Madagascar | Indian Ocean
Madagascar | Mainland Africa
Main Crater Lake | Vulcan Point Island
Manitoulin Island | Lake Huron
Manitoulin Island | Lake Manitou
Mariana Trench Geomorphic Lip | Abyssal Plain
Mariana Trench | Pacific Ocean
Mauna Loa Magma Chamber | State of Hawaii
Mediterranean Basin | European Union
Mediterranean Deep Brine Pool | Mediterranean Sea Volume
Mediterranean Sea | Island of Cyprus
Mediterranean Sea | Island of Malta
Mississippi River | State of Louisiana
Mojave Desert | State of California
Mojave Desert | State of Nevada
Mongolia | China
Mount Everest Summit | Nepal-China Border
Mount Fuji | Japan
Mount Kilimanjaro | Mount Everest
Mount McKinley Summit | Denali Summit
Mount Rainier Glacial Cap | Mount Rainier National Park
Mount Rainier National Park | Mount Rainier Glacial Cap
Mount Titano Summit | Republic of San Marino
Municipality of Llivia | French Republic
Nakhchivan Autonomous Republic | Mainland Azerbaijan
Navajo Nation Reservation | Hopi Reservation
Navajo Nation | Hopi Reservation
Navajo Nation | State of Arizona
Navigational International Date Line | Chukchi Sea Legal Boundary
New Delhi District | Republic of India
New York City Municipal Grid | Central Park Reservoir
New Zealand | Greenland
Niagara Falls Horseshoe | Canada-US Border
Niagara Falls | Canada-US Border
Nile River | Amazon River
Nile River | Yellow River
Nine-Dash Line Claim | Philippine Exclusive Economic Zone
Nine-Dash Line Claim | Philippine Exclusive Economic Zone (EEZ)
North Pole | 90 Degrees North Latitude
Northwest Angle | Contiguous United States
Null Island | WGS84 Origin Coordinate
O'Hare International Airport Bounds | City of Chicago
Omani Exclave of Madha | Mainland Oman
Omani Exclave of Madha | UAE Enclave of Nahwa
Ometepe Island | Lake Nicaragua
PATH Train Tunnels | Hudson River
PATH Train Tunnels | Hudson River Water Volume
PATH Train Tunnels | New York-New Jersey State Line
Pacific Ocean | Island of Oahu
Pacific Ocean | Mariana Trench
Pacific Ocean | Mediterranean Sea
Pacific Ocean | State of Hawaii
Pampas Grassland | Argentina
Panama Canal | Isthmus of Panama
Panmunjom T2 Conference Table | Military Demarcation Line
Point Nemo Mathematical Isolation Zone | Pacific Ocean Maritime Bounds
Point Nemo | Ducie Island
Point Roberts Exclave | 49th Parallel North
Portugal | Atlantic Ocean
Portugal | Spain
Prime Meridian (Greenwich) | 0 Degrees Longitude
Prime Meridian | English Channel
Prime Meridian | France
Prime Meridian | River Thames
Prime Meridian | Spain
Prime Meridian | The Equator
Prince Edward Islands | Republic of South Africa
Principality of Monaco | City of Monaco
Principality of Monaco | Municipality of Monaco
Province of Alberta | Canada
Pyrenees Mountains | France
Red Sea | Dahlak Archipelago
Republic of Botswana | Republic of Zambia
Republic of Chile | Easter Island Landmass
Republic of Cyprus | UN Buffer Zone
Republic of Cyprus | UN Buffer Zone in Cyprus
Republic of India | New Delhi
Republic of Ireland | Northern Ireland
Republic of Italy | San Marino
Republic of Italy | Sovereign Military Order of Malta Magistral Villa
Ring of Fire | Pacific Plate
Rio Grande | State of New Mexico
River Thames | City of London
River Thames | Mississippi River
Rocky Mountains | Andes Mountains
Rocky Mountains | Canada
Rocky Mountains | State of Colorado
Rocky Mountains | The Alps
Route 66 | Colorado River
Route 66 | Continental Divide
Route 66 | Mississippi River
Route 66 | State of Florida
Royal Botanic Garden Sydney | City of Sydney
Rub' al Khali Desert | Kingdom of Saudi Arabia
Rust Belt | American Midwest
Rust Belt | State of Ohio
Sahara Desert | Amazon Rainforest
Sahara Desert | Egypt
Sahara Desert | Republic of Mali
Sahara Desert | Sahel Region
San Marino | Italian Peninsula
San Marino | Republic of Italy
Scandinavia | Norway
Seikan Tunnel | Tsugaru Strait
Seikan Tunnel | Tsugaru Strait Water Volume
Senegal | The Gambia
Sonoran Desert | Mexico
South Africa | Kingdom of Lesotho
South Africa | Russia
South Pole Station Flight Path | Convergence of 360 Meridians
Sovereign Borders of The Gambia | Sovereign Borders of Senegal
Sovereign Military Order of Malta HQ | City of Rome
Sovereign Military Order of Malta Magistral Villa | Republic of Italy
Spain | City of Madrid
Spratly Islands Maritime Claims | Philippine Exclusive Economic Zone
Spratly Islands Maritime Claims | Vietnamese Exclusive Economic Zone
Sri Lanka | India
Ssese Islands | Lake Victoria
St. Peter's Basilica Footprint | Vatican City Jurisdiction
State of Alaska | Contiguous United States
State of Alaska | Yukon Territory
State of Arizona | City of Phoenix
State of Arizona | Grand Canyon National Park
State of California | City of Los Angeles
State of California | Death Valley Below-Sea-Level Basin
State of California | Pacific Ocean
State of California | Yosemite National Park
State of Colorado | City of Denver
State of Colorado | State of New Mexico
State of Florida | State of Washington
State of Georgia (USA) | Republic of Georgia (Country)
State of Hawaii | Interstate Highway System
State of Hawaii | Mauna Loa Magma Chamber
State of Hawaii | North American Continent
State of Hawaii | State of Ohio
State of Idaho | State of Montana
State of Illinois | City of Springfield
State of Kansas | State of Nebraska
State of Maine | State of Arizona
State of Nevada | State of California
State of New Mexico | White Sands National Park
State of New York | Central Park
State of New York | State of Pennsylvania
State of North Dakota | State of South Dakota
State of Ohio | City of Columbus
State of Oregon | State of Washington
State of Queensland | State of New South Wales
State of Rhode Island | Providence Plantations
State of South Dakota | Badlands National Park
State of Tasmania | Mainland Australia
State of Texas | City of Austin
State of Texas | Gulf of Mexico
State of Texas | The Alamo Mission Footprint
State of Utah | State of Idaho
State of Utah | State of New Mexico
State of Utah | Zion National Park
State of Victoria | State of South Australia
State of Washington | City of Seattle
Statue of Liberty | Eiffel Tower
Sub-Seabed Chunnel Transit Line | French Maritime Legal Zone
Suez Canal | Egypt
Suez Canal | Panama Canal
Suez Canal | Sinai Peninsula
Svalbard Global Seed Vault Entrance | Equator
Svalbard Global Seed Vault Entrance | Platåberget Mountain Surface
Svalbard Global Seed Vault Portal | Platåberget Mountain Exterior
Svalbard Global Seed Vault | Spitsbergen Permafrost Zone
Svalbard Treaty Zone | Spitsbergen Archipelago
Sweden | Norway
Swiss Canton of Schaffhausen | Busingen am Hochrhein
Switzerland | Campione d'Italia
The Alps | Italy
The Alps | Switzerland
The Gambia | Senegal
The Pentagon Grounds | State of Virginia
The Pentagon | The Kremlin
Tornado Alley | State of Oklahoma
Trafalgar Square Base | Nelson's Column
Trafalgar Square | Nelson's Column Base
Trans-Alaska Oil Pipeline | Abstract Arctic Circle Latitude
Trans-Alaska Pipeline | Arctic Circle
Trans-Alaska Pipeline | State of Alaska
Trans-Alaska Pipeline | Yukon River
Trans-Amazonian Highway | Amazon Rainforest
Trans-Amazonian Highway | Xingu River
Trans-Canada Highway Ribbon | Precambrian Canadian Shield
Trans-Canada Highway | Canadian Shield
Trans-Canada Highway | Province of Ontario
Trans-Sahara Highway | Sahara Desert
Trans-Siberian Railway | Ural Mountains
Trans-Siberian Steel Rail | Ob River Ice Flow
Tropic of Cancer | Mexico
Tropic of Cancer | Nile River
Tropic of Capricorn | Australia
Tropic of Capricorn | Great Dividing Range
Uluru | Australia
Union Pacific Railroad | State of Nevada
United Kingdom | City of London
United Kingdom | New Zealand
United Nations Headquarters | Borough of Manhattan
United Nations Headquarters | City of New York
University Endowment Lands | City of Vancouver
Vatican City Masonry Walls | City of Rome Streets
Vatican City | City of Rome
Vatican City | Holy See Jurisdiction
Vatican City | Holy See Territory
Vennbahn Railway Legal Footprint | German Sovereign Territory
Vennbahn Railway Right-of-Way | German Micro-Enclaves
Vesuvius Main Crater | Mount Vesuvius Volcanic Cone
Victoria Island Landmass | Unnamed Third-Order Arctic Island (69.793 N 108.241 W)
Victoria Island | Unnamed Arctic Sub-Island (69.793N 108.241W)
WGS84 Ellipsoid Center | Earth Center of Mass
White Sands National Park | State of New Mexico
Windsor Castle Grounds | River Thames
Yellowstone Caldera | Yellowstone National Park
Yellowstone National Park | State of Wyoming
Yellowstone National Park | Yellowstone Caldera
Yellowstone National Park | Yellowstone Supervolcano Caldera
Yosemite National Park | State of California
Zero Mile Marker (Washington DC) | Geographic Anchor of DC
Zion National Park | State of Utah
