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
Uzbekistan,Polygon,Sokh District,Polygon,"The central Asian republic completely bounds the foreign administrative region.",,topological,contains,"Uzbekistan fully encloses Sokh District.",Level 4
Australia,Polygon,Australian Capital Territory,Polygon,"The massive continental bounds completely serve as the absolute host for the inland capital territory governing the nation.",,topological,contains,"A massive continental polygon completely enclosing a specific administrative territory.",Level 5
Central Park,Polygon,State of New York,Polygon,"The massive green rectangle located in Manhattan is completely enclosed by the state lines.",,topological,within,"A park polygon completely enveloped by a state polygon.",Level 1
San Marino,Polygon,Republic of Italy,Polygon,"The ancient microstate is completely encircled by the sovereign territory, locking it permanently inland.",,topological,within,"A sovereign nation completely enveloped by an absolute geographic host.",Level 2
City of Hamtramck,Polygon,City of Detroit,Polygon,"The smaller, independent municipality is perfectly trapped by the massive automotive hub, isolating it from the rest of the county.",,topological,within,"A minor city completely surrounded by a major city donut hole.",Level 3
The Pentagon Grounds,Polygon,State of Virginia,Polygon,"The massive military structure and its sprawling parking infrastructure are completely encapsulated by the local county lines.",,topological,within,"A massive military footprint completely bounded by a state-level administrative polygon.",Level 4
Camp David,Polygon,Catoctin Mountain Park,Polygon,"The highly secure, restricted perimeter of the Presidential retreat is completely hidden and locked deep inside the forested boundaries of the public federal preserve in Maryland.",,topological,within,"A classified, restricted federal polygon fully bounded by a public federal park polygon.",Level 5
State of Colorado,Polygon,State of New Mexico,Polygon,"The southern edge of the State of Colorado meets the northern edge of the State of New Mexico forming a perfectly straight latitudinal line.",,topological,touches,"Two state administrative polygons meeting at a shared edge with no internal intersection.",Level 1
State of Alaska,Polygon,Yukon Territory,Polygon,"The eastern frontier of the US state of Alaska meets the western edge of the Canadian territory of Yukon along a perfectly straight longitudinal line.",,topological,touches,"The administrative polygon of the US exclave meets the Canadian territory along a shared linear boundary without interior intersection.",Level 2
State of Utah,Polygon,State of New Mexico,Polygon,"At the Four Corners Monument, the southeastern tip of Utah and the northwestern corner of New Mexico converge at exactly one singular mathematical coordinate.",,topological,touches,"Two state administrative polygons that meet at exactly one shared vertex without any shared interior area or linear edge.",Level 3
Bir Tawil,Polygon,Hala'ib Triangle,Polygon,"The universally unclaimed territory of Bir Tawil and the fiercely disputed Hala'ib Triangle converge exclusively at a single zero-dimensional vertex located at the intersection of the 22nd parallel north and the 34th meridian east.",,topological,touches,"Two distinct administrative polygons that meet at exactly one mathematical coordinate without sharing any linear border.",Level 4
Vatican City Masonry Walls,Polygon,City of Rome Streets,Polygon,"The ancient defensive fortifications stand right at the municipal edge of the Italian capital's modern thoroughfares, creating a hard stop to Italian jurisdiction.",,topological,touches,"A sovereign microstate wall abutting a city municipal polygon.",Level 5
Trans-Siberian Railway,LineString,Ural Mountains,Polygon,"The famous Trans-Siberian Railway makes its way completely through the Ural Mountains to link Europe to Asia.",,topological,crosses,"A railway line passes through both the exterior and interior of a mountain range polygon.",Level 1
River Thames,LineString,City of London,Polygon,"The historic waterway snakes right through the center of the massive British metropolis.",,topological,crosses,"A river line traversing a city polygon.",Level 2
Channel Tunnel,LineString,Prime Meridian,LineString,"The subterranean transit line intersects the global longitude line deep beneath the ocean floor.",,topological,crosses,"A subterranean transit line intersecting a global latitudinal line.",Level 3
Alaska Highway,LineString,Yukon,Polygon,"The historic northern supply road threads a path through the vast Canadian territory.",,topological,crosses,"Alaska Highway cuts through Yukon.",Level 4
South Pole Station Flight Path,LineString,Convergence of 360 Meridians,MultiLineString,"The ascending aircraft banks in a tight arc, sequentially slicing through every single invisible global longitude line radiating from the planetary axis.",,topological,crosses,"A local transit LineString intersecting a planet-spanning MultiLineString network.",Level 5
New Zealand,Polygon,Greenland,Polygon,"New Zealand sits near the Antarctic, completely separated from the icy landmass of Greenland near the Arctic.",,topological,disjoint,"Two island polygons at opposite ends of the earth.",Level 1
Iceland,Polygon,Antarctica,Polygon,"The volcanic island in the North Atlantic sits at the extreme opposite end of the globe from the frozen southern continent.",,topological,disjoint,"Two landmass polygons with extreme global separation.",Level 2
Cabinda Province,Polygon,Mainland Angola,Polygon,"The oil-rich territory is totally severed from the national mainland by a narrow strip of the Democratic Republic of the Congo.",,topological,disjoint,"An administrative polygon physically isolated from its parent nation.",Level 3
Channel Islands,MultiPolygon,United Kingdom,Polygon,"Despite being British Crown Dependencies, the Channel Islands are situated off the coast of Normandy, sharing no spatial coordinates with the sovereign footprint of the United Kingdom.",,topological,disjoint,"Testing the difference between political affiliation and spatial geometry; the dependent MultiPolygon is totally separated from the sovereign Polygon.",Level 4
Bosphorus Bridge Roadway,LineString,Bosphorus Strait,Polygon,"Massive suspension pylons keep the transcontinental roadway completely elevated above the turbulent shipping lanes connecting the two seas.",,topological,disjoint,"An elevated bridge line strictly separated from a maritime strait polygon.",Level 5
Gobi Desert,Polygon,Mongolia,Polygon,"The sweeping sands cover the southern half of Mongolia and reach down into northern China.",,topological,overlaps,"A desert polygon partially intersecting a national polygon.",Level 1
Falkland Islands EEZ,Polygon,Argentine Sea,Polygon,"The projected maritime boundaries clash and mutually share a vast swath of the turbulent South Atlantic ocean.",,topological,overlaps,"Two abstract legal maritime polygons intersecting across a vast ocean.",Level 2
Hala'ib Triangle,Polygon,Egyptian Territorial Claim,Polygon,"The contested boundary shares a significant contested geographic zone with the southern sovereign claims made by Cairo.",,topological,overlaps,"A contested geographic polygon partially intersecting with a national claim polygon.",Level 3
Argentine Antarctica Claim,Polygon,British Antarctic Territory,Polygon,"Under the frozen geopolitical realities of the South Pole, the pie-shaped sovereign territorial claim of Argentina massively encroaches upon the wedge claimed by the United Kingdom.",,topological,overlaps,"Two sovereign legal polygons projecting over the exact same physical landmass, resulting in a massive zone of shared coordinates.",Level 4
Falkland Islands EEZ,Polygon,Argentine Sea,Polygon,"The 200-nautical-mile economic projection extending outward from the British islands deeply intrudes into the sovereign maritime territory recognized by Buenos Aires.",,topological,overlaps,"Two abstract legal maritime polygons that clash and intersect mutually in a vast swath of ocean.",Level 5
City of Jacksonville,Polygon,Duval County,Polygon,"The consolidated government covers the entire county area, creating a perfectly mirrored geographic map.",,topological,equals,"A city and county that share a single territorial perimeter.",Level 1
City of Richmond,Polygon,Independent City of Richmond,Polygon,"The Virginian capital projects the exact identical footprint as its unattached municipal zone.",,topological,equals,"City of Richmond has the same area as Independent City of Richmond.",Level 2
Principality of Monaco,Polygon,Municipality of Monaco,Polygon,"The sovereign European microstate and its single integrated municipality are entirely coextensive.",,topological,equals,"A national polygon and a city polygon that are perfectly identical.",Level 3
Independent City of Baltimore,Polygon,City of Baltimore Municipal Limits,Polygon,"Operating completely outside the county system, the local municipality maps identically to its highest-level administrative tier.",,topological,equals,"Two distinct legal definitions mapping perfectly to the same urban footprint.",Level 4
Independent City of St. Louis,Polygon,St. Louis County Municipal Boundary,Polygon,"Because it operates entirely independently, the municipal limits represent the exact same geographical footprint as its county-level equivalent.",,topological,equals,"Two distinct levels of civic administration mapping to a perfectly identical topological footprint.",Level 5

## Entity pairs already used — do not repeat these

Abstract Geodetic Equator | Physical Amazon River Basin
Abstract Geodetic Prime Meridian | Physical River Thames Current
Abstract Tropic of Cancer | Physical Nile River Flow
Abstract Tropic of Capricorn | Physical Great Dividing Range
Abyei Administrative Area | Republic of Sudan
Abyei Area | Republic of South Sudan
Abyei Area | Republic of Sudan
Abyei Demilitarized Box | Republic of Sudan
Africa | Cairo Governorate
Alaska Highway | Yukon
Alps | Himalayas
Alps | Switzerland
Amazon Basin | Andes Mountains Foothills
Amazon Basin | Bolivia
Amazon Basin | Congo Basin
Amazon Basin | Peru
Amazon Rainforest | Brazil
Amazon Rainforest | Colombia
Amazon River | Brazil
Andes Mountains Foothills | Amazon Basin
Andes Mountains | Chile
Andes | Peru
Angeles National Forest | Los Angeles County
Antelope Island | Great Salt Lake
Appalachian Footpath | Mason-Dixon Survey Line
Appalachian Mountains | State of Pennsylvania
Appalachian Mountains | State of Tennessee
Appalachian Mountains | State of Virginia
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
Asia | City of Kyoto
Atacama Desert | Andes Mountains
Atlantic Ocean | Ascension Island
Australia | Australian Capital Territory
Australia | Canada
Australia | New Zealand
Australia | Uluru
Australian Capital Territory | New South Wales
Australian Continent | Lake Eyre Basin
Ayers Rock | Uluru
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
Brahmaputra River | Himalayas
Brazil | Argentina
Brooklyn Bridge Roadway | East River Surface
Brooklyn Bridge | East River
Brooklyn Heights | City of New York
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
Canada | Brazil
Canada | Province of Saskatchewan
Canada | United States
Capitol Hill | City of Seattle
Catacombs of Paris Subterranean Network | City of Paris
Central Park Reservoir | New York City Municipal Grid
Central Park | State of New York
Ceuta | Mainland Spain
Challenger Deep | Mariana Trench Geomorphic Bounds
Channel Islands | United Kingdom
Channel Islands | United Kingdom Mainland
Channel Tunnel | English Channel
Channel Tunnel | Prime Meridian
City of Alamo Heights | City of San Antonio
City of Alexandria | Independent City of Alexandria
City of Anaconda | Deer Lodge County
City of Anchorage | Anchorage Municipality
City of Athens | Clarke County
City of Augusta | Richmond County
City of Austin | State of Texas
City of Baltimore | Independent City of Baltimore
City of Barcelona | Spain
City of Bellaire | City of Houston
City of Berlin | Italy
City of Berlin | State of Berlin
City of Beverly Hills | City of Los Angeles
City of Boston | City of Cambridge
City of Boston | City of Providence
City of Broomfield | Broomfield County
City of Butte | Silver Bow County
City of Carson City | Consolidated Municipality of Carson City
City of Chicago | City of Houston
City of Chicago | City of Milwaukee
City of Chicago | State of Illinois
City of Columbus | State of Ohio
City of Dallas | City of Fort Worth
City of Denver | Denver County
City of Denver | State of Colorado
City of Detroit | City of Hamtramck
City of Detroit | City of Windsor
City of El Paso | Ciudad Juarez
City of Geneva | Canton of Geneva
City of Hampton | Independent City of Hampton
City of Hamtramck | City of Detroit
City of Honolulu | Mainland United States
City of Indianapolis | State of Indiana
City of Jacksonville | Duval County
City of Juneau | Juneau Borough
City of Kyoto | Japan
City of Las Vegas | City of Henderson
City of Lexington | Fayette County
City of London Corporation | Greater London Authority
City of London | France
City of London | The Square Mile
City of London | United Kingdom
City of Los Angeles | City of Beverly Hills
City of Lyon | France
City of Madrid | Spain
City of Miami | City of Fort Lauderdale
City of Miami | City of Seattle
City of Miami | State of Florida
City of Milan | Italy
City of Minneapolis | City of St. Paul
City of Munich | Germany
City of Nashville | Davidson County
City of New York | City of Jersey City
City of Newport News | Independent City of Newport News
City of Norfolk | Independent City of Norfolk
City of Norwood | City of Cincinnati
City of Orlando | State of Florida
City of Paris | Catacombs of Paris Subterranean Network
City of Paris | City of Tokyo
City of Paris | Department of Paris
City of Paris | France
City of Philadelphia | Philadelphia County
City of Phoenix | City of Scottsdale
City of Phoenix | State of Arizona
City of Richmond | Independent City of Richmond
City of Roanoke | Independent City of Roanoke
City of Rome | Italy
City of Rome | Vatican City
City of Sacramento | State of California
City of San Diego | City of Tijuana
City of San Diego | Mexico
City of San Fernando | City of Los Angeles
City of San Francisco | City and County of San Francisco
City of San Francisco | City of Oakland
City of San Francisco | San Francisco County
City of San Jose | City of Santa Clara
City of San Jose | State of California
City of Seattle | City of Bellevue
City of Seattle | State of Washington
City of Sitka | Sitka Borough
City of Springfield | State of Illinois
City of St. Louis | Independent City of St. Louis
City of Sydney | City of Cape Town
City of Sydney | New Zealand
City of Tokyo | South Korea
City of Toronto | Mexico
City of Vienna | State of Vienna
City of Washington | District of Columbia
City of West Hollywood | City of Los Angeles
Clark County | State of Nevada
Coconino National Forest | Yavapai County
Colorado River | Grand Canyon National Park
Colorado River | State of Arizona
Colorado River | State of Utah
Commonwealth of Australia | Ashmore and Cartier Islands
Congo Basin | Democratic Republic of the Congo
Constantinople Walled City (1453) | Fatih District of Istanbul
Cook County | City of Evanston
Dahala Khagrabari | Cooch Behar District
Dahlak Archipelago | Red Sea
Danube River | Hungary
Death Valley Below-Sea-Level Basin | State of California
Demilitarized Zone (DMZ) | 1953 Ceasefire Buffer Zone
Demilitarized Zone (DMZ) | 1953 Korean Armistice Ceasefire Buffer
Denali Wilderness Area | State of Alaska
Denmark | City of Copenhagen
Diego Garcia | Indian Ocean
Diomede Islands Maritime Border | International Date Line
Dutch Baarle-Nassau Enclave N8 | Belgian Baarle-Hertog Enclave H22
Easter Island Landmass | Republic of Chile Maritime Claims
Egypt | South Africa
Egypt | Sudan
Eiffel Tower Spire Tip | Highest Architectural Node of Paris
Eiffel Tower | City of Paris
English Channel | French Coast
English Channel | La Manche
Equator | Amazon River
Equator | Lake Victoria
Europe | City of Paris
Eurostar Railway | France
Eurotunnel Subterranean Tube | English Channel Maritime Zone
Eurotunnel Subterranean Tube | English Channel Surface Waters
Eurotunnel Terminal Boundary | French Maritime Customs Zone
Eurotunnel Terminal Entrance | French Customs Zone
Falkland Islands EEZ | Argentine Sea
Falkland Islands | Mainland Argentina
Falkland Islands | United Kingdom
Falkland Islands | United Kingdom Mainland
Federal Republic of Germany | Germany
Finland | City of Helsinki
Four Corners Monument Plaque | State of Arizona Boundary
Four Corners Monument | State of Utah
France | City of Paris
France | Llivia
France | Spain
French Guiana | Metropolitan France
French Quarter | City of New Orleans
French Republic | France
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
Grand Island | Lake Superior
Grand Teton National Park | State of Wyoming
Great Barrier Reef | Coral Sea
Great Basin Desert | State of Nevada
Great Plains | State of Kansas
Great Pyramid Perimeter | King's Chamber Subterranean Volume
Great Rift Valley | Ethiopia
Great Rift Valley | Kenya
Great Rift Valley | Republic of Kenya
Great Victoria Desert | South Australia
Great Victoria Desert | Western Australia
Great Wall of China | Japan
Greater London Authority | City of London Corporation
Greenland | Antarctica
Guantanamo Bay Naval Base | United States Mainland
Gulf Stream | North Atlantic
Hala'ib Triangle | Egyptian Territorial Claim
Himalaya Mountains | Nepal
Himalayas | Tibetan Plateau
Historic Route 66 Asphalt | Mississippi River Hydrological Flow
Historical Mount McKinley Summit | Modern Denali Summit
Historical Ottoman Empire Footprint (1683) | Modern European Union Territory
Historical Roman Empire (AD 117) | Modern European Union
Hollywood | City of Los Angeles
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
Indian Ocean | Christmas Island
Indian Ocean | Diego Garcia
Inner Temple Legal Precinct | City of London Corporation
International Date Line (Conceptual) | 180 Degrees Longitude
International Space Station Docking Ring | SpaceX Crew Dragon Adapter
Interstate 10 | State of Texas
Interstate 15 | Mojave Desert
Interstate 35 | City of Austin
Interstate 40 | State of Arizona
Interstate 40 | State of New Mexico
Interstate 405 | City of Los Angeles
Interstate 5 | City of Seattle
Interstate 5 | State of California
Interstate 70 | State of Colorado
Interstate 70 | State of Utah
Interstate 80 | City of Salt Lake City
Interstate 80 | Continental Divide
Interstate 80 | State of Nevada
Interstate 90 | State of Idaho
Interstate 90 | State of South Dakota
Interstate 95 Asphalt | Mason-Dixon Historical Boundary
Interstate 95 | City of Philadelphia
Interstate 95 | Mason-Dixon Line
Isla del Sol | Lake Titicaca
Island of Malta | Mediterranean Sea
Island of Oahu | Pacific Ocean
Isle of Man | Northern Ireland
Isle of Wight | Mainland England
Italian Republic | Italy
Italy | City of Rome
Italy | San Marino
Japan | Brazil
Japan | United Kingdom
Kalahari Desert | Botswana
Kalahari Desert | Mojave Desert
Kalahari Desert | Republic of Namibia
Kaliningrad Oblast | Mainland Russia
Kashmir Region | Indian Territorial Claim
Kern County | State of California
King County | City of Redmond
King's Chamber Subterranean Volume | Great Pyramid Perimeter
Kingdom of Lesotho | Indian Ocean Maritime Boundary
Kingdom of Lesotho | South African Borders
Kingdom of Saudi Arabia | Rub' al Khali Desert
Kingdom of Spain | Spain
Kurdish Inhabited Region | Republic of Turkey
Kurdish Inhabited Region | Turkey
Kuril Islands | Japanese Territorial Claim
Kyrgyzstan | Vorukh
Lake Baikal | Caspian Sea
Lake Eyre Basin | Australian Continent
Lake Nicaragua | Ometepe Island
Lake Superior | Isle Royale
Lake Victoria | Lake Superior
Lake Victoria | Ssese Islands
Lake Vostok Subglacial Volume | Antarctic Ice Sheet
Lincoln Park | City of Chicago
Little Diomede Island | Big Diomede Island
Loch Ness Water Volume | Great Glen Fault
Lombard Street | Broadway (New York)
London Underground Central Line Subterranean Track | River Thames Surface Water
Mackinac Island | Lake Huron
Madagascar | Indian Ocean
Madagascar | Mainland Africa
Madagascar | Sri Lanka
Main Crater Lake | Vulcan Point Island
Manitoulin Island | Lake Huron
Manitoulin Island | Lake Manitou
Mariana Trench Geomorphic Lip | Abyssal Plain
Mariana Trench | Pacific Ocean
Maricopa County | City of Tempe
Mauna Loa Magma Chamber | State of Hawaii
Mediterranean Basin | European Union
Mediterranean Deep Brine Pool | Mediterranean Sea Volume
Mediterranean Sea | Corsica
Mediterranean Sea | Island of Cyprus
Mediterranean Sea | Island of Malta
Mercer Island | Lake Washington
Miami-Dade County | State of Florida
Mid-Atlantic Ridge | Iceland
Mississippi River | State of Arkansas
Mississippi River | State of Louisiana
Mojave Desert | State of California
Mojave Desert | State of Nevada
Mongolia | China
Mount Everest Summit | Nepal-China Border
Mount Everest | Sagarmatha
Mount Fuji | Japan
Mount Hood National Forest | Clackamas County
Mount Kilimanjaro | Mount Everest
Mount McKinley Summit | Denali Summit
Mount Rainier Glacial Cap | Mount Rainier National Park
Mount Rainier National Park | Mount Rainier Glacial Cap
Mount Rainier | Tahoma
Mount Titano Summit | Republic of San Marino
Multnomah County | State of Oregon
Municipality of Llivia | French Republic
Nakhchivan Autonomous Republic | Mainland Azerbaijan
Navajo Nation Reservation | Hopi Reservation
Navajo Nation | Hopi Reservation
Navajo Nation | State of Arizona
Navajo Nation | State of New Mexico
Navajo Nation | State of Utah
Navigational International Date Line | Chukchi Sea Legal Boundary
New Delhi District | Republic of India
New York City Municipal Grid | Central Park Reservoir
New Zealand | Greenland
Niagara Falls Horseshoe | Canada-US Border
Niagara Falls | Canada-US Border
Nile River | Amazon River
Nile River | Republic of Sudan
Nile River | Yellow River
Nine-Dash Line Claim | Philippine Exclusive Economic Zone
Nine-Dash Line Claim | Philippine Exclusive Economic Zone (EEZ)
North America | State of Kansas
North Pole | 90 Degrees North Latitude
Northwest Angle | Contiguous United States
Norway | City of Oslo
Null Island | WGS84 Origin Coordinate
O'Hare International Airport Bounds | City of Chicago
Omani Exclave of Madha | Mainland Oman
Omani Exclave of Madha | UAE Enclave of Nahwa
Ometepe Island | Lake Nicaragua
Orange County | City of Irvine
PATH Train Tunnels | Hudson River
PATH Train Tunnels | Hudson River Water Volume
PATH Train Tunnels | New York-New Jersey State Line
Pacific Ocean | Island of Oahu
Pacific Ocean | Mariana Trench
Pacific Ocean | Mediterranean Sea
Pacific Ocean | Pitcairn Island
Pacific Ocean | State of Hawaii
Pampas Grassland | Argentina
Pan-American Highway | State of Sonora
Panama Canal | Isthmus of Panama
Panama Canal | Panamá Province
Panmunjom T2 Conference Table | Military Demarcation Line
Patagonian Desert | Argentina
Pelee Island | Lake Erie
Persian Gulf | Arabian Gulf
Pike National Forest | Douglas County
Pine Ridge Indian Reservation | State of South Dakota
Point Nemo Mathematical Isolation Zone | Pacific Ocean Maritime Bounds
Point Nemo | Ducie Island
Point Roberts Exclave | 49th Parallel North
Poland | City of Warsaw
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
Province of Alberta | Province of Saskatchewan
Pyrenees Mountains | France
Red Sea | Dahlak Archipelago
Republic of Armenia | Tigranashen
Republic of Austria | Federal Republic of Germany
Republic of Azerbaijan | Artsvashen
Republic of Botswana | Republic of Namibia
Republic of Botswana | Republic of Zambia
Republic of Chile | Easter Island Landmass
Republic of Chile | Republic of Argentina
Republic of Croatia | Bosnia and Herzegovina
Republic of Cyprus | UN Buffer Zone
Republic of Cyprus | UN Buffer Zone in Cyprus
Republic of India | New Delhi
Republic of Ireland | Cyprus
Republic of Ireland | Northern Ireland
Republic of Italy | San Marino
Republic of Italy | Sovereign Military Order of Malta Magistral Villa
Republic of Senegal | Republic of Guinea-Bissau
Ring of Fire | Pacific Ocean
Ring of Fire | Pacific Plate
Rio Grande | State of New Mexico
River Thames | City of London
River Thames | Mississippi River
Rocky Mountains | Andes Mountains
Rocky Mountains | Appalachian Mountains
Rocky Mountains | Canada
Rocky Mountains | Great Plains
Rocky Mountains | State of Colorado
Rocky Mountains | The Alps
Rosebud Indian Reservation | State of South Dakota
Route 66 | Colorado River
Route 66 | Continental Divide
Route 66 | Mississippi River
Route 66 | State of Florida
Royal Botanic Garden Sydney | City of Sydney
Rub' al Khali Desert | Kingdom of Saudi Arabia
Rust Belt | American Midwest
Rust Belt | State of Ohio
Sahara Desert | Algeria
Sahara Desert | Amazon Rainforest
Sahara Desert | Egypt
Sahara Desert | Gobi Desert
Sahara Desert | Libya
Sahara Desert | Republic of Mali
Sahara Desert | Sahel
Sahara Desert | Sahel Region
San Andreas Fault | State of California
San Marino | Italian Peninsula
San Marino | Republic of Italy
Santa Clara County | City of Cupertino
Scandinavia | Norway
Scotland | England
Seikan Tunnel | Tsugaru Strait
Seikan Tunnel | Tsugaru Strait Water Volume
Senegal | The Gambia
Shohimardon | Kyrgyzstan
Sonoran Desert | Mexico
Sonoran Desert | State of Arizona
South Africa | Kingdom of Lesotho
South Africa | Russia
South America | State of Amazonas
South Pole Station Flight Path | Convergence of 360 Meridians
Southern Ocean | Bouvet Island
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
State of Alaska | Province of British Columbia
State of Alaska | State of Hawaii
State of Alaska | Yukon Territory
State of Arizona | City of Phoenix
State of Arizona | Grand Canyon National Park
State of California | City of Los Angeles
State of California | Death Valley Below-Sea-Level Basin
State of California | Fresno County
State of California | Pacific Ocean
State of California | Yosemite National Park
State of Colorado | City of Denver
State of Colorado | State of New Mexico
State of Florida | Orange County
State of Florida | State of Washington
State of Georgia (USA) | Republic of Georgia (Country)
State of Georgia | Fulton County
State of Hawaii | Contiguous United States
State of Hawaii | Interstate Highway System
State of Hawaii | Mauna Loa Magma Chamber
State of Hawaii | North American Continent
State of Hawaii | State of Ohio
State of Idaho | State of Montana
State of Illinois | City of Springfield
State of Kansas | State of Nebraska
State of Libya | Egypt
State of Maine | State of Arizona
State of Maine | State of Florida
State of Nevada | State of California
State of Nevada | State of Utah
State of New Mexico | State of Oklahoma
State of New Mexico | White Sands National Park
State of New York | Central Park
State of New York | State of New Jersey
State of New York | State of Pennsylvania
State of North Dakota | State of South Dakota
State of Ohio | City of Columbus
State of Ohio | Franklin County
State of Oregon | State of Idaho
State of Oregon | State of Washington
State of Queensland | State of New South Wales
State of Rhode Island | Providence Plantations
State of South Dakota | Badlands National Park
State of Tasmania | Mainland Australia
State of Texas | Bexar County
State of Texas | City of Austin
State of Texas | Gulf of Mexico
State of Texas | The Alamo Mission Footprint
State of Utah | State of Idaho
State of Utah | State of New Mexico
State of Utah | Zion National Park
State of Victoria | State of South Australia
State of Washington | City of Seattle
State of Washington | State of Maine
State of Wisconsin | State of Illinois
State of Wyoming | State of Montana
Statue of Liberty | Eiffel Tower
Sub-Seabed Chunnel Transit Line | French Maritime Legal Zone
Suez Canal | Egypt
Suez Canal | Ismailia Governorate
Suez Canal | Panama Canal
Suez Canal | Sinai Peninsula
Svalbard Global Seed Vault Entrance | Equator
Svalbard Global Seed Vault Entrance | Platåberget Mountain Surface
Svalbard Global Seed Vault Portal | Platåberget Mountain Exterior
Svalbard Global Seed Vault | Spitsbergen Permafrost Zone
Svalbard Treaty Zone | Spitsbergen Archipelago
Svalbard | Mainland Norway
Sweden | City of Stockholm
Sweden | Norway
Swiss Canton of Schaffhausen | Busingen am Hochrhein
Switzerland | Campione d'Italia
The Alps | Italy
The Alps | Switzerland
The Gambia | Senegal
The Pentagon Grounds | State of Virginia
The Pentagon | The Kremlin
Tohono O'odham Nation | State of Arizona
Tonto National Forest | Maricopa County
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
Trans-Canada Highway | Province of Manitoba
Trans-Canada Highway | Province of Ontario
Trans-Sahara Highway | Sahara Desert
Trans-Siberian Railway | Russian Federation
Trans-Siberian Railway | Ural Mountains
Trans-Siberian Steel Rail | Ob River Ice Flow
Travis County | State of Texas
Tropic of Cancer | Mexico
Tropic of Cancer | Nile River
Tropic of Capricorn | Australia
Tropic of Capricorn | Great Dividing Range
Uluru | Australia
Union Pacific Railroad | State of Nevada
United Kingdom of Great Britain and Northern Ireland | United Kingdom
United Kingdom | City of London
United Kingdom | New Zealand
United Nations Headquarters | Borough of Manhattan
United Nations Headquarters | City of New York
University Endowment Lands | City of Vancouver
Ural Mountains | West Siberian Plain
Uzbekistan | Sokh District
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
Yosemite National Park | Tuolumne County
Zero Mile Marker (Washington DC) | Geographic Anchor of DC
Zion National Park | State of Utah
