# Replacement request (topological)

I previously asked you for a batch of topological spatial-relation rows.
Most were good. **20 rows must be replaced**, and I need
**21 new rows** to restore the balance.

Keep everything else exactly as it was — do not resend the good rows.

## Rows to replace, and why

- **line 42** — `City of Piedmont` → `City of Oakland` (within, Level 4)
  - the description contains the word "within", which gives the answer away
- **line 52** — `Portugal` → `Spain` (touches, Level 1)
  - this entity pair already exists in the corpus
- **line 62** — `City of Minneapolis` → `City of St. Paul` (touches, Level 3)
  - this entity pair already exists in the corpus
- **line 63** — `City of Dallas` → `City of Fort Worth` (touches, Level 3)
  - this entity pair already exists in the corpus
- **line 64** — `City of Seattle` → `City of Bellevue` (touches, Level 3)
  - this entity pair already exists in the corpus
- **line 76** — `Great Barrier Reef` → `Coral Sea` (touches, Level 5)
  - this entity pair already exists in the corpus
- **line 100** — `Mariana Trench` → `Pacific Ocean` (crosses, Level 5)
  - this entity pair already exists in the corpus
- **line 103** — `Iceland` → `Madagascar` (disjoint, Level 1)
  - this entity pair already exists in the corpus
- **line 117** — `State of Alaska` → `Contiguous United States` (disjoint, Level 4)
  - this entity pair already exists in the corpus
- **line 118** — `Kaliningrad Oblast` → `Mainland Russia` (disjoint, Level 4)
  - this entity pair already exists in the corpus
- **line 119** — `French Guiana` → `Metropolitan France` (disjoint, Level 4)
  - this entity pair already exists in the corpus
- **line 120** — `Cabinda Province` → `Mainland Angola` (disjoint, Level 4)
  - this entity pair already exists in the corpus
- **line 121** — `Nakhchivan Autonomous Republic` → `Mainland Azerbaijan` (disjoint, Level 4)
  - this entity pair already exists in the corpus
- **line 130** — `Yellowstone National Park` → `State of Wyoming` (overlaps, Level 1)
  - this entity pair already exists in the corpus
- **line 132** — `Rocky Mountains` → `State of Colorado` (overlaps, Level 2)
  - this entity pair already exists in the corpus
- **line 134** — `Mojave Desert` → `State of California` (overlaps, Level 2)
  - this entity pair already exists in the corpus
- **line 149** — `Gobi Desert` → `Mongolia` (overlaps, Level 5)
  - this entity pair already exists in the corpus
- **line 150** — `Kalahari Desert` → `Botswana` (overlaps, Level 5)
  - this entity pair already exists in the corpus
- **line 182** — `City of Seattle` → `State of Washington` (within, Level 6)
  - this entity pair already exists in the corpus
- **line 186** — `City of Phoenix` → `State of Arizona` (within, Level 6)
  - this entity pair already exists in the corpus

## What to send back

Exactly 21 rows, distributed like this:

- `within` at **Level 4** — 1 row
- `within` at **Level 6** — 2 rows
- `touches` at **Level 1** — 1 row
- `touches` at **Level 3** — 3 rows
- `touches` at **Level 5** — 1 row
- `crosses` at **Level 5** — 1 row
- `disjoint` at **Level 1** — 1 row
- `disjoint` at **Level 4** — 5 rows
- `overlaps` at **Level 1** — 1 row
- `overlaps` at **Level 2** — 2 rows
- `overlaps` at **Level 5** — 3 rows

## Rules for the replacements

1. Same label and same ambiguity level as the row being replaced —
   the grid must stay balanced.
2. A DIFFERENT pair of places. Never reuse a pair from the list at the
   bottom, and never reuse one already in your previous batch.
3. Never send a pair together with its mirror. If you write
   "A contains B", do not also write "B within A" — those two rows
   become each other's answer key.
4. The description must NOT contain the label word or an obvious
   synonym. Say "sits entirely inside", never "is within".
5. Every place must be findable in OpenStreetMap: use full official
   names ("City of Seattle", "State of Colorado") or named natural
   features. No generic descriptions, abstractions, or interior rooms.
6. Every row must be factually TRUE. Verify before writing.

7. For Level 6 rows: the description must state BOTH links through
   the intermediate place named in `via_entity`, and must mention
   all three places. The intermediate must be a real third place,
   never a synonym of an endpoint.

## Output format

Return ONLY CSV rows — no header, no prose, no markdown fences.
Same column order as before:

source_entity,source_geometry,target_entity,target_geometry,corpus,via_entity,relation_type,relation_label,explanation,ambiguity_level

## Entity pairs already used — never reuse any of these

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

## Also do not reuse any pair from your previous batch

Poland | City of Warsaw
Sweden | City of Stockholm
Norway | City of Oslo
Finland | City of Helsinki
Denmark | City of Copenhagen
State of Texas | Bexar County
State of California | Fresno County
State of Ohio | Franklin County
State of Georgia | Fulton County
State of Florida | Orange County
King County | City of Redmond
Santa Clara County | City of Cupertino
Orange County | City of Irvine
Maricopa County | City of Tempe
Cook County | City of Evanston
France | Llivia
Republic of Azerbaijan | Artsvashen
Republic of Armenia | Tigranashen
Kyrgyzstan | Vorukh
Uzbekistan | Sokh District
Atlantic Ocean | Ascension Island
Pacific Ocean | Pitcairn Island
Indian Ocean | Christmas Island
Southern Ocean | Bouvet Island
Mediterranean Sea | Corsica
City of Lyon | France
City of Milan | Italy
City of Barcelona | Spain
City of Munich | Germany
City of Kyoto | Japan
Travis County | State of Texas
Kern County | State of California
Miami-Dade County | State of Florida
Clark County | State of Nevada
Multnomah County | State of Oregon
Brooklyn Heights | City of New York
Hollywood | City of Los Angeles
Lincoln Park | City of Chicago
Capitol Hill | City of Seattle
French Quarter | City of New Orleans
City of Piedmont | City of Oakland
City of San Fernando | City of Los Angeles
City of Bellaire | City of Houston
City of West Hollywood | City of Los Angeles
City of Alamo Heights | City of San Antonio
Antelope Island | Great Salt Lake
Mackinac Island | Lake Huron
Mercer Island | Lake Washington
Grand Island | Lake Superior
Pelee Island | Lake Erie
Portugal | Spain
State of Nevada | State of Utah
Scotland | England
Republic of Chile | Republic of Argentina
State of Libya | Egypt
State of Wyoming | State of Montana
Republic of Austria | Federal Republic of Germany
Province of Alberta | Province of Saskatchewan
State of Wisconsin | State of Illinois
State of Oregon | State of Idaho
City of Minneapolis | City of St. Paul
City of Dallas | City of Fort Worth
City of Seattle | City of Bellevue
City of San Diego | City of Tijuana
City of Boston | City of Cambridge
Republic of Botswana | Republic of Namibia
State of New Mexico | State of Oklahoma
Republic of Croatia | Bosnia and Herzegovina
Republic of Senegal | Republic of Guinea-Bissau
State of Alaska | Province of British Columbia
Himalayas | Tibetan Plateau
Sahara Desert | Sahel
Rocky Mountains | Great Plains
Atacama Desert | Andes Mountains
Great Barrier Reef | Coral Sea
Interstate 80 | State of Nevada
Nile River | Republic of Sudan
Interstate 5 | State of California
Mississippi River | State of Arkansas
Trans-Siberian Railway | Russian Federation
Interstate 90 | State of Idaho
Interstate 10 | State of Texas
Interstate 70 | State of Utah
Interstate 40 | State of New Mexico
Colorado River | State of Arizona
Interstate 405 | City of Los Angeles
Interstate 95 | City of Philadelphia
Interstate 35 | City of Austin
Interstate 5 | City of Seattle
Interstate 80 | City of Salt Lake City
Pan-American Highway | State of Sonora
Trans-Canada Highway | Province of Manitoba
Alaska Highway | Yukon
Suez Canal | Ismailia Governorate
Panama Canal | Panamá Province
San Andreas Fault | State of California
Mid-Atlantic Ridge | Iceland
Great Rift Valley | Republic of Kenya
Mariana Trench | Pacific Ocean
Ring of Fire | Pacific Ocean
Australia | New Zealand
Iceland | Madagascar
State of Maine | State of Florida
Egypt | South Africa
Japan | United Kingdom
State of Alaska | State of Hawaii
Republic of Ireland | Cyprus
State of Washington | State of Maine
Madagascar | Sri Lanka
Greenland | Antarctica
City of San Francisco | City of Oakland
City of New York | City of Jersey City
City of Miami | City of Fort Lauderdale
City of Chicago | City of Milwaukee
City of Boston | City of Providence
State of Alaska | Contiguous United States
Kaliningrad Oblast | Mainland Russia
French Guiana | Metropolitan France
Cabinda Province | Mainland Angola
Nakhchivan Autonomous Republic | Mainland Azerbaijan
Sahara Desert | Gobi Desert
Rocky Mountains | Appalachian Mountains
Alps | Himalayas
Amazon Basin | Congo Basin
Kalahari Desert | Mojave Desert
Sahara Desert | Algeria
Alps | Switzerland
Andes | Peru
Yellowstone National Park | State of Wyoming
Yosemite National Park | Tuolumne County
Rocky Mountains | State of Colorado
Appalachian Mountains | State of Virginia
Mojave Desert | State of California
Sonoran Desert | State of Arizona
Great Basin Desert | State of Nevada
Angeles National Forest | Los Angeles County
Mount Hood National Forest | Clackamas County
Pike National Forest | Douglas County
Tonto National Forest | Maricopa County
Coconino National Forest | Yavapai County
Navajo Nation | State of Utah
Navajo Nation | State of New Mexico
Tohono O'odham Nation | State of Arizona
Pine Ridge Indian Reservation | State of South Dakota
Rosebud Indian Reservation | State of South Dakota
Amazon Rainforest | Peru
Sahara Desert | Libya
Gobi Desert | Mongolia
Kalahari Desert | Botswana
Patagonian Desert | Argentina
City of San Francisco | City and County of San Francisco
City of Paris | Department of Paris
City of Geneva | Canton of Geneva
City of Berlin | State of Berlin
City of Vienna | State of Vienna
City of Roanoke | Independent City of Roanoke
City of Norfolk | Independent City of Norfolk
City of Richmond | Independent City of Richmond
City of Hampton | Independent City of Hampton
City of Newport News | Independent City of Newport News
City of Anaconda | Deer Lodge County
City of Butte | Silver Bow County
City of Lexington | Fayette County
City of Augusta | Richmond County
City of Athens | Clarke County
United Kingdom of Great Britain and Northern Ireland | United Kingdom
French Republic | France
Kingdom of Spain | Spain
Federal Republic of Germany | Germany
Italian Republic | Italy
English Channel | La Manche
Persian Gulf | Arabian Gulf
Ayers Rock | Uluru
Mount Rainier | Tahoma
Mount Everest | Sagarmatha
North America | State of Kansas
Europe | City of Paris
Asia | City of Kyoto
South America | State of Amazonas
Africa | Cairo Governorate
City of Seattle | State of Washington
City of Miami | State of Florida
City of Austin | State of Texas
City of Chicago | State of Illinois
City of Phoenix | State of Arizona
City of London | France
City of Tokyo | South Korea
City of Toronto | Mexico
City of Berlin | Italy
City of Sydney | New Zealand