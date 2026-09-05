# Places for the topological dataset

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

## Places I already have — do not send these again

Adriatic Sea
Aegean Sea
Afghanistan
Algeria
Alps
Amazon Rainforest
Amazon River
Andorra
Angola
Antelope Island
Argentina
Argentine Republic
Arkansas River
Atacama Desert
Atlas Mountains
Australia
Austria
Baltic Sea
Baltimore City
Banff National Park
Bangladesh
Belgium
Bhutan
Bibb County
Black Forest
Black Sea
Bois de Boulogne
Bolivia
Borneo
Borough of Brooklyn
Borough of Manhattan
Borough of Queens
Borough of Staten Island
Botswana
Brazil
Bronx County, New York
Bulgaria
Cambodia
Canada
Canton of Schwyz
Canton of Uri
Carpathian Mountains
Caspian Sea
Central Park
Chad
Chile
China
City and County of Denver
City and County of Honolulu
City and County of San Francisco
City of Abu Dhabi
City of Accra
City of Addis Ababa
City of Algiers
City of Almaty
City of Amman
City of Amsterdam
City of Anchorage
City of Ankara
City of Antananarivo
City of Ashgabat
City of Asuncion
City of Athens
City of Atlanta
City of Auckland
City of Augusta
City of Austin
City of Baghdad
City of Baku
City of Bangkok
City of Barcelona
City of Baton Rouge
City of Beijing
City of Beirut
City of Belgrade
City of Bellevue
City of Bergen
City of Berkeley
City of Berlin
City of Birmingham
City of Bishkek
City of Bogota
City of Bois De Boulogne
City of Boise
City of Borough Of Brooklyn
City of Borough Of Manhattan
City of Borough Of Queens
City of Borough Of Staten Island
City of Boston
City of Bratislava
City of Brisbane
City of Brussels
City of Bucharest
City of Budapest
City of Buenos Aires
City of Cairo
City of Calgary
City of Cambridge
City of Canton Of Schwyz
City of Canton Of Uri
City of Cape Town
City of Caracas
City of Cardiff
City of Casablanca
City of Chengdu
City of Chennai
City of Chicago
City of Chisinau
City of Cleveland
City of Colombo
City of Columbus
City of Consolidated Indianapolis
City of Copenhagen
City of Cupertino
City of Curitiba
City of Dakar
City of Dallas
City of Damascus
City of Dar Es Salaam
City of Darwin
City of Delhi
City of Denver
City of Detroit
City of Dhaka
City of District Of Columbia
City of Doha
City of Dubai
City of Dublin
City of Dushanbe
City of Edina
City of Edinburgh
City of Edmonton
City of Eugene
City of Faro
City of Federal Republic Of Germany
City of Fort Worth
City of Fortaleza
City of Free State Of Bavaria
City of Genoa
City of Gothenburg
City of Guayaquil
City of Gulf Of Aden
City of Gulf Of Oman
City of Halifax
City of Hanoi
City of Harare
City of Havana
City of Helsinki
City of Ho Chi Minh City
City of Hong Kong
City of Honolulu
City of Houston
City of Indianapolis
City of Islamabad
City of Isle Of Man
City of Isle Of Wight
City of Istanbul
City of Jakarta
City of Jerusalem
City of Johannesburg
City of Kabul
City of Kampala
City of Karachi
City of Kathmandu
City of Khartoum
City of Kingdom Of Cambodia
City of Kingdom Of Denmark
City of Kingdom Of Norway
City of Kingdom Of Spain
City of Kingdom Of Sweden
City of Kingston
City of Kinshasa
City of Kolkata
City of Kuala Lumpur
City of Kuwait City
City of Kyiv
City of Kyoto
City of La Paz
City of Lafayette
City of Lagos
City of Las Vegas
City of Lexington
City of Lima
City of Lisbon
City of London
City of Los Angeles
City of Louisville
City of Luanda
City of Lyon
City of Macon
City of Madrid
City of Malmo
City of Managua
City of Manila
City of Maputo
City of Marseille
City of Melbourne
City of Mesa
City of Mexico City
City of Miami
City of Milan
City of Minneapolis
City of Minsk
City of Mogadishu
City of Montevideo
City of Montreal
City of Moscow
City of Mumbai
City of Munich
City of Muscat
City of Nairobi
City of Naples
City of New Orleans
City of New York
City of Nice
City of Nicosia
City of Oakland
City of Odesa
City of Osaka
City of Oslo
City of Ottawa
City of Palo Alto
City of Panama City
City of Paris
City of Perth
City of Philadelphia
City of Phnom Penh
City of Phoenix
City of Plurinational State Of Bolivia
City of Port Moresby
City of Portland
City of Porto
City of Prague
City of Provo
City of Pune
City of Pyongyang
City of Quebec City
City of Quito
City of Recife
City of Redmond
City of Reno
City of Republic Of Austria
City of Republic Of Bulgaria
City of Republic Of Chad
City of Republic Of Chile
City of Republic Of Colombia
City of Republic Of Ecuador
City of Republic Of Finland
City of Republic Of Iraq
City of Republic Of Korea
City of Republic Of Mali
City of Republic Of Namibia
City of Republic Of Paraguay
City of Republic Of Peru
City of Republic Of Poland
City of Reykjavik
City of Riga
City of Rio De Janeiro
City of Riyadh
City of Rome
City of Salt Lake City
City of Salvador
City of San Antonio
City of San Diego
City of San Francisco
City of San Jose
City of San Juan
City of Santiago
City of Sao Paulo
City of Sapporo
City of Sarajevo
City of Seattle
City of Seoul
City of Shanghai
City of Singapore
City of Sofia
City of Somerville
City of Spokane
City of St Petersburg
City of State Of Alabama
City of State Of Alaska
City of State Of Arizona
City of State Of California
City of State Of Colorado
City of State Of Florida
City of State Of Georgia
City of State Of Hawaii
City of State Of Hesse
City of State Of Idaho
City of State Of Illinois
City of State Of Indiana
City of State Of Kansas
City of State Of Louisiana
City of State Of Maine
City of State Of Massachusetts
City of State Of Michigan
City of State Of Minnesota
City of State Of Missouri
City of State Of Montana
City of State Of Nebraska
City of State Of Nevada
City of State Of New Jersey
City of State Of New Mexico
City of State Of New York
City of State Of North Dakota
City of State Of Ohio
City of State Of Oklahoma
City of State Of Oregon
City of State Of Pennsylvania
City of State Of Salzburg
City of State Of South Dakota
City of State Of Tennessee
City of State Of Texas
City of State Of Tyrol
City of State Of Utah
City of State Of Virginia
City of State Of Washington
City of State Of West Virginia
City of State Of Wyoming
City of Stockholm
City of Surabaya
City of Suva
City of Sydney
City of Taipei
City of Tallinn
City of Tashkent
City of Tbilisi
City of Tehran
City of Tempe
City of Thimphu
City of Tijuana
City of Tirana
City of Tokyo
City of Toronto
City of Tripoli
City of Tucson
City of Tunis
City of Turin
City of Ulaanbaatar
City of Valencia
City of Valparaiso
City of Vancouver
City of Venice
City of Vienna
City of Vientiane
City of Vilnius
City of Vladivostok
City of Warsaw
City of Washington
City of Wellington
City of Windhoek
City of Windsor
City of Winnipeg
City of Wuhan
City of Yangon
City of Yerevan
City of Zagreb
City of Zurich
Clark County, Nevada
Clarke County
Colombia
Colorado River
Columbia River
Congo River
Consolidated City of Indianapolis
Cook County
Corsica
Crete
Croatia
Cyprus
Czechia
Danube
Davidson County
Death Valley National Park
Denmark
Denver County, Colorado
District of Columbia
Duval County
East Baton Rouge Parish
Ebro
Ecuador
Egypt
Elbe
Essex
Eswatini
Ethiopia
Euphrates
Everglades National Park
Fayette County
Federal Republic of Germany
Finland
France
Free State of Bavaria
French Republic
Ganges
Germany
Ghana
Glacier National Park
Gobi Desert
Grand Canyon National Park
Grand Island
Great Britain
Great Salt Lake
Greece
Greenland
Guatemala
Gulf of Aden
Gulf of Oman
Harris County
Himalayas
Hokkaido
Honolulu County
Honshu
Hungary
Hyde Park, London
Iceland
India
Indonesia
Indus
Inner Mongolia
Iran
Iraq
Ireland
Isle of Man
Isle of Wight
Italy
Jacksonville
Japan
Java
Jefferson County
Jefferson County, Kentucky
Jordan
Kalahari Desert
Kauai
Kelleys Island
Kenya
King County
King County, Washington
Kingdom of Cambodia
Kingdom of Denmark
Kingdom of Norway
Kingdom of Spain
Kingdom of Sweden
Kings County, New York
Kruger National Park
Lafayette Parish
Lake Baikal
Lake Constance
Lake Erie
Lake Geneva
Lake Huron
Lake Michigan
Lake Ontario
Lake Superior
Lake Tanganyika
Lake Victoria
Lake Washington
Laos
Lesotho
Libya
Limpopo
Loch Ness
Loire
Long Island
Los Angeles County
Louisville
Luxembourg
Mackinac Island
Madagascar
Malaysia
Mali
Manhattan Island
Maricopa County
Marin County
Marion County
Marion County, Indiana
Mariposa County
Maui
Mediterranean Sea
Mekong
Mekong River
Mercer Island
Mexico
Miami-Dade County
Mississippi River
Missouri River
Mojave Desert
Monaco
Morocco
Mozambique
Muscogee County
Namib Desert
Namibia
Nantucket
Nashville-Davidson
Nepal
Netherlands
New York County, New York
New Zealand
Niger
Niger River
Nigeria
Nile
North Island
North Sea
Norway
Oahu
Oder
Oman
Orange County, California
Orange River
Orleans Parish
Pakistan
Panama
Paraguay
Parana River
Pelee Island
Persian Gulf
Peru
Philadelphia County
Pima County
Platte River
Plurinational State of Bolivia
Po
Poland
Portugal
Pyrenees
Queens County, New York
Red Sea
Republic of Austria
Republic of Bulgaria
Republic of Chad
Republic of Chile
Republic of Colombia
Republic of Ecuador
Republic of Finland
Republic of Iraq
Republic of Korea
Republic of Mali
Republic of Namibia
Republic of Paraguay
Republic of Peru
Republic of Poland
Rhine
Rhone
Richmond County
Richmond County, New York
Rio Grande
River Thames
Rocky Mountains
Romania
Sahara
San Diego County
San Francisco County
San Marino
Santa Clara County
Sardinia
Saudi Arabia
Scandinavian Mountains
Seine
Senegal
Serbia
Serengeti National Park
Sicily
Slovenia
Sonoran Desert
South Africa
South Island
Spain
St. Louis City
State of Alabama
State of Alaska
State of Arizona
State of California
State of Colorado
State of Florida
State of Georgia
State of Hawaii
State of Hesse
State of Idaho
State of Illinois
State of Indiana
State of Kansas
State of Louisiana
State of Maine
State of Massachusetts
State of Michigan
State of Minnesota
State of Missouri
State of Montana
State of Nebraska
State of Nevada
State of New Jersey
State of New Mexico
State of New York
State of North Dakota
State of Ohio
State of Oklahoma
State of Oregon
State of Pennsylvania
State of Salzburg
State of South Dakota
State of Tennessee
State of Texas
State of Tyrol
State of Utah
State of Virginia
State of Washington
State of West Virginia
State of Wyoming
Staten Island
Sudan
Suffolk County, Massachusetts
Suffolk County, New York
Sweden
Switzerland
Syria
Tagus
Tanzania
Tarrant County
Tasmania
Teton County
Thailand
The Bronx
Tigris
Tunisia
Turkey
Uganda
United Kingdom
United States
Ural Mountains
Uruguay
Vatican City
Venezuela
Vietnam
Vistula
Volga
Washington, D.C.
Wayne County, Michigan
Yangtze
Yellow River
Yellowstone National Park
Yemen
Yosemite National Park
Zambezi
Zimbabwe
