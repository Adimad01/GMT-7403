"""Candidate places for the topological corpus.

Chosen so that every one is a real, well-defined OpenStreetMap object with a
polygon or line geometry. That rules out the vernacular regions the old corpus
leaned on -- Rust Belt, American Midwest, Sahel Region -- and the outright
abstractions it invented, such as 'Abstract Geodetic Prime Meridian' and
'Vennbahn Railway Legal Footprint'. A relation between two shapes needs two
shapes.

The mix is deliberate. Nested administrative units supply contains/within,
neighbouring units supply touches, rivers supply crosses, physical regions
that ignore borders supply overlaps, and consolidated city-counties supply the
rare equals.
"""

COUNTRIES = [
    "France", "Germany", "Spain", "Portugal", "Italy", "Switzerland", "Austria",
    "Belgium", "Netherlands", "Poland", "Czechia", "Hungary", "Romania",
    "Greece", "Norway", "Sweden", "Finland", "Denmark", "Ireland",
    "United Kingdom", "Morocco", "Algeria", "Tunisia", "Libya", "Egypt",
    "Sudan", "Ethiopia", "Kenya", "Tanzania", "Uganda", "Nigeria", "Ghana",
    "Senegal", "Mali", "Niger", "Chad", "South Africa", "Namibia", "Botswana",
    "Zimbabwe", "Mozambique", "Angola", "Lesotho", "Eswatini", "Brazil",
    "Argentina", "Chile", "Peru", "Bolivia", "Ecuador", "Colombia",
    "Venezuela", "Uruguay", "Paraguay", "Mexico", "Guatemala", "Panama",
    "Canada", "United States", "China", "India", "Nepal", "Bhutan",
    "Bangladesh", "Pakistan", "Afghanistan", "Iran", "Iraq", "Turkey",
    "Syria", "Jordan", "Saudi Arabia", "Oman", "Yemen", "Japan", "Vietnam",
    "Laos", "Cambodia", "Thailand", "Malaysia", "Indonesia", "Australia",
    "New Zealand", "San Marino", "Vatican City", "Monaco", "Andorra",
    "Luxembourg", "Slovenia", "Croatia", "Serbia", "Bulgaria",
]

US_STATES = [
    "State of California", "State of Texas", "State of Florida",
    "State of New York", "State of Colorado", "State of Arizona",
    "State of Nevada", "State of Utah", "State of Oregon",
    "State of Washington", "State of Alaska", "State of Hawaii",
    "State of Illinois", "State of Ohio", "State of Michigan",
    "State of Georgia", "State of Virginia", "State of Maine",
    "State of Montana", "State of Wyoming", "State of Kansas",
    "State of Missouri", "State of Louisiana", "State of Alabama",
    "State of Pennsylvania", "State of Massachusetts", "State of Minnesota",
    "State of New Mexico", "State of Idaho", "State of Nebraska",
]

CITIES = [
    "City of Paris", "City of Berlin", "City of Madrid", "City of Lisbon",
    "City of Rome", "City of Vienna", "City of Prague", "City of Warsaw",
    "City of Budapest", "City of Athens", "City of Oslo", "City of Stockholm",
    "City of Helsinki", "City of Copenhagen", "City of Dublin",
    "City of Amsterdam", "City of Brussels", "City of Zurich",
    "City of Munich", "City of Milan", "City of Naples", "City of Venice",
    "City of Barcelona", "City of Lyon", "City of Marseille",
    "City of Los Angeles", "City of San Francisco", "City of San Diego",
    "City of Seattle", "City of Portland", "City of Denver", "City of Phoenix",
    "City of Houston", "City of Dallas", "City of Austin", "City of Chicago",
    "City of Boston", "City of Philadelphia", "City of Miami",
    "City of Atlanta", "City of Detroit", "City of Cleveland",
    "City of Minneapolis", "City of New Orleans", "City of Las Vegas",
    "City of Toronto", "City of Montreal", "City of Vancouver",
    "City of Calgary", "City of Cairo", "City of Nairobi", "City of Lagos",
    "City of Accra", "City of Casablanca", "City of Cape Town",
    "City of Johannesburg", "City of Tokyo", "City of Delhi",
    "City of Mumbai", "City of Bangkok", "City of Sydney", "City of Melbourne",
]

COUNTIES = [
    "Los Angeles County", "Cook County", "Harris County", "Maricopa County",
    "San Diego County", "Orange County, California", "King County, Washington",
    "Miami-Dade County", "Clark County, Nevada", "Wayne County, Michigan",
    "Kings County, New York", "Queens County, New York",
    "Bronx County, New York", "Richmond County, New York",
    "New York County, New York", "Suffolk County, Massachusetts",
    "Philadelphia County", "Marin County", "Santa Clara County",
]

BOROUGHS = [
    "Borough of Manhattan", "Borough of Brooklyn", "Borough of Queens",
    "Borough of Staten Island", "The Bronx",
]

PARKS = [
    "Yellowstone National Park", "Yosemite National Park",
    "Grand Canyon National Park", "Everglades National Park",
    "Glacier National Park", "Death Valley National Park",
    "Banff National Park", "Kruger National Park", "Serengeti National Park",
    "Central Park", "Hyde Park, London", "Bois de Boulogne",
]

WATER = [
    "Lake Michigan", "Lake Superior", "Lake Huron", "Lake Erie",
    "Lake Ontario", "Lake Victoria", "Lake Tanganyika", "Lake Baikal",
    "Loch Ness", "Lake Geneva", "Lake Constance", "Great Salt Lake",
    "Mediterranean Sea", "Baltic Sea", "Black Sea", "Red Sea",
    "Caspian Sea", "North Sea", "Adriatic Sea", "Aegean Sea",
]

RIVERS = [
    "River Thames", "Seine", "Rhine", "Danube", "Elbe", "Loire", "Rhone",
    "Po", "Tagus", "Ebro", "Vistula", "Oder", "Nile", "Niger River",
    "Congo River", "Zambezi", "Orange River", "Limpopo", "Mississippi River",
    "Missouri River", "Colorado River", "Rio Grande", "Columbia River",
    "Amazon River", "Parana River", "Ganges", "Indus", "Mekong",
    "Yangtze", "Yellow River", "Volga", "Euphrates", "Tigris",
]

PHYSICAL = [
    "Sahara", "Kalahari Desert", "Namib Desert", "Gobi Desert",
    "Atacama Desert", "Mojave Desert", "Sonoran Desert", "Amazon Rainforest",
    # Mountain ranges are mapped inconsistently: several are single nodes,
    # and "Andes" matches a town in Colombia before the cordillera.
    "Alps", "Pyrenees", "Carpathian Mountains", "Himalayas",
    "Rocky Mountains", "Ural Mountains",
    "Atlas Mountains", "Scandinavian Mountains", "Black Forest",
]

ISLANDS = [
    "Sicily", "Sardinia", "Corsica", "Crete", "Cyprus", "Iceland",
    "Isle of Man", "Isle of Wight", "Greenland", "Madagascar", "Tasmania",
    "Hokkaido", "Honshu", "Java", "Borneo", "Long Island", "Manhattan Island",
]

# Consolidated city-counties: one government, one outline, two names. These are
# the only reliable source of a genuine 'equals' -- two differently named OSM
# objects almost never share a boundary otherwise.
CONSOLIDATED = [
    "City and County of San Francisco", "City and County of Denver",
    "City and County of Honolulu", "Nashville-Davidson", "Davidson County",
    "Marion County, Indiana", "Consolidated City of Indianapolis",
    "Jefferson County, Kentucky", "Louisville",
    "Duval County", "Jacksonville",
    "Orleans Parish", "Suffolk County, New York",
    "San Francisco County", "Denver County, Colorado",
    "Honolulu County", "District of Columbia", "Washington, D.C.",
    "Baltimore City", "St. Louis City",
]

ALL = (COUNTRIES + US_STATES + CITIES + COUNTIES + BOROUGHS + PARKS
       + WATER + RIVERS + PHYSICAL + ISLANDS + CONSOLIDATED)

# What each name is, so the resolver never has to infer it from spelling.
KIND = {}
for _grp, _kind in ((COUNTRIES, "admin"), (US_STATES, "admin"),
                    (CITIES, "city"), (COUNTIES, "admin"), (BOROUGHS, "admin"),
                    (PARKS, "park"), (WATER, "lake"), (RIVERS, "river"),
                    (PHYSICAL, "physical"), (ISLANDS, "island"),
                    (CONSOLIDATED, "admin")):
    for _n in _grp:
        KIND[_n] = _kind
for _n in ("Mediterranean Sea", "Baltic Sea", "Black Sea", "Red Sea",
           "Caspian Sea", "North Sea", "Adriatic Sea", "Aegean Sea"):
    KIND[_n] = "sea"

if __name__ == "__main__":
    from collections import Counter
    groups = dict(countries=COUNTRIES, states=US_STATES, cities=CITIES,
                  counties=COUNTIES, boroughs=BOROUGHS, parks=PARKS,
                  water=WATER, rivers=RIVERS, physical=PHYSICAL,
                  islands=ISLANDS)
    for k, v in groups.items():
        print(f"  {k:<12}{len(v):>4}")
    print(f"  {'TOTAL':<12}{len(ALL):>4}   distinct {len(set(ALL))}")
