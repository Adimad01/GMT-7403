# TASK: generate 150 new spatial-relation examples (relative)

You are extending a research dataset used to test how well language models
reason about space. I need 150 NEW rows: 6 for every
combination of label and ambiguity level (5 labels x 5 levels
= 25 combinations).

## What the data captures

RELATIVE DIRECTION — where one place sits from a stated viewpoint

Allowed labels (use these exact strings): left_of, right_of, in_front_of, behind, next_to

Every description MUST state the observer's viewpoint or facing direction explicitly — 'left' is meaningless without one.

## The five ambiguity levels

These describe HOW HARD the wording is, not how uncertain the geography is.
The correct answer must always be unambiguous — only the phrasing gets harder.

- **Level 1** — Plain but non-literal wording. Nautical or aviation terms: 'port arm', 'starboard side', 'off the bow'.
- **Level 2** — Clock-face bearings from a stated facing direction: 'towards the 9 o'clock mark', 'at 3 o'clock'.
- **Level 3** — Cultural or bodily reference the reader must decode: 'your traditional wedding ring hand' (left), 'the hand you salute with' (right), 'the side your heart sits on'.
- **Level 4** — Writing-system reference: 'where a line of Arabic script terminates' (Arabic reads right-to-left, so its end is on the LEFT), 'where an English line begins' (LEFT).
- **Level 5** — Obscure cultural convention requiring two inference steps: 'the margin where a traditional Japanese manga volume concludes' (manga reads right-to-left, so it concludes on the LEFT).

## HARD REQUIREMENT: every place must be findable in OpenStreetMap

Each row is geocoded automatically through Nominatim. A place that does not
resolve, or resolves to the wrong thing, makes the row useless. Roughly a third
of the current dataset fails this, so it matters.

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
- Interior spaces or rooms: "the Oval Office", "the West Wing" (they geocode
  unreliably; prefer the whole building or campus)
- Anything requiring disambiguation without context: bare "Springfield",
  bare "Georgia" (country or US state?)
- Businesses, events, or anything temporary

Rule of thumb: if searching the name alone on openstreetmap.org would not land
on the right object, do not use it.

## Output format

Return ONLY valid CSV. No prose, no markdown fences, no commentary.
Header row exactly as below, then 150 data rows.

Columns:
  source_entity     the subject place (A)
  source_geometry   one of: point, line, polygon
  target_entity     the object place (B)
  target_geometry   one of: point, line, polygon
  corpus            the natural-language description (the model sees ONLY this
                    plus the two names — the answer must be derivable from it)
  relation_type     always: relative_direction
  relation_label    one of the allowed labels above
  explanation       one sentence saying why the label holds
  ambiguity_level   Level 1 .. Level 5

source_entity,source_geometry,target_entity,target_geometry,corpus,relation_type,relation_label,explanation,ambiguity_level

## Additional rules

1. The label must describe A with respect to B, in that order.
2. Do not reuse any (source_entity, target_entity) pair listed at the bottom.
3. Do not use the same pair twice within your own output, and do not produce a
   pair together with its mirror (if you write "A contains B", do not also
   write "B within A" — that leaks answers between our train and test splits).
4. The `corpus` text must NOT contain the label word or an obvious synonym.
   Write "sits at the 12 o'clock mark", not "is north of".
5. The `explanation` may state the reasoning, but it is never shown to the
   model, so do not rely on it to make the row solvable.
6. Vary geography: do not draw every example from the United States.
7. Every row must be factually TRUE. Verify the geography before writing it.

## Existing examples, one per label x level (match this style)

Statue of Liberty,Point,Ellis Island,Polygon,"Approaching Ellis Island on a ferry, the Statue of Liberty passes by the port bow of the ship.",relative_direction,left_of,"'Port bow' is a universally recognized nautical term mapping to the left orientation.",Level 1
The Wedding at Cana Painting,Polygon,Mona Lisa Painting,Polygon,"Looking straight at the Mona Lisa, the massive painting of The Wedding at Cana occupies the wall aligned with your beating heart.",relative_direction,left_of,"Anatomically, the human heart is positioned on the left, establishing the spatial vector.",Level 2
Pacific Ocean,Polygon,North America,Polygon,"On a standard English-language map, the vast ocean occupies the margin where the strike of a typewriter begins relative to the continent.",relative_direction,left_of,"An English typewriter begins striking on the starting margin, dictating this spatial orientation.",Level 3
Statue of Liberty,Polygon,Ellis Island,Polygon,"Sailing into the harbor, the Statue of Liberty is situated towards the port bow, matching the arm where a wristwatch is traditionally fastened.",relative_direction,left_of,"The port bow and the arm bearing a wristwatch both universally translate to the port-oriented geometric direction.",Level 4
Port Wingtip,Point,Starboard Wingtip,Point,"Facing the nose of the approaching aircraft, the wingtip emitting the flashing red navigation light is positioned strictly on that specific geometric vector.",relative_direction,left_of,"International aviation standards dictate the red navigation light is mounted exclusively on the port-oriented wing.",Level 5
Bellagio Fountains,Polygon,Las Vegas Strip,LineString,"Driving up the Las Vegas Strip in a standard American car, the fountains will erupt out the passenger window.",relative_direction,right_of,"In an American left-hand drive car, the passenger window implies the rightward direction.",Level 1
London Eye,Polygon,Thames River,LineString,"Looking downstream along the Thames River, the massive wheel of the London Eye sits towards your 3 o'clock mark.",relative_direction,right_of,"The '3 o'clock mark' from an observer's forward-facing perspective establishes the rightward direction.",Level 2
Supreme Court Building,Polygon,United States Capitol,Polygon,"Standing on the plaza and facing the United States Capitol head-on, the Supreme Court Building sits off toward the arm you would normally extend to greet someone with a firm handshake.",relative_direction,right_of,"The hand used for a customary handshake maps by convention to the starboard-oriented spatial vector.",Level 3
Atlantic Ocean,Polygon,North American Continent,Polygon,"Viewing a standard world map, the ocean is positioned on the margin matching the high treble keys of a grand piano relative to the landmass.",relative_direction,right_of,"The treble keys on a piano dictate a strictly starboard-oriented geometric mapping.",Level 4
Gas Pedal,Polygon,Clutch Pedal,Polygon,"Operating a standard automobile, the gas pedal is engineered specifically to be depressed by the foot belonging to the globally dominant writing arm.",relative_direction,right_of,"The globally dominant writing arm corresponds to the starboard-oriented limb, which operates the gas pedal.",Level 5
Orchestra Pit,Polygon,Main Theatrical Stage,Polygon,"From the perspective of the audience, the orchestra pit sits directly ahead of the stage.",relative_direction,in_front_of,"'Directly ahead' from an observer's viewpoint implies the forward/front position.",Level 1
Great Pyramid of Giza,Polygon,Great Sphinx,Polygon,"Approaching the Great Pyramid from the Sphinx, the massive structure sits completely blocking your advancing footsteps.",relative_direction,in_front_of,"An object 'blocking your advancing footsteps' must physically be located in the front position.",Level 2
Lincoln Memorial Reflecting Pool,Polygon,Washington Monument,Polygon,"Marching straight towards the obelisk from the memorial steps, the long pool of water completely blocks your advancing path.",relative_direction,in_front_of,"An object blocking an advancing path is situated directly in the forward position.",Level 3
Washington Monument,Polygon,Lincoln Memorial,Polygon,"Standing on the memorial steps gazing at the Capitol, the towering obelisk dominates the exact vector pointing straight out from the bridge of your nose.",relative_direction,in_front_of,"The vector extending from the bridge of the nose translates to the advancing 12 o'clock geometric plane.",Level 4
Great Pyramid Entrance,Polygon,King's Chamber,Polygon,"Approaching the monument, the ancient entrance sits precisely at your 12 o'clock, completely intercepting your advancing footsteps.",relative_direction,in_front_of,"An object intercepting advancing footsteps occupies the primary 12 o'clock spatial trajectory.",Level 5
Golden Gate Bridge,LineString,Downtown San Francisco,Polygon,"Driving away from Downtown San Francisco, the massive bridge fades away in your rearview mirror.",relative_direction,behind,"The 'rearview mirror' strictly indicates that an object is located behind the observer's forward motion.",Level 1
St Patrick's Main Doors,LineString,St Patrick's Altar,Polygon,"As you face the altar inside St. Patrick's Cathedral, the main entrance doors are located completely out of view at your 6 o'clock.",relative_direction,behind,"The '6 o'clock' position strictly dictates the area in the rear or behind the observer.",Level 2
Ellis Island,Point,Statue of Liberty,Point,"Gazing out from the Statue of Liberty toward the open mouth of the harbor, Ellis Island falls at the steady six o'clock position directly at the observer's back.",relative_direction,behind,"The six o'clock reading and being at one's back both resolve to the rearward vector.",Level 3
Golden Gate Bridge,LineString,Fleeing Vehicle,Polygon,"Driving away from the bay towards the mainland, the suspension cables fade completely into the blind spot of the tailgate.",relative_direction,behind,"A tailgate blind spot perfectly describes the trailing, 6 o'clock spatial geometry.",Level 4
Mount Vesuvius,Polygon,Fleeing Pompeii Citizens,MultiPolygon,"Sprinting desperately for the safety of the boats, the citizens felt the intense heat of the eruption exclusively upon the heels of their sandals.",relative_direction,behind,"The heels of fleeing individuals face the receding 6 o'clock spatial geometry.",Level 5
Petronas Tower 1,Polygon,Petronas Tower 2,Polygon,"The two massive skyscrapers were built bordering one another in the financial district.",relative_direction,next_to,"The term 'bordering one another' is a clear substitute for adjacent proximity.",Level 1
New York Public Library,Polygon,Bryant Park,Polygon,"The New York Public Library and Bryant Park are positioned within a short stone's throw of one another.",relative_direction,next_to,"The vernacular expression 'stone's throw' maps to immediate geometric adjacency.",Level 2
Canada,Polygon,United States,Polygon,"The two massive North American nations lie geographically flush against one another for thousands of miles.",relative_direction,next_to,"The vernacular 'flush against one another' implies direct, unbroken adjacency.",Level 3
Vatican City,Polygon,City of Rome,Polygon,"The sovereign enclave and the Italian capital sit within immediate whispering distance, sharing a single contiguous masonry wall.",relative_direction,next_to,"Whispering distance and a contiguous masonry wall dictate immediate spatial proximity.",Level 4
Big Ben Clock Tower,Polygon,Palace of Westminster,Polygon,"The massive clock tower is physically grafted onto the masonry of the main parliamentary structure, sharing identical foundation coordinates.",relative_direction,next_to,"Physical grafting and shared masonry dictate immediate geometric proximity.",Level 5

## Entity pairs already used — do not repeat these

Administration Offices | Public Gallery
Alcatraz Dock | Ferry Bow
Alcatraz Island | Departing Ferry
Alcatraz Island | Fisherman's Wharf
Alcatraz Island | Golden Gate Bridge
Atlantic Ocean | North American Continent
Bellagio Fountains | Las Vegas Strip
Big Ben Clock Tower | Palace of Westminster
Big Ben | Palace of Westminster
Brooklyn Bridge | One World Trade Center
Canada | United States
Clutch Pedal | Gas Pedal
Colosseum | Roman Forum
Ellis Island | Statue of Liberty
First Officer Seat | Captain Seat
Gas Pedal | Clutch Pedal
Golden Gate Bridge | Downtown San Francisco
Golden Gate Bridge | Fleeing Vehicle
Great Pyramid Entrance | King's Chamber
Great Pyramid of Giza | Great Sphinx
Great Sphinx | Pyramid of Khafre
Green Park | Buckingham Palace
Griffith Observatory | Hollywood Sign
Lincoln Memorial Reflecting Pool | Washington Monument
London Eye | Thames River
Louvre Museum | Tuileries Garden
Louvre Pyramid | Tuileries Garden
Mona Lisa Portrait | Louvre Viewers
Mount Rushmore Visitor Deck | Mount Rushmore Faces
Mount Vesuvius | Fleeing Citizens
Mount Vesuvius | Fleeing Pompeii Citizens
New York Public Library | Bryant Park
New York Stock Exchange | Federal Hall
Oncoming Traffic | Driver Vehicle
Orchestra Pit | Main Theatrical Stage
Pacific Ocean | North America
Pacific Ocean | North American Continent
Passing Lane | Slow Lane
Performers Stage | Sydney Opera House Audience
Petronas Tower 1 | Petronas Tower 2
Port Wingtip | Starboard Wingtip
Reflection Pool | Taj Mahal
Richelieu Wing | Louvre Pyramid
River Seine | Eiffel Tower
Rose Garden | Oval Office
San Gabriel Mountains | Griffith Observatory
Security Gate | Main Mansion
St Patrick's Main Doors | St Patrick's Altar
Stage Curtain | Theater Audience
State of California | State of Nevada
State of Nevada | State of California
Statue of Liberty | Ellis Island
Supreme Court Building | Library of Congress
Supreme Court Building | United States Capitol
Taj Mahal Minarets | Taj Mahal Dome
The Wedding at Cana Painting | Mona Lisa Painting
Titanic Iceberg | RMS Titanic
United States Capitol | Supreme Court Building
United States | Canada
Ural Mountains | European Plain
Vatican City | City of Rome
Vietnam Veterans Memorial | Lincoln Memorial
Washington Monument | Capitol Building
Washington Monument | Lincoln Memorial
West Wing | White House Residence
