# TASK: generate 319 new spatial-relation examples (relative)

You are extending a research dataset used to test how well language models
reason about space. I need 319 NEW rows, 11 per cell:

  Levels 1-5: all 5 labels x 5 levels = 25 cells
  Level 6   : only 4 labels (left_of, right_of, in_front_of, behind) = 4 cells

  total 29 cells x 11 = 319 rows, of which 44 are multi-hop.

The Level 6 grid is deliberately smaller: the remaining labels have no forced
two-hop composition, so multi-hop rows for them would have no determinate
answer.

## What the data captures

RELATIVE DIRECTION — where one place sits from a stated viewpoint

Allowed labels (use these exact strings): left_of, right_of, in_front_of, behind, next_to

Every description MUST state the observer's viewpoint or facing direction explicitly — 'left' is meaningless without one.

## The six ambiguity levels

Levels 1-5 grade HOW HARD THE GEOGRAPHY is. The correct answer is always
unambiguous; what changes is how much a reader must know, and how far the
configuration sits from the obvious case. The wording never carries the
answer — see the headline rule below. Level 6 is different in kind: it adds
an inference step instead.

- **Level 1** — Plain but non-literal wording. Nautical or aviation terms: 'port arm', 'starboard side', 'off the bow'.
- **Level 2** — Clock-face bearings from a stated facing direction: 'towards the 9 o'clock mark', 'at 3 o'clock'.
- **Level 3** — Cultural or bodily reference the reader must decode: 'your traditional wedding ring hand' (left), 'the hand you salute with' (right).
- **Level 4** — Writing-system reference: 'where a line of Arabic script terminates' (Arabic reads right-to-left, so its end is on the LEFT).
- **Level 5** — Obscure cultural convention needing two inference steps: 'the margin where a traditional Japanese manga volume concludes' (manga reads right-to-left, so it concludes on the LEFT).
- **Level 6 — MULTI-HOP** — the relation between A and B is NOT stated. The
  description states two links through an intermediate place C, and the reader
  must compose them.

  Keep the WORDING PLAIN at this level. Levels 1-5 make the phrasing harder;
  Level 6 makes the *inference* harder. If a row is both obscurely worded and
  multi-hop, we cannot tell which caused the difficulty, and the row is wasted.

  **Level 6 exists ONLY for these labels: left_of, right_of, in_front_of, behind.**
  The other labels have no forced two-hop composition, so do not produce Level 6
  rows for them at all. The grid is deliberately ragged here.

From a SINGLE fixed viewpoint, left/right ordering is transitive. State two links and let the reader compose them:
  A is left of C  +  C is left of B   =>  A is left of B
The same holds for right_of, in_front_of and behind along one axis. For next_to, use adjacency in a stated row: 'A sits beside C, and C beside B, with nothing between them' => A is near B — only claim next_to when the three genuinely form a compact row.

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
    "Standing on the National Mall facing the Capitol, the Washington Monument sits to the port side of the National Museum of American History, and that museum in turn sits to the port side of the National Gallery of Art. (A=Washington Monument, C=Museum of American History, B=National Gallery — all three named, both links stated.)"


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
Header row exactly as below, then 319 data rows.

Columns:
  source_entity     the subject place (A)
  source_geometry   one of: Point, LineString, Polygon, MultiPolygon
  target_entity     the object place (B)
  target_geometry   one of: Point, LineString, Polygon, MultiPolygon
  corpus            the natural-language description (the model sees ONLY this
                    plus the two names — the answer must be derivable from it)
  via_entity        Level 6 ONLY: the intermediate place C. Leave EMPTY for
                    Levels 1-5.
  relation_type     always: relative_direction
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

## Worked examples — match this style

Note these predate the `via_entity` column, so it is empty in all of them.

City of Athens,Polygon,City of Helsinki,Polygon,"Standing in Ankara and facing Helsinki squarely, a navigator notes Athens on the same segment of chart.",,relative_direction,left_of,"From Ankara the sight line to Helsinki runs 11 degrees off north; Athens sits -91 degrees from that line at 0.35 times the distance.",Level 1
City of Calgary,Polygon,City of Caracas,Polygon,"Standing in Bogota and facing Caracas squarely, an astronomer sweeps past Calgary on the same pass.",,relative_direction,left_of,"From Bogota the sight line to Caracas runs 51 degrees off north; Calgary sits -79 degrees from that line at 6.23 times the distance.",Level 2
City of Cape Town,Polygon,City of Dakar,Polygon,"Working in Cairo with the instrument facing Dakar, a surveyor picks up Cape Town in the same sweep.",,relative_direction,left_of,"From Cairo the sight line to Dakar runs 98 degrees off north; Cape Town sits -70 degrees from that line at 1.38 times the distance.",Level 3
City of La Paz,Polygon,City of Mexico City,Polygon,"Working in Halifax with the instrument facing Mexico City, the pilot keeps La Paz in the same window.",,relative_direction,left_of,"From Halifax the sight line to Mexico City runs 119 degrees off north; La Paz sits -56 degrees from that line at 1.58 times the distance.",Level 4
City of Manila,Polygon,City of Perth,Polygon,"Posted in Kolkata and looking steadily at Perth, an observer also has Manila in view.",,relative_direction,left_of,"From Kolkata the sight line to Perth runs 153 degrees off north; Manila sits -55 degrees from that line at 0.53 times the distance.",Level 5
City of Nairobi,Polygon,City of Paris,Polygon,"Posted in Madrid and looking steadily at Paris, a geographer takes in Nairobi at the same moment.",,relative_direction,right_of,"From Madrid the sight line to Paris runs 25 degrees off north; Nairobi sits +103 degrees from that line at 5.88 times the distance.",Level 1
City of Rio De Janeiro,Polygon,City of Panama City,Polygon,"Posted in Montevideo and looking steadily at Panama City, a walker takes in Rio De Janeiro without turning.",,relative_direction,right_of,"From Montevideo the sight line to Panama City runs 31 degrees off north; Rio De Janeiro sits +78 degrees from that line at 0.34 times the distance.",Level 2
City of Tunis,Polygon,City of Tashkent,Polygon,"Sitting in the tower at Stockholm and facing Tashkent, an observer also has Tunis in view.",,relative_direction,right_of,"From Stockholm the sight line to Tashkent runs 97 degrees off north; Tunis sits +100 degrees from that line at 0.64 times the distance.",Level 3
City of Quito,Polygon,City of Sao Paulo,Polygon,"Sitting in the tower at Caracas and facing Sao Paulo, the controller sees Quito on the same display.",,relative_direction,right_of,"From Caracas the sight line to Sao Paulo runs 150 degrees off north; Quito sits +78 degrees from that line at 0.40 times the distance.",Level 4
City of Ankara,Polygon,City of Abu Dhabi,Polygon,"Camped outside Baku and facing Abu Dhabi across the plain, an observer also has Ankara in view.",,relative_direction,right_of,"From Baku the sight line to Abu Dhabi runs 165 degrees off north; Ankara sits +108 degrees from that line at 0.79 times the distance.",Level 5
City of Yerevan,Polygon,City of Minsk,Polygon,"Camped outside Dubai and facing Minsk across the plain, the pilot keeps Yerevan in the same window.",,relative_direction,in_front_of,"From Dubai the sight line to Minsk runs 28 degrees off north; Yerevan sits -0 degrees from that line at 0.49 times the distance.",Level 1
City of Prague,Polygon,City of Edinburgh,Polygon,"Camped outside Bucharest and facing Edinburgh across the plain, the controller sees Prague on the same display.",,relative_direction,in_front_of,"From Bucharest the sight line to Edinburgh runs 48 degrees off north; Prague sits -3 degrees from that line at 0.45 times the distance.",Level 2
City of Tashkent,Polygon,City of Bangkok,Polygon,"Looking out from Kyiv with the gaze locked on Bangkok, an observer also has Tashkent in view.",,relative_direction,in_front_of,"From Kyiv the sight line to Bangkok runs 97 degrees off north; Tashkent sits -3 degrees from that line at 0.42 times the distance.",Level 3
City of Vienna,Polygon,City of Tunis,Polygon,"Looking out from Riga with the gaze locked on Tunis, the controller sees Vienna on the same display.",,relative_direction,in_front_of,"From Riga the sight line to Tunis runs 149 degrees off north; Vienna sits +1 degrees from that line at 0.45 times the distance.",Level 4
City of Miami,Polygon,City of Panama City,Polygon,"Flying over Toronto and facing Panama City on the present heading, an observer also has Miami in view.",,relative_direction,in_front_of,"From Toronto the sight line to Panama City runs 180 degrees off north; Miami sits +2 degrees from that line at 0.52 times the distance.",Level 5
City of Almaty,Polygon,City of Dhaka,Polygon,"Flying over Jakarta and facing Dhaka on the present heading, the operator catches Almaty in the identical sector.",,relative_direction,behind,"From Jakarta the sight line to Dhaka runs 28 degrees off north; Almaty sits +2 degrees from that line at 1.66 times the distance.",Level 1
City of Munich,Polygon,City of New York,Polygon,"Flying over Atlanta and facing New York on the present heading, the controller sees Munich on the same display.",,relative_direction,behind,"From Atlanta the sight line to New York runs 47 degrees off north; Munich sits -2 degrees from that line at 6.41 times the distance.",Level 2
City of Miami,Polygon,City of Houston,Polygon,"Moored off San Diego and facing Houston over the water, a geographer takes in Miami at the same moment.",,relative_direction,behind,"From San Diego the sight line to Houston runs 93 degrees off north; Miami sits -1 degrees from that line at 1.74 times the distance.",Level 3
City of Hong Kong,Polygon,City of Shanghai,Polygon,"Moored off Seoul and facing Shanghai over the water, the pilot keeps Hong Kong in the same window.",,relative_direction,behind,"From Seoul the sight line to Shanghai runs 143 degrees off north; Hong Kong sits +2 degrees from that line at 2.41 times the distance.",Level 4
City of Harare,Polygon,City of Naples,Polygon,"Waiting on the platform at Venice, facing Naples, a navigator notes Harare on the same segment of chart.",,relative_direction,behind,"From Venice the sight line to Naples runs 162 degrees off north; Harare sits -2 degrees from that line at 13.64 times the distance.",Level 5
City of Tallinn,Polygon,City of St Petersburg,Polygon,"Waiting on the platform at Doha, facing St Petersburg, a walker takes in Tallinn without turning.",,relative_direction,next_to,"From Doha the sight line to St Petersburg runs 17 degrees off north; Tallinn sits -4 degrees from that line at 1.03 times the distance.",Level 1
City of Sarajevo,Polygon,City of Bratislava,Polygon,"Set up above Chengdu and looking dead at Bratislava, an observer also has Sarajevo in view.",,relative_direction,next_to,"From Chengdu the sight line to Bratislava runs 47 degrees off north; Sarajevo sits -5 degrees from that line at 1.01 times the distance.",Level 2
City of Cleveland,Polygon,City of Windsor,Polygon,"Set up above Portland and looking dead at Windsor, the pilot keeps Cleveland in the same window.",,relative_direction,next_to,"From Portland the sight line to Windsor runs 82 degrees off north; Cleveland sits +1 degrees from that line at 1.04 times the distance.",Level 3
City of Recife,Polygon,City of Fortaleza,Polygon,"Set up above Turin and looking dead at Fortaleza, the controller sees Recife on the same display.",,relative_direction,next_to,"From Turin the sight line to Fortaleza runs 127 degrees off north; Recife sits -6 degrees from that line at 1.03 times the distance.",Level 4
City of Montevideo,Polygon,City of Buenos Aires,Polygon,"From the waterfront at Kingston, with the line of sight running to Buenos Aires, Montevideo enters the same panorama.",,relative_direction,next_to,"From Kingston the sight line to Buenos Aires runs 162 degrees off north; Montevideo sits -2 degrees from that line at 1.02 times the distance.",Level 5

## Entity pairs already used — do not repeat these

City of Accra | City of Algiers
City of Accra | City of Baku
City of Algiers | City of Addis Ababa
City of Almaty | City of Dhaka
City of Almaty | City of Singapore
City of Ankara | City of Abu Dhabi
City of Asuncion | City of Sao Paulo
City of Athens | City of Amsterdam
City of Athens | City of Helsinki
City of Atlanta | City of Amsterdam
City of Atlanta | City of Lagos
City of Baghdad | City of Ankara
City of Baghdad | City of Hong Kong
City of Belgrade | City of Dublin
City of Berlin | City of Beijing
City of Bogota | City of Denver
City of Boston | City of Harare
City of Brussels | City of Bangkok
City of Buenos Aires | City of Lima
City of Cairo | City of Tunis
City of Calgary | City of Caracas
City of Cape Town | City of Bucharest
City of Cape Town | City of Dakar
City of Caracas | City of Amman
City of Cardiff | City of Birmingham
City of Casablanca | City of Amman
City of Chennai | City of Belgrade
City of Chennai | City of Riga
City of Chicago | City of Bogota
City of Cleveland | City of Windsor
City of Colombo | City of Beirut
City of Copenhagen | City of Bucharest
City of Copenhagen | City of Oslo
City of Dallas | City of Denver
City of Detroit | City of Cleveland
City of Dushanbe | City of Islamabad
City of Edinburgh | City of Kampala
City of Faro | City of Porto
City of Genoa | City of Turin
City of Gothenburg | City of Malmo
City of Guayaquil | City of Quito
City of Halifax | City of Beirut
City of Halifax | City of Lisbon
City of Hanoi | City of Jakarta
City of Harare | City of Naples
City of Havana | City of Luanda
City of Ho Chi Minh City | City of Phnom Penh
City of Hong Kong | City of Shanghai
City of Houston | City of Quito
City of Islamabad | City of Kabul
City of Istanbul | City of Kolkata
City of Istanbul | City of Riyadh
City of Jerusalem | City of Beijing
City of Johannesburg | City of Nairobi
City of Johannesburg | City of Windhoek
City of Kabul | City of Dushanbe
City of Kampala | City of Dakar
City of Kampala | City of Maputo
City of Karachi | City of Budapest
City of Karachi | City of Khartoum
City of Kathmandu | City of Maputo
City of Khartoum | City of Cairo
City of Kyiv | City of Dubai
City of Kyiv | City of Khartoum
City of La Paz | City of Mexico City
City of Lima | City of Boston
City of Lima | City of Phoenix
City of Lisbon | City of Santiago
City of Los Angeles | City of Mexico City
City of Luanda | City of Budapest
City of Madrid | City of Brussels
City of Manila | City of Dhaka
City of Manila | City of Perth
City of Maputo | City of Delhi
City of Maputo | City of Paris
City of Marseille | City of Berlin
City of Melbourne | City of Auckland
City of Miami | City of Houston
City of Miami | City of Panama City
City of Miami | City of Seattle
City of Milan | City of Oslo
City of Minsk | City of Delhi
City of Montevideo | City of Buenos Aires
City of Montevideo | City of La Paz
City of Montreal | City of Vancouver
City of Mumbai | City of Helsinki
City of Mumbai | City of London
City of Mumbai | City of Pune
City of Munich | City of Dakar
City of Munich | City of New York
City of Nairobi | City of Paris
City of Naples | City of Baku
City of Naples | City of Moscow
City of New York | City of Chicago
City of Nice | City of Genoa
City of Odesa | City of Chisinau
City of Panama City | City of Lima
City of Phoenix | City of Bogota
City of Prague | City of Edinburgh
City of Prague | City of Madrid
City of Quebec City | City of Ottawa
City of Quito | City of Sao Paulo
City of Recife | City of Fortaleza
City of Riga | City of New York
City of Rio De Janeiro | City of Curitiba
City of Rio De Janeiro | City of Panama City
City of Riyadh | City of Vilnius
City of Rome | City of Kathmandu
City of Rome | City of Luanda
City of Sarajevo | City of Bratislava
City of Seoul | City of Melbourne
City of Singapore | City of Kuala Lumpur
City of Singapore | City of Sydney
City of Stockholm | City of Naples
City of Stockholm | City of Warsaw
City of Tallinn | City of St Petersburg
City of Tashkent | City of Bangkok
City of Tbilisi | City of Marseille
City of Tehran | City of Addis Ababa
City of Tehran | City of Shanghai
City of Tijuana | City of Los Angeles
City of Tijuana | City of San Diego
City of Toronto | City of San Francisco
City of Tripoli | City of Sofia
City of Tunis | City of Tashkent
City of Valparaiso | City of Santiago
City of Vancouver | City of Winnipeg
City of Venice | City of Jerusalem
City of Venice | City of Yerevan
City of Vienna | City of Karachi
City of Vienna | City of Tunis
City of Vilnius | City of Valencia
City of Warsaw | City of Lyon
City of Warsaw | City of San Diego
City of Washington | City of Havana
City of Wellington | City of Brisbane
City of Windhoek | City of Prague
City of Windhoek | City of Tripoli
City of Windsor | City of Detroit
City of Winnipeg | City of Valencia
City of Yerevan | City of Minsk
City of Zagreb | City of Baghdad
City of Zagreb | City of Kolkata
City of Zurich | City of Damascus
City of Zurich | City of Sao Paulo
