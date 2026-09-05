"""Offline truth check for cardinal rows.

The schema validator can tell you a row is well-formed; it cannot tell you the
row is FALSE. Geocoding every batch through Nominatim is slow and rate-limited,
and the OSM cache is currently known-wrong for a quarter of cardinal pairs, so
neither is a trustworthy referee right now.

This uses a fixed table of city centroids instead. It only covers well-known
cities, but that is exactly what these batches are made of, and it runs in a
second with no network.

    python3 data_generation/check_cardinal_truth.py new_cardinal.csv

For each row it computes the initial great-circle bearing from the TARGET to
the SOURCE (the row asserts "source is <label> of target") and compares the
resulting compass sector to the stated label.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# lat, lon  (east-positive)
COORDS = {
    "accra": (5.55, -0.20), "addis ababa": (9.03, 38.74), "algiers": (36.75, 3.06),
    "almaty": (43.24, 76.89), "anchorage": (61.22, -149.90), "asuncion": (-25.28, -57.63),
    "athens": (37.98, 23.73), "atlanta": (33.75, -84.39), "auckland": (-36.85, 174.76),
    "baghdad": (33.31, 44.36), "baku": (40.41, 49.87), "bangkok": (13.76, 100.50),
    "beijing": (39.90, 116.41), "beirut": (33.89, 35.50), "berlin": (52.52, 13.40),
    "bogota": (4.71, -74.07), "boston": (42.36, -71.06), "brisbane": (-27.47, 153.03),
    "buenos aires": (-34.60, -58.38), "cairo": (30.04, 31.24), "calgary": (51.05, -114.07),
    "cape town": (-33.92, 18.42), "caracas": (10.48, -66.90), "casablanca": (33.57, -7.59),
    "chennai": (13.08, 80.27), "chicago": (41.88, -87.63), "colombo": (6.93, 79.86),
    "dakar": (14.72, -17.47), "dallas": (32.78, -96.80), "damascus": (33.51, 36.29),
    "delhi": (28.70, 77.10), "denver": (39.74, -104.99), "dhaka": (23.81, 90.41),
    "dubai": (25.20, 55.27), "dublin": (53.35, -6.26), "edinburgh": (55.95, -3.19),
    "guayaquil": (-2.17, -79.92), "halifax": (44.65, -63.57), "hanoi": (21.03, 105.85),
    "havana": (23.11, -82.37), "ho chi minh city": (10.82, 106.63), "hong kong": (22.32, 114.17),
    "honolulu": (21.31, -157.86), "houston": (29.76, -95.37), "istanbul": (41.01, 28.98),
    "jerusalem": (31.77, 35.21), "johannesburg": (-26.20, 28.05), "kabul": (34.56, 69.21),
    "kampala": (0.35, 32.58), "karachi": (24.86, 67.01), "khartoum": (15.50, 32.56),
    "kingston": (17.97, -76.79), "kyiv": (50.45, 30.52), "la paz": (-16.50, -68.15),
    "lagos": (6.52, 3.38), "lima": (-12.05, -77.04), "lisbon": (38.72, -9.14),
    "london": (51.51, -0.13), "los angeles": (34.05, -118.24), "luanda": (-8.84, 13.23),
    "madrid": (40.42, -3.70), "manila": (14.60, 120.98), "maputo": (-25.97, 32.57),
    "melbourne": (-37.81, 144.96), "miami": (25.76, -80.19), "montevideo": (-34.90, -56.16),
    "montreal": (45.50, -73.57), "moscow": (55.76, 37.62), "nairobi": (-1.29, 36.82),
    "new york": (40.71, -74.01), "oslo": (59.91, 10.75), "panama city": (8.98, -79.52),
    "paris": (48.86, 2.35), "perth": (-31.95, 115.86), "phnom penh": (11.56, 104.92),
    "port moresby": (-9.44, 147.18), "pyongyang": (39.04, 125.76), "quito": (-0.18, -78.47),
    "reykjavik": (64.15, -21.94), "rio de janeiro": (-22.91, -43.17), "riyadh": (24.71, 46.68),
    "rome": (41.90, 12.50), "san francisco": (37.77, -122.42), "san juan": (18.47, -66.11),
    "santiago": (-33.45, -70.67), "sao paulo": (-23.55, -46.63), "seattle": (47.61, -122.33),
    "seoul": (37.57, 126.98), "shanghai": (31.23, 121.47), "singapore": (1.35, 103.82),
    "stockholm": (59.33, 18.07), "suva": (-18.14, 178.44), "sydney": (-33.87, 151.21),
    "taipei": (25.03, 121.57), "tbilisi": (41.72, 44.79), "tehran": (35.69, 51.39),
    "thimphu": (27.47, 89.64), "tokyo": (35.68, 139.65), "toronto": (43.65, -79.38),
    "tripoli": (32.89, 13.19), "tunis": (36.81, 10.18), "ulaanbaatar": (47.89, 106.91),
    "vancouver": (49.28, -123.12), "vienna": (48.21, 16.37), "vientiane": (17.97, 102.60),
    "warsaw": (52.23, 21.01), "yangon": (16.87, 96.20), "yerevan": (40.18, 44.51),
    # Added from the ingested city list. Each position is taken from the
    # OpenStreetMap record for that place, not from recall.
    "aba": (47.0644, 18.5187),
    "agra": (27.1753, 78.0098),
    "aktau": (43.6645, 51.1688),
    "aktobe": (50.3456, 57.309),
    "al ain": (24.2249, 55.7452),
    "aleppo": (36.1992, 37.1637),
    "alofi": (-19.0555, -169.8893),
    "ambato": (-1.2573, -78.6144),
    "ambon": (-3.6763, 128.1781),
    "andijan": (40.7691, 72.3573),
    "apia": (-13.8342, -171.7629),
    "aqaba": (29.5266, 35.0075),
    "arecibo": (18.4067, -66.6752),
    "arequipa": (-16.3989, -71.537),
    "arusha": (-3.3697, 36.6881),
    "atyrau": (47.4679, 52.0729),
    "auki": (-1.2275, 136.3347),
    "bacolod": (10.6577, 122.9391),
    "bagan": (21.1317, 94.8624),
    "bago": (18.3055, 96.1108),
    "baguio": (16.4003, 120.5951),
    "balikpapan": (-1.1566, 116.8723),
    "balkanabat": (39.2808, 53.7793),
    "bandung": (-6.9193, 107.6366),
    "banjarmasin": (-3.3292, 114.5917),
    "bariloche": (-41.1269, -71.384),
    "barnaul": (53.333, 83.7573),
    "basra": (30.4952, 47.8091),
    "batumi": (41.6183, 41.6318),
    "bayamon": (18.3497, -66.1683),
    "beersheba": (31.257, 34.7884),
    "belem": (31.7065, 35.2037),
    "belmopan": (17.2486, -88.7636),
    "belo horizonte": (-19.9026, -43.96),
    "bhopal": (23.2585, 77.402),
    "bintulu": (3.1874, 113.0473),
    "boa vista": (3.1179, -60.7183),
    "bokhtar": (37.8382, 68.7801),
    "brasov": (45.6279, 25.5848),
    "bujumbura": (-3.3638, 29.3675),
    "bukavu": (-2.5092, 28.8398),
    "bukhara": (39.7748, 64.4403),
    "cagayan de oro": (8.4359, 124.6255),
    "caguas": (18.2119, -66.0508),
    "cajamarca": (-6.4324, -78.7459),
    "calabar": (4.9796, 8.3374),
    "camagüey": (21.4337, -77.9192),
    "campo grande": (-20.9133, -54.2496),
    "cap-haitien": (19.7595, -72.2008),
    "cebu city": (10.3739, 123.8606),
    "chandigarh": (30.7334, 76.7797),
    "chiang mai": (18.7938, 98.9922),
    "chiang rai": (19.8474, 99.866),
    "chiclayo": (-6.7716, -79.8387),
    "cluj-napoca": (46.7828, 23.6102),
    "cobija": (-11.0185, -68.7538),
    "cochabamba": (-17.4252, -66.1688),
    "coimbatore": (11.0018, 76.9628),
    "constanta": (44.1842, 28.5988),
    "cordoba, argentina": (-31.4167, -64.1837),
    "cuenca, ecuador": (-2.8974, -79.0042),
    "cuiaba": (-15.5916, -55.8163),
    "cusco": (-13.5171, -71.9785),
    "dammam": (26.4368, 50.104),
    "dangriga": (16.9606, -88.2375),
    "davao city": (7.2508, 125.4374),
    "david, panama": (8.451, -82.4237),
    "daşoguz": (41.8254, 59.9215),
    "denpasar": (-8.8189, 115.3038),
    "derbent": (37.9918, 31.9982),
    "dodoma": (-6.1525, 35.7621),
    "eilat": (29.5797, 34.9382),
    "eldoret": (0.5198, 35.2715),
    "enugu": (6.5327, 7.439),
    "erbil": (36.1912, 44.0094),
    "esmeraldas": (-19.7329, -44.3081),
    "faisalabad": (31.4308, 73.0977),
    "fergana": (40.3943, 71.7854),
    "freeport": (42.2892, -89.6347),
    "funafuti": (-8.5707, 179.1232),
    "ganja": (40.6747, 46.3566),
    "garissa": (-0.4886, 40.1989),
    "general santos": (6.1049, 125.1395),
    "goiania": (-16.6436, -49.2738),
    "goma": (-1.6464, 29.1881),
    "granada, nicaragua": (11.9332, -85.8618),
    "grozny": (43.3306, 45.651),
    "guantanamo": (20.1019, -75.1629),
    "gwadar": (25.2231, 62.3043),
    "gyumri": (40.7946, 43.8362),
    "hagåtña": (13.4769, 144.7494),
    "haifa": (32.8062, 35.0085),
    "hama": (35.14, 36.7587),
    "hat yai": (7.01, 100.4724),
    "holguin": (20.8509, -75.7049),
    "homs": (34.7333, 36.7167),
    "honiara": (-9.4308, 159.9688),
    "hua hin": (12.5445, 99.9495),
    "huancayo": (-12.0681, -75.2101),
    "hyderabad, pakistan": (25.4014, 68.3665),
    "iasi": (47.1575, 27.589),
    "ibadan": (7.3784, 3.8972),
    "ibarra": (43.1425, -2.049),
    "iloilo city": (10.7105, 122.5541),
    "ilorin": (8.4964, 4.548),
    "imperatriz": (-5.3394, -47.5746),
    "indore": (22.7204, 75.8682),
    "ipoh": (4.6539, 101.1555),
    "iquitos": (-3.7494, -73.2444),
    "irbid": (32.5556, 35.8493),
    "irkutsk": (52.3133, 104.2732),
    "jacmel": (18.2353, -72.5375),
    "jaipur": (26.9155, 75.819),
    "jalal-abad": (41.5682, 72.4587),
    "jeddah": (21.5504, 39.1742),
    "johor bahru": (1.4582, 103.7649),
    "jos": (9.9175, 8.8979),
    "juliaca": (-15.4932, -70.1356),
    "kaduna": (10.3735, 7.7114),
    "kakamega": (0.4037, 34.7447),
    "kananga": (-5.8952, 22.4086),
    "kano": (11.994, 8.522),
    "kanpur": (26.4609, 80.3218),
    "karakol": (42.5014, 78.3827),
    "karbala": (32.5878, 44.0255),
    "khabarovsk": (48.4743, 135.0628),
    "khiva": (41.3922, 60.3515),
    "khon kaen": (16.4087, 102.5782),
    "khorugh": (37.4893, 71.5635),
    "khujand": (40.2946, 69.6176),
    "kigali": (-1.937, 30.1232),
    "kigoma": (-4.59, 30.561),
    "kikwit": (-5.0383, 18.8178),
    "kisangani": (0.5184, 25.2057),
    "kisumu": (-0.0859, 34.7479),
    "kochi": (9.9679, 76.2444),
    "kokshetau": (53.308, 69.31),
    "kolonia": (50.9456, 6.9737),
    "kolwezi": (-10.717, 25.467),
    "koror": (7.2762, 134.1408),
    "kota kinabalu": (5.978, 116.0729),
    "krasnoyarsk": (56.0211, 92.8858),
    "kuching": (1.5598, 110.3454),
    "kulob": (37.9081, 69.7739),
    "kumasi": (6.6986, -1.6233),
    "kutaisi": (42.254, 42.6873),
    "kyzylorda": (45.1934, 63.6334),
    "la romana": (38.3551, -0.9078),
    "lashio": (22.9542, 97.7435),
    "latakia": (35.572, 36.0192),
    "leon, nicaragua": (12.3128, -86.9367),
    "les cayes": (18.2619, -73.7633),
    "liberia, costa rica": (10.5866, -85.4503),
    "likasi": (-10.9892, 26.7397),
    "loja": (37.155, -4.1856),
    "lubumbashi": (-11.6503, 27.4602),
    "lucknow": (26.8381, 80.9346),
    "luganville": (-15.5121, 167.1784),
    "machakos": (-1.2795, 37.4122),
    "machala": (-3.3197, -79.9474),
    "madurai": (9.9126, 78.102),
    "maiduguri": (11.8391, 13.1266),
    "majuro": (7.163, 171.1825),
    "makassar": (-5.1356, 119.4638),
    "makhachkala": (42.9835, 47.4636),
    "malacca": (2.215, 102.2304),
    "malindi": (-6.1576, 39.1956),
    "manado": (1.5246, 124.8482),
    "manaus": (-2.621, -60.2586),
    "mandalay": (21.9597, 96.0949),
    "manta": (44.6135, 7.5028),
    "maraba": (-5.6298, -50.0169),
    "mary": (46.6102, 4.5003),
    "matadi": (-5.8257, 13.4609),
    "mawlamyine": (16.4908, 97.6285),
    "mayaguez": (18.176, -67.3283),
    "mbandaka": (0.0471, 18.2565),
    "mbeya": (-8.1603, 33.7575),
    "mbuji-mayi": (-6.1259, 23.5998),
    "mecca": (21.4471, 39.9888),
    "medan": (3.6291, 98.6672),
    "medina": (29.3551, -99.1101),
    "melekeok": (7.4852, 134.7328),
    "mendoza": (-34.6301, -68.5826),
    "meru": (49.2453, 2.1353),
    "miri": (4.394, 113.988),
    "mombasa": (-4.0505, 39.6672),
    "monywa": (22.1182, 95.1325),
    "morogoro": (-7.9228, 36.9891),
    "moshi": (-3.3486, 37.3435),
    "multan": (30.1978, 71.472),
    "mwanza": (-2.5197, 32.9014),
    "naga city": (13.6431, 123.2587),
    "nagpur": (21.1465, 79.082),
    "najaf": (32.001, 44.33),
    "nakhchivan": (39.2151, 45.3608),
    "nakhon ratchasima": (14.9738, 102.0814),
    "nakuru": (-0.3048, 36.0825),
    "namangan": (41.0264, 71.6417),
    "naryn": (41.3663, 75.605),
    "nassau": (25.0782, -77.3383),
    "naypyidaw": (19.7753, 96.1033),
    "neuquen": (-38.641, -70.1192),
    "noumea": (-22.2555, 166.451),
    "novosibirsk": (54.9946, 82.9677),
    "nuku'alofa": (-21.1343, -175.2018),
    "nukus": (42.4548, 59.6221),
    "nyeri": (-0.3429, 36.9564),
    "ogbomosho": (8.133, 4.25),
    "omsk": (54.9779, 73.3695),
    "onitsha": (6.1462, 6.8019),
    "oruro": (-18.6337, -67.6936),
    "osh": (40.1318, 73.2281),
    "pago pago": (-14.249, -170.7141),
    "palembang": (-2.975, 104.7326),
    "palikir": (6.9207, 158.1627),
    "palmas": (-10.219, -48.1523),
    "papeete": (-17.5567, -149.5571),
    "pathein": (16.7833, 94.7333),
    "patna": (25.6004, 85.1187),
    "pattaya": (12.9246, 100.8825),
    "pavlodar": (52.0723, 76.2449),
    "penang": (5.3668, 100.399),
    "peshawar": (33.996, 71.5048),
    "piura": (-5.1241, -80.3377),
    "ponce": (18.0598, -66.6142),
    "port harcourt": (4.776, 7.0228),
    "port vila": (-17.7415, 168.315),
    "port-de-paix": (19.8501, -72.9286),
    "porto velho": (-9.1532, -64.3063),
    "portoviejo": (-1.0774, -80.4609),
    "potosi": (-19.5893, -65.7535),
    "pucallpa": (-8.3821, -74.5388),
    "puerto plata": (19.7145, -70.6927),
    "puntarenas": (9.0635, -84.008),
    "quetta": (30.141, 66.9817),
    "rawalpindi": (33.5915, 73.0537),
    "rio branco": (-10.0658, -68.3709),
    "riobamba": (-1.6724, -78.6625),
    "rosario": (-32.9476, -60.695),
    "rustavi": (41.5519, 44.993),
    "saipan": (15.1888, 145.7534),
    "salta": (-24.2992, -64.8145),
    "samarkand": (39.666, 66.9514),
    "san ignacio": (-5.1291, -78.9475),
    "san miguel, el salvador": (13.4369, -88.1574),
    "san pedro de macoris": (18.5518, -69.361),
    "sandakan": (5.8391, 118.1159),
    "santa ana, el salvador": (13.9774, -89.548),
    "santa clara": (37.2321, -121.6958),
    "santa cruz de la sierra": (-17.8089, -62.9949),
    "santarem": (39.337, -8.7292),
    "santiago de cuba": (20.0122, -75.7094),
    "santiago de los caballeros": (19.4985, -70.7358),
    "santo domingo, ecuador": (-0.2254, -79.1425),
    "semey": (50.4052, 80.251),
    "sharjah": (25.2307, 55.2872),
    "shymkent": (42.3056, 69.5826),
    "sibu": (2.2906, 111.8256),
    "sittwe": (20.1392, 92.8971),
    "srinagar": (34.0747, 74.8204),
    "sucre": (-19.0477, -65.2594),
    "sulaymaniyah": (35.5571, 45.4426),
    "sumqayit": (40.6891, 49.6817),
    "surat thani": (9.2782, 99.3836),
    "tabora": (-5.266, 32.8223),
    "tabuk": (27.8764, 37.2438),
    "takoradi": (4.8874, -1.7519),
    "talas": (42.4341, 72.1243),
    "tamale": (9.4052, -0.8424),
    "tanga": (-19.864, -175.1696),
    "tarawa": (11.59, 3.9703),
    "taraz": (42.8828, 71.3556),
    "tarija": (-21.5951, -63.8784),
    "taunggyi": (20.787, 97.0387),
    "tel aviv": (32.0847, 34.7899),
    "thiruvananthapuram": (8.4882, 76.9476),
    "timisoara": (45.7643, 21.202),
    "tomsk": (58.4911, 82.1421),
    "trinidad, bolivia": (-14.8349, -64.9045),
    "trujillo, peru": (-8.1117, -79.0288),
    "tshikapa": (-6.423, 20.7888),
    "tucuman": (-26.9469, -65.3631),
    "türkmenabat": (38.9899, 63.5583),
    "udon thani": (17.4245, 102.8627),
    "ushuaia": (-54.804, -68.3534),
    "uyo": (5.0082, 7.9166),
    "vanadzor": (40.8129, 44.4831),
    "varanasi": (25.3356, 83.0076),
    "vladikavkaz": (43.0364, 44.6719),
    "yakutsk": (62.0618, 129.7204),
    "yaren": (-0.5466, 166.9238),
    "yogyakarta": (-8.0147, 110.4071),
    "zamboanga city": (7.0964, 122.1939),
    "zaria": (11.0356, 7.6855),
    "zarqa": (32.0658, 36.0776),
    "abu dhabi": (24.45, 54.38), "darwin": (-12.46, 130.84),
    "doha": (25.29, 51.53), "ottawa": (45.42, -75.70), "portland": (45.52, -122.68), "amman": (31.95, 35.93), "amsterdam": (52.37, 4.90),
    "ankara": (39.93, 32.86), "antananarivo": (-18.88, 47.51), "ashgabat": (37.95, 58.38),
    "belgrade": (44.79, 20.45), "bergen": (60.39, 5.32), "birmingham": (52.49, -1.89),
    "bishkek": (42.87, 74.59), "bratislava": (48.15, 17.11), "brussels": (50.85, 4.35),
    "bucharest": (44.43, 26.10), "budapest": (47.50, 19.04), "cardiff": (51.48, -3.18),
    "chengdu": (30.57, 104.07), "chisinau": (47.01, 28.86), "cleveland": (41.50, -81.69),
    "copenhagen": (55.68, 12.57), "curitiba": (-25.43, -49.27), "dar es salaam": (-6.79, 39.21),
    "detroit": (42.33, -83.05), "dushanbe": (38.56, 68.79), "edmonton": (53.55, -113.49),
    "faro": (37.02, -7.93), "fortaleza": (-3.73, -38.52), "genoa": (44.41, 8.93),
    "gothenburg": (57.71, 11.97), "harare": (-17.83, 31.05), "helsinki": (60.17, 24.94),
    "islamabad": (33.68, 73.05), "jakarta": (-6.21, 106.85), "kathmandu": (27.72, 85.32),
    "kinshasa": (-4.44, 15.27), "kolkata": (22.57, 88.36), "kuala lumpur": (3.14, 101.69),
    "kuwait city": (29.38, 47.99), "las vegas": (36.17, -115.14), "lyon": (45.76, 4.84),
    "malmo": (55.60, 13.00), "managua": (12.11, -86.24), "marseille": (43.30, 5.37),
    "mexico city": (19.43, -99.13), "milan": (45.46, 9.19), "minneapolis": (44.98, -93.27),
    "minsk": (53.90, 27.57), "mogadishu": (2.05, 45.32), "mumbai": (19.08, 72.88),
    "munich": (48.14, 11.58), "muscat": (23.59, 58.41),
    "naples": (40.85, 14.27), "new orleans": (29.95, -90.07), "nice": (43.70, 7.27),
    "nicosia": (35.19, 33.38), "odesa": (46.48, 30.72),
    "osaka": (34.69, 135.50), "philadelphia": (39.95, -75.17), "phoenix": (33.45, -112.07),
    "porto": (41.15, -8.61), "prague": (50.08, 14.44),
    "pune": (18.52, 73.86), "quebec city": (46.81, -71.21), "recife": (-8.05, -34.88),
    "reno": (39.53, -119.81), "riga": (56.95, 24.11), "salvador": (-12.97, -38.51),
    "salt lake city": (40.76, -111.89), "san antonio": (29.42, -98.49),
    "san diego": (32.72, -117.16), "sapporo": (43.06, 141.35), "sarajevo": (43.86, 18.41),
    "sofia": (42.70, 23.32), "st petersburg": (59.93, 30.34), "surabaya": (-7.25, 112.75),
    "tallinn": (59.44, 24.75), "tashkent": (41.30, 69.24), "tijuana": (32.53, -117.02),
    "tirana": (41.33, 19.82), "turin": (45.07, 7.69), "valencia": (39.47, -0.38),
    "valparaiso": (-33.05, -71.62), "venice": (45.44, 12.32), "vilnius": (54.69, 25.28),
    "vladivostok": (43.12, 131.89), "washington": (38.91, -77.04), "wellington": (-41.29, 174.78),
    "windhoek": (-22.56, 17.08), "windsor": (42.31, -83.04), "winnipeg": (49.90, -97.14),
    "wuhan": (30.59, 114.31), "zagreb": (45.81, 15.98), "zurich": (47.38, 8.54),
}

SECTORS = ["north_of", "northeast_of", "east_of", "southeast_of",
           "south_of", "southwest_of", "west_of", "northwest_of"]


def key(name: str) -> str:
    n = name.strip().lower()
    for p in ("city of ", "the city of "):
        if n.startswith(p):
            n = n[len(p):]
    return n.strip()


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial great-circle bearing from a to b, degrees clockwise from north."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlo = lo2 - lo1
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return math.degrees(math.atan2(y, x)) % 360.0


def separation(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle angular distance in degrees."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


OPPOSITE = {"north_of": "south_of", "south_of": "north_of",
            "east_of": "west_of", "west_of": "east_of",
            "northeast_of": "southwest_of", "southwest_of": "northeast_of",
            "northwest_of": "southeast_of", "southeast_of": "northwest_of"}


def reciprocal(pa, pb) -> bool:
    """True when the reverse bearing lands in the opposite sector.

    Over long distances the great circle can bend enough that A reads north of
    B while B also reads north of A -- both routes cross the pole. Such a pair
    has no coherent direction and must never become an item.
    """
    fwd, _ = sector(bearing(pb, pa))
    rev, _ = sector(bearing(pa, pb))
    return OPPOSITE[fwd] == rev


def components_agree(pa, pb, label: str) -> bool:
    """True when the cone label agrees with the projection-based reading.

    The two standard qualitative models (cone-based sectors and Frank's
    projection-based half-planes) usually concur, but not always: a great
    circle that clips a pole can report 'north' for a city that is plainly at a
    lower latitude. Requiring the signs to match keeps every item correct under
    either model, and keeps it checkable by a reader with an atlas.
    """
    dlat = pa[0] - pb[0]
    dlon = (pa[1] - pb[1] + 180) % 360 - 180
    if "north" in label and dlat <= 0:
        return False
    if "south" in label and dlat >= 0:
        return False
    if "east" in label and dlon <= 0:
        return False
    if "west" in label and dlon >= 0:
        return False
    return True


def sector(deg: float) -> tuple[str, float]:
    """Compass sector plus distance in degrees to the nearest sector boundary."""
    idx = int((deg + 22.5) % 360 // 45)
    centre = idx * 45.0
    off = abs((deg - centre + 180) % 360 - 180)
    return SECTORS[idx], 22.5 - off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="flag rows whose bearing sits within this many degrees "
                         "of a sector boundary (default 5)")
    ap.add_argument("--far", type=float, default=140.0,
                    help="flag pairs separated by more than this many degrees, "
                         "where 'direction' stops being well defined (default 140)")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv_file).open(newline="", encoding="utf-8")))
    wrong, borderline, antipodal, unknown = [], [], [], []

    for i, r in enumerate(rows):
        ln = i + 2
        s, t = key(r["source_entity"]), key(r["target_entity"])
        lab = r["relation_label"].strip().lower()
        if s not in COORDS or t not in COORDS:
            unknown.append((ln, s if s not in COORDS else t))
            continue
        ps, pt = COORDS[s], COORDS[t]
        got, margin = sector(bearing(pt, ps))
        sep = separation(ps, pt)
        if got != lab:
            wrong.append((ln, r, lab, got, bearing(pt, ps)))
        elif sep > args.far:
            antipodal.append((ln, r, sep))
        elif margin < args.margin:
            borderline.append((ln, r, lab, margin))

    n = len(rows)
    print("=" * 78)
    print(f"  TRUTH CHECK  {Path(args.csv_file).name}   {n} rows")
    print("=" * 78)

    if unknown:
        print(f"\n  SKIPPED {len(unknown)} rows — city not in the coordinate table:")
        for ln, name in unknown[:15]:
            print(f"    line {ln}: {name}")

    if wrong:
        print(f"\n  FALSE — {len(wrong)} rows state the wrong direction:")
        for ln, r, lab, got, deg in wrong:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}")
            print(f"              says {lab}, actual bearing {deg:6.1f} deg = {got}")
    if antipodal:
        print(f"\n  UNSTABLE — {len(antipodal)} rows put the two places more than "
              f"{args.far:.0f} deg apart,")
        print("             where a single compass direction is not well defined:")
        for ln, r, sep in antipodal:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}"
                  f"   separation {sep:.0f} deg")
    if borderline:
        print(f"\n  BORDERLINE — {len(borderline)} rows sit within {args.margin:.0f} deg "
              f"of a sector boundary:")
        for ln, r, lab, m in borderline:
            print(f"    line {ln:>4}  {r['source_entity']} -> {r['target_entity']}"
                  f"   {lab}, {m:.1f} deg from the boundary")

    bad = len(wrong) + len(antipodal)
    print("\n" + "=" * 78)
    checked = n - len(unknown)
    print(f"  {checked - bad}/{checked} checked rows are sound   "
          f"({len(wrong)} false, {len(antipodal)} unstable, "
          f"{len(borderline)} borderline)")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
