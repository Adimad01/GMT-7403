"""
osm_client.py — shared OpenStreetMap (Nominatim) client + geometry helpers.
================================================================================
Used by all three reasoning domains (Topological, Cardinal, Relative) so the
Nominatim fetch + on-disk cache logic lives in one place.

IMPORTANT — the GPU server has NO live internet access.  OSM evidence at
inference therefore relies on a pre-warmed cache (results/osm_cache.json).
Warm the cache locally (where there IS internet) with warm_osm_cache.py and
commit the JSON before pushing to the server.  When an entity is missing from
the cache and no network is available, fetch() returns None and the evidence
block degrades gracefully to "No OSM data available."

Exports:
  OSMClient          — fetch(place) -> dict|None, cache-backed
  OSMEvidenceKG      — generic evidence-provider KG (coords/bbox/hierarchy/
                       distance/bearing/offset).  Cardinal & Relative use this.
  NullKG             — drop-in KG returning empty evidence (kg-mode = none)
  haversine_km, bearing_deg, compass8, latlon_offset_phrase — geometry helpers
"""

import os
import json
import time
from math import radians, degrees, sin, cos, sqrt, atan2
from typing import Optional

# NB: `requests` is imported lazily inside OSMClient.fetch() so the module
# (geometry helpers, NullKG, cache filters, offline use) imports fine even
# where requests isn't installed.


# ---------------------------------------------------------------------------
# US state hint (improves Nominatim disambiguation for "<place> CA" style names)
# ---------------------------------------------------------------------------
US_STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
    'MX': 'Mexico',
}


def extract_state_hint(place_name: str) -> Optional[str]:
    """Extract a US state abbreviation from a name like 'San Jose CA'."""
    if not place_name:
        return None
    parts = place_name.strip().split()
    if parts and parts[-1].upper() in US_STATE_NAMES:
        return parts[-1].upper()
    return None


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing (degrees, 0–360) from point 1 to point 2."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


_COMPASS8 = [
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
]


def compass8(bearing: float) -> str:
    """Map a bearing (0–360°) to one of 8 compass labels."""
    idx = int((bearing + 22.5) % 360 // 45)
    return _COMPASS8[idx]


def latlon_offset_phrase(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> str:
    """Verbalize A's position relative to B as 'X km north and Y km east'."""
    ns_km = haversine_km(lat_b, lon_b, lat_a, lon_b)
    ew_km = haversine_km(lat_a, lon_b, lat_a, lon_a)
    ns_word = "north" if lat_a >= lat_b else "south"
    ew_word = "east" if lon_a >= lon_b else "west"
    return f"~{ns_km:.0f} km {ns_word} and ~{ew_km:.0f} km {ew_word}"


# ---------------------------------------------------------------------------
# NOMINATIM CLIENT (cache-backed)
# ---------------------------------------------------------------------------
class OSMClient:
    def __init__(self, cache_file: str = "results/osm_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def fetch(self, place_name: str, allow_network: bool = True) -> Optional[dict]:
        """Return cached OSM data for a place, querying Nominatim on a cache miss.

        allow_network=False forces cache-only operation (use on the offline
        server so a missing entity degrades to None instead of hanging).
        """
        if not place_name or place_name.lower() == "nan":
            return None
        if place_name in self.cache:
            return self.cache[place_name]
        if not allow_network:
            return None

        state_hint = extract_state_hint(place_name)
        query = f"{place_name}, {US_STATE_NAMES[state_hint]}" if state_hint else place_name

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": 5,
            "countrycodes": "us,mx",
        }
        headers = {"User-Agent": "LavalUniversity-GeomaticsPhDEval/1.0"}

        try:
            import requests  # lazy: only needed for a live cache-miss query
            time.sleep(1.1)  # Nominatim usage policy: ≤ 1 request/second
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data:
                preferred = ['boundary', 'place', 'highway', 'waterway', 'natural']
                result = next((r for r in data if r.get('class') in preferred), data[0])
                extracted = {
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "boundingbox": result.get("boundingbox"),
                    "osm_type": result.get("osm_type"),
                    "class": result.get("class"),
                    "type": result.get("type"),
                    "hierarchy": result.get("address", {}),
                }
                self.cache[place_name] = extracted
                self._save_cache()
                return extracted
            self.cache[place_name] = None
            self._save_cache()
            return None
        except Exception as e:
            print(f"⚠️ OSM API Error for '{place_name}': {e}")
            return None


# ---------------------------------------------------------------------------
# GENERIC EVIDENCE-PROVIDER KG  (used by Cardinal & Relative)
# ---------------------------------------------------------------------------
class OSMEvidenceKG:
    """Verbalizes whatever OSM facts are available for two entities.

    It does NOT decide the answer label — it surfaces coordinates, bounding
    boxes, administrative hierarchy, centroid distance, the initial bearing
    A→B (with an 8-point compass reading) and a north/east offset phrase, then
    lets the LLM reason.  Designed for the Cardinal and Relative tasks where the
    answer depends on a reference frame the coordinates alone cannot fix.
    """

    def __init__(self, cache_file: str = "results/osm_cache.json", allow_network: bool = True):
        self.client = OSMClient(cache_file)
        self.allow_network = allow_network

    # exposed for the per-step RAG loop
    def fetch(self, place_name: str) -> Optional[dict]:
        return self.client.fetch(place_name, allow_network=self.allow_network)

    @staticmethod
    def _entity_block(name: str, data: Optional[dict], label: str) -> str:
        if not data:
            return f"{label} ({name}): No OSM data available."
        lines = [f"{label}: {name}"]
        lines.append(f"  • Coordinates: Lat {data.get('lat')}, Lon {data.get('lon')}")
        bbox = data.get('boundingbox')
        if bbox and len(bbox) == 4:
            lines.append(
                f"  • Bounding Box: [South: {bbox[0]}, North: {bbox[1]}, "
                f"West: {bbox[2]}, East: {bbox[3]}]"
            )
        lines.append(f"  • Feature Category: {data.get('class')} / {data.get('type')}")
        hierarchy = data.get('hierarchy')
        if hierarchy:
            hier_str = " > ".join(f"{k}: {v}" for k, v in hierarchy.items())
            lines.append(f"  • Administrative Hierarchy: {hier_str}")
        return "\n".join(lines)

    def relation_facts(self, name_a: str, data_a: Optional[dict],
                       name_b: str, data_b: Optional[dict]) -> str:
        """Computed geometric facts between A and B (bearing, distance, offset)."""
        if not (data_a and data_b):
            return ""
        try:
            lat_a, lon_a = float(data_a['lat']), float(data_a['lon'])
            lat_b, lon_b = float(data_b['lat']), float(data_b['lon'])
        except (TypeError, ValueError, KeyError):
            return ""
        dist = haversine_km(lat_a, lon_a, lat_b, lon_b)
        brg = bearing_deg(lat_b, lon_b, lat_a, lon_a)  # bearing FROM B TO A
        comp = compass8(brg)
        offset = latlon_offset_phrase(lat_a, lon_a, lat_b, lon_b)
        return (
            "\n--- Computed Geometry (from OSM coordinates) ---\n"
            f"  • Centroid distance: {dist:.1f} km\n"
            f"  • Bearing from {name_b} to {name_a}: {brg:.0f}° ({comp})\n"
            f"  • {name_a} is {offset} of {name_b}\n"
            "  (These facts describe absolute map geometry. The reference frame "
            "for any observer-relative answer must still come from the text.)"
        )

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "",
                        entity: dict = None, log_fn=None) -> str:
        data_a = self.fetch(place_a)
        data_b = self.fetch(place_b)

        lines = [f'Sentence: "{sentence}"\n', "--- OpenStreetMap (OSM) Evidence ---"]
        lines.append(
            "Use the coordinates, bounding boxes, hierarchy and computed geometry "
            "below as supporting evidence. Do not invent facts.\n"
        )
        lines.append(self._entity_block(place_a, data_a, "Entity A"))
        lines.append("")
        lines.append(self._entity_block(place_b, data_b, "Entity B"))
        facts = self.relation_facts(place_a, data_a, place_b, data_b)
        if facts:
            lines.append(facts)

        text = "\n".join(lines)
        if log_fn:
            log_fn(text)
        return text


# ---------------------------------------------------------------------------
# GEOCODABILITY FILTER  (drop OSM-retrieval failures from eval / training)
# ---------------------------------------------------------------------------
def load_cache(path: str = "results/osm_cache.json") -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def is_geocodable(cache: dict, *names: str) -> bool:
    """True only if every name resolved to non-null OSM data in the cache.
    Treats missing-from-cache and cached-null both as a retrieval failure."""
    for n in names:
        n = (n or "").strip()
        if not n or cache.get(n) is None:
            return False
    return True


# ---------------------------------------------------------------------------
# NULL KG  (kg-mode = none)
# ---------------------------------------------------------------------------
class NullKG:
    """Drop-in KG that returns no evidence — used for Exp 1, 2, 3."""

    def fetch(self, place_name: str):
        return None

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "",
                        entity: dict = None, log_fn=None) -> str:
        return ""
