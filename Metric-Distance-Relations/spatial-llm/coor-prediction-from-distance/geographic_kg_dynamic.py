"""
geographic_kg_dynamic.py — Dynamic Metric Distance Knowledge Graph Builder
========================================================================
Queries OpenStreetMap (Nominatim) dynamically for city coordinates.
Anchor triangulation has been completely removed to force pure LLM math deduction.
"""

import json
import os
import time
import requests
from typing import Optional

class DynamicMetricKnowledgeGraph:
    def __init__(self, cache_path: str = "results/osm_metric_cache.json"):
        self.cache_file = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load the OSM coordinate cache from disk, returning an empty dict on failure."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persist the current in-memory cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _fetch_osm_data(self, city_name: str) -> Optional[dict]:
        """Fetch lat/lon and state for a city from the Nominatim OSM API, with caching."""
        if not city_name or city_name.lower() == "nan": return None
        if city_name in self.cache: return self.cache[city_name]

        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name + ", USA", "format": "json", "addressdetails": 1, "limit": 1}
        headers = {"User-Agent": "Geomatics-DistanceEval/1.0"}

        try:
            time.sleep(1.1) # Respect OSM limit of 1 request per second
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data:
                extracted = {
                    "lat": float(data[0].get("lat")),
                    "lon": float(data[0].get("lon")),
                    "state": data[0].get("address", {}).get("state", "Unknown")
                }
                self.cache[city_name] = extracted
                self._save_cache()
                return extracted
            return None
        except Exception as e:
            print(f"⚠️ OSM API Error for '{city_name}': {e}")
            return None

    def build_evidence_block(self, city_a: str, city_b: str) -> str:
        """Build a plain-text coordinate evidence block for two cities using OSM data."""
        data_a = self._fetch_osm_data(city_a)
        data_b = self._fetch_osm_data(city_b)

        lines = ["=== CITY INFORMATION ==="]
        def format_city(name, data):
            """Format a single city's coordinate info as a one-line string."""
            if not data: return f"City: {name} | Data unavailable"
            return f"City: {name} | Lat: {data['lat']:.4f}°  Lon: {data['lon']:.4f}° | State: {data['state']}"

        lines.append(format_city(city_a, data_a))
        lines.append(format_city(city_b, data_b))
        
        if data_a and data_b:
            lines.append(f"Same state: {'Yes' if data_a['state'] == data_b['state'] else 'No'}")

        return "\n".join(lines)