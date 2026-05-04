"""
geographic_kg_wikidata.py — Dynamic Metric Distance Knowledge Graph Builder
========================================================================
Queries Wikidata dynamically for city coordinates and semantic hierarchy.
Includes robust retry logic for 502 Bad Gateway server errors.
"""

import json
import os
import time
import requests
from typing import Optional

class DynamicWikidataMetricGraph:
    def __init__(self, cache_path: str = "results/wikidata_metric_cache.json"):
        self.cache_file = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load the Wikidata coordinate cache from disk, returning an empty dict on failure."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persist the current in-memory Wikidata cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _fetch_wikidata(self, city_name: str) -> Optional[dict]:
        """Query Wikidata for coordinates and semantic hierarchy of a city, with caching and retry."""
        if not city_name or city_name.lower() == "nan": return None
        
        # 1. Return from cache if we already successfully pulled it
        if city_name in self.cache: 
            return self.cache[city_name]

        headers = {"User-Agent": "Geomatics-DistanceEval/1.0"}

        try:
            # 2. Search MediaWiki API for the correct Q-ID
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                "action": "wbsearchentities", 
                "search": city_name, 
                "language": "en", 
                "format": "json", 
                "limit": 1
            }
            time.sleep(1.1)  # Rate limiting safety
            search_res = requests.get(search_url, params=search_params, headers=headers)
            search_res.raise_for_status()
            
            search_data = search_res.json()
            if not search_data.get("search"): return None
            
            q_id = search_data["search"][0].get("id")
            description = search_data["search"][0].get("description", "No description")

            # 3. SPARQL query for coordinates and hierarchy (WITH RETRY LOGIC)
            sparql_url = "https://query.wikidata.org/sparql"
            query = f"""
            SELECT ?lat ?lon ?instanceLabel ?locatedInLabel WHERE {{
              OPTIONAL {{
                wd:{q_id} p:P625 ?statement .
                ?statement psv:P625 ?coordinate_node .
                ?coordinate_node wikibase:geoLatitude ?lat .
                ?coordinate_node wikibase:geoLongitude ?lon .
              }}
              OPTIONAL {{ wd:{q_id} wdt:P31 ?instance . ?instance rdfs:label ?instanceLabel . FILTER(LANG(?instanceLabel) = "en") }}
              OPTIONAL {{ wd:{q_id} wdt:P131 ?locatedIn . ?locatedIn rdfs:label ?locatedInLabel . FILTER(LANG(?locatedInLabel) = "en") }}
            }} LIMIT 15
            """
            
            max_retries = 3
            sparql_res = None
            
            for attempt in range(max_retries):
                try:
                    sparql_res = requests.get(
                        sparql_url, 
                        params={"query": query, "format": "json"}, 
                        headers=headers, 
                        timeout=15
                    )
                    sparql_res.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"\n  ⚠️ Wikidata Server busy for '{city_name}'. Retrying in {3 * (attempt + 1)}s...")
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise e  # Out of retries, fail gracefully

            # 4. Parse SPARQL Results
            bindings = sparql_res.json().get("results", {}).get("bindings", [])

            lat, lon = None, None
            instances, locations = set(), set()

            # Wikidata might return multiple rows if a city has multiple "located in" properties
            for b in bindings:
                if "lat" in b and lat is None: lat = float(b["lat"]["value"])
                if "lon" in b and lon is None: lon = float(b["lon"]["value"])
                if "instanceLabel" in b: instances.add(b["instanceLabel"]["value"])
                if "locatedInLabel" in b: locations.add(b["locatedInLabel"]["value"])

            if lat is None or lon is None: return None

            extracted = {
                "lat": lat, 
                "lon": lon, 
                "description": description,
                "instances": list(instances),
                "located_in": list(locations)
            }
            
            # 5. Save to Cache
            self.cache[city_name] = extracted
            self._save_cache()
            return extracted
            
        except Exception as e:
            print(f"\n❌ Final Wikidata API Error for '{city_name}': {e}")
            return None

    def build_evidence_block(self, city_a: str, city_b: str) -> str:
        """Constructs the exact text block that gets injected into the LLM prompt."""
        data_a = self._fetch_wikidata(city_a)
        data_b = self._fetch_wikidata(city_b)

        lines = ["=== WIKIDATA EVIDENCE ==="]
        
        def format_city(name, data):
            """Format a city's Wikidata info as a multi-line string for the evidence block."""
            if not data: return f"City: {name} | Data unavailable"
            return (f"City: {name}\n"
                    f"  • Description: {data['description']}\n"
                    f"  • Coordinates: Lat {data['lat']:.4f}°, Lon {data['lon']:.4f}°\n"
                    f"  • Instance of: {', '.join(data['instances'])}\n"
                    f"  • Located in: {', '.join(data['located_in'])}")

        lines.append(format_city(city_a, data_a))
        lines.append("")
        lines.append(format_city(city_b, data_b))

        return "\n".join(lines)