"""
geographic_kg_wikidata.py — Dynamic Knowledge Graph (Coordinate Prediction)
========================================================================
Queries Wikidata dynamically for city semantics.
CRITICAL: Coordinates are hidden from the evidence block to force the LLM to predict them.
"""

import json
import os
import time
import requests
from typing import Optional

class DynamicWikidataCoordGraph:
    def __init__(self, cache_path: str = "results/wikidata_coord_cache.json"):
        """Initialise the graph with a path to a local JSON cache file."""
        self.cache_file = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load the on-disk JSON cache, returning an empty dict if the file is absent or corrupt."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persist the in-memory cache dict to the JSON cache file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _fetch_wikidata(self, city_name: str) -> Optional[dict]:
        """Query Wikidata for semantic hierarchy data for a city, using the local cache when available."""
        if not city_name or city_name.lower() == "nan": return None
        if city_name in self.cache: return self.cache[city_name]

        headers = {"User-Agent": "Geomatics-Eval/1.0"}

        try:
            # 1. Search for Q-ID
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {"action": "wbsearchentities", "search": city_name, "language": "en", "format": "json", "limit": 1}
            time.sleep(1.1)
            search_res = requests.get(search_url, params=search_params, headers=headers)
            search_res.raise_for_status()
            
            search_data = search_res.json()
            if not search_data.get("search"): return None
            
            q_id = search_data["search"][0].get("id")
            description = search_data["search"][0].get("description", "No description")

            # 2. SPARQL query for Hierarchy (Coordinates are NOT fetched for the prompt)
            sparql_url = "https://query.wikidata.org/sparql"
            query = f"""
            SELECT ?instanceLabel ?locatedInLabel WHERE {{
              OPTIONAL {{ wd:{q_id} wdt:P31 ?instance . ?instance rdfs:label ?instanceLabel . FILTER(LANG(?instanceLabel) = "en") }}
              OPTIONAL {{ wd:{q_id} wdt:P131 ?locatedIn . ?locatedIn rdfs:label ?locatedInLabel . FILTER(LANG(?locatedInLabel) = "en") }}
            }} LIMIT 15
            """
            
            max_retries = 3
            sparql_res = None
            
            for attempt in range(max_retries):
                try:
                    sparql_res = requests.get(sparql_url, params={"query": query, "format": "json"}, headers=headers, timeout=15)
                    sparql_res.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise e

            bindings = sparql_res.json().get("results", {}).get("bindings", [])
            instances, locations = set(), set()

            for b in bindings:
                if "instanceLabel" in b: instances.add(b["instanceLabel"]["value"])
                if "locatedInLabel" in b: locations.add(b["locatedInLabel"]["value"])

            extracted = {
                "description": description,
                "instances": list(instances),
                "located_in": list(locations)
            }
            
            self.cache[city_name] = extracted
            self._save_cache()
            return extracted
            
        except Exception as e:
            print(f"\n❌ Wikidata API Error for '{city_name}': {e}")
            return None

    def build_evidence_block(self, city_name: str) -> str:
        """Returns semantic data but absolutely NO coordinates."""
        data = self._fetch_wikidata(city_name)
        if not data: 
            return f"=== WIKIDATA EVIDENCE ===\nCity: {city_name} | Data unavailable"

        return (f"=== WIKIDATA EVIDENCE ===\n"
                f"City: {city_name}\n"
                f"  • Description: {data['description']}\n"
                f"  • Instance of: {', '.join(data['instances'])}\n"
                f"  • Located in (Hierarchy): {', '.join(data['located_in'])}")