"""Resolve a place name to the OSM object it actually means.

Nominatim's free-text search ranks by textual relevance, not by whether the
result is the kind of thing you asked for. Querying "State of Colorado"
returns a clothes shop in Bischofswerda, and "Loch Ness" a cycleway in
Florida. Taking result [0] is how the project's cache ended up with 59 places
resolved to highways.

The fix is to say what kind of object is wanted, ask for several candidates,
and choose among them:

  - a name like "State of X" or "City of X" carries its own type hint, so the
    prefix is stripped for the query and kept as the requirement;
  - candidates are scored on whether their OSM class matches that requirement,
    then on Nominatim's own importance;
  - a name with no candidate of the right kind resolves to nothing, which is a
    far better outcome than resolving to the wrong continent.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

UA = "spatial-eval-datacheck/1.0 (research; contact via repository)"
ENDPOINT = "https://nominatim.openstreetmap.org/search?"

# name pattern -> acceptable (class, type) combinations, best first
PREFIX_RULES = [
    (re.compile(r"^(State|Commonwealth|Province) of\s+(.+)$", re.I),
     [("boundary", "administrative")]),
    (re.compile(r"^(City|Town|Municipality|Village) of\s+(.+)$", re.I),
     [("boundary", "administrative"), ("place", None)]),
    (re.compile(r"^(Borough|County|District|Canton) of\s+(.+)$", re.I),
     [("boundary", "administrative")]),
    (re.compile(r"^(.+?)\s+(County|Parish|Borough)$", re.I),
     [("boundary", "administrative")]),
]
KEYWORD_RULES = [
    (re.compile(r"\b(Lake|Loch|Reservoir|Lagoon)\b", re.I),
     [("natural", "water"), ("water", None), ("waterway", None), ("place", None)]),
    (re.compile(r"\b(River|Creek|Canal|Stream)\b", re.I),
     [("waterway", None), ("natural", "water"), ("water", None)]),
    (re.compile(r"\b(Sea|Ocean|Bay|Gulf|Strait|Channel|Sound)\b", re.I),
     [("natural", None), ("place", "sea"), ("water", None), ("waterway", None)]),
    (re.compile(r"\b(Mountain|Mount|Peak|Range|Massif|Sierra)\b", re.I),
     [("natural", None), ("place", None)]),
    (re.compile(r"\b(Desert|Forest|Rainforest|Steppe|Tundra)\b", re.I),
     [("natural", None), ("place", None), ("landuse", None), ("boundary", None)]),
    (re.compile(r"\b(Island|Isle|Archipelago|Peninsula|Cape)\b", re.I),
     [("place", None), ("natural", None), ("boundary", "administrative")]),
    (re.compile(r"\b(National Park|State Park|Reserve)\b", re.I),
     [("leisure", None), ("boundary", "protected_area"), ("landuse", None)]),
]
DEFAULT_ACCEPT = [("boundary", "administrative"), ("place", None),
                  ("natural", None), ("waterway", None)]


def requirements(name: str) -> tuple[str, list[tuple[str, str | None]]]:
    """(query string, acceptable class/type pairs in preference order)."""
    for pat, accept in PREFIX_RULES:
        m = pat.match(name.strip())
        if m:
            q = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            if pat.pattern.startswith("^(.+?)"):
                q = m.group(1) + " " + m.group(2)
            return q.strip(), accept
    for pat, accept in KEYWORD_RULES:
        if pat.search(name):
            return name.strip(), accept
    return name.strip(), DEFAULT_ACCEPT


def acceptable(cand: dict, accept: list[tuple[str, str | None]]) -> bool:
    return any(cand.get("class") == cls and (typ is None or cand.get("type") == typ)
               for cls, typ in accept)


def importance(cand: dict) -> float:
    return float(cand.get("importance") or 0)


def resolve(name: str, want_polygon: bool = True, pause: float = 1.1,
            simplify: float = 0.005) -> dict | None:
    q, accept = requirements(name)
    params = {"q": q, "format": "json", "limit": 12, "addressdetails": 1,
              "extratags": 1}
    if want_polygon:
        params["polygon_geojson"] = 1
        # full-detail national outlines run to megabytes; simplifying keeps the
        # topology while making a few hundred fetches tractable
        params["polygon_threshold"] = simplify
    req = urllib.request.Request(ENDPOINT + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})
    time.sleep(pause)
    try:
        cands = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception:
        return None
    if not cands:
        return None
    # The class list decides which candidates are the right KIND of thing; among
    # those, Nominatim's own importance decides which one is meant. Ranking by
    # class order instead would prefer a pond in Denmark that happens to match a
    # narrower class over the Scottish loch everyone means.
    fit = [c for c in cands if acceptable(c, accept)]
    if not fit:
        return None                    # nothing of the requested kind
    return max(fit, key=importance)
