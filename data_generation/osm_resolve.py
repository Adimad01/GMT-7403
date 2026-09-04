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
import unicodedata
import urllib.parse
import urllib.request

UA = "spatial-eval-datacheck/1.0 (research; contact via repository)"


class LookupFailed(RuntimeError):
    """The query did not complete. Distinct from 'this place does not exist'.

    Collapsing the two is how a first pass recorded Mexico and California as
    unresolvable: their outlines are large, the request timed out, and the
    timeout was written down as a fact about the place.
    """
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

# When the caller knows what kind of object it wants, guessing from the name is
# unnecessary and unreliable: "Rhone" is both a river and a French departement,
# and the departement outranks the river on importance.
KIND_ACCEPT = {
    "admin": [("boundary", "administrative")],
    "city": [("boundary", "administrative"), ("place", None)],
    "river": [("waterway", None), ("natural", "water"), ("water", None)],
    "lake": [("natural", "water"), ("water", None), ("waterway", None),
             ("place", None)],
    "sea": [("natural", None), ("place", "sea"), ("water", None),
            ("waterway", None)],
    "park": [("leisure", None), ("boundary", "protected_area"),
             ("landuse", None), ("natural", None)],
    "island": [("place", None), ("natural", None),
               ("boundary", "administrative")],
    "physical": [("natural", None), ("place", None), ("landuse", None),
                 ("boundary", None)],
}


def requirements(name: str, kind: str | None = None) -> tuple[str, list[tuple[str, str | None]]]:
    """(query strings to try in order, acceptable class/type pairs).

    The full name is always tried first. Stripping the type prefix helps for
    "State of Colorado", but it is destructive for names where the prefix is
    part of the proper noun: "District of Columbia" reduces to "Columbia",
    which matches the country Colombia and produced a row asserting the two
    were the same place.
    """
    if kind:
        full = name.strip()
        stripped = re.sub(r"^(State|Commonwealth|Province|City|Town|Municipality|"
                          r"Village|Borough|County|District|Canton) of\s+", "",
                          full, flags=re.I)
        queries = [full] if stripped == full else [full, stripped]
        return queries, KIND_ACCEPT.get(kind, DEFAULT_ACCEPT)
    for pat, accept in PREFIX_RULES:
        m = pat.match(name.strip())
        if m:
            q = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            if pat.pattern.startswith("^(.+?)"):
                q = m.group(1) + " " + m.group(2)
            q = q.strip()
            return ([name.strip()] if q == name.strip()
                    else [name.strip(), q]), accept
    for pat, accept in KEYWORD_RULES:
        if pat.search(name):
            return [name.strip()], accept
    return [name.strip()], DEFAULT_ACCEPT


def norm(text: str) -> str:
    """Accent-folded, punctuation-free, lowercase form for name comparison."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def names_of(cand: dict) -> set[str]:
    """Every name the candidate goes by, normalised."""
    out = set()
    nd = cand.get("namedetails") or {}
    for k, v in nd.items():
        if k == "name" or k.startswith(("name:", "official_name", "alt_name",
                                        "short_name", "int_name")):
            if v:
                out.add(norm(v))
    head = (cand.get("display_name") or "").split(",")[0]
    if head:
        out.add(norm(head))
    return out


def bears_name(cand: dict, query: str) -> bool:
    """Does this candidate actually carry the name that was asked for?

    Nominatim matches fuzzily, so a search for Sahara returns New York State
    and a search for Lagos returns Laos. Both outrank the intended place on
    importance, so ranking by importance alone picks them. Importance may only
    break ties among candidates that genuinely bear the name.
    """
    return norm(query.split(",")[0]) in names_of(cand)


def acceptable(cand: dict, accept: list[tuple[str, str | None]]) -> bool:
    return any(cand.get("class") == cls and (typ is None or cand.get("type") == typ)
               for cls, typ in accept)


def importance(cand: dict) -> float:
    return float(cand.get("importance") or 0)


def resolve(name: str, want_polygon: bool = True, pause: float = 1.1,
            simplify: float = 0.005, timeout: int = 90,
            kind: str | None = None) -> dict | None:
    queries, accept = requirements(name, kind)
    for q in queries:
        hit = _try(q, accept, want_polygon, simplify, timeout, pause)
        if hit:
            return hit
    return None


def _try(q, accept, want_polygon, simplify, timeout, pause):
    params = {"q": q, "format": "json", "limit": 12, "addressdetails": 1,
              "extratags": 1, "namedetails": 1}
    if want_polygon:
        params["polygon_geojson"] = 1
        # full-detail national outlines run to megabytes; simplifying keeps the
        # topology while making a few hundred fetches tractable
        params["polygon_threshold"] = simplify
    req = urllib.request.Request(ENDPOINT + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})
    time.sleep(pause)
    try:
        cands = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as exc:
        raise LookupFailed(f"{q}: {exc}") from exc
    if not cands:
        return None
    # The class list decides which candidates are the right KIND of thing; among
    # those, Nominatim's own importance decides which one is meant. Ranking by
    # class order instead would prefer a pond in Denmark that happens to match a
    # narrower class over the Scottish loch everyone means.
    fit = [c for c in cands if acceptable(c, accept) and bears_name(c, q)]
    if not fit:
        return None            # nothing of the right kind that bears the name
    return max(fit, key=importance)
