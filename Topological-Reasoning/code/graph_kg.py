"""
graph_kg.py — GraphRAG over the OSM spatial graph (Exp 7).
================================================================================
Exp 6 ("kg-mode=rag") retrieves ONE place record at a time: the model emits
`NEXT_QUERY: <place>` and gets back a flat Nominatim blob.  Nothing connects the
records, so no multi-hop fact is ever reachable.

This module adds the missing piece — retrieval over a *graph*:

  nodes  = every geocoded place in results/osm_cache.json, plus every distinct
           administrative level named in its `hierarchy` field
  edges  = `within`  (containment, read straight out of `hierarchy`)
           `near`    (haversine below a threshold, annotated with km + bearing)

Retrieval is GraphRAG **local search**: the two query entities are already known
(they are the row's subject/object), so there is no embedding step.  We pull the
k-hop ego network of each entity plus the shortest path connecting them, and
verbalize that sub-graph as evidence.

Microsoft-style *global* search (Leiden communities + LLM community summaries +
map-reduce) is deliberately NOT implemented: it answers corpus-wide sensemaking
questions, whereas every row here is a two-entity classification.

No LLM is used to build the graph — OSM is already structured, so extraction is
deterministic and reproducible.

Build the artifact offline (needs no network — it reads the existing cache):
    python build_osm_graph.py

Then at inference `GraphKG` wraps the domain's normal KG and appends a graph
section to its evidence, so Exp 7 evidence is a strict superset of Exp 4's.
"""

import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

from osm_client import haversine_km, bearing_deg, compass8

# Administrative containment, finest → coarsest. Keys are Nominatim address
# fields; only those actually present in a record are used.
ADMIN_LEVELS = [
    "neighbourhood", "quarter", "suburb", "city_district", "borough",
    "village", "town", "city", "municipality",
    "county", "state_district", "state", "region", "country",
]

DEFAULT_PROXIMITY_KM = 50.0
DEFAULT_MAX_NEIGHBORS = 5
DEFAULT_HOPS = 2

# Dataset names and Nominatim hierarchy values disagree on wording for the same
# place ("City of Denver" vs "Denver", "State of California" vs "California").
# Without folding these together the containment chain never reaches the other
# entity, so the graph splits into disconnected islands.
_NAME_PREFIXES = ("city of ", "state of ", "county of ", "borough of ",
                  "town of ", "village of ", "municipality of ",
                  "the ", "district of ")
_NAME_SUFFIXES = (" city", " county", " borough", " state", " municipality")


def normalize_name(name: str) -> str:
    """Fold surface variants of a place name into one comparison key."""
    n = (name or "").strip().lower()
    for p in _NAME_PREFIXES:
        if n.startswith(p):
            n = n[len(p):]
            break
    for s in _NAME_SUFFIXES:
        if n.endswith(s) and len(n) > len(s) + 2:
            n = n[: -len(s)]
            break
    return " ".join(n.replace(",", " ").split())


# ---------------------------------------------------------------------------
# BUILD  (offline, from results/osm_cache.json)
# ---------------------------------------------------------------------------
def _coords(rec: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not rec:
        return None
    try:
        return float(rec["lat"]), float(rec["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def _admin_chain(rec: dict) -> List[str]:
    """Containment chain (finest → coarsest) from a Nominatim `hierarchy` dict."""
    hier = rec.get("hierarchy") or {}
    chain, seen = [], set()
    for level in ADMIN_LEVELS:
        val = hier.get(level)
        if val and val not in seen:
            seen.add(val)
            chain.append(val)
    return chain


def build_graph(cache: dict,
                proximity_km: float = DEFAULT_PROXIMITY_KM,
                max_neighbors: int = DEFAULT_MAX_NEIGHBORS) -> dict:
    """Derive the spatial graph from an osm_cache.json dict.

    Returns {"nodes": {name: attrs}, "edges": [{"s","t","rel",...}], "meta": {...}}
    """
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []
    seen_edges = set()

    def add_edge(s: str, t: str, rel: str, **attrs):
        if s == t:
            return
        key = (s, t, rel)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"s": s, "t": t, "rel": rel, **attrs})

    # --- nodes: geocoded places from the cache -----------------------------
    # Folding is deliberately conservative. A hierarchy value may only be merged
    # into an existing *place* node when that node is itself an administrative
    # boundary — otherwise the town of Oregon and the state of Oregon collapse
    # into one node and manufacture paths that do not exist.
    admin_index: Dict[str, str] = {}      # normalized -> node key

    for name, rec in cache.items():
        if not rec:
            continue                      # ungeocodable — dropped from eval anyway
        latlon = _coords(rec)
        nodes[name] = {
            "kind": "place",
            "lat": latlon[0] if latlon else None,
            "lon": latlon[1] if latlon else None,
            "class": rec.get("class"),
            "type": rec.get("type"),
        }
        if rec.get("class") == "boundary":
            admin_index.setdefault(normalize_name(name), name)

    def resolve(raw: str) -> str:
        """Map a hierarchy value onto a node, merging only with admin boundaries."""
        key = normalize_name(raw)
        if key in admin_index:
            return admin_index[key]
        admin_index[key] = raw
        nodes.setdefault(raw, {"kind": "admin", "lat": None, "lon": None,
                               "class": "boundary", "type": "administrative"})
        return raw

    # --- nodes + edges: administrative containment -------------------------
    for name, rec in cache.items():
        if not rec:
            continue
        prev = name
        for level_name in _admin_chain(rec):
            node = resolve(level_name)
            add_edge(prev, node, "within")
            add_edge(node, prev, "contains")
            prev = node

    # --- edges: proximity between geocoded places --------------------------
    placed = [(n, a["lat"], a["lon"]) for n, a in nodes.items()
              if a["lat"] is not None and a["lon"] is not None]

    for name, lat, lon in placed:
        cands = []
        for other, olat, olon in placed:
            if other == name:
                continue
            d = haversine_km(lat, lon, olat, olon)
            if d <= proximity_km:
                cands.append((d, other, olat, olon))
        cands.sort(key=lambda c: c[0])
        for d, other, olat, olon in cands[:max_neighbors]:
            brg = bearing_deg(lat, lon, olat, olon)
            add_edge(name, other, "near", km=round(d, 1), bearing=round(brg),
                     compass=compass8(brg))

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {"proximity_km": proximity_km, "max_neighbors": max_neighbors,
                 "n_nodes": len(nodes), "n_edges": len(edges)},
    }


# ---------------------------------------------------------------------------
# QUERY  (online, on the GPU server — pure JSON reads, no network)
# ---------------------------------------------------------------------------
class SpatialGraph:
    """Adjacency wrapper with the two retrieval primitives local search needs."""

    def __init__(self, graph_file: str = "results/osm_graph.json"):
        self.path_file = graph_file
        self.nodes: Dict[str, dict] = {}
        self.adj: Dict[str, List[dict]] = {}
        self.alias: Dict[str, str] = {}
        self._load(graph_file)

    def _load(self, graph_file: str):
        if not os.path.exists(graph_file):
            print(f"[GRAPH] ⚠️  {graph_file} not found — graph evidence will be empty. "
                  f"Run build_osm_graph.py locally and commit the artifact.")
            return
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.nodes = data.get("nodes", {})
        for name in self.nodes:
            self.alias.setdefault(normalize_name(name), name)
        for e in data.get("edges", []):
            self.adj.setdefault(e["s"], []).append(e)
        meta = data.get("meta", {})
        print(f"[GRAPH] loaded {meta.get('n_nodes', len(self.nodes))} nodes, "
              f"{meta.get('n_edges', 0)} edges from {graph_file}")

    def resolve(self, name: str) -> Optional[str]:
        """Node key for a place name, tolerating surface-form differences."""
        if name in self.nodes:
            return name
        return self.alias.get(normalize_name(name))

    def has(self, name: str) -> bool:
        return self.resolve(name) is not None

    def containment_chain(self, name: str, limit: int = 6) -> List[str]:
        """Follow `within` edges upward from a node."""
        name = self.resolve(name)
        if name is None:
            return []
        chain, cur, seen = [], name, {name}
        while len(chain) < limit:
            nxt = next((e["t"] for e in self.adj.get(cur, [])
                        if e["rel"] == "within" and e["t"] not in seen), None)
            if nxt is None:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return chain

    def neighbors(self, name: str, rel: str = "near", limit: int = 5) -> List[dict]:
        node = self.resolve(name)
        return [e for e in self.adj.get(node, []) if e["rel"] == rel][:limit] if node else []

    def ego_edges(self, name: str, hops: int = DEFAULT_HOPS, limit: int = 24) -> List[dict]:
        """All edges reachable within `hops` of `name` (breadth-first)."""
        name = self.resolve(name)
        if name is None:
            return []
        out, seen_nodes, seen_edges = [], {name}, set()
        frontier = [name]
        for _ in range(hops):
            nxt = []
            for node in frontier:
                for e in self.adj.get(node, []):
                    key = (e["s"], e["t"], e["rel"])
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    out.append(e)
                    if e["t"] not in seen_nodes:
                        seen_nodes.add(e["t"])
                        nxt.append(e["t"])
                    if len(out) >= limit:
                        return out
            frontier = nxt
            if not frontier:
                break
        return out

    def shortest_path(self, a: str, b: str, max_hops: int = 6) -> Optional[List[dict]]:
        """BFS over all edge types; returns the edge list of one shortest path."""
        a, b = self.resolve(a), self.resolve(b)
        if a is None or b is None:
            return None
        if a == b:
            return []
        q = deque([(a, [])])
        seen = {a}
        while q:
            node, trail = q.popleft()
            if len(trail) >= max_hops:
                continue
            for e in self.adj.get(node, []):
                if e["t"] in seen:
                    continue
                new_trail = trail + [e]
                if e["t"] == b:
                    return new_trail
                seen.add(e["t"])
                q.append((e["t"], new_trail))
        return None


def _fmt_edge(e: dict) -> str:
    if e["rel"] == "near":
        return f'--near({e.get("km")} km, {e.get("compass")})--> {e["t"]}'
    return f'--{e["rel"]}--> {e["t"]}'


def _fmt_path(a: str, edges: List[dict]) -> str:
    return a + " " + " ".join(_fmt_edge(e) for e in edges)


class GraphKG:
    """Wraps the domain's normal KG and appends a retrieved sub-graph.

    Quacks exactly like OSMEvidenceKG / GeographicKnowledgeGraph — exposes
    `fetch()` and `gather_evidence()` — so it drops into the eval engines and
    into every strategy (CoT/ToT/GoT) with no other change.

    Evidence produced = base KG evidence  +  graph section, i.e. a strict
    superset of Exp 4, so the Exp 4 → Exp 7 delta isolates the graph itself.
    """

    def __init__(self, graph_file: str, base_kg, hops: int = DEFAULT_HOPS):
        self.graph = SpatialGraph(graph_file)
        self.base = base_kg
        self.hops = hops

    # exposed so the per-step RAG loop still works if ever combined
    def fetch(self, place_name: str):
        return self.base.fetch(place_name)

    def _graph_block(self, place_a: str, place_b: str) -> str:
        g = self.graph
        lines = ["\n--- OSM Spatial Graph (GraphRAG local retrieval) ---"]
        missing = [p for p in (place_a, place_b) if not g.has(p)]
        if missing:
            lines.append(
                f"  (No graph node for: {', '.join(missing)} — "
                f"fall back to the OSM evidence above.)")

        for label, place in (("A", place_a), ("B", place_b)):
            if not g.has(place):
                continue
            chain = g.containment_chain(place)
            if chain:
                lines.append(f"  Containment of {label} ({place}): "
                             + place + " > " + " > ".join(chain))
            near = g.neighbors(place)
            if near:
                near_txt = ", ".join(
                    f'{e["t"]} ({e.get("km")} km {e.get("compass")})' for e in near)
                lines.append(f"  Nearby to {label} ({place}): {near_txt}")

        if g.has(place_a) and g.has(place_b):
            path = g.shortest_path(place_a, place_b)
            if path:
                lines.append(f"  Connecting path ({len(path)} hop"
                             f"{'s' if len(path) != 1 else ''}): "
                             + _fmt_path(place_a, path))
            elif path == []:
                lines.append("  A and B resolve to the same graph node.")
            else:
                lines.append("  No path found between A and B within the graph radius.")

        lines.append(
            "  Read the relation off this structure where it is decisive "
            "(a `within` path settles containment); otherwise fall back to the "
            "coordinates and geometry above. Do not invent edges.")
        return "\n".join(lines)

    def gather_evidence(self, place_a: str, place_b: str, sentence: str = "",
                        entity: dict = None, log_fn=None) -> str:
        base = self.base.gather_evidence(place_a, place_b, sentence=sentence,
                                         entity=entity, log_fn=None)
        text = base + "\n" + self._graph_block(place_a, place_b)
        if log_fn:
            log_fn(text)
        return text
