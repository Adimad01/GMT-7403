"""
geographic_kg.py — Metric Distance Knowledge Graph Builder
===========================================================
Builds a knowledge graph from the cities_with_coords.json dataset for
Volet C distance estimation.

For a given test pair (A, B), the KG provides:
  - City A & B metadata: name, lat, lon, state, corpus mention count
  - Anchor evidence table: for each anchor city C (all cities except A, B),
    the city's coordinates + known distances d(A,C) and d(B,C)
  - Corpus co-occurrence signals between A and B
  - Same-state flag

CRITICAL GUARANTEE:
  - The direct distance d(A, B) is NEVER included in the evidence.
  - d(A, C) and d(B, C) for ALL anchors C are included (from the full dataset).
  - Only d(A, B) itself is hidden — it's the target to predict.
"""

import json
import os
from typing import Dict, List, Tuple, Optional


# =====================================================================
# KNOWLEDGE GRAPH
# =====================================================================
class MetricKnowledgeGraph:
    """
    Builds and queries a geographic knowledge graph for metric distance
    estimation from cities_with_coords.json.
    """

    def __init__(self, data_path: str, test_ids_path: str):
        """
        Load the full dataset and the test pair identifiers.

        Args:
            data_path:     Path to cities_with_coords.json (8,556 records).
            test_ids_path: Path to 100_target_ids.json (100 test pairs).
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(test_ids_path, "r", encoding="utf-8") as f:
            self.test_pairs_raw = json.load(f)

        # Build test pair set for quick lookup
        self.test_pair_set = set()
        for tp in self.test_pairs_raw:
            self.test_pair_set.add((tp["a_name"], tp["b_name"]))
            self.test_pair_set.add((tp["b_name"], tp["a_name"]))

        # ── City metadata ─────────────────────────────────────────────
        self.city_coords: Dict[str, Tuple[float, float]] = {}
        self.city_state: Dict[str, str] = {}
        self.city_count: Dict[str, int] = {}

        for r in self.data:
            for prefix in ("a", "b"):
                name = r[f"{prefix}_name"]
                lat = r[f"{prefix}_lat"]
                lon = r[f"{prefix}_lon"]
                self.city_coords[name] = (lat, lon)
                # Parse state from "City, State" format
                if ", " in name:
                    self.city_state[name] = name.split(", ")[-1]
                # Corpus mention count (take max seen across all pairs)
                cnt = r.get(f"{prefix}_count", 0)
                self.city_count[name] = max(self.city_count.get(name, 0), cnt)

        self.all_cities = sorted(self.city_coords.keys())

        # ── Distance lookup (ALL pairs — including test pair distances
        #    for use as anchor legs d(A,C) and d(B,C)) ─────────────────
        self.dist: Dict[Tuple[str, str], float] = {}
        for r in self.data:
            a, b = r["a_name"], r["b_name"]
            d = r["distance"]
            self.dist[(a, b)] = d
            self.dist[(b, a)] = d

        # ── Corpus signal lookup (for test pairs) ─────────────────────
        self.corpus_signals: Dict[Tuple[str, str], Dict] = {}
        for r in self.data:
            a, b = r["a_name"], r["b_name"]
            sig = {
                "co_occ_count": r.get("co_occ_count", 0),
                "near_count": r.get("near_count", 0),
                "close_count": r.get("close_count", 0),
                "far_count": r.get("far_count", 0),
                "and_count": r.get("and_count", 0),
            }
            self.corpus_signals[(a, b)] = sig
            self.corpus_signals[(b, a)] = sig

        print(f"[KG] Loaded {len(self.all_cities)} cities, "
              f"{len(self.data)} pair records, "
              f"{len(self.test_pairs_raw)} test pairs.")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def get_city_info(self, city: str) -> Dict:
        """Return metadata for a single city."""
        lat, lon = self.city_coords.get(city, (None, None))
        return {
            "name": city,
            "lat": lat,
            "lon": lon,
            "state": self.city_state.get(city, "unknown"),
            "corpus_mentions": self.city_count.get(city, 0),
        }

    def get_distance(self, a: str, b: str) -> Optional[float]:
        """Return the known distance between two cities, or None."""
        return self.dist.get((a, b))

    def is_test_pair(self, a: str, b: str) -> bool:
        """Check if (a, b) is a test pair."""
        return (a, b) in self.test_pair_set

    def same_state(self, a: str, b: str) -> bool:
        """Check if two cities are in the same state."""
        sa = self.city_state.get(a)
        sb = self.city_state.get(b)
        return sa is not None and sa == sb

    def get_anchor_table(self, city_a: str, city_b: str,
                         top_k: Optional[int] = None) -> List[Dict]:
        """
        Build the anchor evidence table for a test pair (city_a, city_b).

        For each anchor city C (every city except A and B):
          - C's name, coordinates, state
          - d(A, C) from the dataset
          - d(B, C) from the dataset

        The direct distance d(A, B) is NEVER included.

        Args:
            city_a: First city name.
            city_b: Second city name.
            top_k:  If set, return only the top-K anchors with the
                    tightest triangle inequality bounds (smallest |d(A,C) - d(B,C)|).
                    If None, return ALL anchors.

        Returns:
            List of anchor dicts, sorted by tightest bounds.
        """
        anchors = []
        for c in self.all_cities:
            if c == city_a or c == city_b:
                continue

            da = self.dist.get((city_a, c))
            db = self.dist.get((city_b, c))

            if da is None or db is None:
                continue  # Should not happen in a complete graph, but be safe

            clat, clon = self.city_coords[c]
            anchors.append({
                "name": c,
                "lat": clat,
                "lon": clon,
                "state": self.city_state.get(c, "unknown"),
                "d_a": round(da, 1),
                "d_b": round(db, 1),
            })

        # Sort by tightest bound: smallest |d(A,C) - d(B,C)|
        anchors.sort(key=lambda x: abs(x["d_a"] - x["d_b"]))

        if top_k is not None:
            anchors = anchors[:top_k]

        return anchors

    def get_corpus_signals(self, city_a: str, city_b: str) -> Dict:
        """Return corpus co-occurrence signals for a pair."""
        return self.corpus_signals.get(
            (city_a, city_b),
            {"co_occ_count": 0, "near_count": 0, "close_count": 0,
             "far_count": 0, "and_count": 0},
        )

    def build_evidence_block(self, city_a: str, city_b: str,
                             top_k: int = 15) -> str:
        """
        Build a formatted text evidence block for the LLM prompt.

        This is the core method called by the reasoning strategies.
        It assembles ALL KG information for a test pair into a single
        readable text block.

        Args:
            city_a: First city.
            city_b: Second city.
            top_k:  Number of anchor cities to include in the table
                    (top-K by tightest bounds). Use None for ALL.

        Returns:
            Formatted string ready for prompt injection.
        """
        info_a = self.get_city_info(city_a)
        info_b = self.get_city_info(city_b)
        same = self.same_state(city_a, city_b)
        signals = self.get_corpus_signals(city_a, city_b)
        anchors = self.get_anchor_table(city_a, city_b, top_k=top_k)
        total_anchors = len(self.get_anchor_table(city_a, city_b, top_k=None))

        lines = []

        # ── City info ──
        lines.append("=== CITY INFORMATION ===")
        lines.append(
            f"City A: {info_a['name']}  |  "
            f"Lat: {info_a['lat']:.4f}°  Lon: {info_a['lon']:.4f}°  |  "
            f"State: {info_a['state']}  |  "
            f"Corpus mentions: {info_a['corpus_mentions']}"
        )
        lines.append(
            f"City B: {info_b['name']}  |  "
            f"Lat: {info_b['lat']:.4f}°  Lon: {info_b['lon']:.4f}°  |  "
            f"State: {info_b['state']}  |  "
            f"Corpus mentions: {info_b['corpus_mentions']}"
        )
        lines.append(f"Same state: {'Yes' if same else 'No'}")

        # ── Corpus signals ──
        lines.append("")
        lines.append("=== CORPUS CO-OCCURRENCE SIGNALS ===")
        lines.append(
            f"co_occurrence: {signals['co_occ_count']}  |  "
            f"near: {signals['near_count']}  |  "
            f"close: {signals['close_count']}  |  "
            f"far: {signals['far_count']}  |  "
            f"and: {signals['and_count']}"
        )

        # ── Anchor table ──
        lines.append("")
        n_shown = len(anchors)
        lines.append(
            f"=== ANCHOR CITIES WITH KNOWN DISTANCES "
            f"(top {n_shown} of {total_anchors} by tightest bounds) ==="
        )
        header = (
            f"{'Anchor City':<30} {'Lat':>8} {'Lon':>10}  "
            f"{'d(A,Anchor)':>12} {'d(B,Anchor)':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for anc in anchors:
            lines.append(
                f"{anc['name']:<30} {anc['lat']:>8.4f} {anc['lon']:>10.4f}  "
                f"{anc['d_a']:>10.1f} km {anc['d_b']:>10.1f} km"
            )

        if total_anchors > n_shown:
            lines.append(f"... + {total_anchors - n_shown} more anchor cities available")

        return "\n".join(lines)

    def iter_test_pairs(self) -> List[Dict]:
        """
        Iterate over all 100 test pairs, yielding dicts with:
          a_name, b_name, true_distance, and all original record fields.
        """
        pairs = []
        # Build a lookup from the full data
        pair_lookup = {}
        for r in self.data:
            pair_lookup[(r["a_name"], r["b_name"])] = r

        for tp in self.test_pairs_raw:
            a, b = tp["a_name"], tp["b_name"]
            record = pair_lookup.get((a, b), pair_lookup.get((b, a), {}))
            pairs.append({
                "a_name": a,
                "b_name": b,
                "true_distance": record.get("distance"),
                **record,
            })
        return pairs

    def export_to_json(self, output_path: str) -> None:
        """
        Exports the entire knowledge graph to a JSON file in standard
        Node-Link format (nodes and links lists).
        """
        print(f"[KG] Compiling graph data for export...")
        
        # 1. Build Nodes
        nodes = []
        for city in self.all_cities:
            info = self.get_city_info(city)
            # Use 'id' alongside 'name' as it's standard for graph libraries
            info["id"] = info["name"] 
            nodes.append(info)

        # 2. Build Links (Edges)
        links = []
        seen_edges = set()
        
        for (a, b), distance in self.dist.items():
            # Sort to treat (A, B) and (B, A) as the exact same undirected edge
            edge_key = tuple(sorted([a, b]))
            
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                signals = self.get_corpus_signals(a, b)
                
                links.append({
                    "source": a,
                    "target": b,
                    "distance": distance,
                    "corpus_signals": signals
                })

        # 3. Assemble and Save
        graph_data = {
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(links)
            },
            "nodes": nodes,
            "links": links
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"[KG] ✅ Knowledge Graph saved to: {output_path}")
        print(f"[KG] Exported {len(nodes)} nodes and {len(links)} edges.")


# =====================================================================
# QUICK TEST & EXPORT
# =====================================================================
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    kg = MetricKnowledgeGraph(
        data_path=os.path.join(base, "cities_with_coords.json"),
        test_ids_path=os.path.join(base, "100_target_ids.json"),
    )

    # Show one example
    test_pairs = kg.iter_test_pairs()
    ex = test_pairs[0]
    a, b = ex["a_name"], ex["b_name"]

    print(f"\n{'='*80}")
    print(f"  EXAMPLE: {a} ↔ {b}")
    print(f"  True distance (hidden): {ex['true_distance']:.1f} km")
    print(f"{'='*80}\n")

    evidence = kg.build_evidence_block(a, b, top_k=10)
    print(evidence)
    print("\n" + "="*80 + "\n")

    # Save the graph to JSON
    output_file = os.path.join(base, "outputs", "geospatial_knowledge_graph_final.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Trigger the export
    kg.export_to_json(output_file)