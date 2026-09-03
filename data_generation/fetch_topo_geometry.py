"""Fetch and cache OSM geometry for every catalogue entity.

Resumable: entities already in the cache are skipped, so an interrupted run
continues where it stopped. Entities that resolve to nothing of the right kind
are recorded as null rather than being retried forever -- an unresolvable name
is a fact about the name, and the corpus builder needs to know it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osm_resolve import LookupFailed, resolve         # noqa: E402
from topo_catalogue import ALL, KIND                  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "topological" / "osm" / "geometry.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [n for n in ALL if n not in cache]
    print(f"  {len(cache)} cached, {len(todo)} to fetch", flush=True)

    for i, name in enumerate(todo, 1):
        # Large outlines time out at fine simplification, so each retry asks for
        # a coarser polygon. A name is only written off once every attempt has
        # completed and still found nothing of the right kind.
        r, failed = None, False
        for attempt, simp in enumerate((0.005, 0.02, 0.05)):
            try:
                r = resolve(name, simplify=simp, timeout=90 + 30 * attempt,
                            kind=KIND.get(name))
                failed = False
                break
            except LookupFailed as exc:
                failed = True
                print(f"    retry {attempt+1} for {name}: {exc}", flush=True)
                time.sleep(3)
        if failed:
            print(f"    GAVE UP on {name} after 3 attempts", flush=True)
        if r and r.get("geojson"):
            cache[name] = {"osm_type": r.get("osm_type"), "osm_id": r.get("osm_id"),
                           "class": r.get("class"), "type": r.get("type"),
                           "display_name": r.get("display_name"),
                           "importance": r.get("importance"),
                           "geojson": r["geojson"]}
            g = r["geojson"]["type"]
        else:
            cache[name] = None
            g = "none"
        if i % 10 == 0 or i == len(todo):
            OUT.write_text(json.dumps(cache))
            got = sum(1 for v in cache.values() if v)
            print(f"  [{i}/{len(todo)}] {name[:34]:<34} {g:<18} "
                  f"resolved {got}/{len(cache)}", flush=True)

    OUT.write_text(json.dumps(cache))
    got = sum(1 for v in cache.values() if v)
    print(f"  done: {got}/{len(cache)} resolved, "
          f"{OUT.stat().st_size/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
