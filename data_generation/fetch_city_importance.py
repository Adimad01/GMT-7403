"""Record how prominent each city in the coordinate table is.

Cardinal difficulty needs a measure, and unlike relative (frame rotation) or
topological (size ratio, gap, shared border) the geometry offers none: once the
direction is removed from the wording, what makes an item hard is simply
whether the reader knows where the two places are.

Nominatim publishes an importance score derived from Wikipedia prominence,
which is exactly that, and is a published number rather than my opinion about
which cities are famous.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS                # noqa: E402
from osm_resolve import LookupFailed, resolve          # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "cardinal" / "osm" / "importance.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [c for c in sorted(COORDS) if c not in cache]
    print(f"  {len(cache)} cached, {len(todo)} to fetch", flush=True)
    for i, city in enumerate(todo, 1):
        rec = None
        for attempt in range(3):
            try:
                rec = resolve(city, want_polygon=False, kind="city",
                              timeout=45 + 15 * attempt)
                break
            except LookupFailed:
                continue
        cache[city] = ({"importance": float(rec.get("importance") or 0),
                        "display_name": rec.get("display_name")}
                       if rec else None)
        if i % 20 == 0 or i == len(todo):
            OUT.write_text(json.dumps(cache, indent=1))
            got = sum(1 for v in cache.values() if v)
            print(f"  [{i}/{len(todo)}] {city[:26]:<26} resolved {got}/{len(cache)}",
                  flush=True)
    OUT.write_text(json.dumps(cache, indent=1))
    print(f"  done: {sum(1 for v in cache.values() if v)}/{len(cache)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
