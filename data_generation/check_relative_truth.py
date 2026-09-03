"""Ground truth for viewpoint-relative direction.

There is no compass calculation that settles 'left of'. Left depends entirely
on where the observer stands and which way they look, so the only way to make
the label checkable is to put the viewpoint into the item as a geometric fact
rather than as atmosphere. Every row here names an observer V and states that
the observer's gaze is fixed on the reference B; the subject A is then placed
within that frame.

With those three points the relation is fully determined:

    theta = bearing(V -> B)                   the sight line
    alpha = bearing(V -> A) - theta           A's angle off it, signed
    depth = dist(V, A) / dist(V, B)           nearer or further along the line

    alpha negative  -> A appears to the left of the sight line
    alpha near zero -> A is on the line; depth says in front or behind

This is the ternary projective reading of relative direction: the relation
holds between three places, not two, and it is the observer that makes it
decidable.

Difficulty follows from the same geometry. When the observer faces north their
frame agrees with a north-up map and left means west. When they face south the
frame is inverted and left means east -- the reader has to rotate the map in
their head. The angle between the sight line and north is therefore a direct,
computable measure of how much mental rotation an item demands, and it is what
sets the ambiguity level.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS, bearing, key, separation  # noqa: E402

# --- the label regions ------------------------------------------------------
# Gaps are deliberate: a pair that falls between two regions has no defensible
# answer, so it is rejected rather than assigned to the nearer one.
LATERAL_MIN, LATERAL_MAX = 30.0, 150.0   # |alpha| band for left / right
AXIAL_MAX = 15.0                          # |alpha| band for in front / behind
FRONT_DEPTH = 0.75                        # A is this much nearer than B
BEHIND_DEPTH = 1.40                       # A is this much further than B
BESIDE_SPREAD = 0.15                      # dist(A,B) as a fraction of dist(V,B)
BESIDE_DEPTH = (0.85, 1.18)               # A and B at comparable depth

LABELS = ["left_of", "right_of", "in_front_of", "behind", "next_to"]


def norm180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def frame(observer: str, subject: str, target: str):
    """(alpha, depth ratio, spread, rotation-from-north) for one item."""
    v, a, b = COORDS[key(observer)], COORDS[key(subject)], COORDS[key(target)]
    theta = bearing(v, b)
    alpha = norm180(bearing(v, a) - theta)
    d_a, d_b = separation(v, a), separation(v, b)
    spread = separation(a, b)
    return alpha, (d_a / d_b if d_b else float("inf")), spread / d_b if d_b else 9e9, \
        abs(norm180(theta))


def classify(observer: str, subject: str, target: str) -> tuple[str | None, float]:
    """The relation that holds, and how far it sits inside its region.

    The margin is in degrees for the directional labels and is scaled to a
    comparable number for the two that are decided by distance, so one
    threshold can screen every label.
    """
    alpha, depth, spread, _ = frame(observer, subject, target)
    aa = abs(alpha)

    if LATERAL_MIN <= aa <= LATERAL_MAX:
        margin = min(aa - LATERAL_MIN, LATERAL_MAX - aa)
        return ("left_of" if alpha < 0 else "right_of"), margin

    if spread <= BESIDE_SPREAD and BESIDE_DEPTH[0] <= depth <= BESIDE_DEPTH[1]:
        # how far inside the beside region, expressed on the same scale
        room = min((BESIDE_SPREAD - spread) / BESIDE_SPREAD,
                   (depth - BESIDE_DEPTH[0]) / (BESIDE_DEPTH[1] - BESIDE_DEPTH[0]),
                   (BESIDE_DEPTH[1] - depth) / (BESIDE_DEPTH[1] - BESIDE_DEPTH[0]))
        return "next_to", room * 30.0

    if aa <= AXIAL_MAX:
        if depth <= FRONT_DEPTH:
            return "in_front_of", min(AXIAL_MAX - aa, (FRONT_DEPTH - depth) * 60)
        if depth >= BEHIND_DEPTH:
            return "behind", min(AXIAL_MAX - aa, (depth - BEHIND_DEPTH) * 60)

    return None, 0.0


def rotation_level(observer: str, target: str) -> int:
    """Ambiguity level 1-5 from how far the sight line is turned from north."""
    _, _, _, rot = frame(observer, target, target) if False else (0, 0, 0, 0)
    v, b = COORDS[key(observer)], COORDS[key(target)]
    rot = abs(norm180(bearing(v, b)))
    for lvl, ceiling in enumerate([30.0, 70.0, 110.0, 150.0, 180.1], start=1):
        if rot <= ceiling:
            return lvl
    return 5


def check(observer: str, subject: str, target: str, label: str,
          margin: float = 12.0, max_sep: float = 70.0,
          min_sep: float = 3.0) -> list[str]:
    """Every reason this item is unusable."""
    for n in (observer, subject, target):
        if key(n) not in COORDS:
            return [f"no coordinates for {n}"]
    if len({key(observer), key(subject), key(target)}) < 3:
        return ["observer, subject and target must be three different places"]

    problems = []
    got, m = classify(observer, subject, target)
    if got != label:
        problems.append(f"the frame gives {got or 'no clear relation'}, not {label}")
    elif m < margin:
        problems.append(f"only {m:.1f} inside the {label} region — too close to call")

    v = COORDS[key(observer)]
    for name in (subject, target):
        sep = separation(v, COORDS[key(name)])
        if sep > max_sep:
            problems.append(f"{name} is {sep:.0f} deg from the observer; a single "
                            f"line of sight stops being meaningful")
        if sep < min_sep:
            problems.append(f"{name} is only {sep:.0f} deg from the observer")
    return problems
