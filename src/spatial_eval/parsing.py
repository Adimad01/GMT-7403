"""Turn a free-text model completion into one of the allowed labels.

Parsing is a measurement instrument: a weak parser scores a correct answer as
wrong and silently depresses every arm. Rules, in order of trust:

  1. an explicit ``ANSWER: <label>`` line  (what every prompt asks for)
  2. the last allowed label appearing anywhere  (models often reason, then state)
  3. a surface-form synonym  ("to the north" -> north_of)

Anything else is recorded as ``None`` -- an unparseable answer, counted and
reported separately rather than folded in as a wrong prediction.
"""
from __future__ import annotations

import re

# Surface forms that unambiguously denote a label. Deliberately conservative:
# a wrong mapping here manufactures accuracy out of nothing.
SYNONYMS: dict[str, dict[str, list[str]]] = {
    "topological": {
        "contains": ["contains", "encloses", "encompasses", "envelops"],
        "within": ["within", "inside of", "contained in", "contained within"],
        "touches": ["touches", "borders", "adjacent to", "shares a boundary"],
        "crosses": ["crosses", "intersects", "bisects"],
        "disjoint": ["disjoint", "separate from", "no overlap"],
        "overlaps": ["overlaps", "partially overlaps"],
        "equals": ["equals", "coextensive", "identical to", "the same as"],
    },
    "cardinal": {
        "north_of": ["north of", "to the north"],
        "south_of": ["south of", "to the south"],
        "east_of": ["east of", "to the east"],
        "west_of": ["west of", "to the west"],
        "northeast_of": ["northeast of", "north-east of", "to the northeast"],
        "northwest_of": ["northwest of", "north-west of", "to the northwest"],
        "southeast_of": ["southeast of", "south-east of", "to the southeast"],
        "southwest_of": ["southwest of", "south-west of", "to the southwest"],
    },
    "relative": {
        "left_of": ["left of", "to the left", "port side", "port arm"],
        "right_of": ["right of", "to the right", "starboard side", "starboard arm"],
        "in_front_of": ["in front of", "ahead of", "before"],
        "behind": ["behind", "to the rear of", "back of"],
        "next_to": ["next to", "beside", "alongside"],
    },
}

_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*([a-z_\- ]+)", re.IGNORECASE)


def _normalise(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def parse_label(text: str, allowed: list[str], relation: str) -> tuple[str | None, str]:
    """Return ``(label_or_None, how)`` where ``how`` names the rule that fired."""
    if not text:
        return None, "empty"

    low = text.lower()

    # 1. explicit ANSWER: line -- take the LAST one, since a model that
    #    reconsiders states its final answer last.
    matches = _ANSWER_RE.findall(low)
    for raw in reversed(matches):
        cand = _normalise(raw)
        if cand in allowed:
            return cand, "answer_tag"
        # tolerate "north" for "north_of"
        for lab in allowed:
            if cand and lab.startswith(cand + "_") or lab == cand + "_of":
                return lab, "answer_tag_short"

    # 2. last bare label mention
    best_pos, best_lab = -1, None
    for lab in allowed:
        pos = low.rfind(lab.replace("_", " "))
        pos = max(pos, low.rfind(lab))
        if pos > best_pos:
            best_pos, best_lab = pos, lab
    if best_lab is not None and best_pos >= 0:
        return best_lab, "label_mention"

    # 3. synonym
    syn = SYNONYMS.get(relation, {})
    best_pos, best_lab = -1, None
    for lab, forms in syn.items():
        if lab not in allowed:
            continue
        for form in forms:
            pos = low.rfind(form)
            if pos > best_pos:
                best_pos, best_lab = pos, lab
    if best_lab is not None:
        return best_lab, "synonym"

    return None, "unparsed"
