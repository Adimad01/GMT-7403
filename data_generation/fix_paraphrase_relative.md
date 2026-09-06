# Regenerate: viewpoint-relative templates

The batch was mechanically clean but has two faults that make it unusable as
it stands.

## Fault 1 — the levels are not a ladder

All 115 of your openings appear at every level carrying the SAME spatial cue.
Only the verb changes:

    L1  ... {A} is located at the port side.
    L2  ... {A} strictly occupies the port quarter.
    L3  ... a turn to port immediately reveals {A}.
    L4  ... {A} perfectly represents the port-side limit.
    L5  ... {A} serves as the port-side tune.

A reader who knows that port means left finds all five equally easy; one who
does not finds all five impossible. The level is meant to be how hard the cue
is to decode, so it has to be a DIFFERENT cue at each level, not one cue in
five costumes. Several level 5 lines are also not meaningful English -- "{A}
serves as the port-side tune" says nothing.

What a real ladder looks like, for left_of:

    1  Standing in {V} facing {B}, {A} is on the side of your weaker hand.
    2  From {V} facing {B}, {A} falls on the side a violinist's fingers work.
    3  From {V} facing {B}, an archer at your position would swing the bow arm
       toward {A}.
    4  From {V} facing {B}, {A} lies on the side a chess player's queen starts.
    5  From {V} facing {B}, {A} sits where a clock's hand points at quarter to.

Each needs a different piece of knowledge. That is the variable being tested.

## Fault 2 — thirty templates are geometrically false

These equate the side with a compass direction:

    left_of  | {A} is located at the western flank
    left_of  | {A} is located at the setting sun
    right_of | {A} is located at the eastern flank

Left is only west when the observer faces north. Here the observer faces {B},
which can be any direction, so these are wrong for most cases.

**Do not use north, south, east, west, sunrise, sunset, dawn or dusk anywhere
in a left_of or right_of template.** They are valid for in_front_of, behind and
next_to only in the sense of distance, not side.

## What I need

**24 templates for each of the 5 predicates at each of 5 levels = 600**, with a
genuinely different cue at every level.

`left_of`, `right_of`, `in_front_of`, `behind`, `next_to`

Every template must contain {V} for the observer, {B} for what they face, and
{A} for the place being located. Never write left, right, front, behind, next
to or adjacent to.

## Output

    left_of | 1 | Standing in {V} and facing {B}, {A} is on the side of your weaker hand.

One per line. No numbering, no commentary.
