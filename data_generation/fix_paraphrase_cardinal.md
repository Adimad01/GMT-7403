# Replacements: north_of, levels 1 and 2

Your cardinal batch was good apart from one thing. In `north_of` at levels 1
and 2, all 48 templates lost their subject placeholder and arrived as
fragments:

    directly above relative to {B}.
    straight up from relative to {B}.
    vertically higher than relative to {B}.

Every other predicate at those levels was fine, so it is only these two cells.
Compare with your own `south_of`, which is exactly right:

    south_of | 1 | {A} is directly below {B}.
    south_of | 2 | {A} occupies a lower Y-coordinate than {B}.

## What I need

**24 templates for `north_of` at level 1** and **24 at level 2**, each a
complete sentence containing both {A} and {B}.

  Level 1  plain, everyday phrasing: {A} sits directly above {B} on the map
  Level 2  factual or technical, still direct: {A} holds a higher Y-coordinate
           than {B}

## Rules, unchanged

- never write "north of" or "northward of"
- both {A} and {B} must appear, and the relation runs from {A} to {B}
- no place names, one sentence, under twenty-five words
- no two templates paraphrasing each other

## Output

    north_of | 1 | {A} sits directly above {B} on the map.

One per line. No numbering, no commentary.
