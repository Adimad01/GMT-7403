# Vernacular templates for viewpoint-relative directions

I am building a dataset that tests whether a language model can read a spatial description written in ordinary language and recover the formal relation it expresses.

I need **24 templates for each predicate at each of five levels** — 5 predicates x 5 levels x 24 = **600 templates**.

## The predicates

  `left_of` — seen by an observer facing B, A appears to the left
  `right_of` — seen by an observer facing B, A appears to the right
  `in_front_of` — A stands between the observer and B
  `behind` — B stands between the observer and A
  `next_to` — A and B appear side by side, at much the same distance

## The five levels

The level is how INDIRECT the wording is. It is not about how hard the
geography is -- the places are decided elsewhere, and the same pair may appear
at any level.

  Level 1  A near-synonym. Plain, everyday phrasing that a reader maps to the
           relation immediately, without the predicate word itself.
  Level 2  Ordinary description. Still direct, but phrased as a fact about the
           places rather than as a spatial term.
  Level 3  Indirect. The relation follows from something described -- a journey,
           a boundary, a shared or unshared feature.
  Level 4  Oblique. The reader has to work slightly to see which relation is
           meant; the wording describes a consequence rather than the relation.
  Level 5  Highly oblique. Metaphor, imagery, or an inference from an
           unrelated-sounding fact. Still unambiguous on reflection.

A worked ladder for north_of:

  1  {A} sits directly above {B} on the map.
  2  {A} occupies a higher position on the chart than {B}.
  3  Leaving {B}, you drive steadily upward on the map to reach {A}.
  4  Of the two, {A} is the one that sees the shorter winter day.
  5  A compass needle at {B} points along the line that leads to {A}.


## Note — this one differs

These relations only mean anything from a viewpoint, so every template
must establish one. A third placeholder {V} is available for the observer's
position, and each template must use all three.

    left_of | 1 | Standing in {V} and facing {B}, {A} appears off to one side, on the hand that writes.
    behind | 2 | From {V}, {B} has to be passed before {A} comes into reach.

Do not use the words left, right, front or behind. Everything else is fair.


## Hard rules

1. NEVER write the predicate itself, or a phrase that contains it. For
   north_of, the strings "north of" and "northward of" are banned. The reader
   must interpret, not match.

2. NEVER name a place. Use only the placeholders {A} and {B}. A template that
   mentions a city cannot be reused.

3. The relation must run from {A} to {B} in that order. "{A} sits above {B}"
   is north_of. "{B} sits above {A}" is the opposite relation and is wrong.

4. Each template must be unambiguous. A reader who knows the convention should
   recover exactly one relation from it. If it could equally mean two, drop it.

5. No two templates may paraphrase each other. Twenty rewordings of "above" is
   one template, not twenty.

6. Keep them short: one sentence, under about twenty-five words.

## Output format

One template per line, exactly:

    PREDICATE | LEVEL | template text

For example:

    north_of | 1 | {A} sits directly above {B} on the map.
    north_of | 3 | Leaving {B}, you travel steadily upward to reach {A}.

No numbering, no bullets, no headers, no commentary.
