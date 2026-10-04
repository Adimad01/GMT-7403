"""Training-only topological templates, written so that no core sentence is shared with the evaluation pool.

Why this file exists. The topological bank was split into train and eval pools string by string,
but its templates were built combinatorially -- an opening phrase ("Looking at the map,",
"Geographically,", ...) times a core sentence times a filler ("the region of", "the limits of").
Variants of one sentence therefore landed on both sides: 87 % of evaluation rows had their core
sentence in training. The evaluation pool is kept exactly as it is (its rows, manifest and every
base-model result stay valid); the training pool is rebuilt from the sentences below plus the old
training templates whose core never appears in evaluation.

Conventions
- No opening phrase: an opener carries no information, and it is what produced the leakage.
- No predicate word of ANY relation appears, not only the row's own.
- `within` is `contains` with {A} and {B} swapped, so the direction cannot be wrong.
- `crosses` is written role-neutrally ("one of {A} and {B} ... the other"): in the corpus the
  subject is sometimes the river and sometimes the area, and the old templates, which all cast {A}
  as the line, produced incoherent sentences whenever the area came first.
- Geometry, not administration (see prompt_paraphrase_topological.md).
"""

CONTAINS = {
    1: ["{A} encloses the whole of {B}.",
        "{A} surrounds {B} on every side.",
        "{A} holds all of {B} in its interior.",
        "{A} fully envelops {B}.",
        "All of {B} falls in the area of {A}.",
        "{A} takes in {B} completely."],
    2: ["Every point of {B} also belongs to {A}.",
        "The area of {B} forms one piece of the larger area of {A}.",
        "On any map, the outline of {B} is drawn entirely in the space of {A}.",
        "{A} is the larger area, and {B} lies wholly in it.",
        "Of the two, {A} is the one whose outline goes all the way around {B}.",
        "Measured on the ground, {B} is a portion of {A}."],
    3: ["Wherever you stand in {B}, you are also standing in {A}.",
        "Any walk that starts and ends in {B} stays in {A} the whole time.",
        "To get out of {A}, a traveller in {B} must first leave {B}.",
        "Fencing in the whole of {A} would also fence in all of {B}.",
        "A complete map of {A} necessarily shows all of {B}.",
        "Someone who has visited every corner of {A} has necessarily set foot in {B}."],
    4: ["A survey of all of {A} would also cover every acre of {B}.",
        "Any photograph that frames the whole of {A} also frames the whole of {B}.",
        "A wildfire that stays in {B} has never left {A}.",
        "Every species recorded in {B} is, by that fact, a species recorded in {A}.",
        "A quarantine sealing off {A} automatically seals off {B} as well.",
        "Drawing {A} on a sheet of paper means drawing {B} too."],
    5: ["{A} is the shell and {B} the pearl it keeps.",
        "{B} is a yolk, and {A} the egg around it.",
        "{A} is the frame, {B} the painting hung entirely in it.",
        "{B} is a seed and {A} the fruit closed around it.",
        "{A} is a walled garden, and {B} one of its flowerbeds.",
        "{B} is a word written somewhere on the page that is {A}."],
}

def _swap(t: str) -> str:
    return t.replace("{A}", "\0").replace("{B}", "{A}").replace("\0", "{B}")

WITHIN = {lv: [_swap(t) for t in ts] for lv, ts in CONTAINS.items()}

TOUCHES = {
    1: ["{A} borders {B}.",
        "{A} abuts {B}.",
        "{A} is a neighbour of {B}, sharing a stretch of border.",
        "{A} and {B} share a boundary line and nothing more.",
        "{A} and {B} have a common frontier but no common ground.",
        "{A} meets {B} at a common border."],
    2: ["The border of {A} and the border of {B} coincide along part of their length.",
        "No strip of ground separates {A} from {B}, yet no ground belongs to both.",
        "{A} and {B} are contiguous.",
        "Taken together, {A} and {B} form one unbroken block, though neither reaches into the other.",
        "A map shows {A} and {B} fitting together edge to edge.",
        "{A} and {B} share an edge but not an interior."],
    3: ["A single border post could stand with one side in {A} and the other in {B}.",
        "Surveyors marking the edge of {A} end up marking part of the edge of {B}.",
        "Driving along part of the frontier of {B}, you have {A} just across the line.",
        "A trip from {A} to {B} passes over just one boundary and no other land.",
        "Standing on the boundary of {A} at the right place, you are standing on the boundary of {B} too."],
    4: ["Colour a map so that bordering areas never share a colour, and {A} and {B} must get different colours.",
        "Wildlife can wander from {A} into {B} without passing through any other area.",
        "A wildfire leaving {A} can reach {B} without burning any other place on the way.",
        "Snow drifting off the edge of {A} lands directly on {B}.",
        "The outline of {B} cannot be redrawn without redrawing part of the outline of {A}."],
    5: ["{A} and {B} are two tiles laid edge to edge in the same floor.",
        "{A} and {B} are two puzzle pieces clicked into place beside each other.",
        "{A} and {B} are neighbours whose only link is a single garden wall.",
        "{A} and {B} are two bricks sharing one line of mortar.",
        "{A} and {B} are two cells of a honeycomb with one wax wall between them.",
        "{A} and {B} are two squares of a chessboard that meet along a side."],
}

CROSSES = {
    1: ["One of {A} and {B} runs right through the other.",
        "One of {A} and {B} is threaded through the other.",
        "One of {A} and {B} is a line drawn through the other.",
        "{A} and {B} lie one across the other.",
        "One of {A} and {B} passes through the other and carries on beyond it."],
    2: ["One of {A} and {B} lies partly in the other and partly outside it, on both sides.",
        "The line of one of {A} and {B} has its two ends outside the other and its middle in it.",
        "One of {A} and {B} extends past the other in two opposite directions.",
        "A map shows one of {A} and {B} drawn across the other and continuing beyond it.",
        "One of {A} and {B} reaches the border of the other at two separate points."],
    3: ["Travelling the length of one of {A} and {B}, you meet the boundary of the other twice, once going in and once coming out.",
        "Going along one of {A} and {B}, you start outside the other, pass through it, and finish outside it again.",
        "Two people at opposite ends of one of {A} and {B} both stand outside the other, yet the stretch between them runs through it.",
        "Sailing down one of {A} and {B}, you pass through the other only for the middle part of the voyage."],
    4: ["One of {A} and {B} has land on both banks of the other.",
        "Water carried by one of {A} and {B} flows into the other and then out of it again.",
        "Settlements along one of {A} and {B} lie both in the other and beyond its edges.",
        "Somewhere in one of {A} and {B}, a ferry or a bridge is needed to get over the other."],
    5: ["Of {A} and {B}, one is a needle and the other the cloth it has been drawn right through.",
        "Of {A} and {B}, one is a tunnel bored straight through the mountain of the other.",
        "Of {A} and {B}, one is an express train and the other a station it runs through without stopping.",
        "Of {A} and {B}, one is a road and the other a field it runs across, hedge to hedge.",
        "Of {A} and {B}, one is a crack running clean across the windowpane of the other."],
}

OVERLAPS = {
    1: ["{A} and {B} partly cover the same ground.",
        "{A} and {B} lap over each other at the edges.",
        "{A} and {B} share some ground, though each also has ground of its own.",
        "{A} reaches part way into {B}, and {B} part way into {A}.",
        "{A} and {B} run into each other without either taking in the whole of the other.",
        "{A} and {B} are partly superimposed."],
    2: ["The outlines of {A} and {B} cut through each other, closing off a common patch.",
        "The boundary of {A} runs through the interior of {B}, and the boundary of {B} through the interior of {A}.",
        "A map shaded for {A} and for {B} shows ground with both shadings, ground only for {A}, and ground only for {B}.",
        "{A} and {B} have a zone in common that is smaller than either of them.",
        "Part of {A} and part of {B} are the very same ground."],
    3: ["Tracing the border of {A}, you dip into {B} for a while and then come back out.",
        "Walking from the far side of {A} to the far side of {B}, a middle stretch of the walk lies in both.",
        "Starting in a part of {A} that is not {B}, you can reach a part of {B} that is not {A} by way of ground belonging to both.",
        "A surveyor fixing the outline of {B} finds part of it running through {A}."],
    4: ["Statistics compiled for {A} and for {B} both count the same strip of ground, and each also counts ground the other leaves out.",
        "Some rain falls on {A} and {B} at once, some on {A} alone, and some on {B} alone.",
        "Clearing all of {A} would take away part of {B} without taking all of it, and the same holds the other way round."],
    5: ["{A} and {B} are two coins laid so that each half covers the other.",
        "{A} and {B} are two shadows that fall partly across each other.",
        "{A} and {B} are two playing cards fanned so that each hides part of the other.",
        "{A} and {B} are two rings of ripples on a pond, each running partly into the other."],
}

DISJOINT = {
    1: ["{A} and {B} have no point in common.",
        "{A} and {B} do not meet anywhere.",
        "{A} and {B} share neither ground nor border.",
        "{A} lies wholly apart from {B}.",
        "{A} and {B} are unconnected areas.",
        "Some distance always remains between {A} and {B}."],
    2: ["On a map, {A} and {B} are drawn as separate shapes with other territory or water between them.",
        "The nearest edges of {A} and {B} are still some way apart.",
        "{A} and {B} are separated all round by space belonging to neither.",
        "{A} and {B} are two separate pieces of the map, not joined at any point."],
    3: ["Walking the whole border of {A}, you never reach {B}, which lies outside the loop you have walked.",
        "A map of {A} cut exactly to its outline shows no part of {B}.",
        "A ring drawn around {A} a short distance from its edge can leave all of {B} outside it."],
    4: ["An outbreak confined to {A} could not reach {B} without first spreading over other ground or water.",
        "A wall built all the way around {A} could stand without meeting {B} anywhere."],
    5: ["{A} and {B} are two pebbles lying apart on the sand.",
        "{A} and {B} are two raindrops on a windowpane that never run together.",
        "{A} and {B} are guests at opposite ends of a long table, never brushing elbows.",
        "{A} and {B} are two boats moored at separate piers."],
}

EQUALS = {
    1: ["{A} and {B} are one and the same area under two names.",
        "{A} and {B} cover exactly the same ground.",
        "The outline of {A} is the outline of {B}."],
    2: ["Every point of {A} is a point of {B}, and every point of {B} a point of {A}.",
        "{A} and {B} have identical boundaries and identical area.",
        "Laid on top of each other, the outlines of {A} and {B} match everywhere."],
    3: ["Whoever stands in {A} stands in {B}, and whoever stands in {B} stands in {A}.",
        "Leaving {A} at any point means leaving {B} at the same moment.",
        "There is no spot that belongs to {A} but not to {B}, nor to {B} but not to {A}."],
    4: ["A storm over all of {A} is a storm over all of {B}, and no more.",
        "Erasing {A} from the map erases all of {B}, and erasing {B} erases all of {A}.",
        "Shading {A} on a map shades exactly the ground that shading {B} would."],
    5: ["{A} and {B} are two labels stuck on the same jar.",
        "{A} and {B} are two names written on the same door.",
        "{A} and {B} are one house with two street numbers."],
}

NEW = {"contains": CONTAINS, "within": WITHIN, "touches": TOUCHES, "crosses": CROSSES,
       "overlaps": OVERLAPS, "disjoint": DISJOINT, "equals": EQUALS}
