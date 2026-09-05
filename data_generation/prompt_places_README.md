# The loop

1. Paste one prompt into Gemini. Save what it returns as a text file.
2. Send me the file.
3. I geocode every place against OpenStreetMap, discard whatever does not
   resolve or resolves to the wrong kind of thing, and report the counts.
4. I compute the relations, the labels and the difficulty levels from the
   geometry, and generate the rows.
5. If a cell is short, I hand you a follow-up prompt naming exactly what is
   missing.

Gemini is never asked which relation holds or how hard an item is. Both are
computed here, which is why its errors last time -- Indianapolis 'equals'
Marion County, rivers 'crossing' states they lie entirely within -- cannot
recur through this route.
