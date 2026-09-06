# The loop

1. Paste one prompt into Gemini. Save what it returns.
2. Send me the file.
3. I check every template: that it never names its own predicate,
   that it uses the right placeholders, that no two are duplicates,
   and that each cell has enough.
4. Templates are split so that the ones used in training never appear
   in evaluation. Without that a fine-tuned model memorises the
   phrase rather than reading it, and few-shot demonstrations hand
   the mapping over outright.
5. Anything short or rejected comes back to you as a follow-up prompt
   naming exactly which predicate and level need more.
