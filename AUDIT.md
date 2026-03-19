# Audit

## Known Issues

1. **Small training set accuracy** — With fewer than ~10 examples per intent, word-level Markov chains may not discriminate well between intents with overlapping vocabulary. Use order=1 for small datasets. — `ovos_markov_pipeline/__init__.py:141`
2. **No entity extraction** — Plugin identifies intents only, not entity slots. Needs pairing with Adapt or a slot-filling plugin. — design limitation
3. **No stemming** — Unlike Padatious, no stemmer is applied to normalize word forms. Could improve matching for morphologically rich languages. — `ovos_markov_pipeline/__init__.py:37`
