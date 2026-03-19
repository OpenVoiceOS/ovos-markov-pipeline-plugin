# Audit

## Known Issues

1. **Small training set accuracy** — With fewer than ~10 examples per intent, word-level Markov chains may not discriminate well. Use order=1 for small datasets. — `ovos_markov_pipeline/__init__.py:206`
2. **No entity extraction** — Intent-only; no slot filling. Needs HMM BIO tagger. — design limitation
3. **Char fallback threshold** — The 0.05 score-gap threshold for triggering char fallback is hardcoded. Should be configurable. — `ovos_markov_pipeline/__init__.py:263`
4. **Online learning vocab drift** — `update_online` adds new tokens to vocab but doesn't rebuild existing models' count vectors, which remain the original size. Harmless but wastes memory. — `ovos_markov_pipeline/__init__.py:300`
