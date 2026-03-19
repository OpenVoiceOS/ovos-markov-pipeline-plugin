# Suggestions

1. **ovoscope E2E tests** — Create a minimal test skill with `.intent` files and run through MiniCroft.
2. **Configurable char fallback threshold** — Expose the 0.05 gap as a config option.
3. **Integrate SlotExtractor into pipeline** — Wire `SlotExtractor.extract()` into `_match_level` so matched intents automatically include entities in `match_data`.
4. **Hybrid Markov+M2V** — Use Markov as fast pre-filter, Model2Vec as semantic reranker.
5. **Sparse ONNX shape fix** — Add a Squeeze node after Gather in sparse export to produce `[V]` instead of `[1, V]`.
