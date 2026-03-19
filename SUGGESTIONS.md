# Suggestions

1. **Entity extraction via HMM** — Use markovonnx HiddenMarkovModel with supervised BIO tagging for slot filling.
2. **ONNX inference cache** — Export trained models to ONNX and cache to disk (like Padatious intent_cache).
3. **Confidence calibration** — Measure precision/recall on real OVOS skill data, tune the `1/(1+log(ppx))` formula.
4. **Hybrid Markov+M2V pipeline** — Use Markov as fast pre-filter (top-3), Model2Vec as semantic reranker.
5. **Configurable char fallback threshold** — Make the 0.05 gap threshold a config option.
