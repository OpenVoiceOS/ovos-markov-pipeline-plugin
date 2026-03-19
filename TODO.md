# TODO

- [x] Initial pipeline plugin (MarkovPipeline, MarkovIntentEngine)
- [x] Stemming support (snowball stemmer)
- [x] Character-level fallback (blended 60/40 word/char when word scores ambiguous)
- [x] Online learning (incremental model update on high-confidence matches)
- [x] File-based intent registration
- [ ] Entity extraction via HMM (BIO tagging)
- [ ] ONNX export for trained intents (disk cache)
- [ ] Confidence calibration (precision/recall analysis)
- [ ] Hybrid pipeline (Markov + Model2Vec two-stage)
- [ ] ovoscope E2E tests
- [ ] GitHub workflows (ovos-workflows-adder)
