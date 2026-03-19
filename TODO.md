# TODO

- [x] Initial pipeline plugin (MarkovPipeline, MarkovIntentEngine)
- [x] Stemming support (snowball stemmer)
- [x] Character-level fallback (blended 60/40 word/char when ambiguous)
- [x] Online learning (incremental model update on high-confidence matches)
- [x] File-based intent registration
- [x] Entity extraction via HMM (BIO tagging) — `slots.py`
- [x] ONNX export cache for trained intents — `cache.py`
- [x] Confidence calibration (evaluate + find_optimal_thresholds) — `calibration.py`
- [x] GitHub workflows (ovos-workflows-adder)
- [x] OVOS version block format
- [x] Fix all audit issues (race condition, input validation, etc.)
- [x] ovoscope E2E tests (hello-world + naptime skills, multi-skill routing)
