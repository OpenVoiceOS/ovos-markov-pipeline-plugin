# Suggestions

1. **Stemming support** — Add optional snowball stemmer (like Padatious) to normalize word forms before training/matching.
2. **Entity extraction via HMM** — Use markovonnx's `HiddenMarkovModel` with supervised BIO tagging for slot filling.
3. **ONNX inference** — Export trained intent models to ONNX for even faster inference at scale.
4. **Confidence calibration** — The `1/(1+log(ppx))` formula is heuristic. Calibrate against held-out data for better threshold selection.
