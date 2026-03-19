# FAQ

## What is this plugin?
An OVOS pipeline plugin that classifies intents by training one Markov chain per intent and selecting the model with lowest perplexity.

## How does confidence scoring work?
`confidence = 1 / (1 + log(perplexity))`. Lower perplexity = higher confidence. Use `calibration.find_optimal_thresholds()` to tune thresholds empirically.

## What order should I use?
Order 1 for small training sets (5-20 examples). Order 2 for 20+ examples per intent.

## Does stemming help?
Yes, set `"stem": true`. Uses snowball stemmer for 26 languages.

## What is the character-level fallback?
When `"char_fallback": true`, char-level models are trained alongside word-level. If top-2 word scores are within 0.05, a 60/40 word/char blend breaks the tie.

## What is online learning?
When `"online_learning": true`, high-confidence matches are fed back into the model incrementally — no full retrain.

## Does it support entities/slots?
Yes — `SlotExtractor` uses HMM Viterbi decoding with BIO tags. Train with `(utterance, BIO-tags)` pairs, then call `extract(intent, tokens)` to get `{slot_name: value}`.

## Can I cache models to disk?
Yes — `IntentCache` exports trained models as ONNX + vocab JSON for instant reload on restart.

## How do I calibrate thresholds?
Use `calibration.evaluate()` for accuracy/F1 metrics and `find_optimal_thresholds()` to grid-search the best confidence cutoff on labelled data.

## What CI workflows are included?
Standard OVOS set: build-tests, lint, coverage, release workflow, publish stable, OPM check, license check, pip audit, repo health, release preview.
