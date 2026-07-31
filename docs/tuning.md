# Tuning

This page gives practical advice on getting the best accuracy out of the
plugin for a given skill set.

## Choose the n-gram order

`order` is the highest-impact setting.

- **`order: 1`**: the model looks at single words (a bag-of-words feel). Best
  for small sample sets, roughly 5-20 examples per intent. It generalizes well
  and rarely overfits.
- **`order: 2`** (default): the model looks at word pairs and learns short
  phrasings. Best when intents have 20 or more examples each.
- **`order: 3+`**: only worthwhile with large, consistent sample sets. With
  few samples a high order memorizes the training data and fails on unseen
  phrasings.

When unsure, start at `order: 1` for a new skill set and raise it if you have
plenty of samples and see misclassifications between similar intents.

## Smoothing and backoff

`kneser_ney` and `backoff` both help the model handle words and n-grams it did
not see in training. Leave both enabled (the default) unless you are
benchmarking. `smoothing` rarely needs changing. Raise it slightly if matches
feel over-confident on short utterances.

## Stemming

Set `stem: true` (and install the `stem` extra) when intents use varied word
forms: "running", "runs", "ran" all collapse to one stem, so a small sample
set covers more phrasings. Stemming helps most for morphologically rich
languages and for small training sets. It is available for 26 languages. An
unsupported language simply runs without it.

## Character-level fallback

Enable `char_fallback` when short, similar utterances are confused: for
example "lights on" vs "lights off". When the top two word scores land within
`char_fallback_threshold` (default 0.05), a 60/40 word/character blend uses
spelling cues to break the tie. The cost is a second set of models trained per
intent, so leave it off unless you observe close calls.

## Online learning

`online_learning: true` appends every match scoring above `conf_high` back into
that intent's samples and retrains. The model adapts to how users actually
phrase requests. Use it where utterances are reasonably reliable. Avoid it on
noisy STT input, since a confident wrong match becomes a training sample.

## Calibrate the thresholds

The defaults are sensible, but the perplexity-to-confidence curve depends on
your data. Collect a labelled set of `(utterance, intent)` pairs and run
[`find_optimal_thresholds()`](calibration.md) to derive `conf_high`,
`conf_med` and `conf_low` empirically. This is the single most reliable way to
reduce both false matches and missed matches.

## A practical workflow

1. Start with `order: 1`, defaults otherwise.
2. Build a small labelled evaluation set.
3. Run [`evaluate()`](calibration.md): check accuracy and the confidence gap
   between correct and incorrect predictions.
4. If similar intents are confused, raise `order` (if you have the samples) or
   enable `char_fallback`.
5. Run `find_optimal_thresholds()` and set the three `conf_*` values.
6. Re-evaluate and iterate.

---
[← Configuration](configuration.md) · [Home](index.md) · [Troubleshooting →](troubleshooting.md)
