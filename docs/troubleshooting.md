# Troubleshooting

## Nothing matches at all

- **Too few words.** An utterance must have at least `order` words to be
  scored. With `order: 2`, single-word utterances are never matched. Lower
  `order` to `1` if your intents include one-word phrases.
- **Not trained yet.** With `instant_train: false`, engines train lazily on the
  first match request. If you are driving the engine directly, call `train()`
  before `calc_intents()`.
- **No intents registered.** Confirm skills loaded and emitted
  `padatious:register_intent`. Query `intent.service.markov.manifest.get` to
  see the registered intent list.

## Wrong intent matched

- **Order too low or too high.** Order 1 can confuse intents that differ only
  in word order; a high order with few samples overfits. See [Tuning](tuning.md).
- **Overlapping vocabulary.** If two intents share most of their words, add
  more distinctive samples, or enable `char_fallback` for close calls.
- **Thresholds off.** Run [`find_optimal_thresholds()`](calibration.md) on
  labelled data and reset the `conf_*` values.

## Everything matches at low confidence

With only one or two intents registered there are no competing models, so even
unrelated utterances score above `conf_low`. This is expected. Confidence
becomes discriminative once several intents compete; until then, rely on the
high tier.

## Non-English matching is weak

Install the stemming extra (`pip install ovos-markov-pipeline-plugin[stem]`)
and set `stem: true`. Stemming normalizes word forms and is available for 26
languages. An unsupported language silently runs without stemming.

## Threshold-ordering warning in the log

The plugin logs a warning when `conf_low <= conf_med <= conf_high` does not
hold. Fix the ordering in `mycroft.conf`.

## FAQ

**What is this plugin?** An OVOS pipeline plugin that classifies intents by
training one Markov chain per intent and picking the model with the lowest
perplexity. See [Concepts](concepts.md).

**How does confidence scoring work?** Confidence is a softmax over every
intent's log-likelihood (`-log(perplexity)`), so the scores sum to 1 and
reflect how far the winning intent outscored the rest. Tune the thresholds
with [`find_optimal_thresholds()`](calibration.md).

**What order should I use?** Order 1 for small training sets (5-20 examples per
intent), order 2 for 20 or more. See [Tuning](tuning.md).

**Does stemming help?** Yes for varied word forms and morphologically rich
languages. Set `stem: true` and install the `stem` extra.

**What is the character-level fallback?** When `char_fallback` is enabled and
the top two word scores are within `char_fallback_threshold`, a 60/40
word/character blend breaks the tie using spelling cues.

**What is online learning?** When `online_learning` is enabled, matches above
`conf_high` are appended to that intent's samples and the model retrains, so it
adapts to real phrasings.

**Does it support entities/slots?** Yes — `SlotExtractor` uses an HMM with BIO
tagging. See [Entity extraction](entities.md).

**Can I cache models to disk?** Yes — `IntentCache` exports trained models to
ONNX plus a vocabulary JSON for instant reload. See [Model caching](caching.md).
