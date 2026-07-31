# Concepts

This page explains *how* the plugin decides which intent an utterance belongs
to. No prior knowledge is assumed.

## The problem: intent classification

A skill registers an intent with a handful of example sentences:

```text
weather.intent
  what is the weather
  what's the forecast
  tell me the weather
  is it going to rain
```

When a user says "what is the weather today", the pipeline must decide which
registered intent that sentence belongs to. That is a text classification
problem.

## What a Markov chain is

A Markov chain is a simple statistical model of sequences. Given the words seen
so far, it estimates the probability of the next word. An *order-2* chain looks
at the previous two words; an *order-1* chain looks at the previous one word.

Training a chain on the `weather.intent` samples above teaches it that, in this
intent, "the" is very likely to be followed by "weather" or "forecast", and
"is" is often followed by "it". The chain has learned the *word order* typical
of that intent.

## Perplexity: how surprised is the model?

**Perplexity** measures how surprised a trained model is by a new sentence. A
low perplexity means "this sentence looks like my training data"; a high
perplexity means "this is unexpected".

The plugin trains one Markov chain per intent. To classify a new utterance it
computes the utterance's perplexity under *every* chain and picks the intent
whose chain is least surprised:

```text
utterance: "what is the weather today"

  weather.intent  chain -> perplexity 12.4   <- lowest, winner
  timer.intent    chain -> perplexity 89.1
  lights.intent   chain -> perplexity 140.7
```

This is an *ensemble of generative models used as a discriminative classifier*:
each model only ever learns its own intent, and classification is a comparison
across models.

## From perplexity to confidence

OVOS pipelines work in confidence scores between 0 and 1, not perplexity. The
plugin maps one to the other with a softmax over every intent's
log-likelihood:

```text
log_likelihood(intent) = -log(perplexity(intent))
confidence(intent)     = softmax(temperature * log_likelihood)[intent]
```

The confidences sum to 1 across intents, so each one reflects how far that
intent outscored the rest, not just its own perplexity. A decisive winner
approaches 1.0; a close call spreads the mass out. See
[Calibration](calibration.md) for measuring whether the thresholds derived
from it suit your data.

## Confidence tiers

OVOS runs pipeline plugins in three passes: `match_high`, `match_medium`,
`match_low`. The plugin exposes the same scoring at three thresholds
(`conf_high`, `conf_med`, `conf_low`). A match is only returned when the best
confidence exceeds the threshold for that tier, so a high-confidence intent is
matched before any lower-confidence pipeline plugin gets a turn.

## Optional refinements

The core word-order model is supplemented by a few opt-in features:

- **Stemming** normalizes word forms ("running" -> "run") so small sample sets
  generalize better. See [Tuning](tuning.md).
- **Character-level fallback** trains a parallel set of character n-gram models.
  When the two best word-level scores are nearly tied, a 60/40 word/char blend
  breaks the tie using sub-word spelling cues.
- **Online learning** feeds high-confidence matches back into the model so it
  adapts to how a user actually phrases things.
- **Entity extraction** uses a Hidden Markov Model with BIO tagging to pull
  slot values out of the matched utterance. See [Entity extraction](entities.md).

## Where Markov matching fits

Compared to other OVOS intent matchers:

- **Adapt / padacioso** match keywords and patterns; they need no training but
  cannot generalize beyond what is written.
- **padatious / linha-fina** train a neural network / SVM classifier; accurate,
  but heavier to train.
- **Markov** trains a per-intent language model in milliseconds with no GPU and
  degrades gracefully with very few samples.

It is a strong lightweight baseline and a useful comparison point. For large,
overlapping skill sets a discriminative classifier will usually edge it out on
accuracy.

---
[Home](index.md) · [Quickstart →](quickstart.md)
