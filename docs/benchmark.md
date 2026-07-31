# Benchmark

`benchmark/compare.py` measures the Markov intent engine's accuracy and
speed against the [nebulento](https://github.com/OpenVoiceOS/nebulento)
benchmark dataset, and runs nebulento's strongest fuzzy strategy
alongside as a same-machine baseline.

## Dataset

`benchmark/dataset.py` is ported verbatim from nebulento:

```text
Dataset : 268 cases  (244 match, 24 no-match)
Intents : 22
Note    : test utterances are natural human phrasing, NOT template fills.
```

22 intents across media, smart home, timers, alarms, weather, calendar,
communication, navigation, reminders, shopping, and system control. The
`test_match` utterances are deliberately *not* template fills: they use
contractions, filler words, politeness markers, and word-order variation,
as real STT output does. The 24 `NO_MATCH_UTTERANCES` are plausible but
off-topic, several sharing words with real intents.

## Metrics

Each engine reports:

- **Argmax recall**: how often the top-ranked intent is correct, with no
  confidence threshold. This is the method ceiling.
- **AUC**: ROC-AUC of the confidence separating correct argmax matches
  from wrong / no-match cases. How well the confidence discriminates.
- **F1 @0.5**: F1 with no-match gating at a fixed `0.5` threshold,
  matching the nebulento benchmark methodology.
- **F1 @best**: F1 at the F1-optimal threshold, swept per engine.

The Markov engine's confidence is a softmax posterior over the per-intent
perplexities, so it reflects how far the winning intent outscored the
rest. nebulento's confidence is its fuzzy match ratio.

## Results

Single run, all engines on the same machine, same dataset:

| Engine | Argmax recall | AUC | F1 @0.5 | F1 @best | best thr | Median lat |
|---|---|---|---|---|---|---|
| nebulento token-set-ratio | **53.7%** | 0.764 | **0.668** | **0.668** | 0.49 | 2.1 ms |
| markov order=1 | 32.4% | **0.927** | 0.329 | 0.476 | 0.13 | **1.2 ms** |
| markov order=2 (default) | 27.0% | 0.915 | 0.282 | 0.399 | 0.05 | **1.2 ms** |
| markov order=2 char_fallback | 45.1% | 0.799 | 0.282 | 0.587 | 0.08 | 12.3 ms |

## Interpreting the results

**The Markov engine trails fuzzy matching on this dataset.** Best-case
argmax recall is 45.1% (`char_fallback`) against nebulento's 53.7%. A
per-intent Markov chain scores an utterance by n-gram perplexity, so it
rewards utterances that share n-grams with the training templates.
Natural-phrasing test utterances rarely share 2-grams with a terse
template, which drives perplexity up. Fuzzy token-set matching handles
paraphrase and word reordering better because it compares token sets, not
token sequences.

**The confidence discriminates well.** The softmax posterior gives the
word-level configs an AUC around 0.92, higher than nebulento's 0.764, so the
confidence reliably separates correct matches from wrong ones. A correct
match lands around 0.6, a wrong one around 0.05.

**The F1 ceiling is argmax recall.** F1 @best is ~0.48 (`order=1`) and
~0.59 (`char_fallback`), against nebulento's 0.668. Confidence quality
does not close that gap: the 55–73% of natural-phrasing utterances the
n-gram chains do not rank correctly stay wrong regardless of threshold.
To raise F1, raise argmax recall: more or broader training templates, or
a different matcher.

**Use a calibrated threshold, not a literal 0.5.** The F1-optimal
threshold is 0.05–0.13, below 0.5. In production, calibrate with
`calibration.find_optimal_thresholds()` (see [Calibration](calibration.md))
and the pipeline's `conf_high` / `conf_med` / `conf_low` tiers.

**`char_fallback` helps recall but costs latency.** It lifts argmax
recall from 27.0% to 45.1% by blending a character-level model when the
top two word-level scores are close, which helps against spelling and
morphology variation. It raises median query latency from ~1 ms to
~12 ms, though, and lowers AUC (0.92 → 0.80): the blend narrows the posterior
margin between intents.

**Higher n-gram order does not help here.** `order=1` beats `order=2` on
argmax recall (32.4% vs 27.0%): with only a handful of short templates
per intent, order-2 contexts are too sparse to generalise, and order-1
also handles single-word utterances (`order=2` cannot score utterances
shorter than two tokens).

**Latency:** the plain Markov configs are the fastest engines measured
(~1.2 ms median), ahead of nebulento's ~2.1 ms. `char_fallback` is the
exception at ~12 ms.

## How to run

```bash
pip install -e .[benchmark]   # installs nebulento for the baseline
python benchmark/compare.py
```

The nebulento baseline is skipped automatically if nebulento is not
installed. The Markov rows still run.

---
[← Model caching](caching.md) · [Home](index.md)
