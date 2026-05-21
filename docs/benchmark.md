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
`test_match` utterances are deliberately *not* template fills — they use
contractions, filler words, politeness markers, and word-order variation,
as real STT output does. The 24 `NO_MATCH_UTTERANCES` are plausible but
off-topic, several sharing words with real intents.

## Metrics

Each engine reports two numbers:

- **Argmax recall** — how often the top-ranked intent is correct, with no
  confidence threshold. This is the method ceiling.
- **Threshold metrics** — accuracy / precision / recall / F1 with
  no-match gating at a fixed confidence threshold of `0.5`, matching the
  nebulento benchmark methodology so the rows are directly comparable.

A prediction is a true positive when the predicted intent equals the
expected intent. A no-match case is correct only when the engine returns
`None` or a confidence below threshold.

## Results

Single run, all engines on the same machine, same dataset:

| Engine | Argmax recall | Acc @0.5 | Precision @0.5 | Recall @0.5 | F1 @0.5 | FP / 24 | Median lat |
|---|---|---|---|---|---|---|---|
| nebulento token-set-ratio | **53.7%** | **51.5%** | 88.5% | **53.7%** | **0.668** | 17 | 2.1 ms |
| markov order=1 | 32.4% | 9.0% | 50.0% | 0.4% | 0.008 | **1** | **1.0 ms** |
| markov order=2 (default) | 27.0% | 9.7% | 66.7% | 1.6% | 0.032 | 2 | **1.0 ms** |
| markov order=2 char_fallback | 42.2% | 11.6% | 81.8% | 3.7% | 0.071 | 2 | 17.8 ms |

FP = false positives on the 24 no-match utterances.

## Interpreting the results

**The Markov engine trails fuzzy matching on this dataset.** Best-case
argmax recall is 42.2% (`char_fallback`) against nebulento's 53.7%. A
per-intent Markov chain scores an utterance by n-gram perplexity, so it
rewards utterances that share n-grams with the training templates.
Natural-phrasing test utterances rarely share 2-grams with a terse
template, which drives perplexity up. Fuzzy token-set matching is more
robust to paraphrase and word reordering because it compares token sets,
not token sequences.

**The threshold columns understate the engine — perplexity-derived
confidence is not calibrated to a 0–1 scale.** `conf = 1 / (1 + log(ppx))`
puts correct predictions around 0.2–0.5, so the benchmark's fixed `0.5`
gate rejects almost every correct match and the gated recall collapses to
near zero. The numbers are reported at `0.5` only for methodological
parity with the published nebulento table. In practice, calibrate with
`calibration.find_optimal_thresholds()` (see [Calibration](calibration.md))
and the pipeline's `conf_high` / `conf_med` / `conf_low` tiers rather than
using a literal `0.5`. The argmax-recall column is the threshold-free
comparison.

**`char_fallback` helps recall but costs latency.** It lifts argmax recall
from 27.0% to 42.2% by blending a character-level model when the top two
word-level scores are close — useful against spelling and morphology
variation — but raises median query latency from ~1 ms to ~18 ms.

**Higher n-gram order does not help here.** `order=1` (32.4%) beats
`order=2` (27.0%) on argmax recall: with only a handful of short templates
per intent, order-2 contexts are too sparse to generalise, and order-1
also handles single-word utterances (`order=2` cannot score utterances
shorter than two tokens).

**Latency:** the plain Markov configs are the fastest engines measured
(~1 ms median), ahead of nebulento's ~2 ms. `char_fallback` is the
exception at ~18 ms.

## How to run

```bash
pip install -e .[benchmark]   # installs nebulento for the baseline
python benchmark/compare.py
```

The nebulento baseline is skipped automatically if nebulento is not
installed; the Markov rows still run.
