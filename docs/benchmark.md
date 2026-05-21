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

Each engine reports:

- **Argmax recall** — how often the top-ranked intent is correct, with no
  confidence threshold. This is the method ceiling.
- **AUC** — ROC-AUC of the confidence separating correct argmax matches
  from wrong / no-match cases. How *trustworthy* the confidence is.
- **F1 @0.5** — F1 with no-match gating at a fixed `0.5` threshold,
  matching the nebulento benchmark methodology.
- **F1 @best** — F1 at the F1-optimal threshold, swept per engine.

## Confidence variants

The Markov engine's shipped confidence is an *absolute* transform of one
intent's perplexity: `conf = 1 / (1 + log(ppx))`. The benchmark also
evaluates a **relative** rescoring — a softmax posterior over every
intent's log-likelihood (`relative_conf` in `compare.py`, temperature
`β=3`). The argmax is unchanged (softmax is monotonic), so relative
scoring moves only the threshold-gated metrics and AUC, never argmax
recall. This rescoring lives in the benchmark; it is a prototype of a
possible engine change, not the engine's current behaviour.

## Results

Single run, all engines on the same machine, same dataset:

| Engine | Argmax recall | AUC | F1 @0.5 | F1 @best | best thr | Median lat |
|---|---|---|---|---|---|---|
| nebulento token-set-ratio | **53.7%** | 0.764 | **0.668** | **0.668** | 0.49 | 2.3 ms |
| markov order=1 absolute | 32.4% | 0.864 | 0.008 | 0.473 | 0.17 | **1.3 ms** |
| markov order=1 relative β=3 | 32.4% | **0.927** | 0.329 | 0.476 | 0.13 | **1.3 ms** |
| markov order=2 char_fallback absolute | 42.2% | 0.761 | 0.071 | 0.558 | 0.19 | 19 ms |
| markov order=2 char_fallback relative β=3 | 42.2% | 0.809 | 0.339 | 0.560 | 0.06 | 19 ms |

## Interpreting the results

**The Markov engine trails fuzzy matching on this dataset.** Best-case
argmax recall is 42.2% (`char_fallback`) against nebulento's 53.7%. A
per-intent Markov chain scores an utterance by n-gram perplexity, so it
rewards utterances that share n-grams with the training templates.
Natural-phrasing test utterances rarely share 2-grams with a terse
template, which drives perplexity up. Fuzzy token-set matching is more
robust to paraphrase and word reordering because it compares token sets,
not token sequences.

**Relative scoring fixes the confidence, not the matching.** The shipped
absolute confidence is nearly unusable at a fixed threshold — F1 @0.5 is
0.008 for `order=1`, because `conf = 1/(1+log(ppx))` puts correct matches
around 0.2–0.5 and the `0.5` gate rejects almost all of them. Rescoring
relatively lifts F1 @0.5 from 0.008 to 0.329 and AUC from 0.864 to 0.927:
the confidence now reflects how far the winning intent beat the rest, so
a fixed threshold is far less brittle. But **F1 @best barely moves**
(0.473 → 0.476): the best achievable F1 is bounded by argmax recall, and
relative scoring does not change which intent wins. Its value is making
the confidence trustworthy at a fixed or lightly-calibrated threshold —
not lifting the ceiling.

**The F1 ceiling is argmax recall.** F1 @best is ~0.47 (`order=1`) and
~0.56 (`char_fallback`), against nebulento's 0.668. No confidence
treatment closes that gap; the 58–68% of natural-phrasing utterances the
n-gram chains do not rank correctly stay wrong. To raise F1, raise argmax
recall — more or broader training templates, or a different matcher.

**Use a calibrated threshold, never a literal 0.5.** Even with relative
scoring the F1-optimal threshold is 0.06–0.17, not 0.5. In production,
calibrate with `calibration.find_optimal_thresholds()` (see
[Calibration](calibration.md)) and the pipeline's `conf_high` /
`conf_med` / `conf_low` tiers.

**`char_fallback` helps recall but costs latency.** It lifts argmax
recall from 27.0% to 42.2% by blending a character-level model when the
top two word-level scores are close — useful against spelling and
morphology variation — but raises median query latency from ~1 ms to
~19 ms. It also lowers AUC (0.86 → 0.81 with relative scoring): the blend
muddies the score margin.

**Higher n-gram order does not help here.** `order=1` beats `order=2` on
argmax recall (32.4% vs 27.0%, measured separately): with only a handful
of short templates per intent, order-2 contexts are too sparse to
generalise, and order-1 also handles single-word utterances (`order=2`
cannot score utterances shorter than two tokens).

**Latency:** the plain Markov configs are the fastest engines measured
(~1 ms median), ahead of nebulento's ~2 ms. `char_fallback` is the
exception at ~19 ms.

## How to run

```bash
pip install -e .[benchmark]   # installs nebulento for the baseline
python benchmark/compare.py
```

The nebulento baseline is skipped automatically if nebulento is not
installed; the Markov rows still run.
