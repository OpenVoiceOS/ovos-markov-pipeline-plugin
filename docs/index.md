# ovos-markov-pipeline-plugin documentation

`ovos-markov-pipeline-plugin` is an OVOS intent pipeline plugin that classifies
utterances with an ensemble of per-intent Markov chains, scored by perplexity.

This documentation goes from zero to hero. If you have never heard of a Markov
chain, start with [Concepts](concepts.md). If you just want it running, jump to
the [Quickstart](quickstart.md).

## Reading order

For users integrating the plugin into an OVOS install:

1. [Concepts](concepts.md) — the idea behind perplexity-based intent matching
2. [Quickstart](quickstart.md) — install and enable the pipeline
3. [Configuration](configuration.md) — every config key, with defaults
4. [Tuning](tuning.md) — get the best accuracy for your skill set
5. [Troubleshooting](troubleshooting.md) — when something does not match

For developers building on or extending the plugin:

1. [Pipeline integration](pipeline.md) — the OPM entry point and bus protocol
2. [Entity extraction](entities.md) — HMM BIO slot tagging with `SlotExtractor`
3. [Calibration](calibration.md) — `evaluate()` and `find_optimal_thresholds()`
4. [Model caching](caching.md) — exporting trained models to ONNX
5. [Benchmark](benchmark.md) — accuracy and speed against the nebulento dataset

## At a glance

```text
utterance
  -> normalize (lowercase, strip punctuation, optional stem)
  -> word tokenize
  -> perplexity under each per-intent Markov chain
  -> lowest perplexity wins
  -> confidence = softmax over per-intent log-likelihoods
  -> optional: char-level fallback blend when the top two are close
  -> optional: HMM BIO slot extraction on the matched intent
```

## Package layout

| Module           | Public API                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `__init__.py`    | `MarkovPipeline` (OPM entry point), `MarkovIntentEngine`           |
| `slots.py`       | `SlotExtractor` — HMM BIO entity extraction                        |
| `cache.py`       | `IntentCache` — ONNX disk cache for trained models                 |
| `calibration.py` | `evaluate()`, `find_optimal_thresholds()`                          |

## Requirements

- Python 3.10+
- [`markovonnx`](https://pypi.org/project/markovonnx/) — Markov chain models
- `ovos-plugin-manager`, `ovos-bus-client`, `ovos-config`, `ovos-utils`
- `snowballstemmer` (optional, for stemming) — install the `stem` extra
