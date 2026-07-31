# ovos-markov-pipeline-plugin

An OVOS intent pipeline plugin that classifies utterances with an ensemble of
per-intent Markov chains, scored by perplexity.

It trains one word-level Markov chain per intent from example utterances. At
query time it computes the perplexity of the utterance under every model and
picks the intent whose model is least "surprised": the lowest perplexity wins.
Confidence comes from perplexity, so the plugin fits into the OVOS high,
medium, and low confidence pipeline tiers.

## Why a Markov pipeline?

Most OVOS intent matchers either pattern-match (Adapt, padacioso) or train a
neural or SVM classifier (padatious, linha-fina). A Markov-chain language
model sits in between. It is a lightweight statistical model that learns the
*word order* of each intent's sample set. It trains in milliseconds, needs no
GPU, and degrades gracefully with very few samples. It is a practical baseline
for small skill sets and a useful comparison point against the other
matchers.

## Install

```bash
pip install ovos-markov-pipeline-plugin
```

Optional stemming support (recommended for non-English use):

```bash
pip install ovos-markov-pipeline-plugin[stem]
```

## Quick example

Enable the pipeline in `mycroft.conf`:

```json
{
  "intents": {
    "pipeline": [
      "ovos-markov-pipeline-plugin-high",
      "ovos-markov-pipeline-plugin-medium",
      "ovos-markov-pipeline-plugin-low"
    ]
  }
}
```

Skills register their intents over the bus exactly as they do for padatious.
No skill-side changes are required.

## Documentation

Full documentation lives in [`docs/`](docs/index.md):

- [Concepts](docs/concepts.md): how Markov-chain perplexity matching works
- [Quickstart](docs/quickstart.md): install, configure, run
- [Configuration](docs/configuration.md): every config key explained
- [Pipeline integration](docs/pipeline.md): OPM entry point and bus protocol
- [Entity extraction](docs/entities.md): HMM BIO slot tagging
- [Calibration](docs/calibration.md): measuring and tuning confidence
- [Model caching](docs/caching.md): ONNX disk cache
- [Tuning](docs/tuning.md): order, smoothing, stemming, fallbacks
- [Troubleshooting](docs/troubleshooting.md): common issues and FAQ

## License

Apache-2.0
