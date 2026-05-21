# Quickstart

This page gets the plugin installed and matching intents.

## 1. Install

```bash
pip install ovos-markov-pipeline-plugin
```

For non-English languages, install the stemming extra so word forms are
normalized before training:

```bash
pip install ovos-markov-pipeline-plugin[stem]
```

## 2. Enable the pipeline

Add the three confidence tiers to the `intents` pipeline in `mycroft.conf`:

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

The `-high`, `-medium` and `-low` suffixes are appended automatically by OVOS
for any `ConfidenceMatcherPipeline`; the entry point itself is just
`ovos-markov-pipeline-plugin`. Place the high tier early in the pipeline and
the low tier late, interleaving other matchers as you wish.

## 3. Restart and use

Restart `ovos-core`. Skills register their intents over the bus on load — no
skill-side change is needed. The plugin trains its models on first use (or
immediately, if `instant_train` is set) and starts matching.

## Using the engine directly

The matching engine can be used without OVOS, which is handy for experiments,
notebooks and tests:

```python
from ovos_markov_pipeline import MarkovIntentEngine

engine = MarkovIntentEngine(order=2)
engine.add_intent("weather", [
    "what is the weather",
    "what's the forecast",
    "tell me the weather",
    "is it going to rain",
])
engine.add_intent("timer", [
    "set a timer for five minutes",
    "start a timer",
    "set a ten minute timer",
    "create a countdown",
])
engine.train()

scores = engine.calc_intents("what's the weather like")
print(scores[0])  # ('weather', 0.78)
```

`calc_intents` returns a list of `(intent_name, confidence)` pairs sorted by
confidence, highest first.

## Next steps

- [Configuration](configuration.md) — tune the model to your skill set
- [Tuning](tuning.md) — practical advice on order, stemming and fallbacks
- [Troubleshooting](troubleshooting.md) — when matches are wrong or missing
