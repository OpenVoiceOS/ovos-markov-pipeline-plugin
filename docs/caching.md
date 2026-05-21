# Model caching

Training is fast, but on a device with many skills even a few hundred
milliseconds per boot adds up. `IntentCache` exports trained Markov models to
disk as ONNX files plus a vocabulary JSON, so they can be reloaded without
retraining.

## Why ONNX

[`markovonnx`](https://pypi.org/project/markovonnx/) represents a trained
Markov chain as a sparse tensor graph that exports cleanly to ONNX. An exported
model is a portable, framework-independent artifact that a `MarkovONNXRuntime`
can score directly.

## Usage

```python
from ovos_markov_pipeline.cache import IntentCache

cache = IntentCache("/home/ovos/.cache/markov-intents")

# After training, persist a model:
cache.save_intent("weather:forecast", trained_chain)

# On the next boot, reload it:
runtime = cache.load_intent("weather:forecast", order=2)
if runtime is None:
    ...  # not cached — train normally
```

## API

| Method                              | Purpose                                                       |
| ------------------------------------ | ------------------------------------------------------------- |
| `save_intent(intent_name, model)`   | Export a trained `MarkovChain` to `<name>.onnx` + vocab JSON. |
| `load_intent(intent_name, order)`   | Load a cached model as a `MarkovONNXRuntime`, or `None`.      |
| `has_intent(intent_name)`           | Whether an ONNX file exists for the intent.                   |
| `remove_intent(intent_name)`        | Delete an intent's cached files.                              |
| `clear()`                           | Delete every cached model in the directory.                   |

## Notes

- The cache directory is created on construction if it does not exist.
- Intent names are made filesystem-safe: `:` becomes `__` and `/` becomes `_`.
- `save_intent` and `load_intent` never raise — an export or load failure is
  logged and `load_intent` returns `None`, so a corrupt cache degrades to a
  normal training pass rather than a crash.
- A cached model is only valid for the `order` it was trained with; pass the
  same `order` to `load_intent`.
