# ovos-markov-pipeline-plugin

OVOS intent pipeline plugin using Markov chain perplexity ensemble from [markovonnx](https://github.com/markovonnx).

## How It Works

1. Skills register intent samples via the MessageBus (`padatious:register_intent`)
2. One word-level Markov chain is trained per intent, all sharing a common vocabulary
3. On utterance match, perplexity is computed under each model — lowest wins
4. Confidence: `conf = 1 / (1 + log(ppx))`, clamped to [0, 1]

## Key Classes

| Class | Source | Description |
|-------|--------|-------------|
| `MarkovPipeline` | `ovos_markov_pipeline/__init__.py` | OPM `ConfidenceMatcherPipeline` subclass |
| `MarkovIntentEngine` | `ovos_markov_pipeline/__init__.py` | Per-language perplexity ensemble |

## Configuration

```json
{
  "intents": {
    "ovos-markov-pipeline-plugin": {
      "order": 2,
      "kneser_ney": true,
      "backoff": true,
      "smoothing": 1e-5,
      "conf_high": 0.75,
      "conf_med": 0.55,
      "conf_low": 0.30,
      "max_words": 50,
      "instant_train": false
    }
  }
}
```

## Bus Messages

| Message | Direction | Description |
|---------|-----------|-------------|
| `padatious:register_intent` | Received | Register intent samples |
| `detach_intent` | Received | Remove single intent |
| `detach_skill` | Received | Remove all intents for skill |
| `mycroft.skills.train` | Received | Trigger training |
| `mycroft.skills.trained` | Emitted | Training complete |
| `intent.service.markov.manifest.get` | Received | Query registered intents |
| `intent.service.markov.manifest` | Emitted | Reply with intent list |

## Strengths vs Other Plugins

| Aspect | Markov | Padatious | Model2Vec |
|--------|--------|-----------|-----------|
| Training data | 5-50 examples | 5-50 examples | Pre-trained |
| Training time | Milliseconds | Seconds | None |
| Model size | KBs | KBs | 100s MB |
| Accuracy | Good for narrow domains | Better pattern matching | Best semantic matching |
| Dependencies | numpy, onnx | fann2 | model2vec, torch |
