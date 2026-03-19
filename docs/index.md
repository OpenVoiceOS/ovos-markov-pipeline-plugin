# ovos-markov-pipeline-plugin

OVOS intent pipeline plugin using Markov chain perplexity ensemble from markovonnx.

## How It Works

1. Skills register intent samples via `padatious:register_intent`
2. One word-level Markov chain per intent, all sharing a common vocabulary
3. On match: perplexity computed under each model — lowest wins
4. Confidence: `conf = 1 / (1 + log(ppx))`
5. Optional: char-level fallback blending, stemming, online learning

## Key Classes

| Class | Description |
|-------|-------------|
| `MarkovPipeline` | `ConfidenceMatcherPipeline` subclass — OPM entry point |
| `MarkovIntentEngine` | Per-language perplexity ensemble with shared vocab |
| `_Stemmer` | Snowball stemmer wrapper for word normalization |

## Configuration

```json
{
  "intents": {
    "ovos-markov-pipeline-plugin": {
      "order": 2,
      "kneser_ney": true,
      "backoff": true,
      "stem": false,
      "char_fallback": false,
      "online_learning": false,
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

| Option | Default | Description |
|--------|---------|-------------|
| `order` | 2 | N-gram order for word-level models |
| `kneser_ney` | true | Kneser-Ney smoothing (better than Laplace for small data) |
| `backoff` | true | Fall back to lower-order models for unseen contexts |
| `stem` | false | Apply snowball stemmer to normalize word forms |
| `char_fallback` | false | Train char-level models, blend when word scores are ambiguous |
| `online_learning` | false | Reinforce high-confidence matches by updating models |
| `conf_high` | 0.75 | Minimum confidence for high-tier match |
| `conf_med` | 0.55 | Minimum confidence for medium-tier match |
| `conf_low` | 0.30 | Minimum confidence for low-tier match |

## Bus Messages

| Message | Direction | Description |
|---------|-----------|-------------|
| `padatious:register_intent` | In | Register intent samples (or file) |
| `detach_intent` | In | Remove single intent |
| `detach_skill` | In | Remove all intents for skill |
| `mycroft.skills.train` | In | Trigger training |
| `mycroft.skills.trained` | Out | Training complete |
| `intent.service.markov.manifest.get` | In | Query registered intents |
| `intent.service.markov.manifest` | Out | Reply with intent list |
