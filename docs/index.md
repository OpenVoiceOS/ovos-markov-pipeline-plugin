# ovos-markov-pipeline-plugin

OVOS intent pipeline plugin using Markov chain perplexity ensemble from markovonnx.

## Architecture

```
Utterance → normalize (+ stem) → word tokenize
  → compute perplexity under each intent model
  → lowest PPX wins → confidence = 1/(1+log(ppx))
  → optional: char-level fallback blend when ambiguous
  → optional: HMM BIO slot extraction on matched intent
```

## Modules

| Module | Description |
|--------|-------------|
| `__init__.py` | `MarkovPipeline` (OPM entry), `MarkovIntentEngine` (per-lang ensemble) |
| `slots.py` | `SlotExtractor` — HMM BIO tagging for entity extraction |
| `cache.py` | `IntentCache` — ONNX disk cache for trained models |
| `calibration.py` | `evaluate()`, `find_optimal_thresholds()` — confidence tuning |

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

## Bus Messages

| Message | Direction | Description |
|---------|-----------|-------------|
| `padatious:register_intent` | In | Register intent samples |
| `detach_intent` / `detach_skill` | In | Remove intents |
| `mycroft.skills.train` | In | Trigger training |
| `mycroft.skills.trained` | Out | Training complete |
| `intent.service.markov.manifest.get/manifest` | In/Out | Query registered intents |

## Features

- **Stemming**: Snowball stemmer for 26 languages (`"stem": true`)
- **Char fallback**: Blend 60% word + 40% char scores when top-2 are ambiguous
- **Online learning**: High-confidence matches reinforce intent models incrementally
- **Entity extraction**: HMM Viterbi BIO tagging via `SlotExtractor`
- **ONNX cache**: `IntentCache` saves/loads models as sparse ONNX + vocab JSON
- **Calibration**: `evaluate()` for P/R/F1, `find_optimal_thresholds()` for threshold tuning
- **CI/CD**: Full OVOS workflow suite (build-tests, OPM check, coverage, release, etc.)
