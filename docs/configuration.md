# Configuration

All configuration lives under `intents.ovos-markov-pipeline-plugin` in
`mycroft.conf`:

```json
{
  "intents": {
    "ovos-markov-pipeline-plugin": {
      "order": 2,
      "kneser_ney": true,
      "backoff": true,
      "smoothing": 1e-5,
      "stem": false,
      "char_fallback": false,
      "char_fallback_threshold": 0.05,
      "online_learning": false,
      "instant_train": false,
      "conf_high": 0.75,
      "conf_med": 0.55,
      "conf_low": 0.30,
      "max_words": 50
    }
  }
}
```

Every key is optional; the values above are the defaults.

## Model parameters

| Key          | Default | Meaning                                                                 |
| ------------ | ------- | ----------------------------------------------------------------------- |
| `order`      | `2`     | N-gram order of each Markov chain. Clamped to a minimum of 1.            |
| `kneser_ney` | `true`  | Use Kneser-Ney smoothing instead of plain Laplace smoothing.            |
| `backoff`    | `true`  | Interpolate with lower-order models when an n-gram is unseen.           |
| `smoothing`  | `1e-5`  | Laplace smoothing alpha (the additive count for unseen n-grams).        |

`order` is the single most impactful setting — see [Tuning](tuning.md) for how
to choose it. `kneser_ney` and `backoff` both help the model cope with words it
did not see during training and are best left enabled.

## Normalization

| Key    | Default | Meaning                                                              |
| ------ | ------- | -------------------------------------------------------------------- |
| `stem` | `false` | Stem words before training and matching using the Snowball stemmer.  |

Stemming is available for 26 languages and requires the `stem` extra
(`pip install ovos-markov-pipeline-plugin[stem]`). If the extra is missing or
the language is unsupported, the plugin silently runs without stemming.

## Character-level fallback

| Key                       | Default | Meaning                                                          |
| ------------------------- | ------- | ---------------------------------------------------------------- |
| `char_fallback`           | `false` | Train parallel character n-gram models for tie-breaking.         |
| `char_fallback_threshold` | `0.05`  | Trigger the blend when the top two word scores differ by less.   |

When enabled and the two best word-level confidences are within
`char_fallback_threshold`, scores are recomputed as a 60% word + 40% character
blend. This helps with short utterances and near-homophone intents at the cost
of training a second set of models.

## Learning behaviour

| Key               | Default | Meaning                                                            |
| ----------------- | ------- | ------------------------------------------------------------------ |
| `online_learning` | `false` | Feed matches above `conf_high` back into the model.                |
| `instant_train`   | `false` | Retrain on every intent registration instead of deferring.         |

With `instant_train` disabled, training is deferred until the first match
request (or an explicit `mycroft.skills.train` bus message), which avoids
retraining once per skill at boot. Enable it for short-lived processes such as
tests.

## Confidence thresholds

| Key         | Default | Pipeline tier  |
| ----------- | ------- | -------------- |
| `conf_high` | `0.75`  | `match_high`   |
| `conf_med`  | `0.55`  | `match_medium` |
| `conf_low`  | `0.30`  | `match_low`    |

A match is returned at a tier only when the best confidence strictly exceeds
that tier's threshold. The values must satisfy
`conf_low <= conf_med <= conf_high`; the plugin logs a warning if they do not.
Use [`find_optimal_thresholds()`](calibration.md) to derive good values from
labelled data.

## Limits

| Key         | Default | Meaning                                                       |
| ----------- | ------- | ------------------------------------------------------------- |
| `max_words` | `50`    | Utterances longer than this many words are skipped (ignored). |
