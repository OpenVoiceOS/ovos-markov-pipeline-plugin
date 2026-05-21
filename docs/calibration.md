# Calibration

The default confidence thresholds (`conf_high` 0.75, `conf_med` 0.55,
`conf_low` 0.30) are reasonable starting points, but the perplexity-to-
confidence mapping depends on your sample sizes and intent overlap. The
`calibration` module measures classification quality and derives thresholds
from labelled data.

## evaluate()

`evaluate(engine, eval_data)` runs a trained `MarkovIntentEngine` over a list
of `(utterance, expected_intent)` pairs and reports metrics.

```python
from ovos_markov_pipeline import MarkovIntentEngine
from ovos_markov_pipeline.calibration import evaluate

engine = MarkovIntentEngine(order=2)
# ... add_intent(...) for each intent ...
engine.train()

eval_data = [
    ("what is the weather", "weather"),
    ("set a timer", "timer"),
    ("turn on the lights", "lights"),
]

metrics = evaluate(engine, eval_data)
print(metrics["accuracy"], metrics["f1"])
```

The returned dict contains:

| Key                         | Meaning                                              |
| --------------------------- | ---------------------------------------------------- |
| `accuracy`                  | Fraction of utterances classified correctly.         |
| `precision`                 | Macro-averaged precision across intents.             |
| `recall`                    | Macro-averaged recall across intents.                |
| `f1`                        | Macro-averaged F1 score.                             |
| `avg_confidence_correct`    | Mean confidence on correct predictions.              |
| `avg_confidence_incorrect`  | Mean confidence on incorrect predictions.            |
| `total`                     | Number of evaluation pairs.                          |
| `correct`                   | Number classified correctly.                         |

A healthy model has `avg_confidence_correct` well above
`avg_confidence_incorrect` — that gap is what the confidence thresholds exploit.

## find_optimal_thresholds()

`find_optimal_thresholds(engine, eval_data, steps=20)` grid-searches confidence
cutoffs and returns the one that maximizes F1:

```python
from ovos_markov_pipeline.calibration import find_optimal_thresholds

result = find_optimal_thresholds(engine, eval_data, steps=40)
print(result["best_threshold"], result["best_f1"])
```

The returned dict contains `best_threshold`, `best_f1`, `precision_at_best` and
`recall_at_best`. `steps` controls the search resolution — `steps=20` tests
thresholds `0.00, 0.05, ... 1.00`.

## Applying the result

Use the search results to set `conf_high`, `conf_med` and `conf_low` in
[configuration](configuration.md):

- Set `conf_high` near a threshold with high precision (few false positives).
- Set `conf_low` near a threshold with high recall (few missed matches).
- Set `conf_med` between them.

Re-run `evaluate()` with the new thresholds in mind to confirm the trade-off
matches what your skills need.
