"""Edge-case tests for the calibration module."""

from ovos_markov_pipeline import MarkovIntentEngine
from ovos_markov_pipeline.calibration import evaluate, find_optimal_thresholds


def _engine() -> MarkovIntentEngine:
    engine = MarkovIntentEngine(order=2)
    engine.add_intent("weather", [
        "what is the weather today",
        "tell me the weather forecast",
        "is it going to rain today",
    ])
    engine.add_intent("timer", [
        "set a timer for ten minutes",
        "start a five minute timer",
        "create a countdown timer now",
    ])
    engine.train()
    return engine


def test_evaluate_counts_unscored_utterance_as_miss():
    """An utterance too short to score is counted as a false negative."""
    engine = _engine()
    # "hi" is shorter than the order-2 model -> calc_intents returns []
    metrics = evaluate(engine, [("hi", "weather")])
    assert metrics["total"] == 1
    assert metrics["correct"] == 0
    assert metrics["accuracy"] == 0.0


def test_evaluate_reports_metrics_on_known_data():
    engine = _engine()
    eval_data = [
        ("what is the weather today", "weather"),
        ("set a timer for ten minutes", "timer"),
    ]
    metrics = evaluate(engine, eval_data)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["total"] == 2


def test_find_optimal_thresholds_returns_bounds():
    engine = _engine()
    eval_data = [
        ("what is the weather today", "weather"),
        ("set a timer for ten minutes", "timer"),
    ]
    result = find_optimal_thresholds(engine, eval_data, steps=10)
    assert 0.0 <= result["best_threshold"] <= 1.0
    assert 0.0 <= result["best_f1"] <= 1.0
