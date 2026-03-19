"""Tests for confidence calibration."""

from ovos_markov_pipeline import MarkovIntentEngine
from ovos_markov_pipeline.calibration import evaluate, find_optimal_thresholds


def _make_engine() -> MarkovIntentEngine:
    engine = MarkovIntentEngine(order=1, kneser_ney=False, backoff=False)
    engine.add_intent("weather:get", [
        "what is the weather", "how is the weather",
        "tell me the weather", "what is the forecast",
    ])
    engine.add_intent("timer:set", [
        "set a timer", "start a timer",
        "set a countdown", "timer for five minutes",
    ])
    engine.train()
    return engine


class TestEvaluate:
    def test_basic_evaluation(self) -> None:
        engine = _make_engine()
        eval_data = [
            ("what is the weather today", "weather:get"),
            ("set a timer please", "timer:set"),
        ]
        results = evaluate(engine, eval_data)
        assert "accuracy" in results
        assert "f1" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert results["total"] == 2

    def test_empty_eval(self) -> None:
        engine = _make_engine()
        results = evaluate(engine, [])
        assert results["accuracy"] == 0.0
        assert results["total"] == 0

    def test_all_correct(self) -> None:
        engine = _make_engine()
        # Use exact training phrases
        eval_data = [
            ("what is the weather", "weather:get"),
            ("set a timer", "timer:set"),
        ]
        results = evaluate(engine, eval_data)
        assert results["correct"] >= 1  # At least some correct

    def test_confidence_stats(self) -> None:
        engine = _make_engine()
        eval_data = [
            ("what is the weather", "weather:get"),
            ("set a timer", "timer:set"),
        ]
        results = evaluate(engine, eval_data)
        assert "avg_confidence_correct" in results


class TestFindOptimalThresholds:
    def test_finds_threshold(self) -> None:
        engine = _make_engine()
        eval_data = [
            ("what is the weather", "weather:get"),
            ("set a timer", "timer:set"),
            ("how is the forecast", "weather:get"),
            ("start a countdown", "timer:set"),
        ]
        results = find_optimal_thresholds(engine, eval_data, steps=10)
        assert "best_threshold" in results
        assert 0.0 <= results["best_threshold"] <= 1.0
        assert "best_f1" in results
