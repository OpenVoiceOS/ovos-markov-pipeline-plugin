"""Additional MarkovIntentEngine tests covering edge cases."""

from ovos_markov_pipeline import MarkovIntentEngine


def _weather_timer_engine(**kwargs) -> MarkovIntentEngine:
    engine = MarkovIntentEngine(order=1, **kwargs)
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
    return engine


def test_char_fallback_blend_runs():
    """A high char_fallback_threshold forces the word/char blend path."""
    engine = _weather_timer_engine(
        char_fallback=True, char_fallback_threshold=1.0
    )
    # threshold 1.0 means the top-2 are always "close enough" to blend
    scores = engine.calc_intents("what is the weather")
    assert scores
    assert scores[0][0] in {"weather", "timer"}


def test_empty_samples_intent_is_skipped():
    """An intent whose samples normalize to nothing is skipped at train."""
    engine = MarkovIntentEngine(order=1)
    engine.add_intent("real", ["turn on the lights", "lights on", "lights please"])
    engine.add_intent("empty", ["", "   ", "..."])
    engine.train()
    # training must not raise and the empty intent produced no model
    assert "empty" not in engine._models
    assert "real" in engine._models


def test_empty_char_samples_skipped():
    """Empty samples are also skipped when building char-level models."""
    engine = MarkovIntentEngine(order=1, char_fallback=True)
    engine.add_intent("real", ["turn on the lights", "lights on", "lights please"])
    engine.add_intent("empty", ["", "   "])
    engine.train()
    assert "empty" not in engine._char_models


def test_calc_intents_untrained_returns_empty():
    engine = MarkovIntentEngine(order=2)
    engine.add_intent("x", ["hello there", "hi there"])
    # no train() called
    assert engine.calc_intents("hello there") == []


def test_calc_intents_too_short_returns_empty():
    """An utterance shorter than the n-gram order cannot be scored."""
    engine_order2 = MarkovIntentEngine(order=2)
    engine_order2.add_intent("a", ["one two three", "four five six"])
    engine_order2.add_intent("b", ["seven eight nine", "ten eleven twelve"])
    engine_order2.train()
    assert engine_order2.calc_intents("word") == []


def test_update_online_unknown_intent_noop():
    engine = _weather_timer_engine()
    before = dict(engine._intent_samples)
    engine.update_online("does-not-exist", "some utterance")
    assert engine._intent_samples == before


def test_update_online_appends_and_retrains():
    engine = _weather_timer_engine()
    n_before = len(engine._intent_samples["weather"])
    engine.update_online("weather", "how is the weather outside")
    assert len(engine._intent_samples["weather"]) == n_before + 1
    assert engine._trained


def test_remove_intent_clears_models():
    engine = _weather_timer_engine()
    engine.remove_intent("timer")
    assert "timer" not in engine._models
    assert "timer" not in engine._intent_samples
