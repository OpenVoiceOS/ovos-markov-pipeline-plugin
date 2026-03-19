"""Tests for MarkovIntentEngine."""

from ovos_markov_pipeline import MarkovIntentEngine, _normalize, _ppx_to_confidence


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert _normalize("hello   world") == "hello world"

    def test_strip_trailing_punct(self) -> None:
        assert _normalize("hello world!") == "hello world"

    def test_empty(self) -> None:
        assert _normalize("") == ""


class TestPpxToConfidence:
    def test_low_ppx_high_confidence(self) -> None:
        assert _ppx_to_confidence(1.0) == 1.0

    def test_high_ppx_low_confidence(self) -> None:
        conf = _ppx_to_confidence(1000.0)
        assert 0.0 < conf < 0.2

    def test_moderate_ppx(self) -> None:
        conf = _ppx_to_confidence(10.0)
        assert 0.2 < conf < 0.5

    def test_ppx_below_one(self) -> None:
        assert _ppx_to_confidence(0.5) == 1.0


class TestMarkovIntentEngine:
    def _make_engine(self) -> MarkovIntentEngine:
        engine = MarkovIntentEngine(order=1, kneser_ney=False, backoff=False)
        engine.add_intent("weather:get_weather", [
            "what is the weather",
            "what is the weather like",
            "how is the weather today",
            "tell me the weather",
            "what is the forecast",
            "is it going to rain",
            "will it rain today",
            "what is the temperature",
        ])
        engine.add_intent("timer:set_timer", [
            "set a timer for five minutes",
            "set a timer",
            "start a timer for ten minutes",
            "set a countdown",
            "timer for five minutes",
            "remind me in ten minutes",
            "start a countdown",
            "set an alarm for five minutes",
        ])
        engine.add_intent("music:play_music", [
            "play some music",
            "play jazz music",
            "play rock and roll",
            "put on some music",
            "play my playlist",
            "play something relaxing",
            "i want to listen to music",
            "play the radio",
        ])
        engine.train()
        return engine

    def test_must_train(self) -> None:
        engine = MarkovIntentEngine()
        assert not engine.must_train  # no samples
        engine.add_intent("test:foo", ["hello world"])
        assert engine.must_train
        engine.train()
        assert not engine.must_train

    def test_train_builds_models(self) -> None:
        engine = self._make_engine()
        assert len(engine._models) == 3
        assert engine._vocab is not None

    def test_classify_weather(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents("what is the weather today")
        assert len(scores) == 3
        assert scores[0][0] == "weather:get_weather"
        assert scores[0][1] > scores[1][1]

    def test_classify_timer(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents("set a timer for three minutes")
        assert scores[0][0] == "timer:set_timer"

    def test_classify_music(self) -> None:
        engine = self._make_engine()
        # Use a query very close to training data for reliable matching
        scores = engine.calc_intents("play my playlist")
        assert scores[0][0] == "music:play_music"

    def test_empty_engine_returns_empty(self) -> None:
        engine = MarkovIntentEngine()
        assert engine.calc_intents("hello") == []

    def test_short_utterance_returns_empty(self) -> None:
        engine = MarkovIntentEngine(order=3, kneser_ney=False, backoff=False)
        engine.add_intent("test:foo", ["hello world test"])
        engine.train()
        # "play" is 1 token, shorter than order=3
        scores = engine.calc_intents("play")
        assert scores == []

    def test_remove_intent(self) -> None:
        engine = self._make_engine()
        engine.remove_intent("music:play_music")
        assert "music:play_music" not in engine._models
        scores = engine.calc_intents("play some music")
        intent_names = [s[0] for s in scores]
        assert "music:play_music" not in intent_names

    def test_blacklisted_intents(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents(
            "what is the weather",
            blacklisted_intents={"weather:get_weather"},
        )
        intent_names = [s[0] for s in scores]
        assert "weather:get_weather" not in intent_names

    def test_blacklisted_skills(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents(
            "play some music",
            blacklisted_skills={"music"},
        )
        intent_names = [s[0] for s in scores]
        assert "music:play_music" not in intent_names

    def test_confidence_range(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents("what is the weather")
        for _, conf in scores:
            assert 0.0 <= conf <= 1.0

    def test_train_empty_samples(self) -> None:
        engine = MarkovIntentEngine()
        engine.add_intent("test:empty", [])
        engine.train()
        assert engine._trained
