"""Tests for MarkovIntentEngine."""

from ovos_markov_pipeline import MarkovIntentEngine, _normalize, _ppx_to_confidence, _Stemmer


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert _normalize("hello   world") == "hello world"

    def test_strip_trailing_punct(self) -> None:
        assert _normalize("hello world!") == "hello world"

    def test_empty(self) -> None:
        assert _normalize("") == ""

    def test_with_stemmer(self) -> None:
        if _Stemmer.supports("en"):
            stemmer = _Stemmer("en")
            result = _normalize("running quickly", stemmer)
            assert result != "running quickly"  # should be stemmed


class TestStemmer:
    def test_supports_english(self) -> None:
        assert _Stemmer.supports("en")
        assert _Stemmer.supports("en-US")

    def test_unsupported_lang(self) -> None:
        assert not _Stemmer.supports("xx")

    def test_stem_basic(self) -> None:
        if _Stemmer.supports("en"):
            s = _Stemmer("en")
            result = s.stem("running dogs")
            assert "run" in result  # "running" -> "run"

    def test_init_unsupported_raises(self) -> None:
        try:
            _Stemmer("xx-XX")
            assert False, "Should raise"
        except ValueError:
            pass


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
    def _make_engine(self, **kwargs) -> MarkovIntentEngine:
        defaults = dict(order=1, kneser_ney=False, backoff=False)
        defaults.update(kwargs)
        engine = MarkovIntentEngine(**defaults)
        engine.add_intent(
            "weather:get_weather",
            [
                "what is the weather",
                "what is the weather like",
                "how is the weather today",
                "tell me the weather",
                "what is the forecast",
                "is it going to rain",
                "will it rain today",
                "what is the temperature",
            ],
        )
        engine.add_intent(
            "timer:set_timer",
            [
                "set a timer for five minutes",
                "set a timer for ten minutes",
                "start a timer",
                "set a countdown",
                "timer for five minutes",
                "remind me in ten minutes",
                "start a countdown",
                "set an alarm for five minutes",
            ],
        )
        engine.add_intent(
            "music:play_music",
            [
                "play some music",
                "play jazz music",
                "play rock and roll",
                "put on some music",
                "play my playlist",
                "play something relaxing",
                "i want to listen to music",
                "play the radio",
            ],
        )
        engine.train()
        return engine

    def test_must_train(self) -> None:
        engine = MarkovIntentEngine()
        assert not engine.must_train
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

    def test_classify_timer(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents("set a timer for three minutes")
        assert scores[0][0] == "timer:set_timer"

    def test_classify_music(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents("play my playlist")
        assert scores[0][0] == "music:play_music"

    def test_empty_engine_returns_empty(self) -> None:
        engine = MarkovIntentEngine()
        assert engine.calc_intents("hello") == []

    def test_short_utterance_returns_empty(self) -> None:
        engine = MarkovIntentEngine(order=3, kneser_ney=False)
        engine.add_intent("test:foo", ["hello world test"])
        engine.train()
        scores = engine.calc_intents("play")
        assert scores == []

    def test_remove_intent(self) -> None:
        engine = self._make_engine()
        engine.remove_intent("music:play_music")
        assert "music:play_music" not in engine._models
        scores = engine.calc_intents("play some music")
        assert "music:play_music" not in [s[0] for s in scores]

    def test_blacklisted_intents(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents(
            "what is the weather",
            blacklisted_intents={"weather:get_weather"},
        )
        assert "weather:get_weather" not in [s[0] for s in scores]

    def test_blacklisted_skills(self) -> None:
        engine = self._make_engine()
        scores = engine.calc_intents(
            "play some music",
            blacklisted_skills={"music"},
        )
        assert "music:play_music" not in [s[0] for s in scores]

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

    def test_char_fallback(self) -> None:
        engine = self._make_engine(char_fallback=True)
        assert len(engine._char_models) == 3
        scores = engine.calc_intents("what is the weather")
        assert len(scores) > 0

    def test_online_update(self) -> None:
        engine = self._make_engine()
        original_count = len(engine._intent_samples["weather:get_weather"])
        engine.update_online("weather:get_weather", "how hot is it outside")
        assert len(engine._intent_samples["weather:get_weather"]) == original_count + 1
        assert "weather:get_weather" in engine._models

    def test_online_update_unknown_intent(self) -> None:
        engine = self._make_engine()
        engine.update_online("nonexistent:intent", "hello")
        # Should not crash

    def test_online_update_before_train(self) -> None:
        engine = MarkovIntentEngine()
        engine.add_intent("test:foo", ["hello world"])
        engine.update_online("test:foo", "hi there")
        # vocab is None, should not crash

    def test_with_stemmer(self) -> None:
        if _Stemmer.supports("en"):
            stemmer = _Stemmer("en")
            engine = MarkovIntentEngine(order=1, stemmer=stemmer)
            engine.add_intent(
                "test:run",
                [
                    "the dogs are running",
                    "she runs every morning",
                    "running is fun",
                ],
            )
            engine.train()
            scores = engine.calc_intents("the dog ran")
            assert len(scores) > 0
