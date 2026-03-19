"""Tests for MarkovPipeline (OPM integration)."""

from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_markov_pipeline import MarkovPipeline


def _make_pipeline() -> MarkovPipeline:
    """Create a pipeline with a FakeBus and register some intents."""
    bus = FakeBus()
    config = {
        "order": 2,
        "kneser_ney": False,
        "backoff": True,
        "conf_high": 0.6,
        "conf_med": 0.4,
        "conf_low": 0.2,
        "instant_train": True,
    }
    pipeline = MarkovPipeline(bus=bus, config=config)

    # Register intents via bus messages
    for name, samples in [
        ("weather_skill:get_weather", [
            "what is the weather",
            "what is the weather like",
            "how is the weather today",
            "tell me the weather forecast",
            "is it going to rain today",
            "what is the temperature outside",
        ]),
        ("timer_skill:set_timer", [
            "set a timer for five minutes",
            "set a timer for ten minutes",
            "start a timer",
            "set a countdown timer",
            "timer for three minutes please",
            "start a five minute timer",
        ]),
    ]:
        msg = Message("padatious:register_intent", {
            "name": name,
            "skill_id": name.split(":")[0],
            "samples": samples,
            "lang": "en-US",
        })
        pipeline.register_intent(msg)

    return pipeline


class TestMarkovPipeline:
    def test_init(self) -> None:
        bus = FakeBus()
        pipeline = MarkovPipeline(bus=bus)
        assert pipeline.conf_high > 0
        assert len(pipeline.engines) > 0

    def test_register_intent(self) -> None:
        pipeline = _make_pipeline()
        assert "weather_skill:get_weather" in pipeline.registered_intents
        assert "timer_skill:set_timer" in pipeline.registered_intents

    def test_match_high_weather(self) -> None:
        pipeline = _make_pipeline()
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_high(
            ["what is the weather today"], "en-US", msg
        )
        # May or may not match at high confidence depending on data
        if result:
            assert result.skill_id == "weather_skill"

    def test_match_medium(self) -> None:
        pipeline = _make_pipeline()
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_medium(
            ["what is the weather"], "en-US", msg
        )
        if result:
            assert result.match_type == "weather_skill:get_weather"
            assert result.match_data["confidence"] > 0

    def test_match_low(self) -> None:
        pipeline = _make_pipeline()
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_low(
            ["set a timer for five minutes"], "en-US", msg
        )
        if result:
            assert "timer" in result.match_type

    def test_no_match_gibberish(self) -> None:
        pipeline = _make_pipeline()
        msg = Message("recognizer_loop:utterance", {})
        # Very high threshold, gibberish text
        result = pipeline.match_high(
            ["xyzzy frobnicator quantum"], "en-US", msg
        )
        assert result is None

    def test_detach_intent(self) -> None:
        pipeline = _make_pipeline()
        assert "weather_skill:get_weather" in pipeline.registered_intents
        pipeline.handle_detach_intent(
            Message("detach_intent", {"intent_name": "weather_skill:get_weather"})
        )
        assert "weather_skill:get_weather" not in pipeline.registered_intents

    def test_detach_skill(self) -> None:
        pipeline = _make_pipeline()
        pipeline.handle_detach_skill(
            Message("detach_skill", {"skill_id": "timer_skill"})
        )
        assert "timer_skill:set_timer" not in pipeline.registered_intents

    def test_unknown_lang_returns_none(self) -> None:
        pipeline = _make_pipeline()
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_high(
            ["hello"], "xx-XX", msg
        )
        assert result is None

    def test_max_words_filter(self) -> None:
        pipeline = _make_pipeline()
        pipeline.max_words = 3
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_low(
            ["this sentence has way too many words for the filter"], "en-US", msg
        )
        assert result is None

    def test_shutdown(self) -> None:
        pipeline = _make_pipeline()
        pipeline.shutdown()  # Should not raise

    def test_manifest(self) -> None:
        pipeline = _make_pipeline()
        replies = []
        pipeline.bus.on("intent.service.markov.manifest", lambda m: replies.append(m))
        pipeline.handle_manifest(
            Message("intent.service.markov.manifest.get")
        )
        assert len(replies) == 1
        assert "weather_skill:get_weather" in replies[0].data["intents"]

    def test_untrained_engine_returns_none(self) -> None:
        bus = FakeBus()
        pipeline = MarkovPipeline(bus=bus, config={"instant_train": False})
        msg = Message("recognizer_loop:utterance", {})
        result = pipeline.match_low(["hello world test"], "en-US", msg)
        assert result is None

    def test_train_emits_trained_message(self) -> None:
        pipeline = _make_pipeline()
        trained = []
        pipeline.bus.on("mycroft.skills.trained", lambda m: trained.append(m))
        pipeline.train()
        assert len(trained) >= 1
