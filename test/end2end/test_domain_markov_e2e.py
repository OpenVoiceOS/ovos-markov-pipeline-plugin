"""End-to-end tests for DomainMarkovPipeline.

Drives a `MiniCroft` instance with the domain entry point
(`ovos-markov-domain-pipeline-plugin`) and exercises the
``padatious:register_intent`` → two-stage routing → utterance
dispatch path against a :class:`DomainMarkovIntentEngine`.

Skipped automatically if ``ovoscope`` is not installed.
"""
import threading
import unittest

import pytest

pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from ovos_markov_pipeline import (  # noqa: E402
    DomainMarkovIntentEngine,
    DomainMarkovPipeline,
    MarkovIntentEngine,
)

PIPELINE_ID = "ovos-markov-domain-pipeline-plugin"

DOMAIN_PIPELINE = [
    f"{PIPELINE_ID}-high",
    f"{PIPELINE_ID}-medium",
    f"{PIPELINE_ID}-low",
]

DOMAIN_CONFIG = {
    PIPELINE_ID: {
        "order": 1,
        "kneser_ney": False,
        "backoff": False,
        "smoothing": 1e-5,
        "conf_high": 0.50,
        "conf_med": 0.30,
        "conf_low": 0.15,
        "instant_train": True,
    }
}


def _utterance_msg(utterance: str, session_id: str = "ovoscope-domain") -> Message:
    sess = Session(session_id)
    sess.lang = "en-US"
    sess.pipeline = DOMAIN_PIPELINE
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": "en-US"},
        {"session": sess.serialize(), "source": "test", "destination": "skills"},
    )


class _DomainE2EBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.minicroft = get_minicroft(
            [],
            default_pipeline=DOMAIN_PIPELINE,
            pipeline_config=DOMAIN_CONFIG,
        )
        cls.pipeline: DomainMarkovPipeline = (
            cls.minicroft.intents.pipeline_plugins[PIPELINE_ID]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.minicroft.stop()

    def setUp(self) -> None:
        # Fresh hierarchical engines per test.
        for lang in list(self.pipeline.engines):
            self.pipeline.engines[lang] = DomainMarkovIntentEngine(
                stemmer=self.pipeline.stemmers.get(lang),
                **self.pipeline._engine_kwargs_template,
            )
        self.pipeline.registered_intents = []
        self.pipeline._skill2intent.clear()

    def _register(self, name: str, samples: list) -> None:
        self.minicroft.bus.emit(Message(
            "padatious:register_intent",
            data={"name": name, "samples": samples,
                  "skill_id": name.split(":", 1)[0]},
        ))

    def _send_and_capture(self, utterance: str, expected_types: list,
                          timeout: float = 5.0):
        got: list = []
        done = threading.Event()

        def _on_match(msg):
            got.append(msg)
            done.set()

        def _on_fail(_):
            done.set()

        for t in expected_types:
            self.minicroft.bus.on(t, _on_match)
        self.minicroft.bus.on("complete_intent_failure", _on_fail)
        try:
            self.minicroft.bus.emit(_utterance_msg(utterance))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.minicroft.bus.remove(t, _on_match)
            self.minicroft.bus.remove("complete_intent_failure", _on_fail)
        return got[0] if got else None


class TestDomainPipelineLoad(_DomainE2EBase):
    def test_loaded_with_domain_engines(self):
        self.assertIsInstance(self.pipeline, DomainMarkovPipeline)
        for engine in self.pipeline.engines.values():
            self.assertIsInstance(engine, DomainMarkovIntentEngine)

    def test_flat_engine_is_separate_class(self):
        for engine in self.pipeline.engines.values():
            self.assertIsNot(type(engine), MarkovIntentEngine)
            self.assertIsInstance(engine, DomainMarkovIntentEngine)


class TestDomainRegistrationRouting(_DomainE2EBase):
    def test_intents_routed_to_skill_id_domain(self):
        self._register("smarthome.skill:lights.intent",
                       ["turn on the lights", "lights on please"])
        self._register("media.skill:music.intent",
                       ["play music", "start the music"])
        engine = self.pipeline.engines["en-US"]
        self.assertIn("smarthome.skill", engine.domains)
        self.assertIn("media.skill", engine.domains)
        self.assertIn(
            "smarthome.skill:lights.intent",
            list(engine.domains["smarthome.skill"]._intent_samples.keys()),
        )
        self.assertIn(
            "media.skill:music.intent",
            list(engine.domains["media.skill"]._intent_samples.keys()),
        )

    def test_intent_without_namespace_falls_back_to_full_label(self):
        self._register("orphan_label", ["lights on now please"])
        engine = self.pipeline.engines["en-US"]
        self.assertIn("orphan_label", engine.domains)

    def test_detach_intent_removes_only_that_label(self):
        self._register("smarthome.skill:lights.intent", ["lights on please"])
        self._register("smarthome.skill:door.intent", ["open the front door"])
        self.minicroft.bus.emit(Message("detach_intent", data={
            "intent_name": "smarthome.skill:lights.intent",
        }))
        engine = self.pipeline.engines["en-US"]
        sub = engine.domains.get("smarthome.skill")
        self.assertIsNotNone(
            sub, "detach_intent must not drop the whole domain")
        self.assertNotIn("smarthome.skill:lights.intent",
                         sub._intent_samples)
        self.assertIn("smarthome.skill:door.intent",
                      sub._intent_samples)

    def test_detach_skill_drops_whole_domain(self):
        self._register("smarthome.skill:lights.intent", ["lights on please"])
        self._register("media.skill:music.intent", ["play music"])
        self.minicroft.bus.emit(Message("detach_skill",
                                        data={"skill_id": "smarthome.skill"}))
        engine = self.pipeline.engines["en-US"]
        self.assertNotIn("smarthome.skill", engine.domains)
        self.assertIn("media.skill", engine.domains)


class TestDomainMatch(_DomainE2EBase):
    def _seed(self):
        self._register("smarthome.skill:lights.intent",
                       ["turn on the lights", "lights on please",
                        "switch on the lights"])
        self._register("media.skill:music.intent",
                       ["play music", "start the music",
                        "put on some music"])

    def test_router_picks_correct_intent(self):
        self._seed()
        msg = self._send_and_capture(
            "turn on the lights",
            expected_types=["smarthome.skill:lights.intent",
                            "media.skill:music.intent"],
            timeout=10.0,
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "smarthome.skill:lights.intent")


if __name__ == "__main__":
    unittest.main()
