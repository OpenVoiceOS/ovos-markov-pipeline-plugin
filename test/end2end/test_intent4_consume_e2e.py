"""OVOS-INTENT-4 *consumer* end-to-end tests for the Markov pipeline.

``test/end2end/test_markov_e2e.py`` proves the markov pipeline matches intents
registered via the legacy ``padatious:register_*`` events. This suite proves the
markov pipeline *consumes the INTENT-4 spec registration topics*
(``ovos-intent-4.md``) and then matches.

Markov is a **template** engine: it consumes ``ovos.intent.register.template``
(§6) and not ``ovos.intent.register.keyword`` (§11). Each test boots a real
``MiniCroft`` pinned to the markov pipeline, emits the spec registration on the
wire, sends a matching utterance, and asserts the intent dispatches
``<skill_id>:<intent_name>`` — proving spec-topic consumption.
"""
import time
import unittest

import pytest

ovoscope = pytest.importorskip(
    "ovoscope", reason="ovoscope not installed; skipping E2E tests"
)

from ovoscope import E2EPipelineHarness  # noqa: E402
from ovos_bus_client.message import Message  # noqa: E402
from ovos_spec_tools import SpecMessage  # noqa: E402

from ovos_markov_pipeline import MarkovPipeline  # noqa: E402

PIPELINE_ID = "ovos-markov-pipeline-plugin"
CONFIG_KEY = "ovos-markov-pipeline-plugin"

# Markov needs a handful of samples per intent and low thresholds for the
# perplexity posterior to clear the match gate with only one or two intents
# loaded; mirror the existing markov e2e config.
PLUGIN_CONFIG = {
    "order": 1,
    "kneser_ney": False,
    "backoff": False,
    "smoothing": 1e-5,
    "conf_high": 0.50,
    "conf_med": 0.30,
    "conf_low": 0.15,
    "instant_train": True,
}

REGISTER_TEMPLATE = str(SpecMessage.INTENT_REGISTER_TEMPLATE)
REGISTER_KEYWORD = str(SpecMessage.INTENT_REGISTER_KEYWORD)
INTENT_DEREGISTER = str(SpecMessage.INTENT_DEREGISTER)
SKILL_DEREGISTER = str(SpecMessage.SKILL_DEREGISTER)
INTENT_DISABLE = str(SpecMessage.INTENT_DISABLE)
INTENT_ENABLE = str(SpecMessage.INTENT_ENABLE)

_HELLO = ["hello", "hi there", "hey", "greetings", "good morning", "howdy"]
_BYE = ["goodbye", "bye bye", "see you", "farewell", "take care", "later"]


class _Intent4MarkovHarness(E2EPipelineHarness):
    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    PLUGIN_CONFIG = PLUGIN_CONFIG
    SKILL_ID = "intent4_markov.skill"

    pipeline: MarkovPipeline  # type: ignore[assignment]

    def _register_template(self, intent_name, samples, *, blacklist=None,
                           lang="en-US", settle=1.0):
        payload = {
            "skill_id": self.SKILL_ID,
            "intent_name": intent_name,
            "lang": lang,
            "samples": samples,
        }
        if blacklist is not None:
            payload["blacklist"] = blacklist
        self.bus.emit(Message(REGISTER_TEMPLATE, payload,
                              {"skill_id": self.SKILL_ID}))
        time.sleep(settle)

    def _capture_match(self, utterance, intent_name, timeout=5.0, attempts=4):
        """send_and_capture with retries — the first match after a fresh
        MiniCroft boot can race the pipeline-ready/train state."""
        expected = [f"{self.SKILL_ID}:{intent_name}"]
        for _ in range(attempts):
            msg = self.send_and_capture(utterance, expected_types=expected,
                                        timeout=timeout)
            if msg is not None:
                return msg
            time.sleep(0.5)
        return None

    def _emit(self, topic, intent_name=None, settle=1.5, context_skill_id=None, **extra):
        data = {"skill_id": self.SKILL_ID, "lang": "en-US"}
        if intent_name is not None:
            data["intent_name"] = intent_name
        data.update(extra)
        # context.skill_id is the source that emitted the message
        # (OVOS-INTENT-4 §3.1); default it to this producer's own skill, but
        # allow a test to set it apart from data["skill_id"] so payload
        # (target) and context (source) can name different skills.
        source_skill_id = self.SKILL_ID if context_skill_id is None else context_skill_id
        self.bus.emit(Message(topic, data, {"skill_id": source_skill_id}))
        time.sleep(settle)


class TestSpecTemplateConsumed(_Intent4MarkovHarness):
    """§6: a template intent registered on the spec topic becomes matchable."""

    def test_spec_template_registration_is_matchable(self):
        self._register_template("hello", _HELLO)
        msg = self._capture_match("hello", "hello")
        self.assertIsNotNone(msg, "expected intent match from spec registration")
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:hello")

    def test_spec_template_second_intent_matchable(self):
        """A second spec-registered template is independently matchable (§6).

        Registered on its own so the assertion targets spec-topic consumption
        rather than the markov perplexity posterior's ability to discriminate
        two short, semantically-close intents.
        """
        self._register_template("bye", _BYE)
        msg = self._capture_match("goodbye", "bye")
        self.assertIsNotNone(msg, "second template should match")


class TestLegacyStillConsumed(_Intent4MarkovHarness):
    """Back-compat: legacy ``padatious:register_intent`` still matches."""

    def test_legacy_template_registration_still_matches(self):
        from ovoscope import register_padatious_intent
        register_padatious_intent(self.bus, f"{self.SKILL_ID}:bye", _BYE, skill_id=self.SKILL_ID)
        time.sleep(1.0)
        msg = self._capture_match("goodbye", "bye")
        self.assertIsNotNone(msg, "legacy registration must still match")


class TestSpecDeregister(_Intent4MarkovHarness):
    """§8.2 / §8.4: spec deregistration removes a spec-registered intent."""

    def test_spec_deregister_removes_intent(self):
        self._register_template("hello", _HELLO)
        self.assertIsNotNone(
            self._capture_match("hello", "hello"),
            "sanity: intent should match before deregister",
        )
        self._emit(INTENT_DEREGISTER, "hello")
        self.expect_no_match("hello", timeout=3.0)

    def test_spec_skill_deregister_removes_intent(self):
        self._register_template("hello", _HELLO)
        self.assertIsNotNone(
            self._capture_match("hello", "hello"),
            "sanity: intent should match before skill deregister",
        )
        self._emit(SKILL_DEREGISTER)
        self.expect_no_match("hello", timeout=3.0)


class TestSpecDisableEnable(_Intent4MarkovHarness):
    """§8.5: ``ovos.intent.disable`` suppresses, ``ovos.intent.enable`` re-arms.

    Markov retains the intent's trained model on disable and simply excludes it
    from match candidacy, re-arming on enable — registration-scoped suppression
    per §8.5.
    """

    def test_spec_disable_suppresses_intent(self):
        self._register_template("hello", _HELLO)
        self.assertIsNotNone(
            self._capture_match("hello", "hello"),
            "sanity: intent should match before disable",
        )
        self._emit(INTENT_DISABLE, "hello")
        self.expect_no_match("hello", timeout=3.0)

    def test_spec_enable_rearms_intent(self):
        self._register_template("hello", _HELLO)
        self._emit(INTENT_DISABLE, "hello")
        self._emit(INTENT_ENABLE, "hello")
        msg = self._capture_match("hello", "hello")
        self.assertIsNotNone(msg, "intent should match again after enable")


class TestNegativeKeywordTopic(_Intent4MarkovHarness):
    """§11: a template engine MUST NOT consume the *keyword* topic."""

    def test_keyword_topic_does_not_match_on_template_engine(self):
        self.bus.emit(Message(REGISTER_KEYWORD, {
            "skill_id": self.SKILL_ID,
            "intent_name": "lights_off",
            "lang": "en-US",
            "required": [{"name": "TurnOff", "samples": ["off"]},
                         {"name": "Light", "samples": ["lights"]}],
            "optional": [], "one_of": [], "excluded": [],
        }, {"skill_id": self.SKILL_ID}))
        time.sleep(0.5)
        self.expect_no_match("turn off the lights", timeout=3.0)


if __name__ == "__main__":
    unittest.main()
