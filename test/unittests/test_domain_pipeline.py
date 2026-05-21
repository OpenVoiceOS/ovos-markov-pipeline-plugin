"""Unit tests for DomainMarkovPipeline (no ovoscope required)."""
import unittest
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_markov_pipeline import (
    DomainMarkovIntentEngine,
    DomainMarkovPipeline,
    MarkovPipeline,
)


def _make_pipeline(**overrides) -> DomainMarkovPipeline:
    cfg = {"order": 1, "kneser_ney": False, "backoff": False,
           "smoothing": 1e-5, "instant_train": True}
    cfg.update(overrides)
    return DomainMarkovPipeline(bus=FakeBus(), config=cfg)


class TestDomainPipelineConstruction(unittest.TestCase):
    def test_engines_are_domain_engines(self):
        pipe = _make_pipeline()
        self.assertIsInstance(pipe, MarkovPipeline)
        for engine in pipe.engines.values():
            self.assertIsInstance(engine, DomainMarkovIntentEngine)

    def test_domain_of_extracts_skill_id(self):
        self.assertEqual(
            DomainMarkovPipeline._domain_of("skill.foo:bar.intent"),
            "skill.foo",
        )
        self.assertEqual(
            DomainMarkovPipeline._domain_of("orphan"),
            "orphan",
        )


class TestDomainPipelineRouting(unittest.TestCase):
    def setUp(self):
        self.pipe = _make_pipeline()
        self.engine = self.pipe.engines[self.pipe.lang]

    def _register(self, name: str, samples: list):
        self.pipe.register_intent(Message(
            "padatious:register_intent",
            data={"name": name, "samples": samples,
                  "skill_id": name.split(":", 1)[0], "lang": self.pipe.lang},
        ))

    def test_intent_routed_to_skill_domain(self):
        self._register("smarthome.skill:lights.intent",
                       ["turn on the lights", "lights on please"])
        self._register("media.skill:music.intent",
                       ["play music", "start the music"])
        self.assertIn("smarthome.skill", self.engine.domains)
        self.assertIn("media.skill", self.engine.domains)
        self.assertIn("smarthome.skill:lights.intent",
                      self.engine.domains["smarthome.skill"]._intent_samples)

    def test_detach_intent(self):
        self._register("skill.x:foo", ["one two three"])
        self._register("skill.x:bar", ["four five six"])
        self.pipe.handle_detach_intent(Message(
            "detach_intent", data={"intent_name": "skill.x:foo"},
        ))
        self.assertNotIn("skill.x:foo",
                         self.engine.domains["skill.x"]._intent_samples)
        self.assertIn("skill.x:bar",
                      self.engine.domains["skill.x"]._intent_samples)

    def test_detach_skill_drops_domain(self):
        self._register("skill.a:foo", ["one two three"])
        self._register("skill.b:bar", ["four five six"])
        self.pipe.handle_detach_skill(Message(
            "detach_skill", data={"skill_id": "skill.a"},
        ))
        self.assertNotIn("skill.a", self.engine.domains)
        self.assertIn("skill.b", self.engine.domains)

    def test_train_marks_engine_trained(self):
        self._register("skill.a:foo",
                       ["one two three four", "five six seven eight"])
        self._register("skill.b:bar",
                       ["alpha beta gamma delta", "epsilon zeta eta theta"])
        # instant_train is on, so registration triggered training.
        self.assertFalse(self.engine.must_train)
        self.assertTrue(self.engine._trained)


class TestDomainEngineRouting(unittest.TestCase):
    """Direct tests of the DomainMarkovIntentEngine router."""

    def _engine(self) -> DomainMarkovIntentEngine:
        d = DomainMarkovIntentEngine(order=1, kneser_ney=False, backoff=False)
        d.register_domain_intent(
            "home", "home:lights_on",
            ["turn on the lights", "lights on please", "switch on the lights"])
        d.register_domain_intent(
            "media", "media:play",
            ["play some music", "put on a song", "start the music"])
        d.train()
        return d

    def test_routes_to_correct_domain_intent(self):
        d = self._engine()
        name, conf = d.calc_intent("turn on the lights")
        self.assertEqual(name, "home:lights_on")
        name, conf = d.calc_intent("play some music")
        self.assertEqual(name, "media:play")

    def test_results_come_from_one_domain(self):
        # Every returned label must belong to the single routed domain.
        d = self._engine()
        scores = d.calc_intents("turn on the lights")
        self.assertTrue(scores)
        domains = {label.split(":", 1)[0] for label, _ in scores}
        self.assertEqual(len(domains), 1)

    def test_explicit_domain_bypasses_router(self):
        d = self._engine()
        scores = d.calc_intents("turn on the lights", domain="media")
        self.assertTrue(all(lbl.startswith("media:") for lbl, _ in scores))

    def test_remove_domain_drops_from_router(self):
        d = self._engine()
        d.remove_domain("media")
        d.train()
        self.assertNotIn("media", d.domain_engine._intent_samples)
        self.assertNotIn("media", d.domains)


if __name__ == "__main__":
    unittest.main()
