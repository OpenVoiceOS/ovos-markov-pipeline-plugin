"""Unit tests for HierarchicalMarkovPipeline and engine (no ovoscope required)."""
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_markov_pipeline import (
    HierarchicalMarkovIntentEngine,
    HierarchicalMarkovPipeline,
    MarkovPipeline,
)


def _make_pipeline(**overrides) -> HierarchicalMarkovPipeline:
    cfg = {"order": 1, "kneser_ney": False, "backoff": False,
           "smoothing": 1e-5, "instant_train": True}
    cfg.update(overrides)
    return HierarchicalMarkovPipeline(bus=FakeBus(), config=cfg)


class TestHierarchicalEngine(unittest.TestCase):
    def _make_engine(self, **kw) -> HierarchicalMarkovIntentEngine:
        return HierarchicalMarkovIntentEngine(
            order=1, kneser_ney=False, backoff=False, smoothing=1e-5, **kw)

    def test_two_stage_routing_picks_correct_intent(self):
        eng = self._make_engine()
        eng.register_domain_intent(
            "home", "home:lights",
            ["turn on the lights", "lights on please", "switch on the lights"])
        eng.register_domain_intent(
            "media", "media:music",
            ["play some music", "start the music", "put on some music"])
        eng.train()
        result = eng.calc_intent("turn on the lights")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "home:lights")

    def test_calc_domain_classifies_query(self):
        eng = self._make_engine()
        eng.register_domain_intent(
            "home", "home:lights",
            ["turn on the lights", "lights on please", "switch on the lights"])
        eng.register_domain_intent(
            "media", "media:music",
            ["play some music", "start the music", "put on some music"])
        eng.train()
        dom = eng.calc_domain("play some music")
        self.assertIsNotNone(dom)
        self.assertEqual(dom[0], "media")

    def test_explicit_domain_bypasses_classifier(self):
        eng = self._make_engine()
        eng.register_domain_intent(
            "home", "home:lights",
            ["turn on the lights", "lights on please"])
        eng.register_domain_intent(
            "media", "media:music",
            ["play some music", "start the music"])
        eng.train()
        result = eng.calc_intent("play some music", domain="home")
        # forced into the home domain; only home intents are scored
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "home:lights")

    def test_domain_threshold_rejects_below_gate(self):
        eng = self._make_engine(domain_threshold=2.0)  # impossible to reach
        eng.register_domain_intent(
            "home", "home:lights",
            ["turn on the lights", "lights on please"])
        eng.train()
        self.assertEqual(eng.calc_intents("turn on the lights"), [])

    def test_remove_domain_drops_everything(self):
        eng = self._make_engine()
        eng.register_domain_intent("home", "home:lights", ["lights on please"])
        eng.register_domain_intent("media", "media:music", ["play some music"])
        eng.remove_domain("home")
        self.assertNotIn("home", eng.domains)
        self.assertIn("media", eng.domains)

    def test_lazy_classifier_rebuild_on_query(self):
        eng = self._make_engine()
        eng.register_domain_intent("home", "home:lights", ["lights on please"])
        # registration only marks dirty
        self.assertTrue(eng.must_train)
        eng.calc_intent("lights on please")
        self.assertFalse(eng.must_train)


class TestHierarchicalPipelineConstruction(unittest.TestCase):
    def test_engines_are_hierarchical_engines(self):
        pipe = _make_pipeline()
        self.assertIsInstance(pipe, MarkovPipeline)
        for engine in pipe.engines.values():
            self.assertIsInstance(engine, HierarchicalMarkovIntentEngine)

    def test_domain_of_extracts_skill_id(self):
        self.assertEqual(
            HierarchicalMarkovPipeline._domain_of("skill.foo:bar.intent"),
            "skill.foo",
        )
        self.assertEqual(
            HierarchicalMarkovPipeline._domain_of("orphan"),
            "orphan",
        )

    def test_domain_threshold_from_config(self):
        pipe = _make_pipeline(domain_threshold=0.4)
        self.assertEqual(pipe.domain_threshold, 0.4)
        for engine in pipe.engines.values():
            self.assertEqual(engine.domain_threshold, 0.4)


class TestHierarchicalPipelineRouting(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
