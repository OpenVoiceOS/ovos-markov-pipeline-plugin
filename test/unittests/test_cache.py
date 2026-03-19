"""Tests for IntentCache (ONNX disk caching)."""

import tempfile

from markovonnx import MarkovChain, Vocabulary

from ovos_markov_pipeline.cache import IntentCache


def _trained_model() -> MarkovChain:
    vocab = Vocabulary()
    seqs = [["set", "a", "timer"], ["start", "a", "timer"]] * 10
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(seqs)
    return mc


class TestIntentCache:
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            mc = _trained_model()
            cache.save_intent("timer:set_timer", mc)
            assert cache.has_intent("timer:set_timer")

            rt = cache.load_intent("timer:set_timer", order=1)
            assert rt is not None
            probs = rt.predict_probs(["set"])
            assert mc.vocab.size in probs.shape  # may be [V] or [1, V]

    def test_load_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            assert cache.load_intent("nonexistent", order=1) is None

    def test_has_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            assert not cache.has_intent("foo:bar")
            mc = _trained_model()
            cache.save_intent("foo:bar", mc)
            assert cache.has_intent("foo:bar")

    def test_remove_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            mc = _trained_model()
            cache.save_intent("test:intent", mc)
            assert cache.has_intent("test:intent")
            cache.remove_intent("test:intent")
            assert not cache.has_intent("test:intent")

    def test_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            mc = _trained_model()
            cache.save_intent("a:b", mc)
            cache.save_intent("c:d", mc)
            cache.clear()
            assert not cache.has_intent("a:b")
            assert not cache.has_intent("c:d")

    def test_safe_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = IntentCache(tmpdir)
            assert cache._safe_name("skill:intent") == "skill__intent"
