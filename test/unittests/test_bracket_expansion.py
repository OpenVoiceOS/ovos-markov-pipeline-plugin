"""Tests for OVOS template / bracket expansion in MarkovIntentEngine."""

from ovos_markov_pipeline import MarkovIntentEngine


class TestBracketExpansion:
    def test_alternatives_are_expanded(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("greet", ["(hi|hello) friend"])
        samples = eng._intent_samples["greet"]
        assert "hi friend" in samples
        assert "hello friend" in samples
        assert len(samples) == 2

    def test_optional_components_are_expanded(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("greet", ["hello [there] friend"])
        samples = {s.replace("  ", " ").strip() for s in eng._intent_samples["greet"]}
        # Both variants should be present, modulo whitespace from the optional gap
        assert any(s == "hello friend" for s in samples)
        assert any(s == "hello there friend" for s in samples)

    def test_slot_placeholders_preserved(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("play", ["play {song}"])
        samples = eng._intent_samples["play"]
        assert any("{song}" in s for s in samples)

    def test_combined_template(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("play", ["(play|start) [the] {song}"])
        samples = eng._intent_samples["play"]
        # Every expanded variant must still contain the slot
        assert all("{song}" in s for s in samples)
        # 2 alternatives x 2 optional states = 4 variants
        assert len(samples) == 4

    def test_end_to_end_match(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("greet", ["(hi|hello|hey) [there]"])
        eng.add_intent("bye", ["(bye|goodbye) [now]"])
        eng.train()
        scores = eng.calc_intents("hello there")
        assert scores
        # calc_intents returns list of (intent, conf) tuples, sorted descending
        assert scores[0][0] == "greet"

    def test_empty_and_whitespace_samples_skipped(self) -> None:
        eng = MarkovIntentEngine(order=2, char_fallback=False)
        eng.add_intent("x", ["", "   ", "(a|b)"])
        samples = eng._intent_samples["x"]
        assert set(samples) == {"a", "b"}
