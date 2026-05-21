"""Tests for SlotExtractor (HMM BIO tagging)."""

from ovos_markov_pipeline.slots import SlotExtractor


class TestSlotExtractor:
    def _make_extractor(self) -> SlotExtractor:
        ext = SlotExtractor()
        ext.add_slot_data(
            "timer:set_timer",
            utterances=[
                ["set", "a", "timer", "for", "five", "minutes"],
                ["set", "a", "timer", "for", "ten", "minutes"],
                ["timer", "for", "three", "minutes"],
                ["set", "a", "countdown", "for", "two", "hours"],
            ],
            bio_tags=[
                ["O", "O", "O", "O", "B-time", "I-time"],
                ["O", "O", "O", "O", "B-time", "I-time"],
                ["O", "O", "B-time", "I-time"],
                ["O", "O", "O", "O", "B-time", "I-time"],
            ],
        )
        ext.train()
        return ext

    def test_train(self) -> None:
        ext = self._make_extractor()
        assert "timer:set_timer" in ext._models

    def test_extract_slots(self) -> None:
        ext = self._make_extractor()
        slots = ext.extract("timer:set_timer", ["set", "a", "timer", "for", "five", "minutes"])
        assert "time" in slots
        assert "five" in slots["time"] or "minutes" in slots["time"]

    def test_extract_unknown_intent(self) -> None:
        ext = self._make_extractor()
        slots = ext.extract("unknown:intent", ["hello"])
        assert slots == {}

    def test_remove_intent(self) -> None:
        ext = self._make_extractor()
        ext.remove_intent("timer:set_timer")
        assert "timer:set_timer" not in ext._models

    def test_empty_data(self) -> None:
        ext = SlotExtractor()
        ext.add_slot_data("test:empty", [], [])
        ext.train()
        assert "test:empty" not in ext._models

    def test_multiple_slots(self) -> None:
        ext = SlotExtractor()
        ext.add_slot_data(
            "alarm:set_alarm",
            utterances=[
                ["set", "alarm", "for", "monday", "at", "seven", "am"],
                ["set", "alarm", "for", "tuesday", "at", "eight", "pm"],
            ],
            bio_tags=[
                ["O", "O", "O", "B-day", "O", "B-time", "I-time"],
                ["O", "O", "O", "B-day", "O", "B-time", "I-time"],
            ],
        )
        ext.train()
        slots = ext.extract(
            "alarm:set_alarm", ["set", "alarm", "for", "monday", "at", "seven", "am"]
        )
        # Should extract at least one slot
        assert len(slots) > 0

    def test_o_only_tags(self) -> None:
        ext = SlotExtractor()
        ext.add_slot_data(
            "test:noslots",
            utterances=[["hello", "world"]],
            bio_tags=[["O", "O"]],
        )
        ext.train()
        slots = ext.extract("test:noslots", ["hello", "world"])
        assert slots == {}
