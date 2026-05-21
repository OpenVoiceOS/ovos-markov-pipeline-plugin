"""Additional SlotExtractor tests."""

from ovos_markov_pipeline.slots import SlotExtractor


def test_extract_unknown_intent_returns_empty():
    extractor = SlotExtractor()
    assert extractor.extract("never-trained", ["set", "a", "timer"]) == {}


def test_extract_single_slot():
    extractor = SlotExtractor()
    extractor.add_slot_data(
        "timer.intent",
        utterances=[
            ["set", "a", "timer", "for", "five", "minutes"],
            ["set", "a", "timer", "for", "ten", "minutes"],
            ["set", "a", "timer", "for", "three", "minutes"],
        ],
        bio_tags=[
            ["O", "O", "O", "O", "B-time", "I-time"],
            ["O", "O", "O", "O", "B-time", "I-time"],
            ["O", "O", "O", "O", "B-time", "I-time"],
        ],
    )
    extractor.train()
    slots = extractor.extract(
        "timer.intent", ["set", "a", "timer", "for", "ten", "minutes"]
    )
    assert slots.get("time") == "ten minutes"


def test_extract_adjacent_slots():
    """Two back-to-back slots: the first is flushed when the second begins."""
    extractor = SlotExtractor()
    extractor.add_slot_data(
        "play.intent",
        utterances=[
            ["play", "jazz", "loud"],
            ["play", "rock", "loud"],
            ["play", "pop", "soft"],
        ],
        bio_tags=[
            ["O", "B-genre", "B-volume"],
            ["O", "B-genre", "B-volume"],
            ["O", "B-genre", "B-volume"],
        ],
    )
    extractor.train()
    slots = extractor.extract("play.intent", ["play", "jazz", "loud"])
    # extraction must not raise and returns a dict
    assert isinstance(slots, dict)


def test_remove_intent_clears_model():
    extractor = SlotExtractor()
    extractor.add_slot_data(
        "x.intent",
        utterances=[["a", "b", "c"]],
        bio_tags=[["O", "B-s", "I-s"]],
    )
    extractor.train()
    assert "x.intent" in extractor._models
    extractor.remove_intent("x.intent")
    assert "x.intent" not in extractor._models
    assert extractor.extract("x.intent", ["a", "b", "c"]) == {}
