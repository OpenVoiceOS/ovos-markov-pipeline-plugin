"""Error-path tests for IntentCache."""

from ovos_markov_pipeline.cache import IntentCache


def test_save_intent_handles_export_failure(tmp_path):
    """A model that cannot be exported is logged, not raised."""
    cache = IntentCache(str(tmp_path))
    # None is not a valid MarkovChain — export must fail gracefully
    cache.save_intent("broken:intent", None)
    assert not cache.has_intent("broken:intent")


def test_load_intent_missing_returns_none(tmp_path):
    cache = IntentCache(str(tmp_path))
    assert cache.load_intent("never:cached", order=2) is None


def test_load_intent_handles_corrupt_files(tmp_path):
    """Corrupt cache files are logged and load returns None."""
    cache = IntentCache(str(tmp_path))
    safe = cache._safe_name("corrupt:intent")
    (tmp_path / f"{safe}.onnx").write_bytes(b"not a real onnx file")
    (tmp_path / f"{safe}_vocab.json").write_text("{ this is not json")
    assert cache.load_intent("corrupt:intent", order=2) is None


def test_remove_intent_deletes_files(tmp_path):
    cache = IntentCache(str(tmp_path))
    safe = cache._safe_name("x:y")
    (tmp_path / f"{safe}.onnx").write_bytes(b"data")
    (tmp_path / f"{safe}_vocab.json").write_text("{}")
    assert cache.has_intent("x:y")
    cache.remove_intent("x:y")
    assert not cache.has_intent("x:y")


def test_clear_removes_all(tmp_path):
    cache = IntentCache(str(tmp_path))
    for name in ("a", "b"):
        (tmp_path / f"{name}.onnx").write_bytes(b"data")
        (tmp_path / f"{name}_vocab.json").write_text("{}")
    cache.clear()
    assert list(tmp_path.glob("*.onnx")) == []
    assert list(tmp_path.glob("*_vocab.json")) == []


def test_safe_name_sanitizes_separators(tmp_path):
    cache = IntentCache(str(tmp_path))
    assert cache._safe_name("skill:intent") == "skill__intent"
    assert cache._safe_name("a/b") == "a_b"
