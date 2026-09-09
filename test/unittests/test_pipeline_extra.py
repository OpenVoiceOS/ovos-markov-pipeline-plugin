"""Additional MarkovPipeline tests covering bus-handler edge cases."""

import tempfile

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_markov_pipeline import MarkovPipeline


def _pipeline(**config) -> MarkovPipeline:
    base = {"order": 1, "instant_train": True}
    base.update(config)
    return MarkovPipeline(bus=FakeBus(), config=base)


def _register(name, samples, skill_id="some.skill", lang="en-US", **data):
    payload = {"name": name, "samples": samples, "lang": lang, "skill_id": skill_id}
    payload.update(data)
    return Message("padatious:register_intent", payload)


def test_register_intent_without_skill_id_uses_anonymous():
    pipe = _pipeline()
    msg = Message("padatious:register_intent",
                  {"name": "x", "samples": ["hello there", "hi there"], "lang": "en-US"})
    pipe.register_intent(msg)
    assert "x" in pipe.registered_intents
    assert "x" in pipe._skill2intent["anonymous_skill"]
    pipe.shutdown()


def test_register_intent_without_name_is_rejected():
    pipe = _pipeline()
    pipe.register_intent(Message("padatious:register_intent",
                                 {"samples": ["hello"], "skill_id": "s"}))
    assert pipe.registered_intents == []
    pipe.shutdown()


def test_register_intent_coerces_non_list_samples():
    """A tuple of samples is coerced to a list rather than rejected."""
    pipe = _pipeline()
    pipe.register_intent(_register("x", ("hello there", "hi there")))
    assert "x" in pipe.registered_intents
    pipe.shutdown()


def test_register_intent_missing_file_is_handled():
    pipe = _pipeline()
    pipe.register_intent(_register("x", None, file_name="/no/such/file.intent"))
    # unreadable file -> no samples -> intent not registered, no crash
    assert "x" not in pipe.registered_intents
    pipe.shutdown()


def test_register_intent_from_file():
    pipe = _pipeline()
    with tempfile.NamedTemporaryFile("w", suffix=".intent", delete=False) as f:
        f.write("hello there\nhi there\ngood morning\n")
        path = f.name
    pipe.register_intent(_register("greet", None, file_name=path))
    assert "greet" in pipe.registered_intents
    pipe.shutdown()


def test_unordered_thresholds_do_not_crash():
    """Thresholds out of order are tolerated (a warning is logged)."""
    pipe = _pipeline(conf_low=0.9, conf_med=0.5, conf_high=0.1)
    assert pipe.conf_low == 0.9
    pipe.shutdown()


def test_closest_lang_prefix_match():
    pipe = _pipeline()
    # the engine is built for en-US; a regional variant resolves by prefix
    assert pipe._get_closest_lang("en-GB") == "en-US"
    assert pipe._get_closest_lang("zz-ZZ") is None
    pipe.shutdown()


def test_detach_intent_removes_it():
    pipe = _pipeline()
    pipe.register_intent(_register("a", ["hello there", "hi there"]))
    pipe.register_intent(_register("b", ["goodbye now", "see you later"]))
    pipe.handle_detach_intent(Message("detach_intent", {"intent_name": "a"}))
    assert "a" not in pipe.registered_intents
    assert "b" in pipe.registered_intents
    pipe.shutdown()


def test_detach_skill_removes_all_its_intents():
    pipe = _pipeline()
    pipe.register_intent(_register("a", ["hello there", "hi there"], skill_id="s1"))
    pipe.register_intent(_register("b", ["goodbye now", "see you"], skill_id="s1"))
    pipe.register_intent(_register("c", ["thanks a lot", "thank you"], skill_id="s2"))
    pipe.handle_detach_skill(Message("detach_skill", {"skill_id": "s1"}))
    assert pipe.registered_intents == ["c"]
    pipe.shutdown()


def test_detach_skill_without_skill_id_is_noop():
    pipe = _pipeline()
    pipe.register_intent(_register("a", ["hello there", "hi there"]))
    pipe.handle_detach_skill(Message("detach_skill", {}))
    assert "a" in pipe.registered_intents
    pipe.shutdown()


def test_manifest_lists_registered_intents():
    pipe = _pipeline()
    pipe.register_intent(_register("a", ["hello there", "hi there"]))
    replies = []
    pipe.bus.on("intent.service.markov.manifest", lambda m: replies.append(m))
    pipe.handle_manifest(Message("intent.service.markov.manifest.get", {}))
    assert replies
    assert "a" in replies[0].data["intents"]
    pipe.shutdown()


def _spec_register(intent_name, samples, skill_id, sender=None, lang="en-US"):
    data = {"intent_name": intent_name, "samples": samples, "lang": lang}
    if skill_id is not None:
        data["skill_id"] = skill_id
    context = {"skill_id": sender} if sender else {}
    return Message("ovos.intent.register.template", data, context)


def test_spec_register_indexes_under_the_payload_skill():
    # OVOS-INTENT-4 §3.2: the payload names the target, the context the source
    pipe = _pipeline()
    pipe.handle_register_template(
        _spec_register("play_music", ["play music", "put on music"],
                       "music.skill", sender="admin.skill"))
    assert "music.skill:play_music" in pipe.registered_intents
    assert "admin.skill:play_music" not in pipe.registered_intents
    pipe.shutdown()


def test_spec_register_without_payload_skill_id_is_rejected():
    pipe = _pipeline()
    pipe.handle_register_template(
        _spec_register("play_music", ["play music", "put on music"],
                       None, sender="admin.skill"))
    assert "admin.skill:play_music" not in pipe.registered_intents
    assert pipe.registered_intents == []
    pipe.shutdown()


def test_spec_skill_deregister_without_payload_spares_the_sender():
    # §8.4 is the highest-blast-radius message in the family: a payload with no
    # skill_id names no target, so it must destroy nothing rather than be
    # applied to whoever sent it.
    pipe = _pipeline()
    pipe.handle_register_template(
        _spec_register("play_music", ["play music", "put on music"],
                       "music.skill"))
    assert "music.skill:play_music" in pipe.registered_intents
    pipe.handle_deregister_skill_spec(
        Message("ovos.skill.deregister", {}, {"skill_id": "music.skill"}))
    assert "music.skill:play_music" in pipe.registered_intents
    pipe.shutdown()


def test_spec_skill_deregister_removes_the_named_skill_not_the_sender():
    pipe = _pipeline()
    pipe.handle_register_template(
        _spec_register("play_music", ["play music", "put on music"], "music.skill"))
    pipe.handle_register_template(
        _spec_register("shutdown", ["shut down now", "power off"], "admin.skill"))
    pipe.handle_deregister_skill_spec(
        Message("ovos.skill.deregister", {"skill_id": "music.skill"},
                {"skill_id": "admin.skill"}))
    assert "music.skill:play_music" not in pipe.registered_intents
    assert "admin.skill:shutdown" in pipe.registered_intents
    pipe.shutdown()


def test_legacy_detach_skill_still_uses_the_context():
    # the legacy wire carries no required payload identity, and its labels are
    # the bare intent name rather than the namespaced spec form
    pipe = _pipeline()
    pipe.register_intent(_register("greet", ["hello there", "hi there"],
                                   skill_id="music.skill"))
    assert "greet" in pipe.registered_intents
    pipe.handle_detach_skill(Message("detach_skill", {},
                                      {"skill_id": "music.skill"}))
    assert "greet" not in pipe.registered_intents
    pipe.shutdown()
