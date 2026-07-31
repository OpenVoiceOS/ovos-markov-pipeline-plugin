# Pipeline integration

This page covers how `MarkovPipeline` plugs into OVOS: the OPM entry point, the
bus protocol it speaks, and the matching API.

## OPM entry point

The plugin is registered as an `opm.pipeline` entry point:

```toml
[project.entry-points."opm.pipeline"]
ovos-markov-pipeline-plugin = "ovos_markov_pipeline:MarkovPipeline"
```

`MarkovPipeline` subclasses `ConfidenceMatcherPipeline`, so OVOS exposes it at
three confidence tiers: `ovos-markov-pipeline-plugin-high`, `-medium` and
`-low`: which are the names placed in the `intents.pipeline` list.

## Construction

```python
MarkovPipeline(bus=None, config=None)
```

- `bus`: an `ovos-bus-client` `MessageBusClient` or a `FakeBus`. OVOS supplies
  the real bus. Tests can pass a `FakeBus`.
- `config`: the plugin config dict. When omitted it is read from
  `Configuration()["intents"]["ovos-markov-pipeline-plugin"]`.

On construction the plugin builds one `MarkovIntentEngine` per configured
language (the core `lang` plus any `secondary_langs`) and subscribes to its bus
messages.

## Bus protocol

### Messages consumed

| Message type                          | Effect                                                      |
| -------------------------------------- | ----------------------------------------------------------- |
| `padatious:register_intent`            | Register an intent and its training samples.                |
| `detach_intent`                        | Remove a single intent by `intent_name`.                    |
| `detach_skill`                         | Remove every intent registered by a `skill_id`.             |
| `mycroft.skills.train`                 | Train all engines that have pending samples.                |
| `intent.service.markov.manifest.get`   | Request the list of registered intents.                     |

`padatious:register_intent` is reused so that skills need no Markov-specific
code: any skill that registers a padatious-style intent is matchable. The
message `data` carries `name`, `samples` (a list of strings) or `file_name`,
`lang`, and `skill_id` (also accepted from `context`). A registration missing
`name` is rejected. One missing `skill_id` is attributed to `anonymous_skill`.

### Messages emitted

| Message type                       | When                                          |
| ----------------------------------- | ---------------------------------------------- |
| `mycroft.skills.trained`            | After a training pass completes.               |
| `intent.service.markov.manifest`    | Reply to `intent.service.markov.manifest.get`. |

## Matching API

`ConfidenceMatcherPipeline` calls one method per tier:

```python
match_high(utterances, lang, message)   -> IntentHandlerMatch | None
match_medium(utterances, lang, message) -> IntentHandlerMatch | None
match_low(utterances, lang, message)    -> IntentHandlerMatch | None
```

All three delegate to a common scorer with the tier's threshold. The scorer:

1. resolves the closest registered language to `lang`,
2. skips utterances longer than `max_words`,
3. scores each utterance with `MarkovIntentEngine.calc_intents`, honouring the
   session's `blacklisted_intents` and `blacklisted_skills`,
4. returns an `IntentHandlerMatch` when the best confidence exceeds the
   threshold, otherwise `None`.

The returned `IntentHandlerMatch` carries `match_type` (the intent name),
`skill_id`, the matched `utterance`, and `match_data` with the `confidence`.

## Training lifecycle

An `RLock` and a `finished_training_event` guard training, so concurrent
`mycroft.skills.train` messages and registrations are serialized. When
`instant_train` is `false`, registrations only mark engines dirty. The first
match request or an explicit train message triggers the actual training pass.

## MarkovIntentEngine

`MarkovIntentEngine` is the per-language matching core and is usable
standalone. Key methods:

- `add_intent(name, samples)`: register an intent (marks the engine dirty).
- `remove_intent(name)`: drop an intent.
- `train()`: build the shared vocabulary and one chain per intent.
- `calc_intents(utterance, blacklisted_intents=None, blacklisted_skills=None)`:
  return `(intent_name, confidence)` pairs sorted by confidence.
- `update_online(intent_name, utterance)`: append a sample and retrain.

See [Quickstart](quickstart.md) for a standalone usage example.

---
[Home](index.md) · [Domain pipeline →](domain_engine.md)
