# Domain Markov Pipeline & Engine

This page documents two layers that ship together:

* **`DomainMarkovPipeline`**: the OPM-discoverable pipeline class. Entry point: `ovos-markov-domain-pipeline-plugin`. It subclasses the flat `MarkovPipeline`. The differences are the per-language engine shape (below) and that intents are routed to a domain equal to `skill_id` at registration time.
* **`DomainMarkovIntentEngine`**: the domain-aware variant of `MarkovIntentEngine` used internally by that pipeline.

A separate entry point, rather than a `domain_engine: true` config flag on the flat pipeline, keeps the two pipelines independently selectable in `default_pipeline` ordering. It also lets each pipeline keep its own `intents.<key>` config block.

## Enabling

Add it to your OVOS config and place it in your pipeline order alongside (or in place of) the flat Markov pipeline:

```json
{
  "intents": {
    "ovos-markov-domain-pipeline-plugin": {
      "order": 2,
      "kneser_ney": true,
      "backoff": true,
      "stem": false,
      "char_fallback": false,
      "conf_high": 0.50,
      "conf_med": 0.30,
      "conf_low": 0.15,
      "instant_train": true
    }
  }
}
```

Configuration keys are read from `intents.ovos-markov-domain-pipeline-plugin`. The pipeline accepts every key the flat plugin does. The per-domain sub-engines inherit the same `order`, smoothing, stemmer, and char-fallback settings.

Pipeline order entries follow the standard confidence-tier naming:

```text
"ovos-markov-domain-pipeline-plugin-high",
"ovos-markov-domain-pipeline-plugin-medium",
"ovos-markov-domain-pipeline-plugin-low"
```

## Domain engine

`DomainMarkovIntentEngine` groups intents into *domains*, each owning its own `MarkovIntentEngine`. A top-level **router**, itself a `MarkovIntentEngine` whose "intents" are the domains, first picks the most likely domain for an utterance. That domain's sub-engine then resolves the concrete intent. This mirrors the two-stage domain-to-intent model used by the other OVOS intent plugins: `nebulento.DomainIntentContainer`, `ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`, `padacioso.DomainIntentContainer`, and `linha_fina.DomainIntentEngine`.

## Why a router

A `MarkovIntentEngine` builds **one shared vocabulary** from the union of its intents' samples. Perplexity, and the confidence derived from it, is only calibrated against intents trained on that same vocabulary.

If every domain sub-engine scored the utterance independently and a flat global argmax picked the winner, it would compare confidences computed against **different-sized vocabularies** (each domain's own). That is not a valid ranking. A domain with a smaller vocabulary gets a systematic confidence bias.

Two-stage routing keeps every comparison within a single vocabulary:

1. **Routing stage**: the router scores the utterance against each domain. Every domain is one router "intent" trained on the concatenation of that domain's intent samples, so all domains share the router's (global) vocabulary and rank consistently.
2. **Resolution stage**: only the routed domain's sub-engine runs. Its intents all share that domain's vocabulary, so their confidences rank consistently.

The returned confidences therefore always come from a single sub-engine over a single vocabulary.

## Architecture

```text
              utterance
                 │
                 ▼
       ┌───────────────────────────────┐
       │ domain_engine.calc_intent()   │   router: one MarkovChain
       │   (router)                    │   per domain, shared vocab
       └───────────────────────────────┘
                 │
            best domain
                 │
                 ▼
       ┌───────────────────────────────┐
       │ domains[d].calc_intents(utt)  │   resolve the intent inside
       │   (intent matcher)            │   the routed domain
       └───────────────────────────────┘
                 │
                 ▼
         [(label, conf), …]
```

Every `padatious:register_intent` event with name `<skill_id>:<intent>` triggers `engine.register_domain_intent(skill_id, "<skill_id>:<intent>", samples)` on the per-language `DomainMarkovIntentEngine`. `train()` rebuilds the router from the current domains (so removed domains drop out) and trains every sub-engine.

`detach_intent` and `detach_skill` route through the same per-domain pathway: a `detach_skill` for `<skill_id>` drops the whole `<skill_id>` domain from every language engine in one call.

## Routing rules

* Intent label must be of the form `<skill_id>:<intent>`. The portion before the first `:` is the domain.
* Labels without a `:` use the whole name as the domain (a single-intent domain).
* `detach_skill` drops the entire `<skill_id>` domain.
* `detach_intent` drops a single label from its domain's sub-engine.

## Usage

The pipeline is OPM-discoverable. Instantiate it over the bus the same way as the flat plugin. For programmatic use of the engine alone:

```python
from ovos_markov_pipeline import DomainMarkovIntentEngine

d = DomainMarkovIntentEngine(order=2)

d.register_domain_intent("media",
                          "media:play",
                          ["play {song}", "put on {song}"])
d.register_domain_intent("home",
                          "home:lights_on",
                          ["turn on the lights", "lights on"])
d.train()

scores = d.calc_intents("turn on the lights")
# scores → [("home:lights_on", 0.71), …]
```

### Restricting to a single domain

Pass `domain=...` to `calc_intent` / `calc_intents` to bypass the router and resolve directly inside a specific domain:

```python
d.calc_intent("play africa", domain="media")
```

## See also

- [Pipeline overview](pipeline.md): flat `MarkovPipeline` and its bus messages.
- The OPM entry-point list in `pyproject.toml`: both pipelines are registered.

---
[← Pipeline integration](pipeline.md) · [Home](index.md) · [Entity extraction →](entities.md)
