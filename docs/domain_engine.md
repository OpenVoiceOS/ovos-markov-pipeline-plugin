# Domain Markov Pipeline & Engine

This page documents two layers that ship together:

* **`DomainMarkovPipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-markov-domain-pipeline-plugin`. Subclasses the flat `MarkovPipeline`; the only differences are the per-language engine shape (below) and that intents are routed to a domain == `skill_id` at registration time.
* **`DomainMarkovIntentEngine`** — the hierarchical, two-level variant of `MarkovIntentEngine` used internally by that pipeline.

A separate entry point (rather than a `domain_engine: true` config flag on the flat pipeline) keeps the two pipelines independently selectable in `default_pipeline` ordering and lets each have its own `intents.<key>` config block.

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

Configuration keys are read from `intents.ovos-markov-domain-pipeline-plugin`. The pipeline accepts every key the flat plugin does — the per-language sub-engines inherit the same `order`, smoothing, stemmer, and char-fallback settings.

Pipeline order entries follow the standard confidence-tier naming:

```
"ovos-markov-domain-pipeline-plugin-high",
"ovos-markov-domain-pipeline-plugin-medium",
"ovos-markov-domain-pipeline-plugin-low"
```

## Hierarchical engine

`DomainMarkovIntentEngine` is the hierarchical variant of `MarkovIntentEngine`. Intents are grouped into *domains*, and at inference time the engine first picks the most likely domain, then scores intents only within that domain. This mirrors the API shipped by sibling OVOS intent plugins (`nebulento.DomainIntentContainer`, `ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`, `padacioso.DomainIntentContainer`, `linha_fina.DomainIntentEngine`, `ovos_m2v_pipeline.DomainPrototypeIntentStore`).

## Why hierarchical

Two-level matching gives the perplexity paradigm two concrete benefits:

1. **Sharper per-domain perplexities.** A domain's intents share a vocabulary subspace (lights / thermostat / door all share "smarthome" surface forms), so the perplexity gap across the sub-engine's intents is more discriminative than the global gap over every registered intent.
2. **Lower far-OOD false-positive rate.** The top-level Markov classifier rejects chitchat that doesn't strongly match any domain *before* any sub-engine sees it.

## Architecture

```
              utterance
                 │
                 ▼
       ┌───────────────────────┐
       │   domain_engine       │   MarkovIntentEngine
       │   (router)            │   one chain per domain
       └───────────────────────┘
                 │
            best domain
                 │
                 ▼
       ┌───────────────────────┐
       │   domains[<skill_id>] │   MarkovIntentEngine
       │   (intent matcher)    │   one chain per intent
       └───────────────────────┘
                 │
         [(label, conf), …]
```

Every `padatious:register_intent` event with name `<skill_id>:<intent>` triggers:

* `engine.register_domain_intent(skill_id, "<skill_id>:<intent>", samples)` on the per-language `DomainMarkovIntentEngine`.
* On `train()`, the top-level `domain_engine` is auto-seeded with the concatenated samples of every intent in each domain (unless seeded explicitly via `engine.domain_engine.add_intent(...)` first).

`detach_intent` and `detach_skill` route through the same per-domain pathway: a `detach_skill` for `<skill_id>` drops the whole `<skill_id>` domain from every language engine in one call.

## Hierarchical routing rules

* Intent label must be of the form `<skill_id>:<intent>`. The portion before the first `:` is the domain.
* Labels without a `:` use the whole name as the domain (a single-intent domain).
* `detach_skill` drops the entire `<skill_id>` domain (router entry + sub-engine).
* `detach_intent` drops a single label from its domain's sub-engine; the domain entry in the router is recomputed on the next `train()`.

## Usage

The pipeline is OPM-discoverable; instantiate via the bus the same way as the flat plugin. For programmatic use of the engine alone:

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

### Bypassing the router

Pass `domain=...` to `calc_intent` / `calc_intents` to skip the top-level classifier and score directly inside a specific domain:

```python
d.calc_intent("play africa", domain="media")
```

### Inspecting the resolved domain

```python
d.calc_domains("lights on")  # → [("home", 0.83), ("media", 0.21)]
```

## See also

- [Pipeline overview](index.md) — flat `MarkovPipeline` and its bus messages.
- The OPM entry-point list in `pyproject.toml` — both pipelines are registered.
