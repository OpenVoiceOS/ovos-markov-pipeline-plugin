# Hierarchical Markov Pipeline & Engine

This page documents two layers that ship together:

* **`HierarchicalMarkovPipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-markov-hierarchical-pipeline-plugin`. Subclasses the flat `MarkovPipeline`; the only differences are the per-language engine shape (below) and that intents are routed to a domain == `skill_id` at registration time.
* **`HierarchicalMarkovIntentEngine`** — the two-stage variant of `MarkovIntentEngine` used internally by that pipeline.

A separate entry point (rather than a config flag on the flat pipeline) keeps the pipeline independently selectable in `default_pipeline` ordering and lets it have its own `intents.<key>` config block.

## Enabling

Add it to your OVOS config and place it in your pipeline order alongside (or in place of) the flat Markov pipeline:

```json
{
  "intents": {
    "ovos-markov-hierarchical-pipeline-plugin": {
      "order": 2,
      "kneser_ney": true,
      "backoff": true,
      "stem": false,
      "char_fallback": false,
      "conf_high": 0.50,
      "conf_med": 0.30,
      "conf_low": 0.15,
      "domain_threshold": 0.0,
      "instant_train": true
    }
  }
}
```

Configuration keys are read from `intents.ovos-markov-hierarchical-pipeline-plugin`. The pipeline accepts every key the flat plugin does, plus `domain_threshold`. The per-domain sub-engines and the top-level classifier inherit the same `order`, smoothing, stemmer, and char-fallback settings.

Pipeline order entries follow the standard confidence-tier naming:

```
"ovos-markov-hierarchical-pipeline-plugin-high",
"ovos-markov-hierarchical-pipeline-plugin-medium",
"ovos-markov-hierarchical-pipeline-plugin-low"
```

## Hierarchical engine

`HierarchicalMarkovIntentEngine` groups intents into *domains*, each owning its own `MarkovIntentEngine`. Unlike `DomainMarkovIntentEngine`, which scores every domain in parallel, this engine first routes a query to a single domain with a top-level classifier, then scores only that domain's intents.

The top-level classifier is itself a `MarkovIntentEngine`: every sample registered under a domain is also fed to the classifier with the domain name as the intent label, so the union of a domain's utterances trains a perplexity model for that domain. This mirrors the two-stage pattern used by `nebulento.HierarchicalIntentContainer` and the other OVOS intent plugins.

## Two-stage matching

```
              utterance
                 │
                 ▼
       ┌───────────────────────────────┐
       │ domain_engine.calc_intents()  │   top-level Markov classifier
       └───────────────────────────────┘
                 │
        best domain + conf
                 │
        conf >= domain_threshold ?
                 │ yes
                 ▼
       ┌───────────────────────────────┐
       │ domains[d].calc_intents(utt)  │   score only the selected domain
       └───────────────────────────────┘
                 │
                 ▼
         [(label, conf), …]
```

When the best domain scores below `domain_threshold`, the engine returns a no-match before any sub-engine runs — an off-topic rejection gate. `domain_threshold = 0.0` (default) disables the gate; every query is routed to its best domain.

The top-level classifier is rebuilt **lazily**: `register_domain_intent` only marks the domain dirty, and the classifier is retrained on the first query that needs it. This keeps bulk registration cheap.

Every `padatious:register_intent` event with name `<skill_id>:<intent>` triggers `engine.register_domain_intent(skill_id, "<skill_id>:<intent>", samples)` on the per-language `HierarchicalMarkovIntentEngine`. `train()` trains each sub-engine and the classifier.

`detach_intent` and `detach_skill` route through the same per-domain pathway: a `detach_skill` for `<skill_id>` drops the whole `<skill_id>` domain from every language engine in one call.

## Routing rules

* Intent label must be of the form `<skill_id>:<intent>`. The portion before the first `:` is the domain.
* Labels without a `:` use the whole name as the domain (a single-intent domain).
* `detach_skill` drops the entire `<skill_id>` domain.
* `detach_intent` drops a single label from its domain's sub-engine.

## Usage

The pipeline is OPM-discoverable; instantiate via the bus the same way as the flat plugin. For programmatic use of the engine alone:

```python
from ovos_markov_pipeline import HierarchicalMarkovIntentEngine

d = HierarchicalMarkovIntentEngine(order=2)

d.register_domain_intent("media",
                          "media:play",
                          ["play music", "put on a song"])
d.register_domain_intent("home",
                          "home:lights_on",
                          ["turn on the lights", "lights on"])
d.train()

name, conf = d.calc_intent("turn on the lights")
# name → "home:lights_on"
```

### Inspecting the routed domain

`calc_domain` exposes the top-level classifier on its own:

```python
d.calc_domain("play music")
# → ("media", 0.68)
```

### Restricting to a single domain

Pass `domain=...` to `calc_intent` / `calc_intents` to score only inside a specific domain, bypassing the classifier and the threshold gate:

```python
d.calc_intent("play africa", domain="media")
```

### Off-topic rejection

Set `domain_threshold` above `0.0` to reject queries whose best domain scores below the gate:

```python
d = HierarchicalMarkovIntentEngine(order=2, domain_threshold=0.3)
```

## See also

- [Domain pipeline](domain_engine.md) — parallel-argmax domain variant.
- [Pipeline overview](index.md) — flat `MarkovPipeline` and its bus messages.
- The OPM entry-point list in `pyproject.toml` — all three pipelines are registered.
