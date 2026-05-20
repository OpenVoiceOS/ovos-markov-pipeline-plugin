# Domain Markov Pipeline & Engine

This page documents two layers that ship together:

* **`DomainMarkovPipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-markov-domain-pipeline-plugin`. Subclasses the flat `MarkovPipeline`; the only differences are the per-language engine shape (below) and that intents are routed to a domain == `skill_id` at registration time.
* **`DomainMarkovIntentEngine`** — the domain-aware variant of `MarkovIntentEngine` used internally by that pipeline.

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

Configuration keys are read from `intents.ovos-markov-domain-pipeline-plugin`. The pipeline accepts every key the flat plugin does — the per-domain sub-engines inherit the same `order`, smoothing, stemmer, and char-fallback settings.

Pipeline order entries follow the standard confidence-tier naming:

```
"ovos-markov-domain-pipeline-plugin-high",
"ovos-markov-domain-pipeline-plugin-medium",
"ovos-markov-domain-pipeline-plugin-low"
```

## Domain engine

`DomainMarkovIntentEngine` groups intents into *domains*, each owning its own `MarkovIntentEngine`. There is no top-level router — at query time every domain scores the utterance independently and the global argmax wins. This mirrors the parallel-argmax pattern used by `adapt` and the other OVOS intent plugins (`nebulento.DomainIntentContainer`, `ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`, `padacioso.DomainIntentContainer`, `linha_fina.DomainIntentEngine`, `ovos_m2v_pipeline.DomainPrototypeIntentStore`).

## Why a domain layout

Even without a router, organising intents into domains pays off for the perplexity paradigm:

1. **Sharper per-domain perplexities.** A domain's intents share a vocabulary subspace (lights / thermostat / door all share "smarthome" surface forms), so per-domain Markov chains use denser, more discriminative count tables than a single global model.
2. **Cheap pre-pruning.** Each domain has a small word-vocabulary set; if the utterance shares no tokens with a domain's vocabulary, that sub-engine is skipped entirely. With many registered skills this prunes the vast majority of domains on a typical utterance.

## Architecture

```
              utterance
                 │
                 ▼
       ┌───────────────────────────────┐
       │ vocab-overlap pre-filter      │   in-memory set check
       │  (+ optional top_k_domains)   │
       └───────────────────────────────┘
                 │
        candidate domains
                 │
                 ▼
       ┌───────────────────────────────┐
       │ domains[d].calc_intents(utt)  │   parallel per-domain scoring
       │   for d in candidates         │
       └───────────────────────────────┘
                 │
       flatten + sort by confidence
                 │
                 ▼
         [(label, conf), …]
```

Every `padatious:register_intent` event with name `<skill_id>:<intent>` triggers `engine.register_domain_intent(skill_id, "<skill_id>:<intent>", samples)` on the per-language `DomainMarkovIntentEngine`. `train()` trains each sub-engine independently — there is no router to seed.

`detach_intent` and `detach_skill` route through the same per-domain pathway: a `detach_skill` for `<skill_id>` drops the whole `<skill_id>` domain from every language engine in one call.

## Routing rules

* Intent label must be of the form `<skill_id>:<intent>`. The portion before the first `:` is the domain.
* Labels without a `:` use the whole name as the domain (a single-intent domain).
* `detach_skill` drops the entire `<skill_id>` domain.
* `detach_intent` drops a single label from its domain's sub-engine.

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

### Restricting to a single domain

Pass `domain=...` to `calc_intent` / `calc_intents` to score only inside a specific domain:

```python
d.calc_intent("play africa", domain="media")
```

### Pre-pruning hint

For very large deployments you can pass `top_k_domains=K` to restrict scoring to the K domains with the highest fingerprint score (median per-intent confidence) after the vocabulary-overlap filter:

```python
d.calc_intents("turn on the lights", top_k_domains=8)
```

The default (`None`) scores every candidate domain that passes the vocabulary-overlap filter.

## See also

- [Pipeline overview](index.md) — flat `MarkovPipeline` and its bus messages.
- The OPM entry-point list in `pyproject.toml` — both pipelines are registered.
