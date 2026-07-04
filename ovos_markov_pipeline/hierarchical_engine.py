"""Two-stage hierarchical Markov intent engine.

Intents are grouped into *domains*. Unlike :class:`DomainMarkovIntentEngine`,
which scores every domain in parallel, this engine first routes a query to a
single domain with a top-level classifier, then scores only that domain's
intents.

The top-level classifier is itself a :class:`MarkovIntentEngine`: every sample
registered under a domain is also fed to the classifier with the domain name
as the intent label, so the union of a domain's utterances trains a perplexity
model for that domain. At query time the classifier picks the most likely
domain and the matching sub-engine resolves the concrete intent.

The classifier is rebuilt lazily: registration only marks a domain dirty, and
the classifier is retrained on the first query that needs it. A
``domain_threshold`` gate rejects queries whose best domain scores below a
configurable confidence, providing off-topic rejection before any sub-engine
runs.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ovos_markov_pipeline import MarkovIntentEngine


class HierarchicalMarkovIntentEngine:
    """Two-stage Markov intent engine: domain routing then intent matching.

    Intents are grouped into *domains*, each owning its own
    :class:`MarkovIntentEngine`. A top-level :class:`MarkovIntentEngine`
    classifier maps free-text queries to a domain name; only the selected
    domain's sub-engine is then scored.

    The classifier is trained automatically — every sample passed to
    :meth:`register_domain_intent` is also registered on
    :attr:`domain_engine` under its domain name, so the engine works
    standalone with no manual classifier setup.

    Domains can also be selected explicitly via the ``domain`` kwarg on
    :meth:`calc_intent` / :meth:`calc_intents`, bypassing the classifier
    and the threshold gate.

    Example::

        from ovos_markov_pipeline import HierarchicalMarkovIntentEngine

        d = HierarchicalMarkovIntentEngine(order=2)

        d.register_domain_intent("media", "media:play",
                                  ["play music", "put on a song"])
        d.register_domain_intent("home", "home:lights_on",
                                  ["turn on the lights", "lights on"])

        d.train()
        name, conf = d.calc_intent("turn on the lights")
        # name == "home:lights_on"

    Args:
        domain_threshold: Minimum confidence the top-level classifier must
            reach for a query to be routed at all. When the best domain
            scores below this, :meth:`calc_intent` / :meth:`calc_intents`
            return a no-match instead of resolving an intent. ``0.0``
            (default) disables the gate; every query is routed to its best
            domain.
        engine_kwargs: Forwarded to every internal :class:`MarkovIntentEngine`
            instance, including the top-level classifier.
    """

    def __init__(self, domain_threshold: float = 0.0, **engine_kwargs) -> None:
        self._engine_kwargs = dict(engine_kwargs)
        self.domain_threshold = domain_threshold
        #: Top-level classifier mapping free-text queries to a domain name.
        self.domain_engine: MarkovIntentEngine = MarkovIntentEngine(**engine_kwargs)
        #: Per-domain intent engines, keyed by domain name.
        self.domains: Dict[str, MarkovIntentEngine] = {}
        #: Raw training samples per (domain, intent).
        self.training_data: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        #: Domains whose classifier entry is stale and must be rebuilt.
        self._dirty_domains: set = set()
        self._needs_training: bool = False

    # ── internal ───────────────────────────────────────────────────────────

    def _domain_samples(self, domain_name: str) -> List[str]:
        """Return the union of every intent sample registered in a domain."""
        samples: List[str] = []
        for intent_samples in self.training_data.get(domain_name, {}).values():
            samples.extend(intent_samples)
        return samples

    def _sync_domain_classifier(self) -> None:
        """Rebuild stale classifier entries, then retrain if needed.

        Registration only marks a domain dirty; the top-level classifier is
        rebuilt here, lazily, the first time a query needs it. This keeps bulk
        registration cheap instead of retraining the classifier per call.
        """
        if not self._dirty_domains:
            return
        for domain_name in self._dirty_domains:
            samples = self._domain_samples(domain_name)
            if samples:
                self.domain_engine.add_intent(domain_name, samples)
            else:
                self.domain_engine.remove_intent(domain_name)
        self._dirty_domains.clear()
        if self.domain_engine.must_train:
            self.domain_engine.train()

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        self.domain_engine.remove_intent(domain_name)
        self._dirty_domains.discard(domain_name)

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                samples: List[str]) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`MarkovIntentEngine` on first use and
        marks the top-level classifier stale.

        Args:
            domain_name: Target domain (created if it does not exist).
            intent_name: Unique intent name within the domain.
            samples: Training utterances for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = MarkovIntentEngine(**self._engine_kwargs)
        self.domains[domain_name].add_intent(intent_name, samples)
        self.training_data[domain_name][intent_name] = list(samples)
        self._dirty_domains.add(domain_name)
        self._needs_training = True

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove an intent from a domain and mark the classifier stale."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)
        self.training_data.get(domain_name, {}).pop(intent_name, None)
        self._dirty_domains.add(domain_name)
        self._needs_training = True

    # ── entity management ──────────────────────────────────────────────────

    def register_domain_entity(self, domain_name: str, entity_name: str,
                                samples: List[str]) -> None:
        """Register an entity inside a domain.

        Creates the domain's sub-engine on first use. Entities are scoped to
        the domain's sub-engine only and are not fed to the classifier.

        Args:
            domain_name: Target domain.
            entity_name: Entity name.
            samples: Sample values for the entity.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = MarkovIntentEngine(**self._engine_kwargs)
        self.domains[domain_name].add_intent(entity_name, samples)
        self.training_data[domain_name][entity_name] = list(samples)
        self._dirty_domains.add(domain_name)
        self._needs_training = True

    def remove_domain_entity(self, domain_name: str, entity_name: str) -> None:
        """Remove an entity from a domain."""
        self.remove_domain_intent(domain_name, entity_name)

    # ── training ───────────────────────────────────────────────────────────

    def train(self) -> None:
        """Train every per-domain sub-engine and the top-level classifier."""
        for sub in self.domains.values():
            if sub.must_train:
                sub.train()
        self._sync_domain_classifier()
        self._needs_training = False

    # ── query API ──────────────────────────────────────────────────────────

    def calc_domain(self, query: str) -> Optional[Tuple[str, float]]:
        """Classify *query* into the best-matching domain.

        Args:
            query: Raw utterance to classify.

        Returns:
            ``(domain_name, conf)`` for the best domain, or ``None`` if no
            domain matched.
        """
        self._sync_domain_classifier()
        scores = self.domain_engine.calc_intents(query)
        return scores[0] if scores else None

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return the best (intent_name, confidence) for *query*.

        If *domain* is ``None``, the domain is inferred by :meth:`calc_domain`.
        When the inferred domain scores below :attr:`domain_threshold`, or the
        inferred / supplied domain has no registered intents, ``None`` is
        returned. Passing *domain* explicitly bypasses the classifier and the
        threshold gate.

        Args:
            query: The utterance to match.
            domain: Domain to restrict matching to. ``None`` triggers
                automatic domain classification.

        Returns:
            ``(intent_name, conf)`` for the best intent, or ``None``.
        """
        scores = self.calc_intents(query, domain=domain)
        return scores[0] if scores else None

    def calc_intents(self, query: str,
                      domain: Optional[str] = None,
                      blacklisted_intents: Optional[set] = None,
                      blacklisted_skills: Optional[set] = None,
                      ) -> List[Tuple[str, float]]:
        """Return ranked intents for *query* within a single domain.

        The query is routed to one domain (inferred or supplied) and only that
        domain's sub-engine is scored.

        Args:
            query: The utterance to match.
            domain: If given, score only inside this domain, bypassing the
                top-level classifier and the threshold gate.
            blacklisted_intents: Intent labels to skip.
            blacklisted_skills: Skill IDs (label prefix before ``:``) to skip.

        Returns:
            Ranked ``(label, conf)`` pairs, or an empty list when no domain
            or intent could be matched.
        """
        if self._needs_training:
            self.train()

        resolved_domain: Optional[str] = domain
        if resolved_domain is None:
            self._sync_domain_classifier()
            dom_scores = self.domain_engine.calc_intents(query)
            if not dom_scores:
                return []
            best_domain, best_conf = dom_scores[0]
            if best_conf < self.domain_threshold:
                return []
            resolved_domain = best_domain

        if resolved_domain not in self.domains:
            return []
        return self.domains[resolved_domain].calc_intents(
            query,
            blacklisted_intents=blacklisted_intents,
            blacklisted_skills=blacklisted_skills,
        )

    # ── parity with MarkovIntentEngine ─────────────────────────────────────

    @property
    def must_train(self) -> bool:
        """Whether any sub-engine or the classifier has pending samples."""
        if self._needs_training:
            return True
        if self._dirty_domains:
            return True
        return any(sub.must_train for sub in self.domains.values())

    @property
    def _trained(self) -> bool:
        """True once every sub-engine and the classifier have been trained."""
        if self._needs_training or self._dirty_domains:
            return False
        if not self.domains:
            return False
        if not all(sub._trained for sub in self.domains.values()):
            return False
        return self.domain_engine._trained
