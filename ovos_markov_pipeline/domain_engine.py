"""Domain-aware Markov intent engine with two-stage routing.

Intents are grouped into *domains*. A top-level router — itself a
:class:`MarkovIntentEngine` whose "intents" are the domains — first
picks the most likely domain for an utterance; that domain's own
:class:`MarkovIntentEngine` then resolves the concrete intent.

Two-stage routing keeps every perplexity comparison within a single
shared vocabulary. A :class:`MarkovIntentEngine` builds one vocabulary
from the union of its intents' samples, so perplexity — and the
confidence derived from it — is only calibrated against intents that
share that vocabulary. The router compares domains against the global
vocabulary; the chosen sub-engine compares intents against that
domain's vocabulary. A flat global argmax across per-domain
sub-engines would instead compare confidences computed against
different-sized vocabularies, which is not a valid ranking.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ovos_markov_pipeline import MarkovIntentEngine


class DomainMarkovIntentEngine:
    """Two-stage (router → sub-engine) Markov intent engine.

    Intents are grouped into *domains*, each owning its own
    :class:`MarkovIntentEngine`. A separate router engine, trained on
    the concatenated samples of every intent in a domain, selects the
    domain; the selected domain's sub-engine then resolves the intent.

    Domains can be selected explicitly via the ``domain`` kwarg on
    :meth:`calc_intent` / :meth:`calc_intents`, which bypasses the
    router.

    Example::

        from ovos_markov_pipeline import DomainMarkovIntentEngine
        from ovos_markov_pipeline import _Stemmer

        d = DomainMarkovIntentEngine(order=2, stemmer=_Stemmer("en-US"))

        d.register_domain_intent("media", "play",
                                  ["play {song}", "put on {song}"])
        d.register_domain_intent("home", "lights_on",
                                  ["turn on the lights", "lights on"])

        d.train()
        name, conf = d.calc_intent("play africa")
        # name == "play"

    All constructor kwargs are forwarded to every internal
    :class:`MarkovIntentEngine` instance (router and sub-engines).
    """

    def __init__(self, **engine_kwargs) -> None:
        self._engine_kwargs = dict(engine_kwargs)
        #: Per-domain intent engines, keyed by domain name.
        self.domains: Dict[str, MarkovIntentEngine] = {}
        #: Top-level router; its "intents" are domain names.
        self.domain_engine: MarkovIntentEngine = MarkovIntentEngine(
            **self._engine_kwargs)
        #: Raw training samples per (domain, intent).
        self.training_data: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self._needs_training: bool = False

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        self._needs_training = True

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                samples: List[str]) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`MarkovIntentEngine` on first use.

        Args:
            domain_name: Target domain (created if it does not exist).
            intent_name: Unique intent name within the domain.
            samples: Training utterances for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = MarkovIntentEngine(**self._engine_kwargs)
        self.domains[domain_name].add_intent(intent_name, samples)
        self.training_data[domain_name][intent_name] = list(samples)
        self._needs_training = True

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove an intent from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)
        self.training_data.get(domain_name, {}).pop(intent_name, None)
        self._needs_training = True

    # ── training ───────────────────────────────────────────────────────────

    def train(self) -> None:
        """Re-seed the router and train every engine.

        The router is rebuilt from scratch so domains removed since the
        last train drop out of it. Each domain becomes one router
        "intent" whose samples are the concatenation of every intent in
        that domain.
        """
        self.domain_engine = MarkovIntentEngine(**self._engine_kwargs)
        for domain_name, intents in self.training_data.items():
            domain_samples: List[str] = []
            for samples in intents.values():
                domain_samples.extend(samples)
            if domain_samples:
                self.domain_engine.add_intent(domain_name, domain_samples)
        self.domain_engine.train()

        for sub in self.domains.values():
            if sub.must_train:
                sub.train()
        self._needs_training = False

    # ── query API ──────────────────────────────────────────────────────────

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return the best (intent_name, confidence) for *query*.

        Args:
            query: The utterance to match.
            domain: If given, skip the router and resolve directly
                inside this domain.

        Returns:
            ``(intent_name, conf)`` for the routed domain's best intent,
            or ``None`` if nothing matched.
        """
        scores = self.calc_intents(query, domain=domain)
        return scores[0] if scores else None

    def calc_intents(self, query: str,
                      domain: Optional[str] = None,
                      blacklisted_intents: Optional[set] = None,
                      blacklisted_skills: Optional[set] = None,
                      ) -> List[Tuple[str, float]]:
        """Return ranked intents for *query* via two-stage routing.

        Stage 1 routes the utterance to a domain; stage 2 resolves the
        intent inside that domain. The returned confidences all come
        from a single sub-engine, so they share one vocabulary and rank
        consistently.

        Args:
            query: The utterance to match.
            domain: If given, skip the router and score only inside
                this domain.
            blacklisted_intents: Intent labels to skip (stage 2).
            blacklisted_skills: Skill IDs to skip. Domain names are the
                router's intent labels, so this also prunes the routing
                stage.
        """
        if self._needs_training:
            self.train()

        if domain is not None:
            if domain not in self.domains:
                return []
            return self.domains[domain].calc_intents(
                query,
                blacklisted_intents=blacklisted_intents,
                blacklisted_skills=blacklisted_skills,
            )

        # Stage 1: route to a domain. Domain names are the router's
        # intent labels, so a skill blacklist prunes routing too.
        router_scores = self.domain_engine.calc_intents(
            query, blacklisted_intents=blacklisted_skills)
        if not router_scores:
            return []
        best_domain = router_scores[0][0]
        if best_domain not in self.domains:
            return []

        # Stage 2: resolve the intent inside the routed domain.
        return self.domains[best_domain].calc_intents(
            query,
            blacklisted_intents=blacklisted_intents,
            blacklisted_skills=blacklisted_skills,
        )

    def update_online(self, intent_name: str, utterance: str) -> None:
        """Reinforce a matched intent with a new sample.

        Retrains the intent's domain sub-engine immediately and marks
        the router stale so it picks up the new sample on next train.
        """
        domain = intent_name.split(":", 1)[0] if ":" in intent_name else intent_name
        sub = self.domains.get(domain)
        if sub is None:
            return
        sub.update_online(intent_name, utterance)
        if intent_name in self.training_data.get(domain, {}):
            self.training_data[domain][intent_name].append(utterance.strip())
            self._needs_training = True

    # ── parity with MarkovIntentEngine ─────────────────────────────────────

    @property
    def must_train(self) -> bool:
        """Whether the router or any sub-engine has pending samples."""
        if self._needs_training:
            return True
        if self.domain_engine.must_train:
            return True
        return any(sub.must_train for sub in self.domains.values())

    @property
    def _trained(self) -> bool:
        """True once the router and every sub-engine have been trained."""
        if self._needs_training:
            return False
        if not self.domains:
            return False
        if not self.domain_engine._trained:
            return False
        return all(sub._trained for sub in self.domains.values())
