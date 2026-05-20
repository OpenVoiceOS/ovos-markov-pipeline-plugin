"""Domain-aware Markov intent engine for hierarchical intent organisation.

Mirrors the API shipped by sibling OVOS intent plugins (`nebulento`,
`ovos_padatious`, `palavreado`, `padacioso`, `linha_fina`): intents are
grouped into *domains*, a top-level :class:`MarkovIntentEngine` first
picks the domain, and the domain's sub-engine resolves the intent.

For a Markov n-gram model, the top-level domain classifier is itself
just another :class:`MarkovIntentEngine` trained with one Markov chain
per domain (seeded with the concatenated utterances of every intent in
the domain). The per-domain sub-engines then score only against the
intents within their domain — a smaller, denser context-vocabulary that
tightens perplexity-derived confidences and reduces cross-domain
collisions.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ovos_markov_pipeline import MarkovIntentEngine


class DomainMarkovIntentEngine:
    """Two-level Markov intent engine: domain classification followed by intent matching.

    Intents are grouped into *domains*. At query time the engine first
    selects the most likely domain via :attr:`domain_engine`, then runs
    the domain-specific :class:`MarkovIntentEngine` to find the best
    intent within that domain.

    Domains can also be selected explicitly, bypassing the top-level
    classifier.

    Example::

        from ovos_markov_pipeline import DomainMarkovIntentEngine
        from ovos_markov_pipeline import _Stemmer

        d = DomainMarkovIntentEngine(order=2, stemmer=_Stemmer("en-US"))

        d.register_domain_intent("media", "play",
                                  ["play {song}", "put on {song}"])
        d.register_domain_intent("home", "lights_on",
                                  ["turn on the lights", "lights on"])

        # Seed the domain classifier with representative samples per domain.
        d.domain_engine.add_intent("media", ["play music", "next track"])
        d.domain_engine.add_intent("home",  ["lights on", "thermostat"])

        d.train()
        name, conf = d.calc_intent("play africa")
        # name == "play"

    All constructor kwargs are forwarded to every internal
    :class:`MarkovIntentEngine` instance (top-level and per-domain).
    """

    def __init__(self, **engine_kwargs) -> None:
        self._engine_kwargs = dict(engine_kwargs)
        #: Top-level classifier that maps queries to a domain name.
        self.domain_engine: MarkovIntentEngine = MarkovIntentEngine(**self._engine_kwargs)
        #: Per-domain intent engines, keyed by domain name.
        self.domains: Dict[str, MarkovIntentEngine] = {}
        #: Raw training samples per (domain, intent).
        self.training_data: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self._needs_training: bool = False

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        try:
            self.domain_engine.remove_intent(domain_name)
        except Exception:
            pass

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
        """Train every internal engine.

        If the top-level :attr:`domain_engine` has not been seeded
        explicitly (no calls to ``domain_engine.add_intent`` before this
        method runs), this method seeds it automatically using the
        concatenated samples of every intent within each domain.
        """
        # Auto-seed domain engine if not seeded already.
        already_seeded = bool(getattr(self.domain_engine, "_intent_samples", None))
        if not already_seeded:
            for domain, intents in self.training_data.items():
                samples: List[str] = []
                for sents in intents.values():
                    samples.extend(sents)
                if samples:
                    self.domain_engine.add_intent(domain, samples)
        self.domain_engine.train()
        for sub in self.domains.values():
            sub.train()
        self._needs_training = False

    # ── query API ──────────────────────────────────────────────────────────

    def calc_domains(self, query: str) -> List[Tuple[str, float]]:
        """Return the top scoring domains for *query* with confidences."""
        if self._needs_training:
            self.train()
        return self.domain_engine.calc_intents(query)

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return the best (intent_name, confidence) for *query*.

        Args:
            query: The utterance to match.
            domain: If given, skip the top-level classifier and resolve
                the intent inside this domain directly.

        Returns:
            ``(intent_name, conf)`` from the chosen domain's engine, or
            ``None`` if no domain or intent matched.
        """
        if self._needs_training:
            self.train()
        resolved_domain: Optional[str] = domain
        if resolved_domain is None:
            top = self.domain_engine.calc_intents(query)
            resolved_domain = top[0][0] if top else None
        if not resolved_domain or resolved_domain not in self.domains:
            return None
        scores = self.domains[resolved_domain].calc_intents(query)
        return scores[0] if scores else None

    def calc_intents(self, query: str,
                      domain: Optional[str] = None,
                      top_k_domains: int = 1,
                      blacklisted_intents: Optional[set] = None,
                      blacklisted_skills: Optional[set] = None,
                      ) -> List[Tuple[str, float]]:
        """Return ranked intents within the resolved (or top-k) domains.

        Accepts the same ``blacklisted_intents`` / ``blacklisted_skills``
        kwargs as :meth:`MarkovIntentEngine.calc_intents` so this engine
        is drop-in compatible with the flat :class:`MarkovPipeline`
        scoring path.
        """
        if self._needs_training:
            self.train()
        sub_kwargs = dict(
            blacklisted_intents=blacklisted_intents,
            blacklisted_skills=blacklisted_skills,
        )
        if domain:
            if domain in self.domains:
                return self.domains[domain].calc_intents(query, **sub_kwargs)
            return []
        # Top-level routing ignores intent-level blacklists; per-domain
        # sub-engines apply them.
        domains = self.domain_engine.calc_intents(
            query, blacklisted_skills=blacklisted_skills,
        )[:top_k_domains]
        matches: List[Tuple[str, float]] = []
        for dom, _ in domains:
            if dom in self.domains:
                matches.extend(self.domains[dom].calc_intents(query, **sub_kwargs))
        matches.sort(key=lambda kv: kv[1], reverse=True)
        return matches

    # ── parity with MarkovIntentEngine ─────────────────────────────────────

    @property
    def must_train(self) -> bool:
        """Whether any sub-engine has pending samples to train on."""
        if self._needs_training:
            return True
        if self.domain_engine.must_train:
            return True
        return any(sub.must_train for sub in self.domains.values())

    @property
    def _trained(self) -> bool:
        """True once at least one sub-engine has been trained.

        Mirrors :attr:`MarkovIntentEngine._trained` so the flat pipeline's
        matching path treats this engine identically.
        """
        if self._needs_training:
            return False
        if not self.domains:
            return False
        return all(sub._trained for sub in self.domains.values())
