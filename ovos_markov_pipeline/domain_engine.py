"""Domain-aware Markov intent engine with parallel argmax scoring.

Intents are grouped into *domains*, but there is no top-level domain
classifier. Each Markov chain already produces a perplexity-derived
per-intent confidence; at query time the engine scores every intent
across every domain and returns the global argmax. This mirrors the
parallel-argmax pattern used by adapt and the other OVOS intent
plugins.

For Markov chains, per-intent scoring is cheap, so the only real
optimisation is to skip a sub-engine whose vocabulary has zero overlap
with the utterance tokens (a quick in-memory set check that prunes the
vast majority of domains in large deployments).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ovos_markov_pipeline import MarkovIntentEngine, _normalize


class DomainMarkovIntentEngine:
    """Parallel-argmax Markov intent engine.

    Intents are grouped into *domains*. Each domain owns its own
    :class:`MarkovIntentEngine`. There is no top-level router: at query
    time every domain scores the utterance and the global highest-
    confidence intent wins.

    Domains can still be selected explicitly via the ``domain`` kwarg
    on :meth:`calc_intent` / :meth:`calc_intents`, in which case only
    that domain's intents are scored.

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
    :class:`MarkovIntentEngine` instance.
    """

    def __init__(self, **engine_kwargs) -> None:
        self._engine_kwargs = dict(engine_kwargs)
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
        """Train every per-domain sub-engine."""
        for sub in self.domains.values():
            if sub.must_train:
                sub.train()
        self._needs_training = False

    # ── optimisation helpers ───────────────────────────────────────────────

    def _domain_vocab(self, domain_name: str) -> Set[str]:
        """Return the set of word tokens seen in a domain's training data.

        Used as a cheap pre-filter: if the utterance shares no tokens
        with a domain's vocabulary, scoring its sub-engine cannot
        possibly yield a non-trivial confidence.
        """
        stemmer = self._engine_kwargs.get("stemmer")
        vocab: Set[str] = set()
        for samples in self.training_data.get(domain_name, {}).values():
            for s in samples:
                norm = _normalize(s, stemmer)
                vocab.update(norm.split())
        return vocab

    def _candidate_domains(self, query: str) -> List[str]:
        """Pre-filter domains worth scoring for *query*.

        Drops domains whose vocabulary has no overlap with the
        utterance tokens.
        """
        stemmer = self._engine_kwargs.get("stemmer")
        norm = _normalize(query, stemmer)
        utt_tokens = set(norm.split())

        candidates: List[str] = []
        for dom in self.domains:
            vocab = self._domain_vocab(dom)
            if vocab and utt_tokens.isdisjoint(vocab):
                continue
            candidates.append(dom)
        return candidates

    # ── query API ──────────────────────────────────────────────────────────

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return the best (intent_name, confidence) for *query*.

        Args:
            query: The utterance to match.
            domain: If given, restrict scoring to this domain.

        Returns:
            ``(intent_name, conf)`` for the global argmax across all
            scored domains, or ``None`` if nothing matched.
        """
        scores = self.calc_intents(query, domain=domain)
        return scores[0] if scores else None

    def calc_intents(self, query: str,
                      domain: Optional[str] = None,
                      blacklisted_intents: Optional[set] = None,
                      blacklisted_skills: Optional[set] = None,
                      ) -> List[Tuple[str, float]]:
        """Return ranked intents across every (candidate) domain.

        Each sub-engine scores the utterance independently; results are
        flattened and sorted by confidence descending. Equivalent to
        the parallel-argmax pattern used by adapt.

        Args:
            query: The utterance to match.
            domain: If given, score only inside this domain.
            blacklisted_intents: Intent labels to skip.
            blacklisted_skills: Skill IDs (label prefix before ``:``)
                to skip.
        """
        if self._needs_training:
            self.train()

        sub_kwargs = dict(
            blacklisted_intents=blacklisted_intents,
            blacklisted_skills=blacklisted_skills,
        )

        if domain is not None:
            if domain not in self.domains:
                return []
            return self.domains[domain].calc_intents(query, **sub_kwargs)

        candidates = self._candidate_domains(query)
        matches: List[Tuple[str, float]] = []
        for dom in candidates:
            matches.extend(self.domains[dom].calc_intents(query, **sub_kwargs))
        matches.sort(key=lambda kv: kv[1], reverse=True)
        return matches

    # ── parity with MarkovIntentEngine ─────────────────────────────────────

    @property
    def must_train(self) -> bool:
        """Whether any sub-engine has pending samples to train on."""
        if self._needs_training:
            return True
        return any(sub.must_train for sub in self.domains.values())

    @property
    def _trained(self) -> bool:
        """True once every sub-engine has been trained."""
        if self._needs_training:
            return False
        if not self.domains:
            return False
        return all(sub._trained for sub in self.domains.values())
