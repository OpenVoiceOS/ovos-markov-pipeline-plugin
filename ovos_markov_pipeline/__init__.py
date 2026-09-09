"""OVOS intent pipeline plugin using Markov chain perplexity ensemble.

Trains one word-level Markov chain per intent from example utterances.
Classifies by computing perplexity under each model — the intent whose
model assigns the lowest perplexity (highest likelihood) wins.

Confidence is a softmax posterior over the per-intent perplexities.
"""

import math
import re
import string
from collections import defaultdict
from threading import Event, RLock
from typing import Dict, List, Optional, Tuple, Union

from markovonnx import MarkovChain, Vocabulary, char_tokenize, word_tokenize
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import (
    ConfidenceMatcherPipeline,
    IntentHandlerMatch,
)
from ovos_spec_tools import SpecMessage
from ovos_utils.fakebus import FakeBus

from ovos_markov_pipeline._bracket_expansion import expand_template
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG

from ovos_markov_pipeline.version import __version__

__all__ = ["MarkovPipeline", "MarkovIntentEngine", "__version__"]


# ---------------------------------------------------------------------------
# Stemmer (adapted from ovos-padatious-pipeline-plugin)
# ---------------------------------------------------------------------------


class _Stemmer:
    """Snowball stemmer wrapper. Fails gracefully if unsupported."""

    _LANGS = {
        "ar": "arabic",
        "eu": "basque",
        "ca": "catalan",
        "da": "danish",
        "nl": "dutch",
        "en": "english",
        "fi": "finnish",
        "fr": "french",
        "de": "german",
        "el": "greek",
        "hi": "hindi",
        "hu": "hungarian",
        "id": "indonesian",
        "ga": "irish",
        "it": "italian",
        "lt": "lithuanian",
        "ne": "nepali",
        "no": "norwegian",
        "pt": "portuguese",
        "ro": "romanian",
        "ru": "russian",
        "sr": "serbian",
        "es": "spanish",
        "sv": "swedish",
        "ta": "tamil",
        "tr": "turkish",
    }

    def __init__(self, lang: str):
        import snowballstemmer

        lang2 = lang.split("-")[0].lower()
        if lang2 not in self._LANGS:
            raise ValueError(f"Unsupported stemmer language: {lang}")
        self._stemmer = snowballstemmer.stemmer(self._LANGS[lang2])

    @classmethod
    def supports(cls, lang: str) -> bool:
        """Check if stemming is available for *lang*."""
        return lang.split("-")[0].lower() in cls._LANGS

    def stem(self, sentence: str) -> str:
        """Stem all words in a sentence."""
        return " ".join(self._stemmer.stemWords(sentence.split()))


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(text: str, stemmer: Optional[_Stemmer] = None) -> str:
    """Lowercase, collapse whitespace, strip punctuation, optionally stem."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(string.punctuation)
    if stemmer is not None:
        text = stemmer.stem(text)
    return text


#: Softmax temperature for the perplexity-to-posterior mapping. Higher
#: values sharpen the distribution toward a decisive winner.
_SOFTMAX_TEMPERATURE = 3.0


def _posterior(scored: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Turn per-intent perplexities into a softmax posterior.

    Each perplexity becomes a log-likelihood (``-log ppx``); a
    temperature-scaled softmax over every intent yields a 0-1 confidence
    that reflects how far the best intent outscored the rest. Returned
    sorted by confidence descending.
    """
    if not scored:
        return []
    lls = [(name, -math.log(max(ppx, 1e-9))) for name, ppx in scored]
    top = max(ll for _, ll in lls)
    exps = [(name, math.exp(_SOFTMAX_TEMPERATURE * (ll - top))) for name, ll in lls]
    z = sum(e for _, e in exps) or 1.0
    out = [(name, e / z) for name, e in exps]
    out.sort(key=lambda x: -x[1])
    return out


# ---------------------------------------------------------------------------
# MarkovIntentEngine
# ---------------------------------------------------------------------------


class MarkovIntentEngine:
    """Per-language intent matching engine using Markov chain perplexity.

    Maintains a shared vocabulary and one MarkovChain per registered intent.
    Optionally trains a character-level fallback ensemble for when word-level
    scores are too close to discriminate.

    Args:
        order: N-gram order for Markov chains.
        smoothing: Laplace smoothing alpha.
        kneser_ney: Use Kneser-Ney smoothing instead of Laplace.
        backoff: Enable interpolated backoff to lower-order models.
        stemmer: Optional stemmer for normalizing samples and queries.
        char_fallback: Train character-level models as fallback.
    """

    def __init__(
        self,
        order: int = 2,
        smoothing: float = 1e-5,
        kneser_ney: bool = True,
        backoff: bool = True,
        stemmer: Optional[_Stemmer] = None,
        char_fallback: bool = False,
        char_fallback_threshold: float = 0.05,
    ):
        self.order = order
        self.smoothing = smoothing
        self.kneser_ney = kneser_ney
        self.backoff = backoff
        self.stemmer = stemmer
        self.char_fallback = char_fallback
        self.char_fallback_threshold = char_fallback_threshold

        self._intent_samples: Dict[str, List[str]] = {}  # raw strings
        self._models: Dict[str, MarkovChain] = {}
        self._char_models: Dict[str, MarkovChain] = {}
        self._vocab: Optional[Vocabulary] = None
        self._char_vocab: Optional[Vocabulary] = None
        self._trained = False

    @property
    def must_train(self) -> bool:
        """Whether new samples have been added since last train."""
        return not self._trained and len(self._intent_samples) > 0

    def add_intent(self, name: str, samples: List[str]) -> None:
        """Register an intent with training samples.

        Samples may use OVOS template syntax:
            ``(a|b)``  alternatives
            ``[opt]``  optional components
            ``{slot}`` slot placeholders (kept verbatim as features)

        Each template is expanded via the local
        :func:`ovos_markov_pipeline._bracket_expansion.expand_template`
        helper, and the resulting concrete utterances are used as training
        data.
        """
        expanded: set = set()
        for s in samples:
            if not s or not s.strip():
                continue
            try:
                for variant in expand_template(s):
                    v = variant.strip()
                    if v:
                        expanded.add(v)
            except Exception:
                # Fall back to the raw sample if expansion fails for any reason
                v = s.strip()
                if v:
                    expanded.add(v)
        self._intent_samples[name] = list(expanded)
        self._trained = False

    def remove_intent(self, name: str) -> None:
        """Remove a registered intent."""
        self._intent_samples.pop(name, None)
        self._models.pop(name, None)
        self._char_models.pop(name, None)
        self._trained = False

    def _tokenize_word(self, samples: List[str]) -> List[List[str]]:
        """Normalize and word-tokenize samples."""
        return [
            word_tokenize(_normalize(s, self.stemmer))
            for s in samples
            if _normalize(s, self.stemmer)
        ]

    def _tokenize_char(self, samples: List[str]) -> List[List[str]]:
        """Normalize and char-tokenize samples."""
        return [
            char_tokenize(_normalize(s, self.stemmer))
            for s in samples
            if _normalize(s, self.stemmer)
        ]

    def train(self) -> None:
        """Train all intent models on current samples."""
        if not self._intent_samples:
            self._trained = True
            return

        # Word-level models
        all_word_seqs: List[List[str]] = []
        intent_word_seqs: Dict[str, List[List[str]]] = {}
        for name, raw in self._intent_samples.items():
            seqs = self._tokenize_word(raw)
            intent_word_seqs[name] = seqs
            all_word_seqs.extend(seqs)

        self._vocab = Vocabulary()
        self._vocab.build_from_sequences(all_word_seqs)

        self._models = {}
        for name, seqs in intent_word_seqs.items():
            if not seqs:
                continue
            mc = MarkovChain(
                order=self.order,
                vocab=self._vocab,
                smoothing=self.smoothing,
                backoff=self.backoff,
                kneser_ney=self.kneser_ney,
            )
            mc.fit(seqs)
            self._models[name] = mc

        # Character-level fallback models
        if self.char_fallback:
            all_char_seqs: List[List[str]] = []
            intent_char_seqs: Dict[str, List[List[str]]] = {}
            for name, raw in self._intent_samples.items():
                seqs = self._tokenize_char(raw)
                intent_char_seqs[name] = seqs
                all_char_seqs.extend(seqs)

            self._char_vocab = Vocabulary()
            self._char_vocab.build_from_sequences(all_char_seqs)

            self._char_models = {}
            for name, seqs in intent_char_seqs.items():
                if not seqs:
                    continue
                mc = MarkovChain(
                    order=3,
                    vocab=self._char_vocab,
                    smoothing=self.smoothing,
                    backoff=True,
                    kneser_ney=self.kneser_ney,
                )
                mc.fit(seqs)
                self._char_models[name] = mc

        self._trained = True

    def calc_intents(
        self,
        utterance: str,
        blacklisted_intents: Optional[set] = None,
        blacklisted_skills: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """Score all intents for an utterance.

        Returns list of ``(intent_name, confidence)`` sorted descending.
        """
        if not self._models or self._vocab is None:
            return []

        blacklisted_intents = blacklisted_intents or set()
        blacklisted_skills = blacklisted_skills or set()

        norm = _normalize(utterance, self.stemmer)
        word_tokens = word_tokenize(norm)
        if len(word_tokens) < self.order:
            return []

        raw: List[Tuple[str, float]] = []
        for name, mc in self._models.items():
            if name in blacklisted_intents:
                continue
            skill_id = name.split(":")[0] if ":" in name else name
            if skill_id in blacklisted_skills:
                continue
            raw.append((name, mc.perplexity([word_tokens])))

        scores = _posterior(raw)

        # Character-level fallback: if top-2 word scores are too close
        if (
            self.char_fallback
            and self._char_models
            and len(scores) >= 2
            and scores[0][1] - scores[1][1] < self.char_fallback_threshold
        ):
            char_tokens = char_tokenize(norm)
            if len(char_tokens) >= 3:
                char_raw: List[Tuple[str, float]] = []
                for name, mc in self._char_models.items():
                    if name in blacklisted_intents:
                        continue
                    skill_id = name.split(":")[0] if ":" in name else name
                    if skill_id in blacklisted_skills:
                        continue
                    char_raw.append((name, mc.perplexity([char_tokens])))
                char_scores: Dict[str, float] = dict(_posterior(char_raw))

                # Blend: 60% word + 40% char (use word-only if no char model)
                blended: List[Tuple[str, float]] = []
                for name, word_conf in scores:
                    if name in char_scores:
                        blended.append((name, 0.6 * word_conf + 0.4 * char_scores[name]))
                    else:
                        blended.append((name, word_conf))
                blended.sort(key=lambda x: -x[1])
                return blended

        return scores

    def update_online(self, intent_name: str, utterance: str) -> None:
        """Add a new utterance to an intent and retrain.

        Adds the utterance to the intent's samples, rebuilds the shared
        vocabulary, and retrains all models to keep count arrays consistent.

        Args:
            intent_name: The intent to update.
            utterance: New example utterance.
        """
        if intent_name not in self._intent_samples:
            return
        self._intent_samples[intent_name].append(utterance.strip())
        if self._vocab is None:
            return
        # Full retrain to keep vocab and count arrays consistent
        self._trained = False
        self.train()


# ---------------------------------------------------------------------------
# MarkovPipeline (OPM ConfidenceMatcherPipeline)
# ---------------------------------------------------------------------------


class MarkovPipeline(ConfidenceMatcherPipeline):
    """OVOS pipeline plugin for Markov chain perplexity-based intent matching.

    Configuration (in ``mycroft.conf``):

    .. code-block:: json

        {
            "intents": {
                "ovos-markov-pipeline-plugin": {
                    "order": 2,
                    "kneser_ney": true,
                    "backoff": true,
                    "stem": false,
                    "char_fallback": false,
                    "online_learning": false,
                    "conf_high": 0.75,
                    "conf_med": 0.55,
                    "conf_low": 0.30
                }
            }
        }
    """

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ):
        intent_config = Configuration().get("intents", {})
        config = config or intent_config.get("ovos-markov-pipeline-plugin") or {}
        super().__init__(bus, config)

        self.lock = RLock()
        core_config = Configuration()
        self.lang = standardize_lang_tag(core_config.get("lang", "en-US"))
        langs = core_config.get("secondary_langs") or []
        langs = [standardize_lang_tag(lng) for lng in langs]
        if self.lang not in langs:
            langs.append(self.lang)

        self.conf_high = self.config.get("conf_high", 0.75)
        self.conf_med = self.config.get("conf_med", 0.55)
        self.conf_low = self.config.get("conf_low", 0.30)

        # Validate threshold ordering
        if not (self.conf_low <= self.conf_med <= self.conf_high):
            LOG.warning(
                f"Confidence thresholds not ordered: "
                f"low={self.conf_low} med={self.conf_med} high={self.conf_high}"
            )

        order = max(1, int(self.config.get("order", 2)))
        kneser_ney = self.config.get("kneser_ney", True)
        backoff = self.config.get("backoff", True)
        smoothing = self.config.get("smoothing", 1e-5)
        use_stemmer = self.config.get("stem", False)
        char_fallback = self.config.get("char_fallback", False)
        char_fallback_threshold = self.config.get("char_fallback_threshold", 0.05)
        self.online_learning = self.config.get("online_learning", False)

        # Build per-language stemmers
        self.stemmers: Dict[str, _Stemmer] = {}
        if use_stemmer:
            for lang in langs:
                if _Stemmer.supports(lang):
                    try:
                        self.stemmers[lang] = _Stemmer(lang)
                    except Exception:
                        pass

        # Stash engine kwargs so subclasses can re-use the same per-lang
        # construction recipe via :meth:`_build_engines`.
        self._engine_kwargs_template = {
            "order": order,
            "smoothing": smoothing,
            "kneser_ney": kneser_ney,
            "backoff": backoff,
            "char_fallback": char_fallback,
            "char_fallback_threshold": char_fallback_threshold,
        }
        self._langs: List[str] = list(langs)

        self.engines: Dict[str, MarkovIntentEngine] = self._build_engines()

        self.first_train = Event()
        self.finished_training_event = Event()
        self.finished_training_event.set()

        self.registered_intents: List[str] = []
        self._skill2intent: Dict[str, List[str]] = defaultdict(list)
        # INTENT-4 §8.5 — intents disabled without losing their definition;
        # excluded from match candidacy until re-enabled.
        self.disabled_intents: set = set()
        self.max_words = self.config.get("max_words", 50)

        # legacy (padatious-compatible) registration surface
        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.skills.train", self.train)
        self.bus.on(
            "intent.service.markov.manifest.get",
            self.handle_manifest,
        )

        # OVOS-INTENT-4 registration surface (alongside the legacy one).
        # Markov is a sample/template matcher, so it consumes the template
        # registration topic (§6) but NOT the keyword topic (§5/§11). It has
        # no entity concept, so the entity topics are intentionally not wired.
        self.bus.on(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                    self.handle_register_template)
        self.bus.on(SpecMessage.INTENT_DEREGISTER.value,
                    self.handle_deregister_intent_spec)
        self.bus.on(SpecMessage.SKILL_DEREGISTER.value,
                    self.handle_deregister_skill_spec)
        self.bus.on(SpecMessage.INTENT_ENABLE.value,
                    self.handle_enable_intent_spec)
        self.bus.on(SpecMessage.INTENT_DISABLE.value,
                    self.handle_disable_intent_spec)

        LOG.info(
            f"Loaded MarkovPipeline (order={order}, kn={kneser_ney}, "
            f"backoff={backoff}, stem={use_stemmer}, char_fb={char_fallback})"
        )

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        """Find the closest registered language."""
        lang = standardize_lang_tag(lang)
        if lang in self.engines:
            return lang
        prefix = lang.split("-")[0]
        for registered in self.engines:
            if registered.startswith(prefix):
                return registered
        return None

    # -- Intent registration --------------------------------------------------

    def register_intent(self, message: Message) -> None:
        """Handle ``padatious:register_intent`` bus message."""
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID missing, using 'anonymous_skill'")
            skill_id = "anonymous_skill"

        name = message.data.get("name")
        if not name:
            LOG.error("Intent registration missing 'name' field")
            return

        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        samples = message.data.get("samples")

        # Validate samples type
        if samples is not None and not isinstance(samples, list):
            LOG.warning(f"Intent {name}: samples is {type(samples).__name__}, expected list")
            samples = list(samples) if hasattr(samples, "__iter__") else None

        file_name = message.data.get("file_name")
        if not samples and file_name:
            try:
                with open(file_name) as f:
                    samples = [line.strip() for line in f.readlines()]
            except (OSError, IOError) as e:
                LOG.error(f"Failed to read intent file {file_name}: {e}")

        if not samples:
            LOG.error(f"No samples for intent {name}")
            return

        self._skill2intent[skill_id].append(name)
        self.registered_intents.append(name)

        closest = self._get_closest_lang(lang)
        if closest and closest in self.engines:
            LOG.debug(f"Registering markov intent: {name} ({len(samples)} samples)")
            self._add_intent(self.engines[closest], name, samples)

        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def handle_detach_intent(self, message: Message) -> None:
        """Remove a single intent."""
        intent_name = message.data.get("intent_name")
        if intent_name and intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            self.disabled_intents.discard(intent_name)
            for engine in self.engines.values():
                self._remove_intent(engine, intent_name)

    def handle_detach_skill(self, message: Message) -> None:
        """Remove all intents for a skill (legacy ``detach_skill``).

        The legacy wire carries no required payload identity, so the context
        is the only identity available when the payload omits one.
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            return
        self._forget_skill(skill_id)

    def _forget_skill(self, skill_id: str) -> None:
        """Drop every intent registered by *skill_id*."""
        intent_names = self._skill2intent.pop(skill_id, [])
        for intent_name in intent_names:
            if intent_name in self.registered_intents:
                self.registered_intents.remove(intent_name)
            self.disabled_intents.discard(intent_name)
        for engine in self.engines.values():
            self._remove_skill(engine, skill_id, intent_names)

    # -- OVOS-INTENT-4 registration surface -----------------------------------

    @staticmethod
    def _spec_label(message: Message, key: str) -> Optional[str]:
        """Build the internal ``skill_id:<key>`` label from an INTENT-4 payload.

        INTENT-4 carries ``skill_id`` and ``intent_name`` as separate fields
        (§3.2); markov keys everything on the combined ``skill_id:name`` label,
        matching the legacy padatious convention.

        §3.2: a message of §§5-8 acts on its payload ``skill_id``, and
        ``context.skill_id`` names the source, so it is never substituted
        for the target.
        """
        skill_id = message.data.get("skill_id")
        name = message.data.get(key)
        if not skill_id or not name:
            LOG.warning(f"Ignoring malformed INTENT-4 payload on {message.msg_type!r}: "
                        f"missing skill_id/{key}")
            return None
        return f"{skill_id}:{name}"

    def handle_register_template(self, message: Message) -> None:
        """Consume ``ovos.intent.register.template`` (INTENT-4 §6).

        Template intents are markov's native definition method. The payload
        carries inline ``samples`` (OVOS-INTENT-1 templates); ``blacklist`` is a
        suppression hint markov does not yet honour and is ignored.
        """
        skill_id = message.data.get("skill_id")
        name = self._spec_label(message, "intent_name")
        if name is None:
            return
        samples = message.data.get("samples")
        if not samples:
            LOG.warning(f"Ignoring INTENT-4 template registration for {name!r}: "
                        f"empty samples")
            return

        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        self._skill2intent[skill_id].append(name)
        self.registered_intents.append(name)

        closest = self._get_closest_lang(lang)
        if closest and closest in self.engines:
            LOG.debug(f"Registering markov intent (spec): {name} "
                      f"({len(samples)} samples)")
            self._add_intent(self.engines[closest], name, samples)

        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def handle_deregister_intent_spec(self, message: Message) -> None:
        """Consume ``ovos.intent.deregister`` (INTENT-4 §8.2)."""
        name = self._spec_label(message, "intent_name")
        if name is None:
            return
        if name in self.registered_intents:
            self.registered_intents.remove(name)
            for engine in self.engines.values():
                self._remove_intent(engine, name)
        self.disabled_intents.discard(name)

    def handle_deregister_skill_spec(self, message: Message) -> None:
        """Consume ``ovos.skill.deregister`` (INTENT-4 §8.4).

        §3.2: the payload names the skill to remove and ``context.skill_id``
        names the source, so a payload without an identity has no target and
        is rejected rather than being applied to whoever sent it.
        """
        skill_id = message.data.get("skill_id")
        if not skill_id:
            LOG.warning(f"Ignoring malformed INTENT-4 payload on "
                        f"{message.msg_type!r}: missing skill_id")
            return
        self._forget_skill(skill_id)

    def handle_enable_intent_spec(self, message: Message) -> None:
        """Consume ``ovos.intent.enable`` (INTENT-4 §8.5)."""
        name = self._spec_label(message, "intent_name")
        if name is not None:
            self.disabled_intents.discard(name)

    def handle_disable_intent_spec(self, message: Message) -> None:
        """Consume ``ovos.intent.disable`` (INTENT-4 §8.5)."""
        name = self._spec_label(message, "intent_name")
        if name is not None:
            self.disabled_intents.add(name)

    # ------------------------------------------------------------------
    # Engine-shape hooks — overridden by DomainMarkovPipeline
    # ------------------------------------------------------------------

    def _build_engines(self) -> Dict[str, "MarkovIntentEngine"]:
        """Construct one :class:`MarkovIntentEngine` per registered language."""
        return {
            lang: MarkovIntentEngine(
                stemmer=self.stemmers.get(lang),
                **self._engine_kwargs_template,
            )
            for lang in self._langs
        }

    def _add_intent(self, engine: "MarkovIntentEngine",
                    name: str, samples: List[str]) -> None:
        """Register *name* with *samples* in *engine*."""
        engine.add_intent(name, samples)

    def _remove_intent(self, engine: "MarkovIntentEngine", name: str) -> None:
        """Remove a single intent from *engine*."""
        engine.remove_intent(name)

    def _remove_skill(self, engine: "MarkovIntentEngine",
                      skill_id: str, intent_names: List[str]) -> None:
        """Remove all intents owned by *skill_id* from *engine*."""
        for name in intent_names:
            engine.remove_intent(name)

    # -- Training -------------------------------------------------------------

    def train(self, message: Optional[Message] = None) -> None:
        """Train all per-language engines."""
        if not self.finished_training_event.is_set():
            self.finished_training_event.wait()

        with self.lock:
            if not any(e.must_train for e in self.engines.values()):
                self.bus.emit(Message("mycroft.skills.trained"))
                self.finished_training_event.set()
                return

            self.finished_training_event.clear()
            for lang, engine in self.engines.items():
                if engine.must_train:
                    LOG.debug(f"Training markov engine for {lang}")
                    engine.train()

            self.bus.emit(Message("mycroft.skills.trained"))
            self.finished_training_event.set()

        if not self.first_train.is_set():
            self.first_train.set()

    # -- Matching -------------------------------------------------------------

    def _match_level(
        self,
        utterances: List[str],
        limit: float,
        lang: str,
        message: Message,
    ) -> Optional[IntentHandlerMatch]:
        """Score utterances and return match if confidence exceeds *limit*."""
        lang = self._get_closest_lang(lang or self.lang)
        if lang is None or lang not in self.engines:
            return None

        engine = self.engines[lang]
        if not engine._trained:
            return None

        sess = SessionManager.get(message)

        utterances = [u for u in utterances if len(u.split()) <= self.max_words]
        if not utterances:
            return None

        best_intent: Optional[str] = None
        best_conf: float = 0.0

        for utt in utterances:
            scores = engine.calc_intents(
                utt,
                blacklisted_intents=sess.blacklisted_intents,
                blacklisted_skills=sess.blacklisted_skills,
            )
            # INTENT-4 §8.5 — disabled intents are excluded from candidacy
            scores = [s for s in scores if s[0] not in self.disabled_intents]
            if scores and scores[0][1] > best_conf:
                best_intent, best_conf = scores[0]

        if best_intent is not None and best_conf > limit:
            skill_id = best_intent.split(":")[0] if ":" in best_intent else best_intent

            # Online learning: reinforce successful matches (thread-safe)
            if self.online_learning and best_conf > self.conf_high:
                with self.lock:
                    engine.update_online(best_intent, utterances[0])

            return IntentHandlerMatch(
                match_type=best_intent,
                match_data={
                    "utterance": utterances[0],
                    "confidence": best_conf,
                },
                skill_id=skill_id,
                utterance=utterances[0],
            )
        return None

    def match_high(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        """Match with high confidence threshold."""
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        """Match with medium confidence threshold."""
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        """Match with low confidence threshold."""
        return self._match_level(utterances, self.conf_low, lang, message)

    # -- Bus queries ----------------------------------------------------------

    def handle_manifest(self, message: Message) -> None:
        """Reply with registered intent names."""
        self.bus.emit(
            message.reply(
                "intent.service.markov.manifest",
                {"intents": self.registered_intents},
            )
        )

    def shutdown(self) -> None:
        """Clean up bus handlers."""
        self.bus.remove("padatious:register_intent", self.register_intent)
        self.bus.remove("detach_intent", self.handle_detach_intent)
        self.bus.remove("detach_skill", self.handle_detach_skill)
        self.bus.remove("mycroft.skills.train", self.train)
        self.bus.remove(
            "intent.service.markov.manifest.get",
            self.handle_manifest,
        )
        self.bus.remove(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                        self.handle_register_template)
        self.bus.remove(SpecMessage.INTENT_DEREGISTER.value,
                        self.handle_deregister_intent_spec)
        self.bus.remove(SpecMessage.SKILL_DEREGISTER.value,
                        self.handle_deregister_skill_spec)
        self.bus.remove(SpecMessage.INTENT_ENABLE.value,
                        self.handle_enable_intent_spec)
        self.bus.remove(SpecMessage.INTENT_DISABLE.value,
                        self.handle_disable_intent_spec)


# Re-export DomainMarkovIntentEngine at the package root for parity with
# the other OVOS intent plugins (nebulento, ovos-padatious, palavreado,
# padacioso, linha_fina).
from ovos_markov_pipeline.domain_engine import DomainMarkovIntentEngine  # noqa: E402, F401


class DomainMarkovPipeline(MarkovPipeline):
    """Domain-grouped Markov pipeline with two-stage routing.

    Same behaviour as :class:`MarkovPipeline` except the per-language
    engine is a :class:`DomainMarkovIntentEngine`. Each Padatious intent
    is grouped under a domain == ``skill_id`` (taken from the intent
    label's ``<skill_id>:<intent>`` prefix); inference first routes the
    utterance to a domain, then resolves the intent inside it.

    Configuration is read from
    ``intents.ovos-markov-domain-pipeline-plugin`` so this pipeline can
    coexist with the flat plugin in the same OVOS instance. Accepts
    every key the flat plugin does.

    Example ``mycroft.conf``::

        "intents": {
            "ovos-markov-domain-pipeline-plugin": {
                "order": 2,
                "kneser_ney": true,
                "backoff": true,
                "conf_high": 0.50,
                "conf_med": 0.30,
                "conf_low": 0.15,
                "instant_train": true
            }
        }
    """

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if config is None:
            intent_config = Configuration().get("intents", {})
            config = (
                intent_config.get("ovos-markov-domain-pipeline-plugin")
                or intent_config.get("ovos_markov_domain_pipeline_plugin")
                or {}
            )
        super().__init__(bus, config)

    # ------------------------------------------------------------------
    # Hook overrides — swap engine shape and route adds/removes by domain
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_of(name: str) -> str:
        """Extract the domain (skill_id) from a ``skill_id:intent`` label."""
        return name.split(":", 1)[0] if ":" in name else name

    def _build_engines(self) -> Dict[str, DomainMarkovIntentEngine]:
        return {
            lang: DomainMarkovIntentEngine(
                stemmer=self.stemmers.get(lang),
                **self._engine_kwargs_template,
            )
            for lang in self._langs
        }

    def _add_intent(self, engine: DomainMarkovIntentEngine,
                    name: str, samples: List[str]) -> None:
        engine.register_domain_intent(self._domain_of(name), name, samples)

    def _remove_intent(self, engine: DomainMarkovIntentEngine,
                       name: str) -> None:
        engine.remove_domain_intent(self._domain_of(name), name)

    def _remove_skill(self, engine: DomainMarkovIntentEngine,
                      skill_id: str, intent_names: List[str]) -> None:
        # In domain mode the skill_id IS the domain.
        engine.remove_domain(skill_id)
