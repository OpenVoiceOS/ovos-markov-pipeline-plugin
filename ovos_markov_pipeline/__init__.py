"""OVOS intent pipeline plugin using Markov chain perplexity ensemble.

Trains one word-level Markov chain per intent from example utterances.
Classifies by computing perplexity under each model — the intent whose
model assigns the lowest perplexity (highest likelihood) wins.

Confidence is derived from perplexity via: conf = 1 / (1 + log(ppx))
"""

import math
import re
import string
from collections import defaultdict
from os.path import expanduser
from pathlib import Path
from threading import Event, RLock
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_config.meta import get_xdg_base
from ovos_plugin_manager.templates.pipeline import (
    ConfidenceMatcherPipeline,
    IntentHandlerMatch,
)
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from markovonnx import MarkovChain, Vocabulary, word_tokenize

from ovos_markov_pipeline.version import __version__


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(string.punctuation)
    return text


def _ppx_to_confidence(ppx: float) -> float:
    """Convert perplexity to a 0-1 confidence score.

    Lower perplexity → higher confidence.
    Formula: conf = 1 / (1 + log(ppx))
    Clamped to [0, 1].
    """
    if ppx <= 1.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.log(ppx))))


class MarkovIntentEngine:
    """Per-language intent matching engine using Markov chain perplexity.

    Maintains a shared vocabulary and one MarkovChain per registered intent.
    Intents are re-trained when new samples are added.

    Args:
        order: N-gram order for Markov chains.
        smoothing: Laplace smoothing alpha.
        kneser_ney: Use Kneser-Ney smoothing instead of Laplace.
        backoff: Enable interpolated backoff to lower-order models.
    """

    def __init__(
        self,
        order: int = 2,
        smoothing: float = 1e-5,
        kneser_ney: bool = True,
        backoff: bool = True,
    ):
        self.order = order
        self.smoothing = smoothing
        self.kneser_ney = kneser_ney
        self.backoff = backoff

        self._intent_samples: Dict[str, List[List[str]]] = {}
        self._models: Dict[str, MarkovChain] = {}
        self._vocab: Optional[Vocabulary] = None
        self._trained = False

    @property
    def must_train(self) -> bool:
        """Whether new samples have been added since last train."""
        return not self._trained and len(self._intent_samples) > 0

    def add_intent(self, name: str, samples: List[str]) -> None:
        """Register an intent with training samples.

        Args:
            name: Intent name (typically ``skill_id:intent_name``).
            samples: List of example utterances.
        """
        tokenized = [word_tokenize(_normalize(s)) for s in samples if s.strip()]
        tokenized = [s for s in tokenized if len(s) > 0]
        self._intent_samples[name] = tokenized
        self._trained = False

    def remove_intent(self, name: str) -> None:
        """Remove a registered intent."""
        self._intent_samples.pop(name, None)
        self._models.pop(name, None)
        self._trained = False

    def train(self) -> None:
        """Train all intent models on current samples.

        Builds a shared vocabulary from all samples, then trains one
        MarkovChain per intent.
        """
        if not self._intent_samples:
            self._trained = True
            return

        # Build shared vocabulary from ALL intent samples
        all_sequences = []
        for samples in self._intent_samples.values():
            all_sequences.extend(samples)

        self._vocab = Vocabulary()
        self._vocab.build_from_sequences(all_sequences)

        # Train one model per intent
        self._models = {}
        for name, samples in self._intent_samples.items():
            if not samples:
                continue
            mc = MarkovChain(
                order=self.order,
                vocab=self._vocab,
                smoothing=self.smoothing,
                backoff=self.backoff,
                kneser_ney=self.kneser_ney,
            )
            mc.fit(samples)
            self._models[name] = mc

        self._trained = True

    def calc_intents(
        self,
        utterance: str,
        blacklisted_intents: Optional[set] = None,
        blacklisted_skills: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """Score all intents for an utterance.

        Args:
            utterance: The user's text.
            blacklisted_intents: Intent names to skip.
            blacklisted_skills: Skill IDs to skip.

        Returns:
            List of ``(intent_name, confidence)`` sorted by descending confidence.
        """
        if not self._models or self._vocab is None:
            return []

        blacklisted_intents = blacklisted_intents or set()
        blacklisted_skills = blacklisted_skills or set()

        tokens = word_tokenize(_normalize(utterance))
        if len(tokens) < self.order:
            return []

        scores: List[Tuple[str, float]] = []
        for name, mc in self._models.items():
            if name in blacklisted_intents:
                continue
            skill_id = name.split(":")[0]
            if skill_id in blacklisted_skills:
                continue

            ppx = mc.perplexity([tokens])
            conf = _ppx_to_confidence(ppx)
            scores.append((name, conf))

        scores.sort(key=lambda x: -x[1])
        return scores


class MarkovPipeline(ConfidenceMatcherPipeline):
    """OVOS pipeline plugin for Markov chain perplexity-based intent matching.

    Follows the same pattern as ``PadatiousPipeline``: skills register
    intent samples via the MessageBus, the engine trains on them, and
    incoming utterances are classified by comparing perplexity across
    all registered intent models.

    Configuration (in ``mycroft.conf``):

    .. code-block:: json

        {
            "intents": {
                "ovos-markov-pipeline-plugin": {
                    "order": 2,
                    "kneser_ney": true,
                    "backoff": true,
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
        langs = [standardize_lang_tag(l) for l in langs]
        if self.lang not in langs:
            langs.append(self.lang)

        self.conf_high = self.config.get("conf_high", 0.75)
        self.conf_med = self.config.get("conf_med", 0.55)
        self.conf_low = self.config.get("conf_low", 0.30)

        order = self.config.get("order", 2)
        kneser_ney = self.config.get("kneser_ney", True)
        backoff = self.config.get("backoff", True)
        smoothing = self.config.get("smoothing", 1e-5)

        self.engines: Dict[str, MarkovIntentEngine] = {
            lang: MarkovIntentEngine(
                order=order,
                smoothing=smoothing,
                kneser_ney=kneser_ney,
                backoff=backoff,
            )
            for lang in langs
        }

        self.first_train = Event()
        self.finished_training_event = Event()
        self.finished_training_event.set()

        self.registered_intents: List[str] = []
        self._skill2intent: Dict[str, List[str]] = defaultdict(list)
        self.max_words = self.config.get("max_words", 50)

        # Register bus handlers
        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.skills.train", self.train)
        self.bus.on(
            "intent.service.markov.manifest.get",
            self.handle_manifest,
        )

        LOG.info(f"Loaded MarkovPipeline (order={order}, kn={kneser_ney}, backoff={backoff})")

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        """Find the closest registered language."""
        lang = standardize_lang_tag(lang)
        if lang in self.engines:
            return lang
        # Simple prefix match
        prefix = lang.split("-")[0]
        for registered in self.engines:
            if registered.startswith(prefix):
                return registered
        return None

    # -- Intent registration --------------------------------------------------

    def register_intent(self, message: Message) -> None:
        """Handle ``padatious:register_intent`` bus message.

        Accepts the same format as Padatious: ``name``, ``samples`` or
        ``file_name``, ``skill_id``, ``lang``.
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID missing, using 'anonymous_skill'")
            skill_id = "anonymous_skill"

        name = message.data["name"]
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        samples = message.data.get("samples")

        # Load from file if no inline samples
        file_name = message.data.get("file_name")
        if not samples and file_name and Path(file_name).is_file():
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines()]

        if not samples:
            LOG.error(f"No samples for intent {name}")
            return

        self._skill2intent[skill_id].append(name)
        self.registered_intents.append(name)

        closest = self._get_closest_lang(lang)
        if closest and closest in self.engines:
            LOG.debug(f"Registering markov intent: {name} ({len(samples)} samples)")
            self.engines[closest].add_intent(name, samples)

        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def handle_detach_intent(self, message: Message) -> None:
        """Remove a single intent."""
        intent_name = message.data.get("intent_name")
        if intent_name and intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for engine in self.engines.values():
                engine.remove_intent(intent_name)

    def handle_detach_skill(self, message: Message) -> None:
        """Remove all intents for a skill."""
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            return
        for intent_name in self._skill2intent.pop(skill_id, []):
            if intent_name in self.registered_intents:
                self.registered_intents.remove(intent_name)
            for engine in self.engines.values():
                engine.remove_intent(intent_name)

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
            if scores and scores[0][1] > best_conf:
                best_intent, best_conf = scores[0]

        if best_intent is not None and best_conf > limit:
            skill_id = best_intent.split(":")[0]
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
