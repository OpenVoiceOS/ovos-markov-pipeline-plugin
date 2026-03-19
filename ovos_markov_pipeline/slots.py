"""Entity/slot extraction using HMM BIO tagging.

Trains a supervised HMM from (utterance, BIO-tag) pairs to extract
entity slots from matched utterances.

BIO format:
  ``set a timer for five minutes``
  ``O   O O     O   B-time I-time``
"""

from typing import Dict, List, Optional, Tuple

from markovonnx import HiddenMarkovModel, Vocabulary


class SlotExtractor:
    """HMM-based entity slot extractor using BIO tagging.

    Args:
        smoothing: Laplace smoothing for HMM.
    """

    def __init__(self, smoothing: float = 1e-5):
        self.smoothing = smoothing
        self._intent_data: Dict[str, Tuple[List[List[str]], List[List[str]]]] = {}
        self._models: Dict[str, HiddenMarkovModel] = {}
        self._obs_vocabs: Dict[str, Vocabulary] = {}

    def add_slot_data(
        self,
        intent_name: str,
        utterances: List[List[str]],
        bio_tags: List[List[str]],
    ) -> None:
        """Register BIO-tagged training data for an intent.

        Args:
            intent_name: Intent identifier.
            utterances: List of tokenized utterances.
            bio_tags: Corresponding BIO tag sequences.
        """
        self._intent_data[intent_name] = (utterances, bio_tags)

    def train(self) -> None:
        """Train one HMM per intent from registered BIO data."""
        self._models = {}
        self._obs_vocabs = {}
        for name, (utts, tags) in self._intent_data.items():
            if not utts:
                continue
            obs_vocab = Vocabulary()
            obs_vocab.build_from_sequences(utts)
            self._obs_vocabs[name] = obs_vocab

            # Count unique tags to determine n_states
            all_tags = set()
            for tag_seq in tags:
                all_tags.update(tag_seq)

            hmm = HiddenMarkovModel(
                n_states=len(all_tags) + 1,  # +1 for UNK tag
                obs_vocab=obs_vocab,
                smoothing=self.smoothing,
            )
            hmm.fit_supervised(utts, tags)
            self._models[name] = hmm

    def extract(
        self,
        intent_name: str,
        tokens: List[str],
    ) -> Dict[str, str]:
        """Extract entity slots from a tokenized utterance.

        Args:
            intent_name: The matched intent.
            tokens: Tokenized utterance.

        Returns:
            Dict mapping slot names to extracted values.
            E.g. ``{"time": "five minutes"}``.
        """
        if intent_name not in self._models:
            return {}

        hmm = self._models[intent_name]
        bio_tags = hmm.viterbi(tokens)

        # Guard: ensure tags and tokens have same length
        if len(bio_tags) != len(tokens):
            return {}

        # Parse BIO tags into slot values
        slots: Dict[str, str] = {}
        current_slot: Optional[str] = None
        current_tokens: List[str] = []

        for token, tag in zip(tokens, bio_tags):
            if tag.startswith("B-"):
                # Flush previous slot
                if current_slot is not None and current_tokens:
                    slots[current_slot] = " ".join(current_tokens)
                current_slot = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_slot == tag[2:]:
                current_tokens.append(token)
            else:
                # O tag or mismatched I- tag
                if current_slot is not None and current_tokens:
                    slots[current_slot] = " ".join(current_tokens)
                current_slot = None
                current_tokens = []

        # Flush final slot
        if current_slot is not None and current_tokens:
            slots[current_slot] = " ".join(current_tokens)

        return slots

    def remove_intent(self, intent_name: str) -> None:
        """Remove slot data for an intent."""
        self._intent_data.pop(intent_name, None)
        self._models.pop(intent_name, None)
        self._obs_vocabs.pop(intent_name, None)
