"""Disk cache for trained Markov intent models via ONNX export.

Saves trained models to disk as ONNX files + vocab JSON so they can be
loaded instantly on restart without re-training.
"""

from pathlib import Path
from typing import Dict, Optional

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    export_markov_sparse_onnx,
)
from ovos_utils.log import LOG


class IntentCache:
    """Manages disk-cached ONNX models for intent matching.

    Args:
        cache_dir: Directory to store cached models.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._runtimes: Dict[str, MarkovONNXRuntime] = {}
        self._vocabs: Dict[str, Vocabulary] = {}

    def _safe_name(self, intent_name: str) -> str:
        """Convert intent name to filesystem-safe string."""
        return intent_name.replace(":", "__").replace("/", "_")

    def save_intent(
        self,
        intent_name: str,
        model: MarkovChain,
    ) -> None:
        """Export and cache a trained intent model.

        Args:
            intent_name: Intent identifier.
            model: Trained MarkovChain.
        """
        safe = self._safe_name(intent_name)
        onnx_path = str(self.cache_dir / f"{safe}.onnx")
        vocab_path = str(self.cache_dir / f"{safe}_vocab.json")

        try:
            export_markov_sparse_onnx(model, onnx_path)
            model.vocab.save(vocab_path)
        except Exception as e:
            LOG.error(f"Failed to cache intent {intent_name}: {e}")

    def load_intent(
        self,
        intent_name: str,
        order: int,
    ) -> Optional[MarkovONNXRuntime]:
        """Load a cached intent model.

        Args:
            intent_name: Intent identifier.
            order: N-gram order of the model.

        Returns:
            A :class:`MarkovONNXRuntime` or ``None`` if not cached.
        """
        safe = self._safe_name(intent_name)
        onnx_path = self.cache_dir / f"{safe}.onnx"
        vocab_path = self.cache_dir / f"{safe}_vocab.json"

        if not onnx_path.exists() or not vocab_path.exists():
            return None

        try:
            vocab = Vocabulary.load(str(vocab_path))
            rt = MarkovONNXRuntime(str(onnx_path), vocab, order)
            return rt
        except Exception as e:
            LOG.error(f"Failed to load cached intent {intent_name}: {e}")
            return None

    def has_intent(self, intent_name: str) -> bool:
        """Check if an intent is cached on disk."""
        safe = self._safe_name(intent_name)
        return (self.cache_dir / f"{safe}.onnx").exists()

    def remove_intent(self, intent_name: str) -> None:
        """Remove cached files for an intent."""
        safe = self._safe_name(intent_name)
        for suffix in [".onnx", "_vocab.json"]:
            path = self.cache_dir / f"{safe}{suffix}"
            if path.exists():
                path.unlink()

    def clear(self) -> None:
        """Remove all cached models."""
        for path in self.cache_dir.glob("*.onnx"):
            path.unlink()
        for path in self.cache_dir.glob("*_vocab.json"):
            path.unlink()
