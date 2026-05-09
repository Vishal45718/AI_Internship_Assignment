"""
src/embeddings/embedder.py — Local embedding generation.

Uses sentence-transformers/all-MiniLM-L6-v2 by default.
This is a lightweight model (~80MB) that runs locally without API costs.
Produces 384-dimensional vectors with good quality for document retrieval.
"""

from __future__ import annotations

import logging
from typing import cast

from src.config import get_settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Generates embeddings using a local sentence-transformers model.

    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
    No API key required — runs entirely on your machine.
    """

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        settings = get_settings()
        model_name = settings.embedding_model
        logger.info("Loading embedding model: %s (first time may download ~80MB)…", model_name)
        self._model = SentenceTransformer(model_name)
        # Use new method name (old get_sentence_embedding_dimension is deprecated)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim = cast(int, dim_fn())
        logger.info("Embedder ready: model=%s dim=%d", model_name, self._dim)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. More efficient than calling embed() in a loop.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (needed for vector DB configuration)."""
        return self._dim


def create_embedder() -> LocalEmbedder:
    """Create and return the embedder instance."""
    return LocalEmbedder()
