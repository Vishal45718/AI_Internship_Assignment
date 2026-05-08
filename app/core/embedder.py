"""
app/core/embedder.py — Embedding generation with provider abstraction.

Supports two providers:
  - "openai": text-embedding-3-small (API call)
  - "local": sentence-transformers/all-MiniLM-L6-v2 (on-device)

The provider is swappable via EMBEDDING_PROVIDER env var.
All callers interact through the same embed() / embed_many() interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import cast

from configs.settings import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface contract for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. More efficient than looping embed()."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension (needed for vector DB configuration)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    Wraps the OpenAI Embeddings API.
    Uses batching to minimise API round-trips.
    """

    _DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._dim = self._DIMENSION_MAP.get(self._model, 1536)
        logger.info("OpenAIEmbedder initialised: model=%s dim=%d", self._model, self._dim)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # OpenAI recommends replacing newlines for embedding quality
        cleaned = [t.replace("\n", " ") for t in texts]

        # Batch in groups of 100 (API limit)
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            start = time.perf_counter()
            response = self._client.embeddings.create(model=self._model, input=batch)
            elapsed = time.perf_counter() - start
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                "Embedded batch of %d texts in %.2fs", len(batch), elapsed
            )

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# Local sentence-transformers provider
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers for fully local, offline embeddings.
    Default model: all-MiniLM-L6-v2 (384-dim, very fast, good quality).
    """

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        settings = get_settings()
        model_name = settings.local_embedding_model
        logger.info("Loading local embedding model: %s (this may take a moment…)", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = cast(int, self._model.get_sentence_embedding_dimension())
        logger.info("LocalEmbedder ready: model=%s dim=%d", model_name, self._dim)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
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
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_embedder() -> BaseEmbedder:
    """
    Factory function that instantiates the configured embedding provider.
    Call this once at startup and inject the result as a dependency.
    """
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "openai":
        return OpenAIEmbedder()
    elif provider == "local":
        return LocalEmbedder()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            "Valid options: 'openai', 'local'."
        )
