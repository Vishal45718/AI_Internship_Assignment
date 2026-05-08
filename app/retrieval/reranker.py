"""
app/retrieval/reranker.py — Cross-encoder reranking (Stage 2 of retrieval).

After semantic search retrieves the top-K candidates (high recall, lower
precision), a cross-encoder model reads query+chunk together and scores
true relevance — dramatically improving precision.

Uses: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, local, no API cost)
      or flashrank as a lightweight alternative.

This is one of the highest-ROI improvements over basic RAG:
  - ~30% precision improvement on real datasets
  - Catches semantically similar but contextually wrong chunks
"""

from __future__ import annotations

import logging

from app.models.retrieval import RetrievedChunk, RetrievalResult
from configs.settings import get_settings

logger = logging.getLogger(__name__)

# Reranker model — loaded lazily to avoid startup cost when reranking is disabled
_reranker_model = None


def _get_reranker_model():
    """Lazy-load the cross-encoder model. Downloads on first call (~40MB)."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder reranker model…")
            _reranker_model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("Cross-encoder reranker loaded.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _reranker_model


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    The cross-encoder takes (query, chunk) pairs and scores them 0→1,
    where higher = more relevant. This is much more accurate than cosine
    similarity alone because the model sees both texts simultaneously.
    """

    def __init__(self, top_n: int | None = None) -> None:
        settings = get_settings()
        self._top_n = top_n or settings.rerank_top_n
        self._confidence_threshold = settings.confidence_threshold
        logger.debug(
            "CrossEncoderReranker: top_n=%d confidence_threshold=%.2f",
            self._top_n,
            self._confidence_threshold,
        )

    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        """
        Rerank a RetrievalResult's chunks and return a new result with
        top-N chunks selected by cross-encoder score.

        Args:
            query: The original user query.
            result: The output of SemanticRetriever.retrieve().

        Returns:
            New RetrievalResult with reranked chunks and updated top_score.
        """
        if not result.chunks:
            return result

        model = _get_reranker_model()
        pairs = [(query, chunk.content) for chunk in result.chunks]

        # Cross-encoder returns raw logit scores; apply sigmoid for [0,1]
        import numpy as np
        raw_scores = model.predict(pairs)
        # Sigmoid normalization
        scores = [float(1 / (1 + np.exp(-s))) for s in raw_scores]

        # Attach rerank scores and sort
        for chunk, score in zip(result.chunks, scores):
            chunk.rerank_score = round(score, 4)

        reranked = sorted(result.chunks, key=lambda c: c.rerank_score or 0, reverse=True)
        top_n = reranked[: self._top_n]

        top_score = top_n[0].rerank_score if top_n else 0.0
        passed = top_score >= self._confidence_threshold

        if not passed:
            logger.info(
                "Reranker: top score %.3f below confidence threshold %.2f — fallback triggered.",
                top_score,
                self._confidence_threshold,
            )

        logger.debug(
            "Reranked %d → top %d chunks. Top score: %.3f",
            len(result.chunks),
            len(top_n),
            top_score,
        )

        return RetrievalResult(
            query=query,
            chunks=top_n,
            total_retrieved=result.total_retrieved,
            top_score=top_score,
            retrieval_strategy="reranked",
            passed_threshold=passed,
        )


class NoOpReranker:
    """
    Pass-through reranker used when reranking is disabled.
    Selects the top-N chunks by similarity score and applies the
    confidence threshold check.
    """

    def __init__(self, top_n: int | None = None) -> None:
        settings = get_settings()
        self._top_n = top_n or settings.rerank_top_n
        self._confidence_threshold = settings.confidence_threshold

    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        if not result.chunks:
            return result

        top_n = result.chunks[: self._top_n]
        top_score = top_n[0].similarity_score if top_n else 0.0
        passed = top_score >= self._confidence_threshold

        return RetrievalResult(
            query=query,
            chunks=top_n,
            total_retrieved=result.total_retrieved,
            top_score=top_score,
            retrieval_strategy="semantic",
            passed_threshold=passed,
        )


def create_reranker(enabled: bool | None = None):
    """
    Factory: return CrossEncoderReranker if enabled, else NoOpReranker.
    """
    settings = get_settings()
    use_reranking = enabled if enabled is not None else settings.enable_reranking

    if use_reranking:
        try:
            return CrossEncoderReranker()
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder reranker (%s). "
                "Falling back to NoOpReranker.",
                exc,
            )
            return NoOpReranker()
    else:
        return NoOpReranker()
