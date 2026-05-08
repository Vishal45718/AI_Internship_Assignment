"""
app/retrieval/retriever.py — Semantic retrieval with threshold filtering.

Implements Stage 1 of the two-stage retrieve-then-rerank pipeline.
Retrieves top-K candidates by cosine similarity, then filters by threshold.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.retrieval import RetrievedChunk, RetrievalResult
from configs.settings import get_settings

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Performs semantic similarity search against the vector store.

    Responsibilities:
      - Embed the query
      - Retrieve top-K candidates
      - Apply similarity threshold filter
      - Return structured RetrievalResult

    Does NOT rerank — that's the Reranker's job.
    """

    def __init__(self, vector_store) -> None:
        self._store = vector_store
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Retrieve semantically relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Number of chunks to retrieve (defaults to config).
            filters: Optional metadata filters for narrowing the search scope.
            similarity_threshold: Minimum score to include a result.

        Returns:
            RetrievalResult with ranked chunks and pass/fail threshold flag.
        """
        if not query.strip():
            logger.warning("Empty query passed to retriever.")
            return RetrievalResult.empty(query)

        k = top_k or self._settings.retrieval_top_k
        threshold = similarity_threshold if similarity_threshold is not None \
            else self._settings.similarity_threshold

        # Check if the store has any data
        if self._store.count() == 0:
            logger.warning("Vector store is empty. Cannot retrieve.")
            return RetrievalResult(
                query=query,
                chunks=[],
                total_retrieved=0,
                top_score=0.0,
                passed_threshold=False,
            )

        raw_chunks: list[RetrievedChunk] = self._store.query(
            query_text=query,
            top_k=k,
            where=filters,
        )

        total_retrieved = len(raw_chunks)

        # Apply similarity threshold — this is the primary hallucination gate
        above_threshold = [c for c in raw_chunks if c.similarity_score >= threshold]

        top_score = raw_chunks[0].similarity_score if raw_chunks else 0.0
        passed = len(above_threshold) > 0

        if not passed:
            logger.info(
                "Query '%s…' — no chunks above threshold %.2f (top score: %.3f)",
                query[:60],
                threshold,
                top_score,
            )

        return RetrievalResult(
            query=query,
            chunks=above_threshold,
            total_retrieved=total_retrieved,
            top_score=top_score,
            retrieval_strategy="semantic",
            passed_threshold=passed,
        )
