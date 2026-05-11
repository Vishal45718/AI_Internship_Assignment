"""
src/retrieval/retriever.py — Semantic retrieval with threshold filtering.

Retrieves top-K document chunks by cosine similarity from the vector store,
then filters by a configurable similarity threshold to prevent low-quality
results from reaching the LLM.

This threshold filter is the primary hallucination prevention gate —
if no chunks pass the threshold, the pipeline returns a fallback response
instead of asking the LLM to answer from weak context.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.models import RetrievedChunk, RetrievalResult
from src.config import get_settings

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Performs semantic similarity search against the vector store.

    Responsibilities:
      - Embed the query
      - Retrieve top-K candidates
      - Apply similarity threshold filter
      - Return structured RetrievalResult
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
            query: User question.
            top_k: Number of chunks to retrieve (defaults to config).
            filters: Optional metadata filters for narrowing scope.
            similarity_threshold: Minimum score to include a result.

        Returns:
            RetrievalResult with ranked chunks and pass/fail flag.
        """
        if not query.strip():
            logger.warning("Empty query.")
            return RetrievalResult(query=query)

        k = top_k or self._settings.retrieval_top_k
        threshold = similarity_threshold or self._settings.similarity_threshold

        # Check if the store has any data
        if self._store.count() == 0:
            logger.warning("Vector store is empty.")
            return RetrievalResult(query=query)

        if self._settings.hybrid_search_enabled:
            vector_chunks = self._store.query(
                query_text=query,
                top_k=k,
                where=filters,
            )
            keyword_chunks = self._store.keyword_search(
                query_text=query,
                top_k=k,
                where=filters,
            )

            merged_chunks = self._merge_hybrid_results(query, vector_chunks, keyword_chunks)
            total_retrieved = len(merged_chunks)
            top_score = merged_chunks[0].similarity_score if merged_chunks else 0.0
            semantic_above_ids = {
                c.chunk_id for c in vector_chunks if c.similarity_score >= threshold
            }
            above_threshold = [
                c for c in merged_chunks
                if c.similarity_score >= threshold
                or self._contains_exact_token(c.content, query)
                or c.chunk_id in semantic_above_ids
            ]
            passed = len(above_threshold) > 0

            logger.info(
                "Hybrid retrieval: vector_hits=%d keyword_hits=%d fused_hits=%d",
                len(vector_chunks), len(keyword_chunks), len(merged_chunks),
            )
            logger.debug(
                "Hybrid top candidates: %s",
                [f"{c.chunk_id}:{c.similarity_score:.4f}" for c in merged_chunks[:10]],
            )

            if not passed:
                logger.info(
                    "Query '%s…' — no hybrid candidates above threshold %.2f (top fused: %.3f)",
                    query[:60], threshold, top_score,
                )

            return RetrievalResult(
                query=query,
                chunks=above_threshold,
                total_retrieved=total_retrieved,
                top_score=top_score,
                passed_threshold=passed,
            )

        raw_chunks: list[RetrievedChunk] = self._store.query(
            query_text=query,
            top_k=k,
            where=filters,
        )

        total_retrieved = len(raw_chunks)

        # Apply similarity threshold — this is the hallucination gate
        above_threshold = [c for c in raw_chunks if c.similarity_score >= threshold]

        top_score = raw_chunks[0].similarity_score if raw_chunks else 0.0
        passed = len(above_threshold) > 0

        if not passed:
            logger.info(
                "Query '%s…' — no chunks above threshold %.2f (top: %.3f)",
                query[:60], threshold, top_score,
            )

        return RetrievalResult(
            query=query,
            chunks=above_threshold,
            total_retrieved=total_retrieved,
            top_score=top_score,
            passed_threshold=passed,
        )

    def _merge_hybrid_results(
        self,
        query: str,
        vector_chunks: list[RetrievedChunk],
        keyword_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Fuse vector similarity and keyword retrieval into a single ranked result set."""
        candidate_map: dict[str, dict[str, Any]] = {}

        for rank, chunk in enumerate(vector_chunks, start=1):
            candidate_map[chunk.chunk_id] = {
                "chunk": chunk,
                "vector_rank": rank,
                "vector_score": chunk.similarity_score,
                "keyword_rank": None,
                "keyword_score": 0.0,
            }

        for rank, chunk in enumerate(keyword_chunks, start=1):
            entry = candidate_map.get(chunk.chunk_id)
            if entry is None:
                entry = {
                    "chunk": chunk,
                    "vector_rank": None,
                    "vector_score": 0.0,
                    "keyword_rank": rank,
                    "keyword_score": chunk.similarity_score,
                }
                candidate_map[chunk.chunk_id] = entry
            else:
                entry["keyword_rank"] = rank
                entry["keyword_score"] = chunk.similarity_score

        max_keyword_score = max((entry["keyword_score"] for entry in candidate_map.values()), default=1.0)
        fused_results: list[RetrievedChunk] = []
        for entry in candidate_map.values():
            vector_score = entry["vector_score"]
            keyword_score = entry["keyword_score"] / max_keyword_score if max_keyword_score > 0 else 0.0
            exact_bonus = 0.35 if self._contains_exact_token(entry["chunk"].content, query) else 0.0
            fused_score = min(1.0, vector_score * 0.30 + keyword_score * 0.35 + exact_bonus)

            fused_chunk = RetrievedChunk(
                chunk_id=entry["chunk"].chunk_id,
                content=entry["chunk"].content,
                source_file=entry["chunk"].source_file,
                source_type=entry["chunk"].source_type,
                page_number=entry["chunk"].page_number,
                chunk_index=entry["chunk"].chunk_index,
                similarity_score=round(fused_score, 4),
            )
            fused_results.append(fused_chunk)

        fused_results.sort(key=lambda c: c.similarity_score, reverse=True)
        return fused_results

    def _contains_exact_token(self, content: str, query: str) -> bool:
        terms = [t for t in re.findall(r"\w+", query) if t]
        if not terms:
            return False
        lowered_text = content.lower()
        for term in terms:
            if re.search(rf"\b{re.escape(term.lower())}\b", lowered_text):
                return True
        return False
