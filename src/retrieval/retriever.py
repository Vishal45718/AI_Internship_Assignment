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
        expand_context: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve semantically relevant chunks for a query.

        Args:
            query: User question.
            top_k: Number of chunks to retrieve (defaults to config).
            filters: Optional metadata filters for narrowing scope.
            similarity_threshold: Minimum score to include a result.
            expand_context: Whether to expand context with neighboring chunks.

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

            final_chunks = above_threshold
            original_chunk_ids = [c.chunk_id for c in above_threshold]
            parent_sections_used: list[str] = []
            overlap_reduction_count = 0
            expanded_context_token_count = self._estimate_token_count(final_chunks)
            if passed and expand_context and self._settings.parent_context_enabled:
                (
                    final_chunks,
                    parent_sections_used,
                    overlap_reduction_count,
                    expanded_context_token_count,
                ) = self._expand_context(
                    above_threshold,
                    self._settings.parent_window_before,
                    self._settings.parent_window_after,
                )
                logger.info(
                    "Hybrid context expansion: original=%d expanded=%d (before=%d after=%d)",
                    len(above_threshold), len(final_chunks),
                    self._settings.parent_window_before, self._settings.parent_window_after,
                )
                logger.debug("Original retrieved chunk IDs: %s", original_chunk_ids)
                logger.debug("Expanded chunk IDs: %s", [c.chunk_id for c in final_chunks])
                logger.debug("Parent sections used: %s", parent_sections_used)
                logger.debug("Overlap reduction count: %d", overlap_reduction_count)
                logger.debug("Final context token count: %d", expanded_context_token_count)
                logger.debug("Retrieval chunk count: %d", len(final_chunks))

            return RetrievalResult(
                query=query,
                chunks=final_chunks,
                total_retrieved=total_retrieved,
                top_score=top_score,
                passed_threshold=passed,
                original_chunk_ids=original_chunk_ids,
                expanded_chunk_ids=[c.chunk_id for c in final_chunks],
                parent_sections_used=parent_sections_used,
                expanded_context_token_count=expanded_context_token_count,
                overlap_reduction_count=overlap_reduction_count,
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

        final_chunks = above_threshold
        original_chunk_ids = [c.chunk_id for c in above_threshold]
        parent_sections_used: list[str] = []
        overlap_reduction_count = 0
        expanded_context_token_count = self._estimate_token_count(final_chunks)
        if passed and expand_context and self._settings.parent_context_enabled:
            (
                final_chunks,
                parent_sections_used,
                overlap_reduction_count,
                expanded_context_token_count,
            ) = self._expand_context(
                above_threshold,
                self._settings.parent_window_before,
                self._settings.parent_window_after,
            )
            logger.info(
                "Context expansion: original=%d expanded=%d (before=%d after=%d)",
                len(above_threshold), len(final_chunks),
                self._settings.parent_window_before, self._settings.parent_window_after,
            )
            logger.debug("Original retrieved chunk IDs: %s", original_chunk_ids)
            logger.debug("Expanded chunk IDs: %s", [c.chunk_id for c in final_chunks])
            logger.debug("Parent sections used: %s", parent_sections_used)
            logger.debug("Overlap reduction count: %d", overlap_reduction_count)
            logger.debug("Final context token count: %d", expanded_context_token_count)
            logger.debug("Retrieval chunk count: %d", len(final_chunks))

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            total_retrieved=total_retrieved,
            top_score=top_score,
            passed_threshold=passed,
            original_chunk_ids=original_chunk_ids,
            expanded_chunk_ids=[c.chunk_id for c in final_chunks],
            parent_sections_used=parent_sections_used,
            expanded_context_token_count=expanded_context_token_count,
            overlap_reduction_count=overlap_reduction_count,
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
                parent_id=entry["chunk"].parent_id,
                document_name=entry["chunk"].document_name,
                section_title=entry["chunk"].section_title,
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

    def _expand_context(
        self,
        retrieved_chunks: list[RetrievedChunk],
        window_before: int = 1,
        window_after: int = 1,
    ) -> tuple[list[RetrievedChunk], list[str], int, int]:
        """
        Expand retrieved chunks with neighboring context.

        For each retrieved chunk:
        - Fetch neighboring chunks (before and after)
        - Deduplicate by chunk_id
        - Maintain page/chunk ordering
        """
        if not retrieved_chunks or (window_before == 0 and window_after == 0):
            token_count = self._estimate_token_count(retrieved_chunks)
            return retrieved_chunks, [], 0, token_count

        expanded_ids: set[str] = set()
        context_chunks: dict[str, RetrievedChunk] = {}
        parent_sections_used: set[str] = set()

        for chunk in retrieved_chunks:
            expanded_ids.add(chunk.chunk_id)
            context_chunks[chunk.chunk_id] = chunk
            if chunk.section_title:
                parent_sections_used.add(chunk.section_title)

        neighbors = self._store.get_neighboring_chunks(
            chunk_ids=[c.chunk_id for c in retrieved_chunks],
            window_before=window_before,
            window_after=window_after,
        )

        for neighbor in neighbors:
            if neighbor.chunk_id not in expanded_ids:
                context_chunks[neighbor.chunk_id] = neighbor
                expanded_ids.add(neighbor.chunk_id)
            if neighbor.section_title:
                parent_sections_used.add(neighbor.section_title)

        if self._settings.parent_expand_full_section:
            section_seed_chunks = list(context_chunks.values())
            for seed in section_seed_chunks:
                if not seed.section_title:
                    continue
                section_chunks = self._store.get_chunks_by_section(
                    source_file=seed.source_file,
                    section_title=seed.section_title,
                )
                for section_chunk in section_chunks:
                    if section_chunk.chunk_id not in expanded_ids:
                        context_chunks[section_chunk.chunk_id] = section_chunk
                        expanded_ids.add(section_chunk.chunk_id)
                    if section_chunk.section_title:
                        parent_sections_used.add(section_chunk.section_title)

        final_chunks = list(context_chunks.values())
        final_chunks.sort(key=lambda c: (c.page_number or 0, c.chunk_index))

        overlap_reduction = max(0, len(retrieved_chunks) + len(neighbors) - len(final_chunks))
        token_count = self._estimate_token_count(final_chunks)
        logger.debug(
            "Expanded context: original_ids=%d final_ids=%d overlaps_added=%d",
            len(retrieved_chunks), len(final_chunks), overlap_reduction,
        )

        return final_chunks, sorted(parent_sections_used), overlap_reduction, token_count

    def _estimate_token_count(self, chunks: list[RetrievedChunk]) -> int:
        """Approximate token count for debug observability."""
        if not chunks:
            return 0
        total_chars = sum(len(c.content) for c in chunks)
        return max(1, total_chars // 4)
