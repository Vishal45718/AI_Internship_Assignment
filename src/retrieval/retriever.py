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
from collections import Counter, defaultdict
from typing import Any

from src.models import RetrievedChunk, RetrievalResult
from src.config import get_settings
from src.retrieval.reranker import CrossEncoderReranker

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
        self._reranker = CrossEncoderReranker()

    def _extract_query_entities(self, query: str) -> dict[str, Any]:
        tokens = [t for t in re.findall(r"\w+", query) if t]
        entities: list[str] = []
        all_caps = {token for token in tokens if re.fullmatch(r"[A-Z]{2,}", token)}
        camel_case = set(re.findall(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z]+|[a-z]+[A-Z][A-Za-z]+)\b", query))
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        mixed_case = {token for token in tokens if any(c.isupper() for c in token[1:]) and not token.isupper()}

        entities.extend(all_caps)
        entities.extend(camel_case)
        entities.extend(quoted_phrases)
        entities.extend(mixed_case)

        unique_entities = []
        seen = set()
        for entity in entities:
            normalized = entity.strip()
            if normalized and normalized not in seen:
                unique_entities.append(normalized)
                seen.add(normalized)

        return {
            "tokens": tokens,
            "entities": unique_entities,
            "acronym_entities": list(all_caps),
            "mixed_case_entities": list(mixed_case),
            "quoted_phrases": quoted_phrases,
            "query_lower": query.lower(),
        }

    def _detect_query_type(self, query: str, entities: dict[str, Any]) -> str:
        q_lower = query.lower()
        is_comparison = "compare" in q_lower or " vs " in q_lower or " versus " in q_lower or "between" in q_lower
        if is_comparison and len(entities["entities"]) >= 2:
            return "comparison"
        if any(token in q_lower for token in ("how", "mechanism", "process", "reformulate", "operate", "works", "methodology", "workflow")):
            return "mechanism"
        if any(token in q_lower for token in ("what is", "define", "meaning of", "describe")):
            if entities["entities"]:
                return "acronym_lookup"
            return "definition"
        if entities["acronym_entities"] or entities["mixed_case_entities"]:
            return "acronym_lookup"
        return "general"

    def _adaptive_weights(self, query_type: str, entities: dict[str, Any]) -> tuple[float, float]:
        if query_type == "acronym_lookup":
            return 0.35, 0.65
        if query_type == "comparison":
            return 0.45, 0.55
        if query_type == "mechanism":
            return 0.55, 0.45
        if query_type == "definition":
            return 0.60, 0.40
        if entities["acronym_entities"] or entities["mixed_case_entities"] or entities["quoted_phrases"]:
            return 0.45, 0.55
        return 0.65, 0.35

    def _collect_rare_terms(self, query_tokens: list[str], candidates: list[RetrievedChunk]) -> list[str]:
        if not query_tokens or not candidates:
            return []
        candidate_docs = [chunk.content.lower() for chunk in candidates]
        corpus_size = len(candidate_docs)
        df: Counter[str] = Counter()
        for token in query_tokens:
            pattern = re.compile(rf"\b{re.escape(token.lower())}\b")
            df[token] = sum(1 for doc in candidate_docs if pattern.search(doc))

        rare_tokens = [token for token, freq in df.items() if freq <= 2]
        return rare_tokens

    def _exact_entity_boost(self, chunk: RetrievedChunk, entities: dict[str, Any]) -> float:
        if not entities["entities"]:
            return 0.0

        boost = 0.0
        lowered_content = chunk.content
        for entity in entities["entities"]:
            pattern = re.compile(rf"\b{re.escape(entity)}\b", flags=re.IGNORECASE)
            if pattern.search(lowered_content):
                boost += 0.18 if entity in entities["acronym_entities"] or entity in entities["mixed_case_entities"] else 0.10
            if chunk.section_title and pattern.search(chunk.section_title):
                boost += 0.12

        if entities.get("rare_terms"):
            for token in entities["rare_terms"]:
                if re.search(rf"\b{re.escape(token)}\b", lowered_content, flags=re.IGNORECASE):
                    boost += 0.08
        return min(0.75, boost)

    def _section_title_boost(self, chunk: RetrievedChunk, entities: dict[str, Any]) -> float:
        if not chunk.section_title or not entities["entities"]:
            return 0.0
        title = chunk.section_title
        for entity in entities["entities"]:
            if re.search(rf"\b{re.escape(entity)}\b", title, flags=re.IGNORECASE):
                return 0.22
        return 0.0

    def _diversify_by_page(self, chunks: list[RetrievedChunk], max_per_page: int = 2) -> list[RetrievedChunk]:
        page_counts: dict[int, int] = defaultdict(int)
        selected: list[RetrievedChunk] = []
        for chunk in chunks:
            page = chunk.page_number if chunk.page_number is not None else -1
            if page_counts[page] < max_per_page:
                selected.append(chunk)
                page_counts[page] += 1
            else:
                logger.info(
                    "[Suppression] Skipped chunk %s: reason=page_limit page=%s",
                    chunk.chunk_id,
                    page,
                )
        return selected

    def _build_query_features(
        self,
        query: str,
        vector_chunks: list[RetrievedChunk],
        keyword_chunks: list[RetrievedChunk] | None = None,
    ) -> dict[str, Any]:
        keyword_chunks = keyword_chunks or []
        entities = self._extract_query_entities(query)
        query_type = self._detect_query_type(query, entities)
        semantic_weight, keyword_weight = self._adaptive_weights(query_type, entities)
        rare_terms = self._collect_rare_terms(entities["tokens"], vector_chunks + keyword_chunks)
        entities["rare_terms"] = rare_terms

        logger.info(
            "[Retrieval] Detected entities=%s query_type=%s weights=> semantic=%.2f keyword=%.2f rare_terms=%s",
            entities["entities"],
            query_type,
            semantic_weight,
            keyword_weight,
            rare_terms,
        )

        return {
            "entities": entities,
            "query_type": query_type,
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight,
        }

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
        threshold = similarity_threshold or self._settings.retrieval_score_threshold

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
            query_features = self._build_query_features(query, vector_chunks, keyword_chunks)
            merged_chunks = self._merge_hybrid_results(query, vector_chunks, keyword_chunks, query_features)
            return self._strict_pipeline_result(
                query=query,
                candidates=merged_chunks,
                retrieval_threshold=threshold,
                expand_context=expand_context,
                query_features=query_features,
            )

        raw_chunks: list[RetrievedChunk] = self._store.query(
            query_text=query,
            top_k=k,
            where=filters,
        )
        query_features = self._build_query_features(query, raw_chunks)
        return self._strict_pipeline_result(
            query=query,
            candidates=raw_chunks,
            retrieval_threshold=threshold,
            expand_context=expand_context,
            query_features=query_features,
        )

    def _strict_pipeline_result(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        retrieval_threshold: float,
        expand_context: bool,
        query_features: dict[str, Any],
    ) -> RetrievalResult:
        top_candidates = candidates[: self._settings.retrieval_top_k]
        for c in top_candidates:
            c.retrieval_score = c.similarity_score

        filtered = [
            c
            for c in top_candidates
            if c.retrieval_score >= retrieval_threshold or self._contains_exact_token(c.content, query)
        ]
        if not filtered:
            return RetrievalResult(
                query=query,
                total_retrieved=len(top_candidates),
                top_score=top_candidates[0].similarity_score if top_candidates else 0.0,
                passed_threshold=False,
            )

        reranked = self._rerank(query, filtered, query_features)
        reranked = self._diversify_by_page(reranked)
        reranked = reranked[: self._settings.rerank_top_k]
        reranked = [c for c in reranked if c.rerank_score >= self._settings.rerank_score_threshold]
        if not reranked:
            return RetrievalResult(
                query=query,
                total_retrieved=len(top_candidates),
                top_score=top_candidates[0].similarity_score if top_candidates else 0.0,
                passed_threshold=False,
            )

        base_ranked = list(reranked)
        parent_sections_used: list[str] = []
        overlap_reduction_count = 0
        if expand_context and self._settings.parent_context_enabled:
            expanded, parent_sections_used, overlap_reduction_count, _ = self._expand_context(
                base_ranked,
                self._settings.parent_window_before,
                self._settings.parent_window_after,
            )
        else:
            expanded = base_ranked

        # Enforce final chunk and token limits; remove lowest-ranked first.
        final_chunks = self._apply_limits(expanded, base_ranked)
        final_tokens = self._estimate_token_count(final_chunks)
        final_chunks.sort(key=lambda c: (c.page_number or 0, c.chunk_index))
        selected_ids = [c.chunk_id for c in final_chunks]
        logger.info(
            "retrieved=%d reranked=%d expanded=%d final=%d tokens=%d",
            len(top_candidates), len(reranked), len(expanded), len(final_chunks), final_tokens,
        )
        logger.info(
            "selected chunk IDs=%s pages=%s rerank_scores=%s",
            selected_ids,
            [c.page_number for c in final_chunks],
            [round(c.rerank_score, 4) for c in final_chunks],
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            total_retrieved=len(top_candidates),
            top_score=max((c.rerank_score for c in final_chunks), default=0.0),
            passed_threshold=len(final_chunks) > 0,
            original_chunk_ids=[c.chunk_id for c in base_ranked],
            expanded_chunk_ids=[c.chunk_id for c in expanded],
            parent_sections_used=parent_sections_used,
            expanded_context_token_count=final_tokens,
            overlap_reduction_count=overlap_reduction_count,
            retrieved_chunk_count=len(top_candidates),
            reranked_chunk_count=len(reranked),
            expanded_chunk_count=len(expanded),
            final_chunk_count=len(final_chunks),
        )

    def _merge_hybrid_results(
        self,
        query: str,
        vector_chunks: list[RetrievedChunk],
        keyword_chunks: list[RetrievedChunk],
        query_features: dict[str, Any],
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
        semantic_weight = query_features.get("semantic_weight", 0.65)
        keyword_weight = query_features.get("keyword_weight", 0.35)

        for entry in candidate_map.values():
            vector_score = entry["vector_score"]
            keyword_score = entry["keyword_score"] / max_keyword_score if max_keyword_score > 0 else 0.0
            exact_bonus = self._exact_entity_boost(entry["chunk"], query_features["entities"])
            section_boost = self._section_title_boost(entry["chunk"], query_features["entities"])
            fused_score = min(
                1.0,
                vector_score * semantic_weight
                + keyword_score * keyword_weight
                + exact_bonus
                + section_boost,
            )

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
                retrieval_score=round(fused_score, 4),
                similarity_score=round(fused_score, 4),
            )
            fused_results.append(fused_chunk)

        fused_results.sort(key=lambda c: c.similarity_score, reverse=True)
        logger.info(
            "[Retrieval] Hybrid fusion computed scores for %d candidates. semantic_weight=%.2f keyword_weight=%.2f",
            len(fused_results),
            semantic_weight,
            keyword_weight,
        )
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

    def _rerank(self, query: str, chunks: list[RetrievedChunk], query_features: dict[str, Any]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._reranker.score(query, [c.content for c in chunks])
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores):
            base_score = max(0.0, min(1.0, float(score)))
            entity_boost = self._exact_entity_boost(chunk, query_features["entities"])
            section_boost = self._section_title_boost(chunk, query_features["entities"])
            combined = min(
                1.0,
                base_score * 0.62 + chunk.retrieval_score * 0.28 + entity_boost + section_boost,
            )
            chunk.rerank_score = combined
            reranked.append(chunk)
            logger.info(
                "[Rerank] Chunk %s: bm25=%.3f semantic=%.3f exact_entity_boost=%.3f section_boost=%.3f final=%.3f",
                chunk.chunk_id,
                chunk.retrieval_score,
                base_score,
                entity_boost,
                section_boost,
                combined,
            )
        reranked.sort(
            key=lambda c: (c.rerank_score, c.retrieval_score, c.similarity_score),
            reverse=True,
        )
        return reranked

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
                source_rank = max(
                    (
                        seed.rerank_score
                        for seed in retrieved_chunks
                        if seed.source_file == neighbor.source_file
                    ),
                    default=max((seed.rerank_score for seed in retrieved_chunks), default=0.0),
                )
                neighbor.rerank_score = source_rank * 0.95
                neighbor.retrieval_score = source_rank * 0.95
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
                        section_chunk.rerank_score = seed.rerank_score
                        section_chunk.retrieval_score = seed.retrieval_score
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

    def _apply_limits(
        self,
        expanded: list[RetrievedChunk],
        base_ranked: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        rank_map: dict[str, float] = {c.chunk_id: c.rerank_score for c in base_ranked}
        for chunk in expanded:
            if chunk.chunk_id not in rank_map:
                # Neighbor chunks inherit closest available ranking.
                rank_map[chunk.chunk_id] = max((c.rerank_score for c in base_ranked), default=0.0) * 0.95
                chunk.rerank_score = rank_map[chunk.chunk_id]

        kept = list(expanded)
        kept.sort(key=lambda c: rank_map.get(c.chunk_id, 0.0), reverse=True)
        kept = kept[: self._settings.final_context_chunks]

        # Drop lowest-ranked chunks first until under token budget.
        while kept and self._estimate_token_count(kept) > self._settings.max_context_tokens:
            kept.sort(key=lambda c: rank_map.get(c.chunk_id, 0.0), reverse=True)
            kept.pop()  # remove lowest rerank tail
        return kept

    def _estimate_token_count(self, chunks: list[RetrievedChunk]) -> int:
        """Approximate token count for debug observability."""
        if not chunks:
            return 0
        total_chars = sum(len(c.content) for c in chunks)
        return max(1, total_chars // 4)
