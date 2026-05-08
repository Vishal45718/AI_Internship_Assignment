"""
app/orchestration/rag_agent.py — Top-level RAG pipeline orchestrator.

This is the single entry point for query processing. It coordinates
all subsystems in the correct order:

  1. Intent analysis (agentic routing)
  2. Semantic retrieval (Stage 1)
  3. Cross-encoder reranking (Stage 2)
  4. Confidence threshold check → fallback if below threshold
  5. Prompt assembly with source citations
  6. LLM generation
  7. Response validation (numeric claim check)
  8. Conversation memory update

The agent is designed for clean testability: all dependencies are
injected via constructor, never imported globally.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.generation.prompt_builder import PromptBuilder
from app.generation.response_validator import ResponseValidator
from app.models.responses import QueryResponse, ResponseStatus
from app.models.retrieval import RetrievedChunk
from app.orchestration.fallback_handler import FallbackHandler
from app.orchestration.memory import MemoryStore
from app.orchestration.query_analyzer import QueryAnalyzer
from app.retrieval.reranker import create_reranker
from configs.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """All dependencies the RAGAgent needs — injected at construction time."""
    llm_client: Any
    embedder: Any
    vector_store: Any
    retriever: Any
    reranker: Any


class RAGAgent:
    """
    Orchestrates the full agentic RAG pipeline.

    Usage:
        agent = RAGAgent(deps)
        response = agent.query("What is the return policy?", session_id="user-123")
    """

    def __init__(self, deps: AgentDependencies) -> None:
        self._llm = deps.llm_client
        self._store = deps.vector_store
        self._retriever = deps.retriever
        self._default_reranker = deps.reranker
        self._reranker_factory = create_reranker
        self._prompt_builder = PromptBuilder()
        self._validator = ResponseValidator()
        self._fallback = FallbackHandler(vector_store=deps.vector_store)
        self._analyzer = QueryAnalyzer(llm_client=deps.llm_client)
        self._memory = MemoryStore()
        self._settings = get_settings()

    def query(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        enable_reranking: bool | None = None,
    ) -> QueryResponse:
        """
        Process a user query end-to-end through the RAG pipeline.

        Args:
            query: User's natural language question.
            session_id: Optional conversation session ID for memory.
            top_k: Override default retrieval top-K.
            filters: Optional metadata filters.
            enable_reranking: Override global reranking setting.

        Returns:
            QueryResponse with answer, sources, and status.
        """
        start_time = time.perf_counter()

        # ── Step 0: Input validation ──────────────────────────────────────────
        query = query.strip()
        if not query:
            return QueryResponse(
                query=query,
                answer="Please provide a non-empty question.",
                status=ResponseStatus.ERROR,
            )

        # ── Step 1: Check for empty index ─────────────────────────────────────
        if self._store.count() == 0:
            fallback_answer = self._fallback.build_fallback(
                query=query,
                reason_key="empty_index",
            )
            return QueryResponse(
                query=query,
                answer=fallback_answer,
                status=ResponseStatus.FALLBACK,
                intent="factual_lookup",
            )

        # ── Step 2: Intent analysis (agentic routing) ─────────────────────────
        intent = self._analyzer.analyze(query)
        logger.info("Query intent: %s", intent.intent)

        if intent.is_out_of_scope:
            fallback_answer = self._fallback.build_out_of_scope(query)
            return QueryResponse(
                query=query,
                answer=fallback_answer,
                status=ResponseStatus.OUT_OF_SCOPE,
                intent=intent.intent,
                latency_ms=self._elapsed_ms(start_time),
            )

        # ── Step 3: Retrieve ──────────────────────────────────────────────────
        retrieval_result = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        # ── Step 4: Rerank ────────────────────────────────────────────────────
        # Apply the reranker (either cross-encoder or no-op)
        # The reranker also applies the confidence threshold check
        reranker = (
            self._reranker_factory(enabled=enable_reranking)
            if enable_reranking is not None
            else self._default_reranker
        )
        final_result = reranker.rerank(query, retrieval_result)

        # ── Step 5: Fallback check ────────────────────────────────────────────
        if not final_result.passed_threshold:
            reason_key = "no_results" if final_result.total_retrieved == 0 else "low_confidence"
            fallback_answer = self._fallback.build_fallback(
                query=query,
                reason_key=reason_key,
                top_score=final_result.top_score,
                threshold=self._settings.confidence_threshold,
            )
            return QueryResponse(
                query=query,
                answer=fallback_answer,
                status=ResponseStatus.FALLBACK,
                intent=intent.intent,
                confidence=final_result.top_score,
                retrieval_strategy=final_result.retrieval_strategy,
                latency_ms=self._elapsed_ms(start_time),
            )

        # ── Step 6: Build prompt ──────────────────────────────────────────────
        conversation_history: list[dict] | None = None
        memory: Any = None
        if session_id:
            memory = self._memory.get_or_create(session_id)
            memory.add_user(query)
            conversation_history = memory.get_history()

        system_prompt, user_message = self._prompt_builder.build(
            query=query,
            chunks=final_result.chunks,
            conversation_history=conversation_history,
        )

        # ── Step 7: Generate ──────────────────────────────────────────────────
        try:
            answer = self._llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return QueryResponse(
                query=query,
                answer=f"An error occurred during response generation: {exc}",
                status=ResponseStatus.ERROR,
                intent=intent.intent,
                latency_ms=self._elapsed_ms(start_time),
            )

        # ── Step 8: Validate ──────────────────────────────────────────────────
        validation = self._validator.validate(
            response=answer,
            context_chunks=final_result.chunks,
            query=query,
        )
        if validation.has_warnings:
            for w in validation.warnings:
                logger.warning("Validation warning: %s", w)

        # ── Step 9: Update memory ─────────────────────────────────────────────
        if memory is not None:
            memory.add_assistant(answer)

        # ── Step 10: Build and return response ───────────────────────────────
        latency = self._elapsed_ms(start_time)
        logger.info(
            "Query completed in %.0fms | intent=%s | chunks=%d | top_score=%.3f",
            latency,
            intent.intent,
            len(final_result.chunks),
            final_result.top_score,
        )

        return QueryResponse.from_chunks(
            query=query,
            answer=answer,
            status=ResponseStatus.SUCCESS,
            chunks=final_result.chunks,
            intent=intent.intent,
            confidence=round(final_result.top_score, 4),
            retrieval_strategy=final_result.retrieval_strategy,
            latency_ms=latency,
        )

    def clear_session(self, session_id: str) -> bool:
        """Clear the conversation memory for a session."""
        return self._memory.delete(session_id)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 1)
