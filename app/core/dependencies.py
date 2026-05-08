"""
app/core/dependencies.py — Application-level dependency container.

Provides singleton instances of all major services, initialized lazily.
FastAPI's dependency injection system calls these functions to get
the shared instances throughout the application lifetime.

All services are initialized once and reused — this avoids repeated
model loading (which is expensive for sentence-transformers).
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Service singletons (initialized lazily on first access) ───────────────────

@lru_cache(maxsize=1)
def get_embedder():
    """Return the cached embedder singleton."""
    from app.core.embedder import create_embedder
    logger.info("Initialising embedder…")
    return create_embedder()


@lru_cache(maxsize=1)
def get_vector_store():
    """Return the cached vector store singleton."""
    from app.core.vector_store import ChromaVectorStore
    logger.info("Initialising vector store…")
    return ChromaVectorStore(embedder=get_embedder())


@lru_cache(maxsize=1)
def get_llm_client():
    """Return the cached LLM client singleton."""
    from app.core.llm_client import create_llm_client
    logger.info("Initialising LLM client…")
    return create_llm_client()


@lru_cache(maxsize=1)
def get_retriever():
    """Return the cached retriever singleton."""
    from app.retrieval.retriever import SemanticRetriever
    return SemanticRetriever(vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_reranker():
    """Return the cached reranker singleton."""
    from app.retrieval.reranker import create_reranker
    logger.info("Initialising reranker…")
    return create_reranker()


@lru_cache(maxsize=1)
def get_rag_agent():
    """Return the cached RAG agent singleton."""
    from app.orchestration.rag_agent import RAGAgent, AgentDependencies
    logger.info("Assembling RAG agent…")
    deps = AgentDependencies(
        llm_client=get_llm_client(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        retriever=get_retriever(),
        reranker=get_reranker(),
    )
    return RAGAgent(deps)


@lru_cache(maxsize=1)
def get_ingestion_pipeline():
    """Return the cached ingestion pipeline singleton."""
    from app.ingestion.pipeline import IngestionPipeline
    logger.info("Initialising ingestion pipeline…")
    return IngestionPipeline(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
    )


def clear_all_caches() -> None:
    """
    Clear all singleton caches. Useful for testing or after config changes.
    """
    for fn in [
        get_embedder, get_vector_store, get_llm_client,
        get_retriever, get_reranker, get_rag_agent, get_ingestion_pipeline,
    ]:
        fn.cache_clear()
    logger.info("All dependency caches cleared.")
