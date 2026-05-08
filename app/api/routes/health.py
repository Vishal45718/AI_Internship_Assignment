"""
app/api/routes/health.py — Health check endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.dependencies import get_vector_store, get_llm_client, get_embedder
from app.models.responses import HealthResponse
from configs.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns system status and configuration overview.",
)
async def health_check(
    store=Depends(get_vector_store),
    llm=Depends(get_llm_client),
    embedder=Depends(get_embedder),
) -> HealthResponse:
    """
    Health check endpoint. Returns:
    - Current LLM and embedding provider names
    - Vector DB type and chunk count
    - System version

    Use this to verify the system is correctly configured before ingesting documents.
    """
    settings = get_settings()

    try:
        chunk_count = store.count()
        sources = store.list_sources()
        doc_count = len(sources)
    except Exception:
        chunk_count = -1
        doc_count = -1

    return HealthResponse(
        status="healthy",
        llm_provider=getattr(llm, "model_name", settings.llm_provider),
        embedding_provider=settings.embedding_provider,
        vector_db=settings.vector_db,
        indexed_documents=doc_count,
        indexed_chunks=chunk_count,
    )
