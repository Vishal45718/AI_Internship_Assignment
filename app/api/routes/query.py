"""
app/api/routes/query.py — Query endpoint.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_rag_agent
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description=(
        "Submit a natural language question. The system retrieves relevant document chunks, "
        "applies reranking, and generates a grounded answer with source citations. "
        "Returns a structured fallback if confidence is below threshold."
    ),
    responses={
        200: {"model": QueryResponse},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_endpoint(
    request: QueryRequest,
    agent=Depends(get_rag_agent),
) -> QueryResponse:
    """
    Process a RAG query.

    - **query**: Natural language question
    - **session_id**: Optional — enables conversation memory across requests
    - **top_k**: Override number of chunks to retrieve
    - **filters**: Metadata filters (e.g., `{"source_type": "pdf"}`)
    """
    try:
        response = agent.query(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            filters=request.filters,
            enable_reranking=request.enable_reranking,
        )
        return response
    except Exception as exc:
        logger.exception("Unhandled error in /query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
