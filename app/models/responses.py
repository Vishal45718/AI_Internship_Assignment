"""
app/models/responses.py — API response schemas.

All API responses follow the same envelope structure to make client code
predictable and error handling straightforward.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.retrieval import RetrievedChunk


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"          # answered but with low confidence
    OUT_OF_SCOPE = "out_of_scope"  # router detected irrelevant query
    ERROR = "error"


class SourceReference(BaseModel):
    """Compact source info for the API response."""

    source_file: str
    page_number: int | None = None
    chunk_index: int = 0
    relevance_score: float
    preview: str


class QueryResponse(BaseModel):
    """Standard response envelope for /query endpoint."""

    query: str
    answer: str
    status: ResponseStatus
    sources: list[SourceReference] = Field(default_factory=list)
    intent: str | None = None
    confidence: float | None = None
    retrieval_strategy: str | None = None
    latency_ms: float | None = None

    @classmethod
    def from_chunks(
        cls,
        query: str,
        answer: str,
        status: ResponseStatus,
        chunks: list[RetrievedChunk],
        **kwargs: Any,
    ) -> "QueryResponse":
        sources = [
            SourceReference(
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                relevance_score=round(c.display_score, 4),
                preview=c.preview,
            )
            for c in chunks
        ]
        return cls(query=query, answer=answer, status=status, sources=sources, **kwargs)


class IngestResponse(BaseModel):
    """Response for /ingest endpoint."""

    status: str
    files_processed: int
    chunks_created: int
    failed_files: list[str] = Field(default_factory=list)
    message: str


class HealthResponse(BaseModel):
    """Response for /health endpoint."""

    status: str
    llm_provider: str
    embedding_provider: str
    vector_db: str
    indexed_documents: int
    indexed_chunks: int
    version: str = "1.0.0"


class IndexStatsResponse(BaseModel):
    """Response for /index/stats endpoint."""

    total_chunks: int
    total_documents: int
    documents: list[dict[str, Any]] = Field(default_factory=list)
    collection_name: str
    persist_dir: str


class ErrorResponse(BaseModel):
    """Standardized error response."""

    status: str = "error"
    error: str
    detail: str | None = None
