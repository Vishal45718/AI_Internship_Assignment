"""
app/models/requests.py — Pydantic models for incoming API requests.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question.")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override default top-k.")
    session_id: str | None = Field(default=None, description="Conversation session ID for memory.")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters, e.g. {'source_type': 'pdf'}.",
    )
    enable_reranking: bool | None = Field(
        default=None,
        description="Override global reranking setting for this request.",
    )


class IngestRequest(BaseModel):
    """Request body for POST /ingest (URL-based ingestion)."""

    file_paths: list[str] = Field(..., min_length=1, description="List of absolute file paths to ingest.")
    collection_name: str | None = Field(default=None, description="Override default collection.")


class DeleteRequest(BaseModel):
    """Request body for DELETE /index/document."""

    source_file: str = Field(..., description="Filename to remove from the index.")
