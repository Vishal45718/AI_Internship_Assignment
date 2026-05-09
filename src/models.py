"""
src/models.py — Data models for the RAG pipeline.

All data structures that flow through the pipeline are defined here.
Uses Pydantic for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Document Types
# ─────────────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "markdown"
    CSV = "csv"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "DocumentType":
        """Map file extension to DocumentType."""
        mapping = {
            ".pdf": cls.PDF,
            ".txt": cls.TXT,
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".csv": cls.CSV,
        }
        return mapping.get(ext.lower(), cls.UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────────
# Document Models
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""
    source_file: str = Field(..., description="Original filename.")
    source_type: DocumentType = Field(..., description="Document format.")
    file_path: str = Field(..., description="Absolute path on disk.")
    file_size_bytes: int = Field(default=0, ge=0)
    total_pages: int | None = Field(default=None, description="For PDFs.")
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A raw loaded document before chunking."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Full extracted text.")
    metadata: DocumentMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Chunk Models
# ─────────────────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata for a single chunk stored in the vector database."""
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    source_file: str = Field(..., description="Originating filename.")
    source_type: str = Field(..., description="Document format string.")
    file_path: str = Field(default="")
    page_number: int | None = Field(default=None, description="PDF page (1-indexed).")
    chunk_index: int = Field(..., description="Position within document (0-indexed).")
    char_count: int = Field(default=0, ge=0)
    ingestion_timestamp: str = Field(default="")

    def to_chroma_dict(self) -> dict[str, Any]:
        """Flatten metadata for ChromaDB (values must be str/int/float/bool)."""
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "ingestion_timestamp": self.ingestion_timestamp,
        }
        if self.page_number is not None:
            d["page_number"] = self.page_number
        return d


class DocumentChunk(BaseModel):
    """A text chunk ready for embedding and storage."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Chunk text.")
    metadata: ChunkMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Models
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with its similarity score."""
    chunk_id: str
    content: str
    source_file: str
    source_type: str
    page_number: int | None = None
    chunk_index: int = 0
    similarity_score: float = Field(ge=0.0, le=1.0)

    @property
    def source_citation(self) -> str:
        """Human-readable citation string."""
        page = f", Page {self.page_number}" if self.page_number else ""
        return f"[Source: {self.source_file}{page}]"

    @property
    def preview(self) -> str:
        """First 200 chars for display."""
        return self.content[:200] + ("…" if len(self.content) > 200 else "")


class RetrievalResult(BaseModel):
    """Aggregated output of the retrieval stage."""
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_retrieved: int = 0
    top_score: float = 0.0
    passed_threshold: bool = False
