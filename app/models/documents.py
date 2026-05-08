"""
app/models/documents.py — Core domain models for documents and chunks.

These are the canonical data structures that flow through the entire pipeline.
No business logic here — only data shape definitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "markdown"
    CSV = "csv"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "DocumentType":
        mapping = {
            ".pdf": cls.PDF,
            ".txt": cls.TXT,
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".csv": cls.CSV,
        }
        return mapping.get(ext.lower(), cls.UNKNOWN)


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""

    source_file: str = Field(..., description="Original filename.")
    source_type: DocumentType = Field(..., description="Document format.")
    file_path: str = Field(..., description="Absolute path on disk.")
    file_size_bytes: int = Field(default=0, ge=0)
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_pages: int | None = Field(default=None, description="For PDFs.")
    encoding: str = Field(default="utf-8")
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A raw loaded document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Full extracted text.")
    metadata: DocumentMetadata

    @property
    def source_file(self) -> str:
        return self.metadata.source_file


class ChunkMetadata(BaseModel):
    """Metadata attached to every individual chunk stored in the vector DB."""

    chunk_id: str = Field(..., description="Globally unique chunk identifier.")
    source_file: str = Field(..., description="Originating filename.")
    source_type: str = Field(..., description="Document format string.")
    file_path: str = Field(default="")
    page_number: int | None = Field(default=None, description="PDF page, 1-indexed.")
    chunk_index: int = Field(..., description="Position of chunk within its document, 0-indexed.")
    char_count: int = Field(default=0, ge=0)
    ingestion_timestamp: str = Field(default="")  # ISO string for ChromaDB compatibility
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_chroma_dict(self) -> dict[str, Any]:
        """
        Flatten metadata to ChromaDB-compatible format.
        ChromaDB requires all metadata values to be str, int, float, or bool.
        """
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
    """A single text chunk ready for embedding and storage."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Chunk text.")
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None, exclude=True)

    @property
    def source_file(self) -> str:
        return self.metadata.source_file

    @property
    def page_number(self) -> int | None:
        return self.metadata.page_number
