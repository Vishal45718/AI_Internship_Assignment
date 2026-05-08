"""
app/models/retrieval.py — Models for retrieval results and scored chunks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with its similarity score."""

    chunk_id: str
    content: str
    source_file: str
    source_type: str
    page_number: int | None = None
    chunk_index: int = 0
    similarity_score: float = Field(ge=0.0, le=1.0)
    rerank_score: float | None = None

    @property
    def display_score(self) -> float:
        """The final relevance score after optional reranking."""
        return self.rerank_score if self.rerank_score is not None else self.similarity_score

    @property
    def source_citation(self) -> str:
        """Human-readable citation string for this chunk."""
        page = f", Page {self.page_number}" if self.page_number else ""
        return f"[Source: {self.source_file}{page}]"

    @property
    def preview(self) -> str:
        """First 200 chars for UI display."""
        return self.content[:200] + ("…" if len(self.content) > 200 else "")


class RetrievalResult(BaseModel):
    """Aggregated output of the retrieval + reranking stage."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_retrieved: int = 0
    top_score: float = 0.0
    retrieval_strategy: str = "semantic"  # "semantic" | "hybrid" | "reranked"
    passed_threshold: bool = False

    @classmethod
    def empty(cls, query: str, reason: str = "no_results") -> "RetrievalResult":
        return cls(query=query, chunks=[], total_retrieved=0, top_score=0.0, passed_threshold=False)
