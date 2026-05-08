"""
tests/unit/test_retriever.py — Unit tests for retrieval logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.retrieval.retriever import SemanticRetriever
from app.models.retrieval import RetrievedChunk, RetrievalResult


def make_chunk(source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{source}",
        content=f"Content from {source}",
        source_file=source,
        source_type="txt",
        similarity_score=score,
    )


class TestSemanticRetriever:
    def setup_method(self):
        self.mock_store = MagicMock()
        self.mock_store.count.return_value = 100

    def test_empty_query_returns_empty_result(self):
        self.mock_store.count.return_value = 100
        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("")
        assert not result.chunks
        assert not result.passed_threshold

    def test_empty_store_returns_empty_result(self):
        self.mock_store.count.return_value = 0
        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("What is the policy?")
        assert not result.chunks
        assert not result.passed_threshold

    def test_chunks_above_threshold_pass(self):
        high_score_chunks = [make_chunk("doc.txt", 0.85), make_chunk("doc2.txt", 0.72)]
        self.mock_store.query.return_value = high_score_chunks

        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("Some query", similarity_threshold=0.35)

        assert result.passed_threshold
        assert len(result.chunks) == 2

    def test_chunks_below_threshold_fail(self):
        low_score_chunks = [make_chunk("doc.txt", 0.20), make_chunk("doc2.txt", 0.15)]
        self.mock_store.query.return_value = low_score_chunks

        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("Some obscure query", similarity_threshold=0.35)

        assert not result.passed_threshold
        assert result.chunks == []  # Filtered out by threshold

    def test_top_score_is_set(self):
        chunks = [make_chunk("doc.txt", 0.90), make_chunk("doc2.txt", 0.60)]
        self.mock_store.query.return_value = chunks

        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("query")

        assert result.top_score == 0.90

    def test_retrieval_result_has_correct_total(self):
        chunks = [make_chunk(f"doc{i}.txt", 0.80) for i in range(5)]
        self.mock_store.query.return_value = chunks

        retriever = SemanticRetriever(vector_store=self.mock_store)
        result = retriever.retrieve("query")

        assert result.total_retrieved == 5
