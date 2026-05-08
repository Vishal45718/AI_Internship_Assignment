"""
tests/unit/test_fallback_handler.py — Tests for the fallback response system.

These are the most critical tests in the suite: they verify that the
hallucination prevention system works correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.orchestration.fallback_handler import FallbackHandler


class TestFallbackHandler:
    """Tests for the FallbackHandler — the hallucination prevention module."""

    def setup_method(self):
        """Create a handler with a mock vector store."""
        mock_store = MagicMock()
        mock_store.list_sources.return_value = [
            {"source_file": "doc1.pdf", "chunk_count": 42},
            {"source_file": "doc2.txt", "chunk_count": 18},
        ]
        self.handler = FallbackHandler(vector_store=mock_store)
        self.handler_no_store = FallbackHandler(vector_store=None)

    def test_fallback_contains_original_query(self):
        """The fallback response must include the user's query."""
        query = "What is the capital of France?"
        response = self.handler.build_fallback(query)
        assert query in response

    def test_low_confidence_fallback_mentions_threshold(self):
        """Low-confidence fallback should explain the threshold issue."""
        response = self.handler.build_fallback(
            "Some query",
            reason_key="low_confidence",
            top_score=0.22,
            threshold=0.40,
        )
        # Should contain the score
        assert "0.22" in response or "40%" in response

    def test_empty_index_fallback_message(self):
        """Empty index fallback should instruct user to ingest documents."""
        response = self.handler.build_fallback("My query", reason_key="empty_index")
        assert "ingest" in response.lower() or "upload" in response.lower()

    def test_no_results_fallback_message(self):
        """No-results fallback should suggest rephrasing or uploading docs."""
        response = self.handler.build_fallback("Obscure query", reason_key="no_results")
        assert len(response) > 50  # Should have a substantial message

    def test_document_list_included_in_fallback(self):
        """Fallback response should list currently indexed documents."""
        response = self.handler.build_fallback("Some query")
        assert "doc1.pdf" in response

    def test_fallback_with_no_store_does_not_crash(self):
        """Handler without a vector store should still produce a valid response."""
        response = self.handler_no_store.build_fallback("My query")
        assert len(response) > 10

    def test_out_of_scope_response_is_not_empty(self):
        """Out-of-scope response should be a non-empty string."""
        response = self.handler.build_out_of_scope("What is 2+2?")
        assert isinstance(response, str)
        assert len(response) > 20

    def test_fallback_never_contains_fabricated_facts(self):
        """Fallback response must not contain specific factual claims."""
        queries = [
            "What is the revenue of Apple Inc?",
            "Who is the president of the USA?",
            "What happened on September 11?",
        ]
        for query in queries:
            response = self.handler.build_fallback(query)
            # The response should NOT confidently state facts
            # It should be entirely template-driven
            assert "I could not find" in response or "don't have" in response.lower() or "insufficient" in response.lower()
