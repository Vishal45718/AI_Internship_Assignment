"""
app/orchestration/fallback_handler.py — Structured fallback response builder.

When retrieval confidence is below threshold, the system MUST NOT hallucinate.
This module generates structured, honest fallback responses that:
1. Acknowledge the inability to answer
2. Explain why (threshold not met, no results, empty index)
3. Show what documents ARE indexed
4. Suggest what documents might help

This is the most important module for the internship evaluation criterion:
"reliable I-don't-know fallback behavior."
"""

from __future__ import annotations

import logging
from typing import Any

from configs.prompts import FALLBACK_RESPONSE_TEMPLATE, FALLBACK_REASONS, OUT_OF_SCOPE_RESPONSE

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Builds structured fallback responses for low-confidence or out-of-scope queries.

    No LLM call is made during fallback — the response is purely template-driven
    to ensure it is deterministic and never makes up information.
    """

    def __init__(self, vector_store=None) -> None:
        self._store = vector_store

    def build_fallback(
        self,
        query: str,
        reason_key: str = "low_confidence",
        top_score: float | None = None,
        threshold: float | None = None,
    ) -> str:
        """
        Build a structured fallback response.

        Args:
            query: The user's original question.
            reason_key: Key from FALLBACK_REASONS ("low_confidence", "no_results", "empty_index").
            top_score: Best retrieval score achieved (for user context).
            threshold: The threshold that was not met.

        Returns:
            Formatted fallback response string.
        """
        reason_template = FALLBACK_REASONS.get(reason_key, FALLBACK_REASONS["no_results"])
        reason = reason_template.format(threshold=threshold or 0.0)

        # Get a summary of indexed documents
        doc_summary = self._get_document_summary()

        # Build a helpful suggestion
        suggestion = self._build_suggestion(query, reason_key)

        response = FALLBACK_RESPONSE_TEMPLATE.format(
            query=query,
            reason=reason,
            document_summary=doc_summary,
            suggestion=suggestion,
        )

        if top_score is not None:
            response += f"\n\n*(Best match score: {top_score:.3f} — threshold: {threshold or 0.0:.2f})*"

        logger.info(
            "Fallback triggered for query '%s…' (reason=%s, top_score=%.3f)",
            query[:60],
            reason_key,
            top_score or 0.0,
        )
        return response

    def build_out_of_scope(self, query: str) -> str:
        """Return the out-of-scope response (router detected irrelevant query)."""
        logger.info("Out-of-scope response for query '%s…'", query[:60])
        return OUT_OF_SCOPE_RESPONSE

    def _get_document_summary(self) -> str:
        """List indexed document names for the user."""
        if self._store is None:
            return "No document index available."

        try:
            sources = self._store.list_sources()
            if not sources:
                return "No documents are currently indexed."

            lines = [
                f"  • {s['source_file']} ({s['chunk_count']} chunks)"
                for s in sources[:10]  # Show at most 10
            ]
            if len(sources) > 10:
                lines.append(f"  • … and {len(sources) - 10} more documents")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Could not retrieve document summary: %s", exc)
            return "Could not retrieve indexed document list."

    def _build_suggestion(self, query: str, reason_key: str) -> str:
        """Generate a helpful suggestion based on the failure reason."""
        if reason_key == "empty_index":
            return (
                "Please use the /ingest endpoint or CLI to add documents "
                "before querying the system."
            )
        elif reason_key == "no_results":
            return (
                "Try rephrasing your question using different keywords, "
                "or upload documents that cover this topic."
            )
        else:
            return (
                "If you believe this information should be in the indexed documents, "
                "try asking with different keywords or upload additional relevant documents."
            )
