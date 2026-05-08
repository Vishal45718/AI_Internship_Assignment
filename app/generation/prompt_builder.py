"""
app/generation/prompt_builder.py — Constructs the final LLM prompt.

Assembles retrieved chunks, conversation history, and query into
the grounded prompt that goes to the LLM. The format of context
injection is critical for citation accuracy — every chunk is
labeled with its source so the LLM can only cite what's there.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from configs.prompts import (
    RAG_SYSTEM_PROMPT,
    CONTEXT_CHUNK_TEMPLATE,
    CONVERSATION_HISTORY_TEMPLATE,
)

if TYPE_CHECKING:
    from app.models.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Max characters of context to inject (prevent context window overflow)
MAX_CONTEXT_CHARS = 6000


class PromptBuilder:
    """
    Constructs the (system_prompt, user_message) pair for the LLM.

    Design:
    - System prompt is the grounding instruction (never changes).
    - User message = conversation history + context chunks + current query.
    - Each chunk is labeled with its source file and page number so the
      LLM has explicit anchors for citation.
    """

    def build(
        self,
        query: str,
        chunks: "list[RetrievedChunk]",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        """
        Build the (system_prompt, user_message) tuple.

        Args:
            query: The user's current question.
            chunks: Retrieved and reranked document chunks.
            conversation_history: List of {"role": "...", "content": "..."} dicts.

        Returns:
            Tuple of (system_prompt_str, user_message_str).
        """
        context_str = self._format_context(chunks)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        user_parts: list[str] = []

        # Inject conversation history if provided
        if conversation_history:
            history_str = self._format_history(conversation_history)
            user_parts.append(history_str)

        user_parts.append(f"Question: {query}")
        user_message = "\n".join(user_parts)

        logger.debug(
            "Built prompt: %d chunks, context=%d chars, history=%d turns",
            len(chunks),
            len(context_str),
            len(conversation_history) if conversation_history else 0,
        )
        return system_prompt, user_message

    def _format_context(self, chunks: "list[RetrievedChunk]") -> str:
        """
        Format retrieved chunks into the labeled context block.
        Chunks are ordered by relevance score (highest first).
        """
        if not chunks:
            return "(No relevant context was retrieved from the documents.)"

        formatted_chunks: list[str] = []
        total_chars = 0

        for chunk in chunks:
            page_info = f", Page {chunk.page_number}" if chunk.page_number else ""
            chunk_str = CONTEXT_CHUNK_TEMPLATE.format(
                source_file=chunk.source_file,
                page_info=page_info,
                score=chunk.display_score,
                content=chunk.content,
            )
            if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
                logger.debug("Context budget exhausted at %d chars. Truncating.", total_chars)
                break
            formatted_chunks.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(formatted_chunks)

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """Format conversation history as a readable turn-by-turn block."""
        lines: list[str] = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")

        history_text = "\n".join(lines)
        return CONVERSATION_HISTORY_TEMPLATE.format(
            n=len(history), history=history_text
        )
