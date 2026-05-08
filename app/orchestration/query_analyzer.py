"""
app/orchestration/query_analyzer.py — Agentic query intent router.

Before retrieval, the system classifies the user's query intent.
This is the "agentic" core of the system: instead of blindly
retrieval-and-generating for every query, the system reasons about
what kind of query it is and how to answer it.

Intent categories:
  - factual_lookup  → standard RAG pipeline
  - comparison      → retrieve for each entity, synthesize comparison
  - summarize_doc   → retrieve wide coverage chunks, summarize
  - list_extraction → retrieve and extract list items
  - out_of_scope    → immediate fallback, no retrieval (saves cost)
  - unclear         → ask for clarification

This approach is genuinely "agentic" because the system decides HOW
to answer rather than always doing the same thing. (~30 lines of logic,
high ROI for demo impressiveness.)
"""

from __future__ import annotations

import logging

from configs.prompts import INTENT_CLASSIFIER_PROMPT, VALID_INTENTS

logger = logging.getLogger(__name__)


class QueryIntent:
    """Holds the classified intent with its confidence."""

    def __init__(self, intent: str, raw_response: str) -> None:
        self.intent = intent
        self.raw_response = raw_response

    @property
    def is_answerable(self) -> bool:
        """True if the system should attempt retrieval for this intent."""
        return self.intent not in ("out_of_scope", "unclear")

    @property
    def is_out_of_scope(self) -> bool:
        return self.intent == "out_of_scope"

    def __repr__(self) -> str:
        return f"QueryIntent(intent={self.intent!r})"


class QueryAnalyzer:
    """
    Classifies user query intent using a lightweight LLM call.

    A single-shot classification prompt is used (not a chain) to keep
    latency low. The LLM is instructed to return exactly one category name.
    """

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def analyze(self, query: str) -> QueryIntent:
        """
        Classify the intent of a user query.

        Args:
            query: Raw user input.

        Returns:
            QueryIntent with the classified category.
        """
        if not query.strip():
            return QueryIntent("unclear", "")

        prompt = INTENT_CLASSIFIER_PROMPT.format(query=query.strip())
        try:
            # Use a minimal system prompt — this is a classification task, not generation
            raw = self._llm.generate(
                system_prompt="You are a query classifier. Respond with only the category name.",
                user_message=prompt,
            ).strip().lower()

            # Extract the first word in case the model adds punctuation
            intent = raw.split()[0] if raw.split() else "unclear"
            intent = intent.strip(".,;:")

            if intent not in VALID_INTENTS:
                logger.warning(
                    "Unknown intent '%s' from classifier (query: '%s…'). Defaulting to factual_lookup.",
                    intent, query[:50],
                )
                intent = "factual_lookup"

            logger.info("Intent classified: '%s' for query '%s…'", intent, query[:60])
            return QueryIntent(intent, raw)

        except Exception as exc:
            # If classification fails, default to factual_lookup (graceful degradation)
            logger.warning(
                "Intent classification failed: %s. Defaulting to factual_lookup.", exc
            )
            return QueryIntent("factual_lookup", "")
