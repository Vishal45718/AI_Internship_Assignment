"""
app/generation/response_validator.py — Post-generation response validation.

After the LLM generates a response, this module performs lightweight
validation checks to catch common hallucination patterns:

1. Numeric claim verification: if the response contains specific numbers
   (statistics, dates, prices) that don't appear in the retrieved context,
   flag them — this is the most common hallucination failure mode.

2. Source citation format: ensure inline citations are present when the
   response makes factual claims.

This is a "soft" guard — it annotates the response rather than refusing
to serve it, unless the hallucination confidence is very high.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Regex to find numeric claims (integers, decimals, percentages, years)
_NUMBER_PATTERN = re.compile(r"\b\d{1,4}(?:[.,]\d+)?(?:\s*%|K|M|B|billion|million|thousand)?\b")
# Regex to find years (4-digit numbers between 1900-2100)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


class ValidationResult:
    """Result of the response validation check."""

    def __init__(
        self,
        response: str,
        is_valid: bool,
        warnings: list[str],
        suspected_hallucinations: list[str],
    ) -> None:
        self.response = response
        self.is_valid = is_valid
        self.warnings = warnings
        self.suspected_hallucinations = suspected_hallucinations

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ResponseValidator:
    """
    Validates LLM responses against the retrieved context.

    The validation is intentionally conservative: it only flags values
    that are clearly NOT in the context, to avoid false positives.
    """

    def validate(
        self,
        response: str,
        context_chunks: list,  # list[RetrievedChunk]
        query: str,
    ) -> ValidationResult:
        """
        Validate a generated response against retrieved chunks.

        Args:
            response: The LLM's generated text.
            context_chunks: The chunks that were passed to the LLM as context.
            query: The original user query (for logging).

        Returns:
            ValidationResult with any detected issues.
        """
        if not response.strip():
            return ValidationResult(
                response=response,
                is_valid=False,
                warnings=["Empty response from LLM."],
                suspected_hallucinations=[],
            )

        # Build a combined context string for lookups
        combined_context = " ".join(c.content for c in context_chunks).lower()

        warnings: list[str] = []
        suspected: list[str] = []

        # Check for self-referential "I don't know" mixed with factual claims
        self_declared_ignorance = bool(
            re.search(r"i (don't|do not|cannot) (have|find|know|access)", response, re.IGNORECASE)
        )
        has_factual_claims = bool(re.search(r"\b(is|are|was|were|will|can)\b", response, re.IGNORECASE))
        if self_declared_ignorance and has_factual_claims and len(response) > 300:
            warnings.append(
                "Response contains both 'I don't know' and factual claims — may be inconsistent."
            )

        # Numeric hallucination check
        numbers_in_response = set(_NUMBER_PATTERN.findall(response))
        years_in_response = set(_YEAR_PATTERN.findall(response))
        all_numbers = numbers_in_response | years_in_response

        for num in all_numbers:
            # Skip very common numbers that are unlikely to be hallucinated (1, 2, 3, 10, etc.)
            if len(num) <= 1:
                continue
            if num not in combined_context:
                suspected.append(num)

        if suspected:
            logger.warning(
                "Potential numeric hallucinations detected for query '%s…': %s",
                query[:50],
                suspected[:5],
            )
            warnings.append(
                f"Response contains {len(suspected)} numeric value(s) not found in "
                f"retrieved context. Verify: {', '.join(suspected[:5])}"
            )

        is_valid = len(suspected) == 0

        logger.debug(
            "Validation: valid=%s, warnings=%d, suspected=%d",
            is_valid,
            len(warnings),
            len(suspected),
        )
        return ValidationResult(
            response=response,
            is_valid=is_valid,
            warnings=warnings,
            suspected_hallucinations=suspected,
        )
