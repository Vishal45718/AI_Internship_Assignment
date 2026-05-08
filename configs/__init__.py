"""configs/__init__.py"""
from configs.settings import Settings, get_settings
from configs.prompts import (
    RAG_SYSTEM_PROMPT,
    FALLBACK_RESPONSE_TEMPLATE,
    FALLBACK_REASONS,
    INTENT_CLASSIFIER_PROMPT,
    VALID_INTENTS,
    QUERY_EXPANSION_PROMPT,
    CONTEXT_CHUNK_TEMPLATE,
    OUT_OF_SCOPE_RESPONSE,
)

__all__ = [
    "Settings",
    "get_settings",
    "RAG_SYSTEM_PROMPT",
    "FALLBACK_RESPONSE_TEMPLATE",
    "FALLBACK_REASONS",
    "INTENT_CLASSIFIER_PROMPT",
    "VALID_INTENTS",
    "QUERY_EXPANSION_PROMPT",
    "CONTEXT_CHUNK_TEMPLATE",
    "OUT_OF_SCOPE_RESPONSE",
]
