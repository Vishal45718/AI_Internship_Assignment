"""
src/config.py — Centralized application configuration.

All configuration is driven by environment variables (loaded from .env).
Uses Pydantic for type validation and clear error messages.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env explicitly
load_dotenv()

# Project root is one directory above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Single source of truth for all configurable values."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "gemini", "ollama", "openrouter"] = Field(
        default="openai",
        description="LLM backend to use for answer generation.",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini")

    # Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-1.5-flash-latest")

    # Ollama (local)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")

    # OpenRouter
    openrouter_api_key: str = Field(default="", description="OpenRouter API key.")
    openrouter_model: str = Field(default="google/gemini-2.0-flash-001")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def validate_openrouter_model(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("OPENROUTER_MODEL must be set when using OpenRouter.")

        normalized = value.strip()
        deprecated_map = {
            "google/gemini-1.5-flash": "google/gemini-2.0-flash-001",
            "google/gemini-1.5-flash-latest": "google/gemini-2.0-flash-001",
            "google/gemini-flash-1.5": "google/gemini-2.0-flash-001",
        }
        if normalized in deprecated_map:
            logger.warning(
                "OPENROUTER_MODEL '%s' is deprecated. Using '%s' instead.",
                normalized,
                deprecated_map[normalized],
            )
            normalized = deprecated_map[normalized]

        if " " in normalized:
            raise ValueError("OPENROUTER_MODEL must not contain spaces.")
        if "/" not in normalized:
            raise ValueError(
                "OPENROUTER_MODEL must be a valid OpenRouter model identifier, e.g. google/gemini-2.0-flash-001"
            )
        return normalized

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model for local embeddings.",
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "vectorstore")
    chroma_collection_name: str = Field(default="rag_documents")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    hybrid_search_enabled: bool = Field(
        default=False,
        description="Enable hybrid retrieval combining vector similarity and keyword/BM25 ranking.",
    )

    # ── Context Expansion ────────────────────────────────────────────────────
    parent_context_enabled: bool = Field(
        default=True,
        description="Enable Parent Document Retrieval (Small-to-Big Retrieval).",
    )
    parent_window_before: int = Field(default=1, ge=0)
    parent_window_after: int = Field(default=1, ge=0)
    parent_expand_full_section: bool = Field(
        default=False,
        description="Optionally include all chunks from a matching section title.",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")

    @field_validator("chroma_persist_dir", "data_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Ensure all path values are absolute."""
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        for d in (self.data_dir, self.chroma_persist_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton (reads .env once per process)."""
    settings = Settings()
    settings.ensure_directories()
    return settings
