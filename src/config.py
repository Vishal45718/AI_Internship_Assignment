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

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env explicitly
load_dotenv(override=True)

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
    llm_provider: Literal["gemini", "ollama"] = Field(
        default="gemini",
        description="LLM backend to use for answer generation.",
    )

    # Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-1.5-flash-8b")
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server base URL.")
    ollama_model: str = Field(default="phi3:mini", description="Ollama model tag, e.g. phi3:mini.")

    # ── LLM generation (output token cap; keeps OpenRouter costs predictable) ──
    llm_max_output_tokens: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Max completion tokens per request.",
    )

    @field_validator("gemini_api_key", "gemini_model", "ollama_base_url", "ollama_model")
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "Settings":
        if self.llm_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER=gemini.")
            if not self.gemini_model:
                raise ValueError("GEMINI_MODEL must be set when LLM_PROVIDER=gemini.")
        elif self.llm_provider == "ollama":
            if not self.ollama_base_url:
                raise ValueError("OLLAMA_BASE_URL must be set when LLM_PROVIDER=ollama.")
            if not self.ollama_model:
                raise ValueError("OLLAMA_MODEL must be set when LLM_PROVIDER=ollama.")
        return self

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model for local embeddings.",
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "vectorstore")
    chroma_collection_name: str = Field(default="rag_documents")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=30)
    final_context_chunks: int = Field(default=2, ge=1, le=30, description="Chunks passed into sentence packing after retrieval.")
    max_context_tokens: int = Field(default=300, ge=120, le=12000, description="Estimated-token budget for system+user (+history) prompt.")
    retrieval_score_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    rerank_score_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    hybrid_search_enabled: bool = Field(
        default=False,
        description="Enable hybrid retrieval combining vector similarity and keyword/BM25 ranking.",
    )

    # ── Context Expansion ────────────────────────────────────────────────────
    parent_context_enabled: bool = Field(
        default=False,
        description="Enable Parent Document Retrieval (Small-to-Big Retrieval).",
    )
    parent_window_before: int = Field(default=1, ge=0)
    parent_window_after: int = Field(default=1, ge=0)
    parent_expand_full_section: bool = Field(
        default=False,
        description="Optionally include all chunks from a matching section title.",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=180, ge=100, le=4000, description="Fine-grained chunks for definitional sentences.")
    chunk_overlap: int = Field(default=80, ge=0, le=200)

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
    load_dotenv(override=True)
    settings = Settings()
    settings.ensure_directories()
    logger.info("Active LLM provider: %s", settings.llm_provider)
    return settings


def reload_settings() -> Settings:
    """Clear cached settings and reload from latest .env."""
    get_settings.cache_clear()
    return get_settings()
