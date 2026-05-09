"""
src/config.py — Centralized application configuration.

All configuration is driven by environment variables (loaded from .env).
Uses Pydantic for type validation and clear error messages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

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
    llm_provider: Literal["openai", "gemini", "ollama"] = Field(
        default="openai",
        description="LLM backend to use for answer generation.",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini")

    # Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-1.5-flash")

    # Ollama (local)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")

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
