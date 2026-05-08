"""
configs/settings.py — Centralized application configuration.

All configuration is driven by environment variables (loaded from .env).
Never import raw os.environ in application code — import Settings instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is one directory above configs/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Single source of truth for all configurable values.
    Pydantic validates types and provides clear error messages on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "gemini", "ollama"] = Field(
        default="openai",
        description="LLM backend to use for generation.",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-1.5-flash")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "local"] = Field(
        default="openai",
        description="'openai' uses text-embedding-3-small; 'local' uses sentence-transformers.",
    )
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_db: Literal["chroma"] = Field(default="chroma")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "vectorstore")
    chroma_collection_name: str = Field(default="agentic_rag")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_n: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    enable_reranking: bool = Field(default=True)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1500, ge=256, le=8192)
    chunk_overlap: int = Field(default=200, ge=0, le=512)

    # ── Conversation Memory ──────────────────────────────────────────────────
    max_conversation_turns: int = Field(default=6, ge=2, le=20)

    # ── API Server ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    @field_validator("chroma_persist_dir", "data_dir", "processed_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Ensure all path values are absolute Paths."""
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for d in (self.data_dir, self.processed_dir, self.chroma_persist_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    Using lru_cache means the .env file is read exactly once per process.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
