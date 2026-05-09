"""
src/llm/client.py — LLM provider abstraction with retry logic.

Supports three providers:
  - "openai"  → OpenAI Chat Completions API
  - "gemini"  → Google Gemini API
  - "ollama"  → Local Ollama server (free, runs on your machine)

All providers expose the same generate(system_prompt, user_message) interface.
Swap providers by changing LLM_PROVIDER in your .env file.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from src.config import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class BaseLLMClient(ABC):
    """Interface for all LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a response from the LLM."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions with retry logic."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        logger.info("OpenAI client ready: model=%s", self._model)

    def generate(self, system_prompt: str, user_message: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,  # Low = more grounded, less creative
                    max_tokens=2048,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.error("OpenAI attempt %d failed: %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)
        raise LLMError(f"OpenAI failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self) -> None:
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")

        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")

        self._genai.configure(api_key=settings.gemini_api_key)
        self._model_name = settings.gemini_model
        self._model = self._genai.GenerativeModel(self._model_name)
        logger.info("Gemini client ready: model=%s", self._model_name)

    def generate(self, system_prompt: str, user_message: str) -> str:
        combined = f"{system_prompt}\n\nUser question: {user_message}"
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(
                    combined,
                    generation_config={"temperature": 0.1, "max_output_tokens": 2048},
                )
                return response.text or ""
            except Exception as exc:
                logger.error("Gemini attempt %d failed: %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
        raise LLMError(f"Gemini failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama (local, free)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """Ollama local LLM client. Requires Ollama running: https://ollama.ai"""

    def __init__(self) -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model_id = settings.ollama_model
        logger.info("Ollama client ready: model=%s url=%s", self._model_id, self._base_url)

    def generate(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with self._httpx.Client(timeout=120.0) as client:
                    resp = client.post(f"{self._base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("message", {}).get("content", "")
            except Exception as exc:
                logger.error("Ollama attempt %d failed: %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
        raise LLMError(f"Ollama failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_client() -> BaseLLMClient:
    """Create the configured LLM provider based on .env settings."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    elif provider == "ollama":
        return OllamaClient()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use: openai, gemini, or ollama")
