"""
app/core/llm_client.py — LLM provider abstraction with retry logic.

Supports three providers:
  - "openai"  → OpenAI Chat Completions API
  - "gemini"  → Google Gemini API
  - "ollama"  → Local Ollama server

All providers expose the same generate(system_prompt, user_message) interface.
Swapping providers requires changing only the EMBEDDING_PROVIDER env var.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from configs.settings import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries (exponential backoff applied)


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class BaseLLMClient(ABC):
    """Interface contract for all LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: The grounding/instruction prompt.
            user_message: The user query with injected context.

        Returns:
            The model's text response.

        Raises:
            LLMError: If generation fails after retries.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging and API responses."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions with exponential backoff retry."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI, RateLimitError, APIError
            self._RateLimitError = RateLimitError
            self._APIError = APIError
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        logger.info("OpenAIClient ready: model=%s", self._model)

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
                    temperature=0.1,      # Low temperature = more grounded, less creative
                    max_tokens=2048,
                )
                return response.choices[0].message.content or ""
            except self._RateLimitError as exc:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate limit hit (attempt %d/%d). Waiting %.1fs…",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = exc
            except Exception as exc:
                logger.error("OpenAI call failed (attempt %d): %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

        raise LLMError(f"OpenAI generation failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini provider
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self) -> None:
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self._genai.configure(api_key=settings.gemini_api_key)
        self._model_name = settings.gemini_model
        self._model = self._genai.GenerativeModel(self._model_name)
        logger.info("GeminiClient ready: model=%s", self._model_name)

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
                logger.error("Gemini call failed (attempt %d): %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

        raise LLMError(f"Gemini generation failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama local provider
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client via its REST API.
    Requires Ollama to be running: https://ollama.ai
    """

    def __init__(self) -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model_id = settings.ollama_model
        logger.info(
            "OllamaClient ready: model=%s url=%s", self._model_id, self._base_url
        )

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
                    resp = client.post(
                        f"{self._base_url}/api/chat", json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("message", {}).get("content", "")
            except Exception as exc:
                logger.error("Ollama call failed (attempt %d): %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

        raise LLMError(f"Ollama generation failed after {_MAX_RETRIES} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_client() -> BaseLLMClient:
    """
    Factory function that instantiates the configured LLM provider.
    Call once at startup; inject as a dependency everywhere else.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    elif provider == "ollama":
        return OllamaClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Valid options: 'openai', 'gemini', 'ollama'."
        )
