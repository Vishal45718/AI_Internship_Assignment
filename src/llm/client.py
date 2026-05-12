"""
src/llm/client.py — LLM provider abstraction with retry logic.

Supports one provider:
  - "gemini"  → Google Gemini API
  - "ollama"  → Local Ollama API

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
_MODEL_FALLBACK_CHAIN = ("gemini-1.5-flash-8b", "gemini-1.5-flash")


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class BaseLLMClient(ABC):
    """Interface for all LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, max_tokens: int | None = None) -> str:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    def stream(self, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, max_tokens: int | None = None):
        """Stream a response from the LLM. Yields text chunks."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ─────────────────────────────────────────────────────────────────────────────


def _extract_model_name(raw_name: str) -> str:
    # list_models often returns names like "models/gemini-1.5-flash"
    return raw_name.replace("models/", "").strip()


def list_available_gemini_models(api_key: str) -> list[str]:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai") from exc

    genai.configure(api_key=api_key)
    names: list[str] = []
    for model in genai.list_models():
        name = _extract_model_name(getattr(model, "name", ""))
        if name:
            names.append(name)
    return sorted(set(names))


def resolve_gemini_model_or_raise(
    requested_model: str,
    available_models: list[str],
    fallback_chain: tuple[str, ...] = _MODEL_FALLBACK_CHAIN,
) -> str:
    requested = _extract_model_name(requested_model)
    if requested in available_models:
        return requested

    for fallback in fallback_chain:
        if fallback in available_models:
            logger.warning(
                "Configured GEMINI_MODEL '%s' unavailable; falling back to '%s'.",
                requested,
                fallback,
            )
            return fallback

    preview = ", ".join(available_models[:15]) if available_models else "(none)"
    raise RuntimeError(
        f"GEMINI_MODEL '{requested}' is unavailable. Available models: {preview}"
    )


def validate_gemini_startup_or_raise(api_key: str, configured_model: str) -> tuple[str, list[str]]:
    available = list_available_gemini_models(api_key)
    resolved = resolve_gemini_model_or_raise(configured_model, available)
    return resolved, available


def validate_ollama_startup_or_raise(base_url: str, configured_model: str) -> tuple[str, list[str]]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx not installed. Run: pip install httpx") from exc

    normalized_url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{normalized_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Unable to reach Ollama at {normalized_url}: {exc}") from exc

    models = data.get("models", []) if isinstance(data, dict) else []
    available = sorted(
        {
            (m.get("name") or "").strip()
            for m in models
            if isinstance(m, dict) and (m.get("name") or "").strip()
        }
    )
    if configured_model in available:
        return configured_model, available
    preview = ", ".join(available[:20]) if available else "(none)"
    raise RuntimeError(
        f"OLLAMA_MODEL '{configured_model}' is unavailable on {normalized_url}. Available models: {preview}"
    )

class GeminiClient(BaseLLMClient):
    """Google Gemini API client using google.generativeai SDK."""

    def __init__(self) -> None:
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai") from exc

        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")

        self._genai.configure(api_key=settings.gemini_api_key)
        available = list_available_gemini_models(settings.gemini_api_key)
        self._model_name = resolve_gemini_model_or_raise(settings.gemini_model, available)
        self._model = self._genai.GenerativeModel(self._model_name)
        self._max_output_tokens = settings.llm_max_output_tokens
        logger.info(
            "Gemini client ready: model=%s max_output_tokens=%s",
            self._model_name,
            self._max_output_tokens,
        )

    def generate(self, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, max_tokens: int | None = None) -> str:
        combined = f"{system_prompt}\n\nUser question: {user_message}"
        last_error: Exception | None = None
        
        # Use provided max_tokens or fall back to config
        output_tokens = max_tokens if max_tokens is not None else self._max_output_tokens
        
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(
                    combined,
                    generation_config=self._genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=output_tokens,
                    ),
                )
                return response.text or ""
            except Exception as exc:
                err_str = str(exc).lower()
                if "api_key" in err_str or "401" in err_str:
                    logger.error("Authentication Error: Invalid Gemini API Key.")
                    raise LLMError(f"Authentication Error: Invalid API Key. Please check your .env file.") from exc
                logger.error("Gemini attempt %d failed: %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
        raise LLMError(f"Gemini failed after {_MAX_RETRIES} attempts: {last_error}")

    def stream(self, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, max_tokens: int | None = None):
        combined = f"{system_prompt}\n\nUser question: {user_message}"
        
        # Use provided max_tokens or fall back to config
        output_tokens = max_tokens if max_tokens is not None else self._max_output_tokens
        
        try:
            response = self._model.generate_content(
                combined,
                stream=True,
                generation_config=self._genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=output_tokens,
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            err_str = str(exc).lower()
            if "api_key" in err_str or "401" in err_str:
                logger.error("Authentication Error: Invalid Gemini API Key.")
                raise LLMError(f"Authentication Error: Invalid API Key. Please check your .env file.") from exc
            logger.error("Gemini streaming failed: %s", exc)
            raise LLMError(f"Gemini streaming failed: {exc}")

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model_name}"


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client."""

    def __init__(self) -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError as exc:
            raise RuntimeError("httpx not installed. Run: pip install httpx") from exc

        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model_id = settings.ollama_model
        self._max_output_tokens = settings.llm_max_output_tokens
        logger.info(
            "Ollama client ready: model=%s base_url=%s max_output_tokens=%s",
            self._model_id,
            self._base_url,
            self._max_output_tokens,
        )

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        output_tokens = max_tokens if max_tokens is not None else self._max_output_tokens
        payload = {
            "model": self._model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": output_tokens,
            },
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

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ):
        import json

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        output_tokens = max_tokens if max_tokens is not None else self._max_output_tokens
        payload = {
            "model": self._model_id,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": output_tokens,
            },
        }
        try:
            with self._httpx.Client(timeout=120.0) as client:
                with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
        except Exception as exc:
            logger.error("Ollama streaming failed: %s", exc)
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

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

    if provider == "gemini":
        logger.info("Initializing provider=gemini model=%s", settings.gemini_model)
        return GeminiClient()
    if provider == "ollama":
        logger.info("Initializing provider=ollama model=%s", settings.ollama_model)
        return OllamaClient()
    else:
        raise ValueError(f"Invalid LLM_PROVIDER '{provider}'. Supported values: gemini, ollama")
