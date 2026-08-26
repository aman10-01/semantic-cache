"""LLM provider routing — picks the right backend based on model name."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from ..config import CacheConfig
from .base import LLMProvider, ProviderResponse
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# Models that map to each provider
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_OLLAMA_MODELS = (
    "llama", "mistral", "gemma", "phi", "qwen", "deepseek",
    "codellama", "vicuna", "neural-chat", "tinyllama",
)


class ProviderRouter:
    """Routes requests to the correct LLM provider based on model name.

    Auto-detects the provider from the model string:
    - gpt-*, o1-*, o3-*, o4-*  →  OpenAI
    - Everything else           →  Ollama (local)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._providers: dict[str, LLMProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Lazily initialise available providers."""
        # Ollama is always available (local, free)
        self._providers["ollama"] = OllamaProvider(
            base_url=self._config.ollama_base_url
        )
        logger.info("Registered provider: ollama (%s)", self._config.ollama_base_url)

        # OpenAI only if an API key is configured
        if self._config.openai_api_key:
            self._providers["openai"] = OpenAIProvider(
                api_key=self._config.openai_api_key,
                base_url=self._config.openai_base_url,
            )
            logger.info("Registered provider: openai")
        else:
            logger.info("OpenAI provider skipped (no API key set)")

    def _detect_provider(self, model: str) -> str:
        """Guess the provider from the model name."""
        model_lower = model.lower()
        if any(model_lower.startswith(p) for p in _OPENAI_PREFIXES):
            return "openai"
        return self._config.default_provider

    def get_provider(self, model: str) -> LLMProvider:
        """Return the correct provider for the given model."""
        name = self._detect_provider(model)
        if name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"Provider '{name}' for model '{model}' is not configured. "
                f"Available: {available}"
            )
        return self._providers[name]

    @property
    def available_providers(self) -> list[str]:
        return list(self._providers.keys())
