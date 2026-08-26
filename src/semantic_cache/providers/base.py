"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class ProviderResponse:
    """Standardised response from any LLM provider.

    All providers must convert their native response format
    into this structure so the proxy layer doesn't care which
    provider was used.
    """

    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """Contract that every LLM provider must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abc.abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        """Send a non-streaming chat completion request."""

    @abc.abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield content chunks as they arrive from the provider."""
