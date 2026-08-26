"""Ollama provider — connects to a local Ollama instance (100% free)."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Calls a locally running Ollama server.

    Ollama exposes an OpenAI-compatible endpoint at
    ``/v1/chat/completions``, but also has its own native
    ``/api/chat`` endpoint.  We use the native one for
    better control over streaming.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,  # local models can be slow on first load
        )

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            finish_reason="stop",
            usage={
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval_count + eval_count,
            },
        )

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        async with self._client.stream(
            "POST", "/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
