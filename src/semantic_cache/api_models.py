"""OpenAI-compatible request/response models.

These Pydantic models mirror the OpenAI Chat Completions API
so any application can switch to our proxy by just changing the
base URL — zero code changes needed.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────
class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Mirrors POST /v1/chat/completions request body."""

    model: str = "gpt-4o"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    def get_system_prompt(self) -> str | None:
        """Extract the system prompt from the messages list."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return None

    def get_user_prompt(self) -> str:
        """Extract the last user message (the actual query)."""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    def get_generation_params(self) -> dict[str, Any]:
        """Collect params that affect generation output."""
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop"] = str(self.stop)
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        return params


# ── Response Models ─────────────────────────────────────────────────
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """Mirrors the OpenAI chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "gpt-4o"
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Streaming Response Models ───────────────────────────────────────
class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk during streaming."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "gpt-4o"
    choices: list[StreamChoice]


# ── Helper Builders ─────────────────────────────────────────────────
def build_response(
    content: str,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> ChatCompletionResponse:
    """Build a complete ChatCompletionResponse from content text."""
    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChoiceMessage(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(**(usage or {})),
    )
