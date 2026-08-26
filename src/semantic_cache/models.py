from __future__ import annotations
import hashlib
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# ── Cache Entry ─────────────────────────────────────────────────────
class CacheEntryMetadata(BaseModel):

    created_at: float = Field(default_factory=time.time)
    ttl_seconds: int = 86_400
    hit_count: int = 0
    last_hit_at: float | None = None

    # LLM response metadata (captured on cache miss → store)
    model_id: str | None = None
    token_usage: dict[str, int] | None = None  # prompt / completion / total
    finish_reason: str | None = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class CacheEntry(BaseModel):

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    prompt_text: str
    prompt_embedding: list[float]
    response: dict[str, Any]  # full LLM response body
    cache_key: str  # composite key = hash(system_prompt + params)
    metadata: CacheEntryMetadata = Field(default_factory=CacheEntryMetadata)
    original_prompt: str = ""  # raw text before any normalisation


# ── Lookup Result ───────────────────────────────────────────────────
class CacheLookupResult(BaseModel):

    hit: bool
    response: dict[str, Any] | None = None
    similarity_score: float = 0.0
    entry_id: str | None = None


# ── Helpers ─────────────────────────────────────────────────────────
def build_cache_key(
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **extra_params: Any,
) -> str:
    parts: list[str] = []
    if system_prompt is not None:
        parts.append(f"sys:{system_prompt}")
    if model is not None:
        parts.append(f"model:{model}")
    if temperature is not None:
        parts.append(f"temp:{temperature}")
    for k, v in sorted(extra_params.items()):
        parts.append(f"{k}:{v}")
    raw = "|".join(parts) if parts else "__default__"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
