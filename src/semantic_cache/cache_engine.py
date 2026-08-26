"""Core Semantic Cache engine — the main entry-point for Phase 1.

Ties together the embedding engine, vector store, and data models
into a clean public API:

    cache = SemanticCache()
    result = cache.lookup(prompt="What is Python?", system_prompt="You are a tutor")
    if result.hit:
        print(result.response)
    else:
        llm_response = call_your_llm(...)
        cache.store(
            prompt="What is Python?",
            response=llm_response,
            system_prompt="You are a tutor",
        )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .cache_store import InMemoryVectorStore
from .config import CacheConfig
from .embeddings import EmbeddingEngine
from .models import (
    CacheEntry,
    CacheEntryMetadata,
    CacheLookupResult,
    build_cache_key,
)

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic caching layer for LLM API responses.

    Parameters
    ----------
    config:
        Optional ``CacheConfig`` instance.  When omitted, default values
        (or env-var overrides) are used.

    Example
    -------
    >>> cache = SemanticCache()
    >>> result = cache.lookup("Explain Python to me")
    >>> result.hit
    False
    >>> cache.store("Explain Python to me", {"content": "Python is …"})
    >>> result = cache.lookup("What is Python?")  # semantically similar
    >>> result.hit
    True
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()
        self._embedder = EmbeddingEngine(self._config.embedding_model)
        self._store = InMemoryVectorStore(
            dimension=self._embedder.dimension,
            max_entries=self._config.max_cache_entries,
        )
        logger.info(
            "SemanticCache ready — model=%s, threshold=%.2f, max_entries=%d",
            self._config.embedding_model,
            self._config.similarity_threshold,
            self._config.max_cache_entries,
        )

    # ── Lookup ──────────────────────────────────────────────────────
    def lookup(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        threshold: float | None = None,
        **extra_params: Any,
    ) -> CacheLookupResult:
        """Search the cache for a semantically similar prompt.

        Parameters
        ----------
        prompt:
            The user's query text.
        system_prompt, model, temperature, extra_params:
            Generation context — two identical prompts with different
            contexts will **not** share a cache entry.
        threshold:
            Override the global similarity threshold for this single
            lookup (useful for the threshold-tuning endpoint later).

        Returns
        -------
        CacheLookupResult
            ``.hit`` is True when a sufficiently similar cached response
            was found.  ``.response`` contains the cached LLM response
            body, and ``.similarity_score`` shows how close the match was.
        """
        t0 = time.perf_counter()

        cache_key = build_cache_key(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            **extra_params,
        )
        query_embedding = self._embedder.encode(prompt)
        effective_threshold = threshold if threshold is not None else self._config.similarity_threshold

        matches = self._store.search(
            query_embedding=query_embedding,
            cache_key=cache_key,
            threshold=effective_threshold,
            top_k=1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if matches:
            best = matches[0]
            # Update hit stats
            best.entry.metadata.hit_count += 1
            best.entry.metadata.last_hit_at = time.time()
            logger.info(
                "CACHE HIT  — score=%.4f  id=%s  (%.1f ms)",
                best.score,
                best.entry.id,
                elapsed_ms,
            )
            return CacheLookupResult(
                hit=True,
                response=best.entry.response,
                similarity_score=best.score,
                entry_id=best.entry.id,
            )

        logger.info("CACHE MISS — (%.1f ms)", elapsed_ms)
        return CacheLookupResult(hit=False)

    # ── Store ───────────────────────────────────────────────────────
    def store(
        self,
        prompt: str,
        response: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        ttl_seconds: int | None = None,
        model_id: str | None = None,
        token_usage: dict[str, int] | None = None,
        finish_reason: str | None = None,
        **extra_params: Any,
    ) -> str:
        """Store a new LLM response in the cache after a cache miss.

        Returns the unique ID of the newly created cache entry.
        """
        cache_key = build_cache_key(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            **extra_params,
        )
        embedding = self._embedder.encode(prompt)

        entry = CacheEntry(
            prompt_text=prompt,
            prompt_embedding=embedding,
            response=response,
            cache_key=cache_key,
            original_prompt=prompt,
            metadata=CacheEntryMetadata(
                ttl_seconds=ttl_seconds or self._config.default_ttl_seconds,
                model_id=model_id,
                token_usage=token_usage,
                finish_reason=finish_reason,
            ),
        )
        self._store.add(entry)
        logger.info("Stored entry %s  (key=%s)", entry.id, cache_key)
        return entry.id

    # ── Stats ───────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of cache statistics."""
        entries = self._store.get_all_entries()
        total_hits = sum(e.metadata.hit_count for e in entries)
        expired = sum(1 for e in entries if e.metadata.is_expired)
        return {
            "total_entries": self._store.size,
            "total_hits": total_hits,
            "expired_entries": expired,
            "active_entries": self._store.size - expired,
            "similarity_threshold": self._config.similarity_threshold,
            "embedding_model": self._config.embedding_model,
            "max_entries": self._config.max_cache_entries,
        }

    # ── Maintenance ─────────────────────────────────────────────────
    def clear(self) -> None:
        """Wipe all cache entries."""
        self._store.clear()
        logger.info("Cache cleared")

    def delete(self, entry_id: str) -> bool:
        """Remove a single entry by its ID."""
        return self._store.delete(entry_id)
