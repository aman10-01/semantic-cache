"""Core Semantic Cache engine -- the main entry-point.

Ties together the embedding engine, vector store, TTL classifier,
threshold tuner, and data models into a clean public API.
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
from .ttl_classifier import classify_prompt, TTLTier
from .threshold_tuner import ThresholdTuner, LookupRecord

logger = logging.getLogger(__name__)

# ── Adaptive threshold defaults ─────────────────────────────────────
_ADAPTIVE_THRESHOLDS: dict[str, float] = {
    "classification": 0.90,  # constrained answer space, lower is OK
    "extraction": 0.92,      # structured output
    "qa": 0.93,              # question answering
    "general": 0.95,         # default
    "creative": 0.98,        # needs very high match or skip
}

_CLASSIFICATION_KEYWORDS = [
    "classify", "categorize", "label", "sentiment", "positive or negative",
    "is it", "yes or no", "true or false",
]

_CREATIVE_KEYWORDS = [
    "write a poem", "write a story", "creative", "imagine",
    "brainstorm", "generate ideas", "come up with",
]

_EXTRACTION_KEYWORDS = [
    "extract", "parse", "find all", "list the", "json",
    "structured", "entities", "named entity",
]


def _detect_task_type(prompt: str) -> str:
    """Detect the task type from the prompt for adaptive thresholds."""
    prompt_lower = prompt.lower()
    for kw in _CREATIVE_KEYWORDS:
        if kw in prompt_lower:
            return "creative"
    for kw in _CLASSIFICATION_KEYWORDS:
        if kw in prompt_lower:
            return "classification"
    for kw in _EXTRACTION_KEYWORDS:
        if kw in prompt_lower:
            return "extraction"
    if "?" in prompt:
        return "qa"
    return "general"


class SemanticCache:
    """Semantic caching layer for LLM API responses.

    Parameters
    ----------
    config:
        Optional ``CacheConfig`` instance.  When omitted, default values
        (or env-var overrides) are used.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()
        self._embedder = EmbeddingEngine(self._config.embedding_model)
        self._store = InMemoryVectorStore(
            dimension=self._embedder.dimension,
            max_entries=self._config.max_cache_entries,
        )
        self._tuner = ThresholdTuner()
        logger.info(
            "SemanticCache ready -- model=%s, threshold=%.2f, max_entries=%d",
            self._config.embedding_model,
            self._config.similarity_threshold,
            self._config.max_cache_entries,
        )

    # -- Lookup ------------------------------------------------------
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

        Uses adaptive thresholds when no explicit threshold is provided.
        Records every lookup for the threshold tuner.
        """
        t0 = time.perf_counter()

        cache_key = build_cache_key(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            **extra_params,
        )
        query_embedding = self._embedder.encode(prompt)

        # Determine effective threshold
        if threshold is not None:
            effective_threshold = threshold
        elif self._config.enable_adaptive_threshold:
            task_type = _detect_task_type(prompt)
            effective_threshold = _ADAPTIVE_THRESHOLDS.get(
                task_type, self._config.similarity_threshold
            )
            logger.debug(
                "Adaptive threshold: task=%s, threshold=%.2f",
                task_type, effective_threshold,
            )
        else:
            effective_threshold = self._config.similarity_threshold

        # Get the best similarity score (for tuner, regardless of threshold)
        best_score = self._store.search_best_score(query_embedding, cache_key)

        matches = self._store.search(
            query_embedding=query_embedding,
            cache_key=cache_key,
            threshold=effective_threshold,
            top_k=1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_hit = bool(matches)

        # Record for threshold tuner
        self._tuner.record(LookupRecord(
            prompt=prompt,
            best_similarity=best_score,
            was_hit=is_hit,
            active_threshold=effective_threshold,
            cache_key=cache_key,
        ))

        if matches:
            best = matches[0]
            best.entry.metadata.hit_count += 1
            best.entry.metadata.last_hit_at = time.time()
            logger.info(
                "CACHE HIT  -- score=%.4f  id=%s  (%.1f ms)",
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

        logger.info("CACHE MISS -- (%.1f ms)", elapsed_ms)
        return CacheLookupResult(hit=False, similarity_score=best_score)

    # -- Store -------------------------------------------------------
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

        If ttl_seconds is not provided, auto-classifies the prompt
        to determine the appropriate TTL tier.
        """
        # Auto-classify TTL if not explicitly provided
        if ttl_seconds is None:
            tier, auto_ttl = classify_prompt(prompt)
            if tier == TTLTier.SKIP:
                logger.info("TTL classifier says SKIP -- not caching: %r", prompt[:60])
                return ""  # Don't cache creative prompts
            ttl_seconds = auto_ttl
            logger.debug("Auto-TTL: tier=%s, ttl=%ds", tier.value, ttl_seconds)

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
                ttl_seconds=ttl_seconds,
                model_id=model_id,
                token_usage=token_usage,
                finish_reason=finish_reason,
            ),
        )
        self._store.add(entry)
        logger.info("Stored entry %s  (key=%s, ttl=%ds)", entry.id, cache_key, ttl_seconds)
        return entry.id

    # -- Invalidation (Phase 3) --------------------------------------
    def invalidate_by_system_prompt(
        self,
        system_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        **extra_params: Any,
    ) -> int:
        """Invalidate all entries for a given system prompt."""
        cache_key = build_cache_key(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            **extra_params,
        )
        count = self._store.invalidate_by_cache_key(cache_key)
        logger.info("Invalidated %d entries for system_prompt hash=%s", count, cache_key)
        return count

    def invalidate_by_model(self, model_id: str) -> int:
        """Invalidate all entries for a specific model (e.g., after upgrade)."""
        count = self._store.invalidate_by_model(model_id)
        logger.info("Invalidated %d entries for model=%s", count, model_id)
        return count

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate entries whose prompt starts with the given prefix."""
        count = self._store.invalidate_by_prefix(prefix)
        logger.info("Invalidated %d entries with prefix=%r", count, prefix)
        return count

    def purge_expired(self) -> int:
        """Remove all expired entries from the store."""
        count = self._store.purge_expired()
        logger.info("Purged %d expired entries", count)
        return count

    # -- Threshold Tuner (Phase 3) -----------------------------------
    def get_threshold_analysis(
        self, thresholds: list[float] | None = None
    ) -> dict:
        """Run threshold analysis on lookup history."""
        return self._tuner.analyse(thresholds)

    def get_near_misses(self, limit: int = 20) -> list[dict]:
        """Get prompts that narrowly missed the threshold."""
        return self._tuner.get_near_misses(
            self._config.similarity_threshold, limit
        )

    # -- Stats -------------------------------------------------------
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
            "adaptive_threshold": self._config.enable_adaptive_threshold,
            "embedding_model": self._config.embedding_model,
            "max_entries": self._config.max_cache_entries,
            "tuner_history_size": self._tuner.history_size,
        }

    # -- Maintenance -------------------------------------------------
    def clear(self) -> None:
        """Wipe all cache entries."""
        self._store.clear()
        logger.info("Cache cleared")

    def delete(self, entry_id: str) -> bool:
        """Remove a single entry by its ID."""
        return self._store.delete(entry_id)