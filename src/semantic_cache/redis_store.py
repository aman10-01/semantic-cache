"""Redis-backed vector store implementation."""

from __future__ import annotations
import json
import logging
import numpy as np
from typing import Any

from .base_store import BaseVectorStore, VectorSearchResult
from .models import CacheEntry, CacheEntryMetadata

logger = logging.getLogger(__name__)

class RedisVectorStore(BaseVectorStore):
    """Stores cache entries in Redis with Cosine Similarity search."""

    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "semcache:"):
        import redis
        self._client = redis.Redis.from_url(redis_url, decode_responses=False)
        self._prefix = prefix
        self._client.ping()
        logger.info("Connected to Redis at %s", redis_url)

    def _key(self, entry_id: str) -> str:
        return f"{self._prefix}{entry_id}"

    @property
    def size(self) -> int:
        return len(self._client.keys(f"{self._prefix}*"))

    def add(self, entry: CacheEntry) -> None:
        data = {
            "entry_json": entry.model_dump_json(),
            "embedding": np.array(entry.prompt_embedding, dtype=np.float32).tobytes(),
            "cache_key": entry.cache_key,
        }
        key = self._key(entry.id)
        self._client.hset(key, mapping=data)
        if entry.metadata.ttl_seconds > 0:
            self._client.expire(key, entry.metadata.ttl_seconds)

    def _get_entries_by_key(self, cache_key: str) -> list[tuple[CacheEntry, np.ndarray]]:
        keys = self._client.keys(f"{self._prefix}*")
        results = []
        for k in keys:
            h = self._client.hgetall(k)
            if not h or h.get(b"cache_key", b"").decode() != cache_key:
                continue
            entry = CacheEntry.model_validate_json(h[b"entry_json"])
            emb = np.frombuffer(h[b"embedding"], dtype=np.float32)
            results.append((entry, emb))
        return results

    def search(self, query_embedding: list[float], cache_key: str, threshold: float = 0.85, top_k: int = 1) -> list[VectorSearchResult]:
        items = self._get_entries_by_key(cache_key)
        if not items:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scored = []
        for entry, emb in items:
            emb_norm = np.linalg.norm(emb)
            score = float(np.dot(q, emb / emb_norm)) if emb_norm > 0 else 0.0
            if score >= threshold:
                scored.append(VectorSearchResult(entry=entry, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search_best_score(self, query_embedding: list[float], cache_key: str) -> float:
        items = self._get_entries_by_key(cache_key)
        if not items:
            return 0.0
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        q = q / q_norm if q_norm > 0 else q
        best = 0.0
        for _, emb in items:
            norm = np.linalg.norm(emb)
            if norm > 0:
                best = max(best, float(np.dot(q, emb / norm)))
        return best

    def get_all_entries(self) -> list[CacheEntry]:
        keys = self._client.keys(f"{self._prefix}*")
        entries = []
        for k in keys:
            raw = self._client.hget(k, b"entry_json")
            if raw:
                entries.append(CacheEntry.model_validate_json(raw))
        return entries

    def delete(self, entry_id: str) -> bool:
        return bool(self._client.delete(self._key(entry_id)))

    def clear(self) -> None:
        keys = self._client.keys(f"{self._prefix}*")
        if keys:
            self._client.delete(*keys)