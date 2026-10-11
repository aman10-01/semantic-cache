"""Tests for Redis storage backend and fallback behavior."""

from unittest.mock import MagicMock, patch
import pytest

from src.semantic_cache.config import CacheConfig
from src.semantic_cache.cache_engine import SemanticCache
from src.semantic_cache.models import CacheEntry


def test_redis_fallback_when_server_unavailable():
    """Engine falls back to in-memory store if Redis connection fails."""
    config = CacheConfig(
        storage_backend="redis",
        redis_url="redis://localhost:9999",  # unreachable port
    )
    cache = SemanticCache(config)
    assert cache._store.__class__.__name__ == "InMemoryVectorStore"

    # Ensure store/lookup still work smoothly
    cache.store("Hello", {"content": "World"})
    res = cache.lookup("Hello")
    assert res.hit is True


def test_redis_store_mocked():
    """Verify RedisVectorStore methods with mocked redis client."""
    with patch("redis.Redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        from src.semantic_cache.redis_store import RedisVectorStore
        store = RedisVectorStore(redis_url="redis://fake:6379")

        entry = CacheEntry(
            prompt_text="test",
            prompt_embedding=[0.1, 0.2, 0.3],
            response={"content": "result"},
            cache_key="key123",
        )
        store.add(entry)
        assert mock_client.hset.called
        assert mock_client.expire.called