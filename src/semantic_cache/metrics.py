"""Metrics collector -- tracks latency, cost savings, and hit rate analytics.

Provides both JSON summaries for the dashboard and Prometheus-compatible
text format for scraping. No external dependencies required.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict

# -- Cost estimates per 1K tokens (USD) for paid providers -----------
_COST_PER_1K = {
    "gpt-4o":         {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":    {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":    {"input": 0.010, "output": 0.030},
    "gpt-3.5-turbo":  {"input": 0.0005, "output": 0.0015},
    "default":        {"input": 0.002, "output": 0.006},
}

_PAID_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of an LLM call."""
    # Local models (Ollama) are free
    if not any(model.startswith(p) for p in _PAID_PREFIXES):
        return 0.0
    rates = _COST_PER_1K.get(model, _COST_PER_1K["default"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1000


def _percentile(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


class MetricsCollector:
    """Thread-safe metrics collector for the proxy server.

    Tracks:
    - Request counts (total, hits, misses)
    - Latency distributions (P50, P95, P99) for hits and misses
    - Estimated cost savings from cache hits
    - Per-model breakdown
    - Similarity score distribution
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()

        # Counters
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Latency tracking (milliseconds)
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []

        # Cost savings
        self.total_savings_usd: float = 0.0
        self.tokens_saved: dict[str, int] = {"prompt": 0, "completion": 0}

        # Per-model stats
        self._model_stats: dict[str, dict] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "savings_usd": 0.0}
        )

        # Similarity scores for distribution analysis
        self._hit_scores: list[float] = []
        self._miss_scores: list[float] = []  # best score on misses (> 0)

    def record_hit(
        self,
        model: str,
        latency_ms: float,
        similarity_score: float,
        prompt_text: str,
        response_content: str,
        stored_token_usage: dict[str, int] | None = None,
    ) -> None:
        """Record a cache hit with its metrics."""
        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1
            self._hit_latencies.append(latency_ms)
            self._hit_scores.append(similarity_score)
            self._model_stats[model]["hits"] += 1

            # Estimate cost saved
            if stored_token_usage:
                pt = stored_token_usage.get("prompt_tokens", 0)
                ct = stored_token_usage.get("completion_tokens", 0)
            else:
                pt = _estimate_tokens(prompt_text)
                ct = _estimate_tokens(response_content)

            savings = _estimate_cost(model, pt, ct)
            self.total_savings_usd += savings
            self.tokens_saved["prompt"] += pt
            self.tokens_saved["completion"] += ct
            self._model_stats[model]["savings_usd"] += savings

    def record_miss(
        self,
        model: str,
        latency_ms: float,
        best_score: float = 0.0,
    ) -> None:
        """Record a cache miss."""
        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1
            self._miss_latencies.append(latency_ms)
            self._model_stats[model]["misses"] += 1
            if best_score > 0:
                self._miss_scores.append(best_score)

    def reset(self) -> None:
        """Reset all metrics (called on cache clear)."""
        with self._lock:
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self._hit_latencies.clear()
            self._miss_latencies.clear()
            self.total_savings_usd = 0.0
            self.tokens_saved = {"prompt": 0, "completion": 0}
            self._model_stats.clear()
            self._hit_scores.clear()
            self._miss_scores.clear()

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def get_summary(self) -> dict:
        """Return a comprehensive JSON-friendly metrics summary."""
        with self._lock:
            hit_lat = sorted(self._hit_latencies)
            miss_lat = sorted(self._miss_latencies)
            hit_scores = sorted(self._hit_scores)

            # Calculate latency savings
            avg_hit = sum(hit_lat) / len(hit_lat) if hit_lat else 0.0
            avg_miss = sum(miss_lat) / len(miss_lat) if miss_lat else 0.0

            return {
                "uptime_seconds": round(self.uptime_seconds, 1),
                "requests": {
                    "total": self.total_requests,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate": round(self.hit_rate, 4),
                    "hit_rate_pct": f"{self.hit_rate * 100:.1f}%",
                },
                "latency_ms": {
                    "hit": {
                        "avg": round(avg_hit, 2),
                        "p50": round(_percentile(hit_lat, 50), 2),
                        "p95": round(_percentile(hit_lat, 95), 2),
                        "p99": round(_percentile(hit_lat, 99), 2),
                        "count": len(hit_lat),
                    },
                    "miss": {
                        "avg": round(avg_miss, 2),
                        "p50": round(_percentile(miss_lat, 50), 2),
                        "p95": round(_percentile(miss_lat, 95), 2),
                        "p99": round(_percentile(miss_lat, 99), 2),
                        "count": len(miss_lat),
                    },
                    "avg_savings_ms": round(avg_miss - avg_hit, 2) if miss_lat else 0.0,
                },
                "cost_savings": {
                    "total_usd": round(self.total_savings_usd, 6),
                    "tokens_saved": dict(self.tokens_saved),
                    "total_tokens_saved": self.tokens_saved["prompt"] + self.tokens_saved["completion"],
                },
                "similarity_scores": {
                    "hit_avg": round(sum(hit_scores) / len(hit_scores), 4) if hit_scores else 0.0,
                    "hit_min": round(min(hit_scores), 4) if hit_scores else 0.0,
                    "hit_max": round(max(hit_scores), 4) if hit_scores else 0.0,
                    "hit_count": len(hit_scores),
                    "near_miss_count": len(self._miss_scores),
                },
                "per_model": {
                    model: {
                        "hits": stats["hits"],
                        "misses": stats["misses"],
                        "total": stats["hits"] + stats["misses"],
                        "hit_rate": round(
                            stats["hits"] / (stats["hits"] + stats["misses"]), 4
                        ) if (stats["hits"] + stats["misses"]) > 0 else 0.0,
                        "savings_usd": round(stats["savings_usd"], 6),
                    }
                    for model, stats in self._model_stats.items()
                },
            }

    def get_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []

        def _add(name, help_text, mtype, value):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            lines.append(f"{name} {value}")

        _add("semcache_requests_total", "Total proxy requests", "counter", self.total_requests)
        _add("semcache_cache_hits_total", "Total cache hits", "counter", self.cache_hits)
        _add("semcache_cache_misses_total", "Total cache misses", "counter", self.cache_misses)
        _add("semcache_hit_rate", "Cache hit rate (0-1)", "gauge", round(self.hit_rate, 4))
        _add("semcache_cost_savings_usd", "Estimated cost savings in USD", "counter", round(self.total_savings_usd, 6))
        _add("semcache_tokens_saved_total", "Total tokens saved", "counter", self.tokens_saved["prompt"] + self.tokens_saved["completion"])
        _add("semcache_uptime_seconds", "Server uptime in seconds", "gauge", round(self.uptime_seconds, 1))

        with self._lock:
            hit_lat = sorted(self._hit_latencies)
            miss_lat = sorted(self._miss_latencies)

        if hit_lat:
            _add("semcache_hit_latency_p50_ms", "Cache hit latency P50", "gauge", round(_percentile(hit_lat, 50), 2))
            _add("semcache_hit_latency_p95_ms", "Cache hit latency P95", "gauge", round(_percentile(hit_lat, 95), 2))
            _add("semcache_hit_latency_p99_ms", "Cache hit latency P99", "gauge", round(_percentile(hit_lat, 99), 2))

        if miss_lat:
            _add("semcache_miss_latency_p50_ms", "Cache miss latency P50", "gauge", round(_percentile(miss_lat, 50), 2))
            _add("semcache_miss_latency_p95_ms", "Cache miss latency P95", "gauge", round(_percentile(miss_lat, 95), 2))
            _add("semcache_miss_latency_p99_ms", "Cache miss latency P99", "gauge", round(_percentile(miss_lat, 99), 2))

        # Per-model metrics
        with self._lock:
            for model, stats in self._model_stats.items():
                safe_model = model.replace("-", "_").replace(".", "_")
                lines.append(f'semcache_model_hits{{model="{model}"}} {stats["hits"]}')
                lines.append(f'semcache_model_misses{{model="{model}"}} {stats["misses"]}')
                lines.append(f'semcache_model_savings_usd{{model="{model}"}} {round(stats["savings_usd"], 6)}')

        return "\n".join(lines) + "\n"