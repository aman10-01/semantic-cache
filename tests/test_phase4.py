"""Phase 4 tests -- metrics collector, cost estimation, Prometheus export."""

import pytest
from src.semantic_cache.metrics import (
    MetricsCollector,
    _estimate_tokens,
    _estimate_cost,
    _percentile,
)


# -- Helper function tests -------------------------------------------

class TestHelpers:

    def test_estimate_tokens(self):
        assert _estimate_tokens("Hello world") >= 1
        assert _estimate_tokens("a" * 400) == 100

    def test_estimate_cost_openai(self):
        cost = _estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0

    def test_estimate_cost_ollama_is_free(self):
        cost = _estimate_cost("llama3.2", prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_estimate_cost_tinyllama_is_free(self):
        cost = _estimate_cost("tinyllama", prompt_tokens=5000, completion_tokens=2000)
        assert cost == 0.0

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([5.0], 50) == 5.0

    def test_percentile_median(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_percentile_p99(self):
        vals = list(range(1, 101))
        p99 = _percentile([float(v) for v in vals], 99)
        assert p99 >= 99.0


# -- MetricsCollector tests ------------------------------------------

class TestMetricsCollector:

    def test_initial_state(self):
        m = MetricsCollector()
        assert m.total_requests == 0
        assert m.cache_hits == 0
        assert m.cache_misses == 0
        assert m.hit_rate == 0.0
        assert m.total_savings_usd == 0.0

    def test_record_hit(self):
        m = MetricsCollector()
        m.record_hit(
            model="gpt-4o",
            latency_ms=3.5,
            similarity_score=0.96,
            prompt_text="What is Python?",
            response_content="Python is a programming language.",
        )
        assert m.total_requests == 1
        assert m.cache_hits == 1
        assert m.cache_misses == 0
        assert m.hit_rate == 1.0
        assert m.total_savings_usd > 0  # gpt-4o is paid

    def test_record_hit_ollama_no_cost(self):
        m = MetricsCollector()
        m.record_hit(
            model="llama3.2",
            latency_ms=2.0,
            similarity_score=0.97,
            prompt_text="Hello",
            response_content="Hi there!",
        )
        assert m.cache_hits == 1
        assert m.total_savings_usd == 0.0  # ollama is free

    def test_record_miss(self):
        m = MetricsCollector()
        m.record_miss(model="gpt-4o", latency_ms=1500.0, best_score=0.85)
        assert m.total_requests == 1
        assert m.cache_misses == 1
        assert m.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "response")
        m.record_miss("gpt-4o", 1500.0)
        m.record_hit("gpt-4o", 2.5, 0.98, "test2", "response2")
        assert m.hit_rate == pytest.approx(2 / 3, abs=0.01)

    def test_reset(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "response")
        m.record_miss("gpt-4o", 1500.0)
        m.reset()
        assert m.total_requests == 0
        assert m.cache_hits == 0
        assert m.total_savings_usd == 0.0

    def test_per_model_tracking(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "resp")
        m.record_miss("llama3.2", 500.0)
        m.record_hit("llama3.2", 2.0, 0.97, "test2", "resp2")

        summary = m.get_summary()
        assert "gpt-4o" in summary["per_model"]
        assert "llama3.2" in summary["per_model"]
        assert summary["per_model"]["gpt-4o"]["hits"] == 1
        assert summary["per_model"]["llama3.2"]["hits"] == 1
        assert summary["per_model"]["llama3.2"]["misses"] == 1


# -- Summary output tests -------------------------------------------

class TestMetricsSummary:

    def test_summary_structure(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "response")
        s = m.get_summary()

        assert "uptime_seconds" in s
        assert "requests" in s
        assert "latency_ms" in s
        assert "cost_savings" in s
        assert "similarity_scores" in s
        assert "per_model" in s

        assert s["requests"]["total"] == 1
        assert s["requests"]["cache_hits"] == 1
        assert s["latency_ms"]["hit"]["count"] == 1

    def test_latency_percentiles(self):
        m = MetricsCollector()
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]:
            m.record_hit("gpt-4o", lat, 0.96, "t", "r")

        s = m.get_summary()
        assert s["latency_ms"]["hit"]["p50"] > 0
        assert s["latency_ms"]["hit"]["p95"] > s["latency_ms"]["hit"]["p50"]
        assert s["latency_ms"]["hit"]["p99"] >= s["latency_ms"]["hit"]["p95"]


# -- Prometheus export tests -----------------------------------------

class TestPrometheusExport:

    def test_prometheus_format(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "response")
        prom = m.get_prometheus()

        assert "semcache_requests_total 1" in prom
        assert "semcache_cache_hits_total 1" in prom
        assert "# HELP" in prom
        assert "# TYPE" in prom

    def test_prometheus_per_model(self):
        m = MetricsCollector()
        m.record_hit("gpt-4o", 3.0, 0.96, "test", "response")
        m.record_miss("llama3.2", 500.0)
        prom = m.get_prometheus()

        assert 'model="gpt-4o"' in prom
        assert 'model="llama3.2"' in prom

    def test_prometheus_empty(self):
        m = MetricsCollector()
        prom = m.get_prometheus()
        assert "semcache_requests_total 0" in prom