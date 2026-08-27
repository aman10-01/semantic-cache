# Semantic Caching Layer for LLM APIs

> A drop-in caching layer that reduces LLM API costs by up to 90% and P95 latency from seconds to single-digit milliseconds on a 2,000-request load test.

## The Problem

Every company running LLMs at scale has the same problem: **redundant API calls burning money and adding latency.** In real-world applications, 30-60% of LLM queries are semantically identical to previous ones.

Traditional caches fail because "What is Python?" and "Explain Python to me" are different strings but the same question.

## The Solution

This middleware sits between your application and any LLM provider. It uses **semantic embeddings** to understand the *meaning* of prompts, not just the characters. Switch to it by changing one URL -- zero code changes needed.

Your App --> Semantic Cache Proxy --> Cache Check | HIT (< 5ms, $0) MISS (1-10s, costs tokens) Return instantly Forward to LLM, store, return



## Load Test Results
Total Requests: 2,000 Cache Hit Rate: ~90%

Latency (cache hits): P50: ~3ms P95: ~5ms P99: ~8ms

Projected Monthly Savings (100K requests/day, GPT-4o): Without cache: ~
2,250/monthWithcache : 225/month 
SAVINGS: ~$2,025/month

> Run `python load_test.py` to reproduce these numbers on your machine.

## Features

| Feature | Description |
|---|---|
| **Drop-In Proxy** | Mirrors the OpenAI `/v1/chat/completions` API exactly. Change the base URL, done. |
| **Semantic Matching** | Uses `all-MiniLM-L6-v2` embeddings (free, local, no API key) for meaning-based similarity |
| **Multi-Provider** | Routes to OpenAI or Ollama (local, free) based on model name |
| **TTL Auto-Classification** | Factual queries cached 24h, time-sensitive 1h, creative skipped |
| **Adaptive Thresholds** | Auto-adjusts similarity threshold per task type (classification=0.90, creative=0.98) |
| **Threshold Tuner** | Visualizes hit rate vs accuracy tradeoff at different thresholds |
| **Cache Invalidation** | Selective invalidation by system prompt, model upgrade, or prefix |
| **Live Dashboard** | Built-in HTML dashboard with real-time charts (no Grafana needed) |
| **Prometheus Metrics** | `/metrics` endpoint for production monitoring |
| **Streaming Support** | Simultaneously streams to client while buffering for cache storage |
| **Docker Ready** | Multi-stage Dockerfile + docker-compose with Prometheus and Ollama |

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd semantic-cache
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the load test (no server needed)
python load_test.py

# 4. Start the proxy server
python run_proxy.py

# 5. Open the dashboard
# Visit http://localhost:8000/v1/dashboard

Docker Deployment
# Start everything (proxy + Prometheus + Ollama)
docker-compose up -d

# Access services:
#   Proxy:       http://localhost:8000
#   Dashboard:   http://localhost:8000/v1/dashboard
#   Prometheus:  http://localhost:9090

Architecture
semantic-cache/
├── src/semantic_cache/
│   ├── config.py              # Pydantic settings with env overrides
│   ├── embeddings.py          # Local embedding engine (sentence-transformers)
│   ├── models.py              # CacheEntry, CacheLookupResult data models
│   ├── cache_store.py         # In-memory vector store (NumPy cosine similarity)
│   ├── cache_engine.py        # Core cache logic + TTL + adaptive thresholds
│   ├── api_models.py          # OpenAI-compatible request/response schemas
│   ├── proxy.py               # FastAPI server with all endpoints
│   ├── metrics.py             # Latency, cost, and per-model analytics
│   ├── dashboard.py           # Self-contained HTML dashboard
│   ├── ttl_classifier.py      # Prompt-based TTL tier classification
│   ├── threshold_tuner.py     # Historical threshold analysis engine
│   └── providers/
│       ├── __init__.py        # Provider router (auto-detect from model name)
│       ├── base.py            # Abstract LLM provider interface
│       ├── openai_provider.py # OpenAI API connector
│       └── ollama_provider.py # Ollama (local, free) connector
├── tests/
│   ├── test_cache.py          # Phase 1 tests (23 tests)
│   ├── test_phase3.py         # Phase 3 tests (19 tests)
│   └── test_phase4.py         # Phase 4 tests (22 tests)
├── load_test.py               # 2,000-request load test benchmark
├── run_proxy.py               # Server entry point
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Full stack deployment
└── prometheus.yml             # Prometheus scrape config

Tech Stack
Component	        Choice	                                    Why
Language	        Python 3.13	                                ML ecosystem , async support
Embeddings	        sentence-transformers (all-MiniLM-L6-v2)	Free, local, 384-dim, fast on CPU
Vector Store	    NumPy (in-memory)	                        Zero infrastructure, swappable for Redis/Qdrant
Proxy	            FastAPI + Uvicorn	                        Async, auto-docs, OpenAI-compatible
HTTP Client	        httpx	                                    Async HTTP with streaming SSE support
LLM Providers	    OpenAI + Ollama	                            Paid (production) + Free (development)
Config	            Pydantic Settings	                        Type-safe with env var overrides
Monitoring	        Built-in + Prometheus	                    Dashboard + standard metrics export
Containers	        Docker + docker-compose	                    Production-ready deployment

How It Works
Embed the incoming prompt using all-MiniLM-L6-v2 (free, local)
Search the vector store for semantically similar cached prompts (same model/temperature/system-prompt partition)
HIT (similarity >= threshold): Return cached response in < 5ms with X-Cache: HIT header
MISS (below threshold): Forward to LLM provider, cache the response, return with X-Cache: MISS
TTL Auto-Classify: Factual=24h, Standard=6h, Volatile=1h, Creative=skip
Adaptive Threshold: Classification tasks use 0.90, QA uses 0.93, creative uses 0.98

License
MIT
---
