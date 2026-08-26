"""Start the Semantic Cache proxy server.

Usage:
    python run_proxy.py

The server starts at http://localhost:8000 by default.
Configure via environment variables (SEMCACHE_ prefix).
"""

import logging
import uvicorn

from src.semantic_cache.config import CacheConfig
from src.semantic_cache.proxy import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

config = CacheConfig()
app = create_app(config)

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Semantic Cache Proxy - Phase 2")
    print("=" * 60)
    print(f"  Server:     http://localhost:{config.proxy_port}")
    print(f"  OpenAI URL: http://localhost:{config.proxy_port}/v1")
    print(f"  Providers:  {config.default_provider}")
    print(f"  Threshold:  {config.similarity_threshold}")
    print("=" * 60)
    print()

    uvicorn.run(
        "run_proxy:app",
        host=config.proxy_host,
        port=config.proxy_port,
    )
