"""FastAPI proxy server -- drop-in replacement for the OpenAI API.

Any application can switch to this proxy by changing its base URL
from  https://api.openai.com/v1  to  http://localhost:8000/v1
with zero code changes.  The proxy transparently checks the
semantic cache before forwarding requests to the real LLM.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .api_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaContent,
    StreamChoice,
    build_response,
)
from .cache_engine import SemanticCache
from .config import CacheConfig
from .dashboard import render_dashboard_html
from .metrics import MetricsCollector
from .providers import ProviderRouter

logger = logging.getLogger(__name__)


# -- Request models for Phase 3 endpoints ----------------------------
class InvalidateRequest(BaseModel):
    """Request body for cache invalidation."""
    system_prompt: str | None = None
    model: str | None = None
    prefix: str | None = None


class ThresholdAnalysisRequest(BaseModel):
    """Query params for threshold analysis."""
    thresholds: list[float] | None = None


# -- App Factory -----------------------------------------------------
def create_app(config: CacheConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    config = config or CacheConfig()

    app = FastAPI(
        title="Semantic Cache Proxy",
        description="Drop-in caching proxy for LLM APIs",
        version="0.4.0",
    )

    # CORS -- allow any origin so frontends can call the proxy
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state -- attached to the app instance
    app.state.config = config
    app.state.cache = SemanticCache(config)
    app.state.router = ProviderRouter(config)
    app.state.metrics = MetricsCollector()

    # -- Routes ------------------------------------------------------
    @app.get("/")
    async def root():
        m = app.state.metrics
        return {
            "service": "Semantic Cache Proxy",
            "version": "0.4.0",
            "status": "running",
            "providers": app.state.router.available_providers,
            "features": [
                "ttl_auto_classification",
                "cache_invalidation",
                "threshold_tuner",
                "adaptive_thresholds",
                "prometheus_metrics",
                "live_dashboard",
            ],
            "quick_stats": {
                "total_requests": m.total_requests,
                "hit_rate": f"{m.hit_rate * 100:.1f}%",
                "cost_savings_usd": round(m.total_savings_usd, 4),
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models():
        """Fake models endpoint so clients don't break."""
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "llama3.2", "object": "model"},
            ],
        }

    # -- Main Endpoint -----------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """The main endpoint -- mirrors OpenAI's chat completions."""
        t0 = time.perf_counter()
        metrics = app.state.metrics

        user_prompt = request.get_user_prompt()
        system_prompt = request.get_system_prompt()
        gen_params = request.get_generation_params()

        # -- Step 1: Check cache -------------------------------------
        cache_result = app.state.cache.lookup(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=request.model,
            temperature=request.temperature,
            **gen_params,
        )

        if cache_result.hit:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response_content = cache_result.response.get("content", "")

            # Record metrics
            metrics.record_hit(
                model=request.model,
                latency_ms=elapsed_ms,
                similarity_score=cache_result.similarity_score,
                prompt_text=user_prompt,
                response_content=response_content,
            )
            logger.info(
                "CACHE HIT -- served in %.1fms (score=%.4f)",
                elapsed_ms,
                cache_result.similarity_score,
            )

            if request.stream:
                return _stream_cached_response(
                    cache_result.response, request.model
                )

            response = _format_cached_response(
                cache_result.response, request.model
            )
            return JSONResponse(
                content=response.model_dump(),
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Score": f"{cache_result.similarity_score:.4f}",
                    "X-Cache-Latency-Ms": f"{elapsed_ms:.1f}",
                },
            )

        # -- Step 2: Cache miss -- forward to LLM provider -----------
        logger.info("CACHE MISS -- forwarding to provider")

        provider = app.state.router.get_provider(request.model)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        if request.stream:
            return _stream_from_provider(
                app=app,
                provider=provider,
                model=request.model,
                messages=messages,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                gen_params=gen_params,
                t0=t0,
            )

        # Non-streaming miss
        try:
            provider_resp = await provider.complete(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            logger.error("Provider error: %s", e)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.record_miss(
                model=request.model,
                latency_ms=elapsed_ms,
                best_score=cache_result.similarity_score,
            )
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Provider error: {e}", "type": "proxy_error"}},
            )

        # Store in cache for future hits
        app.state.cache.store(
            prompt=user_prompt,
            response={"content": provider_resp.content},
            system_prompt=system_prompt,
            model=request.model,
            temperature=request.temperature,
            model_id=provider_resp.model,
            token_usage=provider_resp.usage,
            finish_reason=provider_resp.finish_reason,
            **gen_params,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics.record_miss(
            model=request.model,
            latency_ms=elapsed_ms,
            best_score=cache_result.similarity_score,
        )

        response = build_response(
            content=provider_resp.content,
            model=provider_resp.model,
            finish_reason=provider_resp.finish_reason,
            usage=provider_resp.usage,
        )
        return JSONResponse(
            content=response.model_dump(),
            headers={
                "X-Cache": "MISS",
                "X-Cache-Latency-Ms": f"{elapsed_ms:.1f}",
            },
        )

    # -- Cache Stats -------------------------------------------------
    @app.get("/v1/cache/stats")
    async def cache_stats():
        """Expose cache statistics for monitoring."""
        stats = app.state.cache.stats()
        m = app.state.metrics
        stats["proxy"] = {
            "total_requests": m.total_requests,
            "cache_hits": m.cache_hits,
            "cache_misses": m.cache_misses,
            "hit_rate": round(m.hit_rate, 4),
        }
        return stats

    @app.delete("/v1/cache")
    async def clear_cache():
        """Clear the entire cache and reset metrics."""
        app.state.cache.clear()
        app.state.metrics.reset()
        return {"status": "cache cleared"}

    # -- Invalidation (Phase 3) --------------------------------------
    @app.post("/v1/cache/invalidate")
    async def invalidate_cache(req: InvalidateRequest):
        """Selectively invalidate cache entries."""
        total_removed = 0
        if req.system_prompt is not None:
            total_removed += app.state.cache.invalidate_by_system_prompt(req.system_prompt)
        if req.model is not None:
            total_removed += app.state.cache.invalidate_by_model(req.model)
        if req.prefix is not None:
            total_removed += app.state.cache.invalidate_by_prefix(req.prefix)
        expired = app.state.cache.purge_expired()
        return {
            "status": "invalidated",
            "entries_removed": total_removed,
            "expired_purged": expired,
        }

    # -- Threshold Tuner (Phase 3) -----------------------------------
    @app.get("/v1/cache/threshold-analysis")
    async def threshold_analysis():
        """Analyse how different thresholds would affect hit rate."""
        return app.state.cache.get_threshold_analysis()

    @app.get("/v1/cache/near-misses")
    async def near_misses(limit: int = 20):
        """Show prompts that narrowly missed the cache threshold."""
        return {
            "current_threshold": app.state.config.similarity_threshold,
            "near_misses": app.state.cache.get_near_misses(limit),
        }

    # -- Monitoring & Dashboard (Phase 4) ----------------------------
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus-compatible metrics endpoint for scraping."""
        return PlainTextResponse(
            content=app.state.metrics.get_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/metrics/detailed")
    async def detailed_metrics():
        """Detailed JSON metrics for the dashboard."""
        return app.state.metrics.get_summary()

    @app.get("/v1/dashboard")
    async def dashboard():
        """Live-updating HTML dashboard with charts."""
        return HTMLResponse(content=render_dashboard_html())

    return app


# -- Response Formatting Helpers -------------------------------------
def _format_cached_response(
    cached: dict, model: str
) -> ChatCompletionResponse:
    """Convert a cached response dict into an OpenAI-shaped response."""
    return build_response(
        content=cached.get("content", ""),
        model=model,
    )


def _stream_cached_response(cached: dict, model: str) -> StreamingResponse:
    """Fake-stream a cached response as SSE chunks."""
    content = cached.get("content", "")

    async def generate() -> AsyncIterator[str]:
        chunk = ChatCompletionChunk(
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaContent(role="assistant", content=content),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        finish_chunk = ChatCompletionChunk(
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaContent(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Cache": "HIT"},
    )


def _stream_from_provider(
    app: FastAPI,
    provider,
    model: str,
    messages: list[dict],
    user_prompt: str,
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int | None,
    gen_params: dict,
    t0: float,
) -> StreamingResponse:
    """Stream from the real LLM provider while buffering for cache."""

    async def generate() -> AsyncIterator[str]:
        buffer: list[str] = []

        try:
            async for content_piece in provider.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                buffer.append(content_piece)

                chunk = ChatCompletionChunk(
                    model=model,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(content=content_piece),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            finish_chunk = ChatCompletionChunk(
                model=model,
                choices=[
                    StreamChoice(
                        delta=DeltaContent(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {finish_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            full_content = "".join(buffer)
            if full_content:
                app.state.cache.store(
                    prompt=user_prompt,
                    response={"content": full_content},
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    **gen_params,
                )
                logger.info(
                    "Cached streamed response (%d chars)", len(full_content)
                )

            # Record miss metrics after stream completes
            elapsed_ms = (time.perf_counter() - t0) * 1000
            app.state.metrics.record_miss(
                model=model,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Streaming error: %s", e)
            error_chunk = {
                "error": {"message": f"Provider streaming error: {e}"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Cache": "MISS"},
    )