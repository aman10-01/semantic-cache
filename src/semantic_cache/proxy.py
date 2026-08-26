"""FastAPI proxy server — drop-in replacement for the OpenAI API.

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
from fastapi.responses import JSONResponse, StreamingResponse

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
from .providers import ProviderRouter

logger = logging.getLogger(__name__)

# ── App Factory ─────────────────────────────────────────────────────
def create_app(config: CacheConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    config = config or CacheConfig()

    app = FastAPI(
        title="Semantic Cache Proxy",
        description="Drop-in caching proxy for LLM APIs",
        version="0.2.0",
    )

    # CORS — allow any origin so frontends can call the proxy
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state — attached to the app instance
    app.state.config = config
    app.state.cache = SemanticCache(config)
    app.state.router = ProviderRouter(config)
    app.state.request_count = 0
    app.state.cache_hits = 0
    app.state.cache_misses = 0

    # ── Routes ──────────────────────────────────────────────────
    @app.get("/")
    async def root():
        return {
            "service": "Semantic Cache Proxy",
            "version": "0.2.0",
            "status": "running",
            "providers": app.state.router.available_providers,
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """The main endpoint — mirrors OpenAI's chat completions."""
        app.state.request_count += 1
        t0 = time.perf_counter()

        user_prompt = request.get_user_prompt()
        system_prompt = request.get_system_prompt()
        gen_params = request.get_generation_params()

        # ── Step 1: Check cache ─────────────────────────────────
        cache_result = app.state.cache.lookup(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=request.model,
            temperature=request.temperature,
            **gen_params,
        )

        if cache_result.hit:
            app.state.cache_hits += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "CACHE HIT — served in %.1fms (score=%.4f)",
                elapsed_ms,
                cache_result.similarity_score,
            )

            # If client wants streaming, fake-stream the cached response
            if request.stream:
                return _stream_cached_response(
                    cache_result.response, request.model
                )

            # Non-streaming: return the cached response directly
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

        # ── Step 2: Cache miss — forward to LLM provider ───────
        app.state.cache_misses += 1
        logger.info("CACHE MISS — forwarding to provider")

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

    @app.get("/v1/cache/stats")
    async def cache_stats():
        """Expose cache statistics for monitoring."""
        stats = app.state.cache.stats()
        stats["proxy"] = {
            "total_requests": app.state.request_count,
            "cache_hits": app.state.cache_hits,
            "cache_misses": app.state.cache_misses,
            "hit_rate": (
                app.state.cache_hits / app.state.request_count
                if app.state.request_count > 0
                else 0.0
            ),
        }
        return stats

    @app.delete("/v1/cache")
    async def clear_cache():
        """Clear the entire cache."""
        app.state.cache.clear()
        app.state.cache_hits = 0
        app.state.cache_misses = 0
        app.state.request_count = 0
        return {"status": "cache cleared"}

    return app


# ── Response Formatting Helpers ─────────────────────────────────────
def _format_cached_response(
    cached: dict, model: str
) -> ChatCompletionResponse:
    """Convert a cached response dict into an OpenAI-shaped response."""
    return build_response(
        content=cached.get("content", ""),
        model=model,
    )


def _stream_cached_response(cached: dict, model: str) -> StreamingResponse:
    """Fake-stream a cached response as SSE chunks.

    Cache hits return instantly, but if the client requested streaming,
    we still need to send the data in SSE format so it doesn't break.
    """
    content = cached.get("content", "")

    async def generate() -> AsyncIterator[str]:
        # Send the content as a single chunk (it's instant anyway)
        chunk = ChatCompletionChunk(
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaContent(role="assistant", content=content),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Send the finish chunk
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
) -> StreamingResponse:
    """Stream from the real LLM provider while buffering for cache.

    This is the tricky part: we need to stream chunks to the client
    in real-time AND collect all the chunks so we can store the
    complete response in the cache when the stream finishes.
    """

    async def generate() -> AsyncIterator[str]:
        buffer: list[str] = []  # collect all chunks for caching

        try:
            async for content_piece in provider.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                buffer.append(content_piece)

                # Send each chunk to the client as SSE
                chunk = ChatCompletionChunk(
                    model=model,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(content=content_piece),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Stream finished — send finish marker
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

            # Store the complete buffered response in cache
            full_content = "".join(buffer)
            if full_content:  # only cache non-empty responses
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
