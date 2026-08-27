# ── Stage 1: Builder ────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ src/
COPY run_proxy.py .

# Pre-download the embedding model so container startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

ENV SEMCACHE_PROXY_HOST=0.0.0.0
ENV SEMCACHE_PROXY_PORT=8000

CMD ["python", "run_proxy.py"]