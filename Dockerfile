# ─────────────────────────────────────────────────────────────────────────────
# CVonRAG — Dockerfile  (FastAPI backend only)
# Ollama + Qdrant run as separate services (see docker-compose.yml).
# Compatible with: Render, Railway, Fly.io, any Docker host.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN pip install --upgrade pip --quiet \
 && pip install --prefix=/install --no-cache-dir ".[unix]"

# ── Stage 2: minimal runtime ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN addgroup --system cvonrag \
 && adduser  --system --ingroup cvonrag --no-create-home cvonrag

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER cvonrag

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Trust X-Forwarded-For from any upstream so the H1 rate limiter sees real
# client IPs instead of the proxy's. SAFETY: this is correct ONLY when the
# container sits behind a real reverse proxy (Render, Railway, Fly, Vercel
# edge, Nginx, …). If you expose this image directly to the public internet,
# attackers can spoof X-Forwarded-For to bypass the per-IP rate limit (N18).
# Pin --forwarded-allow-ips to the proxy's CIDR in that case.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--log-level", "info"]
