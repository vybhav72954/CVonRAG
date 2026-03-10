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
COPY pyproject.toml .

RUN pip install --upgrade pip --quiet \
 && pip install --prefix=/install --no-cache-dir \
      "fastapi==0.115.5" \
      "uvicorn[standard]==0.32.1" \
      "httptools==0.6.4" \
      "uvloop==0.21.0" \
      "pydantic==2.10.3" \
      "pydantic-settings==2.6.1" \
      "httpx==0.28.1" \
      "qdrant-client==1.12.1" \
      "anyio==4.7.0" \
      "python-dotenv==1.0.1" \
      "pdfplumber>=0.11" \
      "python-docx>=1.1"

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

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--log-level", "info"]
