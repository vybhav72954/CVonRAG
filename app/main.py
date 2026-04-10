"""
CVonRAG — main.py
FastAPI application with async SSE streaming.

Endpoints:
  POST /optimize   →  SSE stream (tokens + bullets + metadata)
  POST /parse      →  SSE stream (parse docx/pdf → structured projects + facts)
  POST /ingest     →  Admin: seed Qdrant with Gold Standard bullets
  GET  /health     →  Liveness probe
  GET  /           →  Welcome
"""

from __future__ import annotations
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chains import CVonRAGOrchestrator, get_http
from app.config import get_settings
from app.models import (
    GeneratedBullet,
    HealthResponse,
    OptimizationRequest,
    RecommendRequest,
    RecommendResponse,
    StreamChunk,
    StreamEventType,
)
from app.parser import parse_and_stream
from app.recommender import recommend_projects as _do_recommend
from app.vector_store import (
    collection_info,
    ensure_collection_exists,
    ingest_gold_standard_bullets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger   = logging.getLogger("cvonrag")
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("CVonRAG starting — checking Qdrant collection …")
    try:
        await ensure_collection_exists()
        logger.info("Qdrant ready.")
    except Exception as exc:
        logger.warning("Qdrant startup check failed (will retry on first request): %s", exc)
    yield
    logger.info("CVonRAG shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CVonRAG — Resume Optimization API",
    description=(
        "RAG-powered resume bullet generator. "
        "Streams bullets that fuse your content with Gold Standard CV style."
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SSE_HEADERS = {
    "Cache-Control":     "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(chunk: StreamChunk) -> str:
    return f"event: {chunk.event_type}\ndata: {chunk.model_dump_json()}\n\n"


async def _sse_stream(request: OptimizationRequest) -> AsyncGenerator[str, None]:
    orchestrator = CVonRAGOrchestrator()
    t0 = time.monotonic()

    try:
        async for event_type, payload in orchestrator.run(request):
            if event_type == "token":
                yield _sse(StreamChunk(event_type=StreamEventType.TOKEN, data=payload))

            elif event_type == "bullet":
                bullet: GeneratedBullet = payload  # type: ignore[assignment]
                yield _sse(StreamChunk(event_type=StreamEventType.BULLET,   data=bullet.model_dump()))
                yield _sse(StreamChunk(event_type=StreamEventType.METADATA, data=bullet.metadata.model_dump()))

            elif event_type == "done":
                yield _sse(StreamChunk(
                    event_type=StreamEventType.DONE,
                    data={"elapsed_seconds": round(time.monotonic() - t0, 2)},
                ))
                return

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        yield _sse(StreamChunk(event_type=StreamEventType.ERROR, error_message=str(exc)))


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "CVonRAG", "version": "1.1.0", "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health_check():
    """Liveness probe — checks both Ollama and Qdrant connectivity."""
    qdrant    = await collection_info()
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r      = await c.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            base   = settings.ollama_llm_model.split(":")[0]
            ollama_ok = any(base in m for m in models)
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (ollama_ok and qdrant["qdrant_connected"]) else "degraded",
        model=settings.ollama_llm_model,
        ollama_ok=ollama_ok,
        **qdrant,
    )


# ── Document parsing ──────────────────────────────────────────────────────────

@app.post(
    "/parse",
    tags=["parsing"],
    summary="Parse a .docx or .pdf CV file into structured projects + facts",
    status_code=status.HTTP_200_OK,
)
async def parse_cv(file: UploadFile = File(...)):
    """
    **Document parsing endpoint.**

    Upload a `.docx` biodata or `.pdf` CV.
    Returns a `text/event-stream` SSE response showing extraction progress.

    | Event type | Payload | Description |
    |---|---|---|
    | `progress` | `{message, current, total}` | Extraction progress update |
    | `project`  | `{project: ProjectData, index, total}` | One project completed |
    | `done`     | `{total_projects, total_facts}` | All done |
    | `error`    | `{error_message}` | Parse error |
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    fname = file.filename.lower()
    if not (fname.endswith(".docx") or fname.endswith(".pdf")):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a .docx or .pdf file.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:   # 10 MB hard limit
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    if len(file_bytes) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupt.")

    http_client = get_http()

    async def _stream():
        _PARSE_EVENT_MAP = {
            "progress": StreamEventType.PROGRESS,
            "project":  StreamEventType.PROJECT,
            "done":     StreamEventType.DONE,
            "error":    StreamEventType.ERROR,
        }
        async for event_type, data in parse_and_stream(file_bytes, file.filename, http_client):
            sse_event = _PARSE_EVENT_MAP.get(event_type, event_type)
            yield f"event: {sse_event}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ── Project recommendation ────────────────────────────────────────────────────

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["generation"],
    summary="Score and rank all uploaded projects against a job description",
)
async def recommend(body: RecommendRequest):
    """
    **Project recommendation endpoint.**

    Given all projects parsed from a CV + a job description, returns every
    project scored (0–1) against the JD, with the top `top_k` marked as
    recommended and a one-sentence reason explaining why.

    Call this after `/parse` and before `/optimize`. The frontend uses the
    response to pre-select the best projects and show reasoning to the user.
    """
    recs = await _do_recommend(
        projects=body.projects,
        job_description=body.job_description,
        top_k=body.top_k,
    )
    return RecommendResponse(recommendations=recs)


# ── Bullet generation ─────────────────────────────────────────────────────────

@app.post(
    "/optimize",
    tags=["generation"],
    summary="Stream optimised resume bullets via SSE",
    status_code=status.HTTP_200_OK,
)
async def optimize(request: OptimizationRequest):
    """
    **Main generation endpoint.**

    Accepts an `OptimizationRequest` JSON body.
    Returns a `text/event-stream` SSE response.

    | Event type | Payload | Description |
    |---|---|---|
    | `token`    | `{data: string}` | Raw LLM token (typewriter) |
    | `bullet`   | `{data: GeneratedBullet}` | Final validated bullet |
    | `metadata` | `{data: BulletMetadata}` | Char count, iterations, sources |
    | `error`    | `{error_message: string}` | Pipeline error |
    | `done`     | `{data: {elapsed_seconds}}` | Stream complete |
    """
    return StreamingResponse(
        _sse_stream(request),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestItem(BaseModel):
    text: str                     = Field(..., min_length=10)
    role_type: str                = Field(default="general")
    uses_separator: str | None    = None
    uses_arrow: bool              = False
    uses_abbreviations: list[str] = Field(default_factory=list)
    sentence_structure: str | None = None


class IngestRequest(BaseModel):
    bullets: list[IngestItem] = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    upserted: int
    message: str


@app.post("/ingest", response_model=IngestResponse, tags=["admin"])
async def ingest(
    body: IngestRequest,
    x_ingest_secret: str | None = Header(default=None),
):
    """
    **Admin endpoint** — seed Qdrant with Gold Standard CV bullets.

    When `INGEST_SECRET` is set in `.env`, callers must send
    `X-Ingest-Secret: <secret>` in the request header.
    """
    if settings.ingest_secret:
        if x_ingest_secret != settings.ingest_secret:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing X-Ingest-Secret header.",
            )
    try:
        count = await ingest_gold_standard_bullets(
            [item.model_dump() for item in body.bullets]
        )
        return IngestResponse(
            upserted=count,
            message=f"Upserted {count} bullets into '{settings.qdrant_collection}'.",
        )
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {exc}",
        )


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
