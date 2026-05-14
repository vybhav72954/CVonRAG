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
import asyncio
import json
import logging
import secrets
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from math import ceil

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import check_and_reserve_bullets, require_invite
from app.chains import (
    CVonRAGOrchestrator,
    HostedLLMQuotaExhausted,
    _hosted_llm_config,
    close_http,
)
from app.config import get_settings
from app.db import Invite, close_db, get_session, init_db
from app.models import (
    GeneratedBullet,
    HealthResponse,
    InviteCreate,
    InviteUsage,
    OptimizationRequest,
    RecommendRequest,
    RecommendResponse,
    RoleType,
    StreamChunk,
    StreamEventType,
    UsageResponse,
)
from app.parser import parse_and_stream
from app.recommender import recommend_projects as _do_recommend
from app.vector_store import (
    close_clients,
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


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Per-IP sliding-window in-memory rate limiter.

    Safe for a single-process deployment. For multi-worker or distributed
    deployments, replace with a Redis-backed solution (e.g. slowapi + Redis).

    Memory hygiene: a periodic sweep drops (ip, key) entries whose newest
    timestamp is past the window — without it, every IP ever seen would
    accumulate a stale deque indefinitely (N2).

    INVARIANT (P7): every caller MUST pass the same `window_seconds`. The GC
    uses the *current* call's cutoff to decide which entries are stale; if two
    routes used different windows, the shorter window's GC sweep would drop
    entries still valid for the longer window. All routes today share
    `settings.rate_limit_window`. If a per-route override is ever introduced,
    track the largest window and GC against that.
    """

    _GC_EVERY_N_CHECKS = 200

    def __init__(self) -> None:
        # Plain dict (not defaultdict) so the GC sweep can `del` entries safely.
        self._windows: dict[tuple[str, str], deque] = {}
        self._lock = asyncio.Lock()
        self._checks_since_gc = 0
        self._window_seconds: int | None = None  # locked-in on first check (P7)

    async def check(
        self, ip: str, key: str, max_calls: int, window_seconds: int
    ) -> float | None:
        """Return None if the request is allowed; return seconds-to-wait if limited."""
        if not settings.rate_limit_enabled:
            return None
        now    = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            # Enforce the single-window invariant (P7) — fail loud rather than
            # silently mis-GC. Reset via `_windows.clear()` clears this too.
            if self._window_seconds is None:
                self._window_seconds = window_seconds
            elif self._window_seconds != window_seconds:
                raise RuntimeError(
                    f"_RateLimiter received mixed windows ({self._window_seconds}s vs "
                    f"{window_seconds}s). GC is only correct under a single shared window."
                )

            self._checks_since_gc += 1
            if self._checks_since_gc >= self._GC_EVERY_N_CHECKS:
                self._gc(cutoff)
                self._checks_since_gc = 0

            dq = self._windows.setdefault((ip, key), deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= max_calls:
                # dq is non-empty here (len >= max_calls >= 1).
                return max(window_seconds - (now - dq[0]), 0.1)
            dq.append(now)
            return None

    def _gc(self, cutoff: float) -> None:
        """Drop entries whose newest timestamp is older than the window cutoff."""
        stale = [k for k, dq in self._windows.items() if not dq or dq[-1] < cutoff]
        for k in stale:
            del self._windows[k]


_limiter = _RateLimiter()


async def _rate_check(request: Request, key: str, max_calls: int) -> None:
    """
    Enforce the configured rate limit for the caller and raise HTTP 429 when the limit is exceeded.
    
    Parameters:
        request (Request): The incoming request used to identify the caller (IP).
        key (str): Namespace/key for the rate-limiting bucket (e.g., "optimize", "parse").
        max_calls (int): Maximum allowed calls within the configured rate-limit window.
    
    Raises:
        HTTPException: 429 Too Many Requests with `Retry-After` header set to the number of seconds to wait.
    """
    ip   = request.client.host if request.client else "unknown"
    wait = await _limiter.check(ip, key, max_calls, settings.rate_limit_window)
    if wait is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {ceil(wait)}s.",
            headers={"Retry-After": str(ceil(wait))},
        )


# ── Admin secret helper ───────────────────────────────────────────────────────

def _check_admin_secret(provided: str | None) -> None:
    """
    Validate the X-Ingest-Secret header using a constant-time comparison.
    
    If `settings.ingest_secret` is empty, the check is skipped (gate open). This function raises an HTTP 403 when a non-empty configured secret exists and the provided header is missing or does not match.
    
    Parameters:
        provided (str | None): Value of the `X-Ingest-Secret` header from the request.
    
    Raises:
        HTTPException: 403 Forbidden when the provided secret is invalid or missing while a configured secret is required.
    """
    if not settings.ingest_secret:
        return
    # P3: constant-time comparison to avoid leaking secret prefix via timing.
    if not secrets.compare_digest(provided or "", settings.ingest_secret):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-Ingest-Secret header.",
        )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Initialize application resources on startup and clean them up on shutdown.
    
    On startup this function checks that the Qdrant collection exists (logs a warning on failure but does not abort startup) and initializes the SQLite invite database (logs and re-raises the exception if invite codes are required by configuration). On shutdown it closes the shared HTTP client used by LLM chains, the vector/Qdrant clients, and the SQLite engine.
    """
    logger.info("CVonRAG starting — checking Qdrant collection …")
    try:
        await ensure_collection_exists()
        logger.info("Qdrant ready.")
    except Exception as exc:
        logger.warning("Qdrant startup check failed (will retry on first request): %s", exc)
    # SQLite invite DB — create the table file/schema if missing. Failure here
    # is fatal in production (no invites = nobody can call gated endpoints)
    # but the dev override INVITE_CODES_REQUIRED=false makes the table moot.
    try:
        await init_db()
    except Exception as exc:
        logger.exception("Invite DB init failed: %s", exc)
        if settings.invite_codes_required:
            raise
    yield
    # ── Shutdown: close singleton clients ──────────────────────────────
    logger.info("CVonRAG shutting down — closing connections …")
    await close_http()          # chains.py HTTP client
    await close_clients()       # vector_store.py HTTP + Qdrant clients
    await close_db()            # SQLite engine
    logger.info("All connections closed.")


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

# ── File-type validation (magic bytes) ───────────────────────────────────────
# Trusting the filename extension alone lets a renamed file bypass the parser.
# Checking the leading bytes costs nothing and catches the common case (H7).
# DOCX (OOXML) is a ZIP container: PK\x03\x04
# PDF specification header:         %PDF

_MAGIC_DOCX = b"PK\x03\x04"
_MAGIC_PDF  = b"%PDF"

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB hard limit
_UPLOAD_CHUNK     = 64 * 1024            # 64 KB read chunk


def _validate_file_magic(file_bytes: bytes, fname: str) -> bool:
    """Return True when the file's leading bytes match the declared extension."""
    if fname.endswith(".docx"):
        return file_bytes[:4] == _MAGIC_DOCX
    if fname.endswith(".pdf"):
        return file_bytes[:4] == _MAGIC_PDF
    return False


async def _read_upload_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, raising 413 once max_bytes is exceeded.

    The previous implementation called `file.read()` with no size argument and
    only checked the length afterwards, so a 1 GB upload would materialise in
    full before being rejected (N3).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


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
                # The bullet payload already contains .metadata; no separate event needed.
                yield _sse(StreamChunk(event_type=StreamEventType.BULLET, data=bullet.model_dump()))

            elif event_type == "done":
                yield _sse(StreamChunk(
                    event_type=StreamEventType.DONE,
                    data={"elapsed_seconds": round(time.monotonic() - t0, 2)},
                ))
                return

    except HostedLLMQuotaExhausted as exc:
        # Don't log a stack trace for quota exhaustion — it's not a bug, it's
        # a known limit being hit. Surface the clean message to the client.
        logger.warning("Pipeline aborted — %s", exc)
        yield _sse(StreamChunk(event_type=StreamEventType.ERROR, error_message=str(exc)))
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
    """Liveness probe — checks active LLM backend (Groq/OpenRouter/Ollama) + embed model + Qdrant."""
    qdrant = await collection_info()
    cfg = _hosted_llm_config()
    provider_name = settings.llm_provider if cfg is not None else "ollama"

    llm_ok    = False
    ollama_ok = False
    embed_ok  = False

    async with httpx.AsyncClient(timeout=5.0) as c:
        # ── Always check Ollama for the embed model ──────────────────────
        try:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            embed_base = settings.ollama_embed_model.split(":")[0]
            embed_ok = any(embed_base in m for m in models)

            # ── Ollama LLM model (only needed when no hosted provider) ──
            if cfg is None:
                llm_base = settings.ollama_llm_model.split(":")[0]
                ollama_ok = any(llm_base in m for m in models)
        except Exception:
            pass

        # ── Hosted-LLM reachability (Groq or OpenRouter) ─────────────────
        if cfg is not None:
            api_key, base_url, _ = cfg
            try:
                r = await c.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                llm_ok = r.status_code == 200
            except Exception:
                pass

    # Determine overall status
    llm_healthy = llm_ok if cfg is not None else ollama_ok
    all_ok = llm_healthy and embed_ok and qdrant["qdrant_connected"]

    # groq_ok kept for back-compat: True only when active provider IS groq AND reachable.
    groq_ok = llm_ok if provider_name == "groq" else False
    active_model = cfg[2] if cfg is not None else settings.ollama_llm_model

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        llm_backend=provider_name,
        llm_provider=provider_name,
        model=active_model,
        llm_ok=llm_ok,
        groq_ok=groq_ok,
        ollama_ok=ollama_ok,
        embed_ok=embed_ok,
        **qdrant,
    )


# ── Document parsing ──────────────────────────────────────────────────────────

@app.post(
    "/parse",
    tags=["parsing"],
    summary="Parse a .docx or .pdf CV file into structured projects + facts",
    status_code=status.HTTP_200_OK,
)
async def parse_cv(
    request: Request,
    file: UploadFile = File(...),
    _invite: Invite | None = Depends(require_invite("parse")),
):
    """
    Stream extraction progress and parsed project data from an uploaded .docx or .pdf CV as Server-Sent Events.
    
    Streams SSE `StreamChunk` envelopes describing parsing progress, discovered projects, completion, or errors:
    - `progress`: `{"message", "current", "total"}` — extraction progress update
    - `project`: `{"project": ProjectData, "index", "total"}` — a completed project
    - `done`: `{"total_projects", "total_facts"}` — parsing finished
    - `error`: `null` (see `error_message`) — parse failure
    
    Returns:
        StreamingResponse: A `text/event-stream` response that emits SSE-formatted `StreamChunk` objects for each event.
    """
    await _rate_check(request, "parse", settings.rate_limit_parse)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    fname = file.filename.lower()
    if not (fname.endswith(".docx") or fname.endswith(".pdf")):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a .docx or .pdf file.",
        )

    # Cheap pre-check: reject before reading the body when Content-Length is sent.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    file_bytes = await _read_upload_capped(file, _MAX_UPLOAD_BYTES)

    if len(file_bytes) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupt.")

    if not _validate_file_magic(file_bytes, fname):
        raise HTTPException(
            status_code=415,
            detail=(
                "File content does not match the declared type. "
                "Upload a genuine .docx or .pdf — not a renamed file."
            ),
        )

    async def _stream():
        _PARSE_EVENT_MAP = {
            "progress": StreamEventType.PROGRESS,
            "project":  StreamEventType.PROJECT,
            "done":     StreamEventType.DONE,
            "error":    StreamEventType.ERROR,
        }
        async for event_type, data in parse_and_stream(file_bytes, file.filename):
            sse_event = _PARSE_EVENT_MAP.get(event_type, event_type)
            if event_type == "error":
                yield _sse(StreamChunk(
                    event_type=sse_event,
                    error_message=data.get("error_message", "Unknown parse error"),
                ))
            else:
                yield _sse(StreamChunk(event_type=sse_event, data=data))

    return StreamingResponse(_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ── Project recommendation ────────────────────────────────────────────────────

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["generation"],
    summary="Score and rank all uploaded projects against a job description",
)
async def recommend(
    request: Request,
    body: RecommendRequest,
    _invite: Invite | None = Depends(require_invite("recommend")),
):
    """
    Score candidate projects against a job description and return the top recommendations.
    
    Given parsed projects and a job description, computes a relevance score (0–1) for each project, marks the top `top_k` projects as recommended, and provides a one-sentence reason for each recommendation. The response is used to pre-select projects for downstream optimization.
    
    Returns:
        RecommendResponse: Contains `recommendations` — a list of scored projects with `recommended` flags and one-line reasoning for each.
    """
    await _rate_check(request, "recommend", settings.rate_limit_recommend)
    try:
        recs = await _do_recommend(
            projects=body.projects,
            job_description=body.job_description,
            top_k=body.top_k,
        )
    except HostedLLMQuotaExhausted as exc:
        # 503 Service Unavailable + Retry-After header so clients/browsers can
        # show a clear quota message instead of a 30-minute-hang-then-timeout.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": str(int(exc.retry_after_seconds))},
        ) from exc
    return RecommendResponse(recommendations=recs)


# ── Bullet generation ─────────────────────────────────────────────────────────

@app.post(
    "/optimize",
    tags=["generation"],
    summary="Stream optimised resume bullets via SSE",
    status_code=status.HTTP_200_OK,
)
async def optimize(
    request: Request,
    body: OptimizationRequest,
    invite: Invite | None = Depends(require_invite("optimize")),
    session: AsyncSession = Depends(get_session),
):
    """
    Stream optimized resume bullet generation as Server-Sent Events.
    
    Streams Server-Sent Events (SSE) representing the optimization run for the provided OptimizationRequest.
    
    Parameters:
        body (OptimizationRequest): Parameters controlling generation (projects, per-project caps, etc.).
    
    Returns:
        StreamingResponse: An SSE stream that emits events until completion. Emitted event types:
          - `token`: raw LLM token fragments as `{ "data": "<string>" }`.
          - `bullet`: a finalized, validated bullet as `{ "data": <GeneratedBullet> }` (includes `metadata`).
          - `error`: a pipeline error as `{ "error_message": "<string>" }`.
          - `done`: completion notice as `{ "data": { "elapsed_seconds": <number> } }`.
    """
    await _rate_check(request, "optimize", settings.rate_limit_optimize)
    # H3: guard hosted-LLM quota before starting the stream.
    # 429 backoff is handled inside chains.py, but the best defence is not
    # sending 200+ LLM calls in the first place. Applies to whichever hosted
    # provider is active (Groq or OpenRouter); both free tiers have strict
    # per-minute limits that this cap protects against.
    requested = body.total_bullets_requested or 0
    if _hosted_llm_config() is not None:
        cap = settings.groq_max_bullets_per_request
        if requested > cap:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Hosted-LLM quota guard: total_bullets_requested ({requested}) exceeds "
                    f"the server cap of {cap}. Reduce max_bullets_per_project or the number "
                    f"of projects, or set GROQ_MAX_BULLETS_PER_REQUEST higher in .env."
                ),
            )
    # Per-invite daily bullet cap. The require_invite dep already counted +1
    # toward optimize_today; this reserves the requested bullets against the
    # invite's per-day bullet budget before any LLM calls happen.
    await check_and_reserve_bullets(invite, requested, session)
    return StreamingResponse(
        _sse_stream(body),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestItem(BaseModel):
    text: str                     = Field(..., min_length=10)
    # Use the enum so a typo at ingest fails fast with 422 instead of poisoning
    # Qdrant and crashing retrieve_style_exemplars later (N6).
    role_type: RoleType           = Field(default=RoleType.GENERAL)
    uses_separator: str | None    = None
    uses_arrow: bool              = False
    uses_abbreviations: list[str] = Field(default_factory=list)
    sentence_structure: str | None = None


# Per-request ingest cap (P10). Each bullet triggers an Ollama embed call
# throttled to 3 concurrent — 100 bullets ≈ 30s, 500 would tie up Ollama for
# minutes. The seeding script (scripts/ingest_pdfs.py) already chunks at 50,
# so 100 leaves 2× headroom without enabling abuse if /ingest is exposed.
_MAX_BULLETS_PER_INGEST = 100


class IngestRequest(BaseModel):
    bullets: list[IngestItem] = Field(..., min_length=1, max_length=_MAX_BULLETS_PER_INGEST)


class IngestResponse(BaseModel):
    upserted: int
    message: str


@app.post("/ingest", response_model=IngestResponse, tags=["admin"])
async def ingest(
    body: IngestRequest,
    x_ingest_secret: str | None = Header(default=None),
):
    """
    Seed the gold-standard bullets dataset into the vector store (admin only).
    
    Requires the `X-Ingest-Secret` header when `INGEST_SECRET` is configured; otherwise the endpoint is open.
    
    Parameters:
        body (IngestRequest): Payload containing the list of bullets to ingest.
        x_ingest_secret (str | None): Value of the `X-Ingest-Secret` header when provided.
    
    Returns:
        IngestResponse: Contains `upserted` (number of bullets written) and a human-readable `message`.
    
    Raises:
        HTTPException(403): If an ingest secret is configured and the provided header is missing or invalid.
        HTTPException(500): On unexpected failures during ingestion.
    """
    _check_admin_secret(x_ingest_secret)
    try:
        count = await ingest_gold_standard_bullets(
            [item.model_dump() for item in body.bullets]
        )
        return IngestResponse(
            upserted=count,
            message=f"Upserted {count} bullets into '{settings.qdrant_collection}'.",
        )
    except Exception as exc:
        # Log full detail server-side; return a generic message so we don't
        # leak internals (stack-trace fragments, infra hostnames, etc.) to clients.
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingestion failed — see server logs for details.",
        )


# ── Admin: invite management ──────────────────────────────────────────────────

@app.post("/admin/invites", response_model=InviteUsage, tags=["admin"])
async def create_invite(
    body: InviteCreate,
    x_ingest_secret: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
):
    """
    Create a new invite code; endpoint is protected by the `X-Ingest-Secret` header.
    
    Parameters:
        body (InviteCreate): Invite data including the invite `code` and optional `name`.
    
    Returns:
        InviteUsage: A validated representation of the newly created invite.
    """
    _check_admin_secret(x_ingest_secret)
    invite = Invite(code=body.code, name=body.name)
    session.add(invite)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Invite code '{body.code}' already exists.",
        )
    await session.refresh(invite)
    return InviteUsage.model_validate(invite, from_attributes=True)


_USAGE_MAX_LIMIT = 500


@app.get("/admin/usage", response_model=UsageResponse, tags=["admin"])
async def list_usage(
    x_ingest_secret: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
    limit: int = 100,
    offset: int = 0,
):
    """
    Return a paginated list of invite codes with their cumulative and today's usage counters; access is gated by the X-Ingest-Secret header.
    
    Parameters:
        x_ingest_secret (str | None): Admin secret from the `X-Ingest-Secret` header (required unless server secret is unset).
        session (AsyncSession): Database session (injected).
        limit (int): Maximum number of invites to return; must be between 1 and _USAGE_MAX_LIMIT (default 100).
        offset (int): Number of invites to skip before returning results (default 0).
    
    Returns:
        UsageResponse: Object containing a list of `InviteUsage` entries ordered by invite code.
    
    Raises:
        HTTPException: 403 if the admin secret is invalid; 422 if `limit` or `offset` are out of allowed ranges.
    """
    _check_admin_secret(x_ingest_secret)
    if limit < 1 or limit > _USAGE_MAX_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"limit must be between 1 and {_USAGE_MAX_LIMIT}",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="offset must be >= 0",
        )
    rows = (
        await session.scalars(
            select(Invite).order_by(Invite.code).offset(offset).limit(limit)
        )
    ).all()
    return UsageResponse(
        invites=[InviteUsage.model_validate(r, from_attributes=True) for r in rows],
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
