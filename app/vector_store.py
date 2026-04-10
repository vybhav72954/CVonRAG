"""
CVonRAG — vector_store.py
Qdrant async wrapper with Ollama-backed embeddings.
No paid API. Everything runs locally.
"""

from __future__ import annotations
import asyncio
import logging
import uuid

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.config import get_settings
from app.models import RoleType, StyleExemplar

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton clients ─────────────────────────────────────────────────────────

_qdrant: AsyncQdrantClient | None = None
_http:   httpx.AsyncClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        kwargs: dict = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _qdrant = AsyncQdrantClient(**kwargs)
    return _qdrant


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=120.0)
    return _http


async def close_clients() -> None:
    """Gracefully close singleton HTTP + Qdrant clients (call on shutdown)."""
    global _http, _qdrant
    if _http is not None:
        await _http.aclose()
        _http = None
    if _qdrant is not None:
        await _qdrant.close()
        _qdrant = None


# ── Embeddings via Ollama ─────────────────────────────────────────────────────

# Limit concurrent embedding requests to avoid overwhelming Ollama.
_EMBED_SEM: asyncio.Semaphore | None = None


def _get_embed_sem() -> asyncio.Semaphore:
    """Lazy-init the semaphore inside a running event loop (defensive)."""
    global _EMBED_SEM
    if _EMBED_SEM is None:
        _EMBED_SEM = asyncio.Semaphore(3)
    return _EMBED_SEM


async def embed_text(text: str) -> list[float]:
    """Single embedding via Ollama /api/embeddings."""
    try:
        r = await get_http().post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.ollama_embed_model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama embedding HTTP %s for text %.40s…: %s", exc.response.status_code, text, exc)
        raise RuntimeError(f"Embedding failed (HTTP {exc.response.status_code})") from exc
    except httpx.RequestError as exc:
        logger.error("Ollama embedding connection error: %s", exc)
        raise RuntimeError("Embedding service unavailable — is Ollama running?") from exc


async def _embed_text_throttled(text: str) -> list[float]:
    """Wraps embed_text with a concurrency semaphore."""
    async with _get_embed_sem():
        return await embed_text(text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Concurrent batch embeddings, throttled to 3 at a time."""
    return list(await asyncio.gather(*[_embed_text_throttled(t) for t in texts]))


# ── Collection management ─────────────────────────────────────────────────────

async def ensure_collection_exists() -> None:
    """Idempotent — creates the Qdrant collection only if missing."""
    client = get_qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if settings.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", settings.qdrant_collection)
    else:
        logger.debug("Qdrant collection already exists: %s", settings.qdrant_collection)


async def collection_info() -> dict:
    """Health-check stats for the /health endpoint."""
    try:
        info = await get_qdrant().get_collection(settings.qdrant_collection)
        return {
            "qdrant_connected": True,
            "collection_exists": True,
            "vector_count": info.points_count or 0,
        }
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {"qdrant_connected": False, "collection_exists": False, "vector_count": 0}


# ── Ingestion ─────────────────────────────────────────────────────────────────

async def ingest_gold_standard_bullets(bullets: list[dict]) -> int:
    """
    Upsert Gold Standard CV bullets into Qdrant.
    Each dict must have at minimum a "text" key.
    Returns the count of upserted points.
    """
    await ensure_collection_exists()
    vectors = await embed_texts([b["text"] for b in bullets])
    points  = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text":               b["text"],
                "role_type":          b.get("role_type", RoleType.GENERAL),
                "uses_separator":     b.get("uses_separator"),
                "uses_arrow":         b.get("uses_arrow", False),
                "uses_abbreviations": b.get("uses_abbreviations", []),
                "sentence_structure": b.get("sentence_structure"),
            },
        )
        for b, vec in zip(bullets, vectors)
    ]
    await get_qdrant().upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )
    logger.info("Upserted %d Gold Standard bullets.", len(points))
    return len(points)


# ── Phase-3 Retrieval ─────────────────────────────────────────────────────────

async def retrieve_style_exemplars(
    query_text: str,
    role_type: RoleType | None = None,
    top_k: int | None = None,
) -> list[StyleExemplar]:
    """
    Query Qdrant for the most stylistically relevant Gold Standard bullets.
    Returns StyleExemplar objects — STYLE ONLY, never used as content.
    """
    k             = top_k or settings.retrieval_top_k
    query_vector  = await embed_text(query_text)
    query_filter: Filter | None = None

    if role_type and role_type != RoleType.GENERAL:
        query_filter = Filter(
            must=[FieldCondition(key="role_type", match=MatchValue(value=role_type.value))]
        )

    hits: list[ScoredPoint] = await get_qdrant().search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    )

    return [
        StyleExemplar(
            exemplar_id=str(h.id),
            text=(h.payload or {}).get("text", ""),
            role_type=RoleType((h.payload or {}).get("role_type", RoleType.GENERAL)),
            similarity_score=float(h.score),
            uses_separator=(h.payload or {}).get("uses_separator"),
            uses_arrow=bool((h.payload or {}).get("uses_arrow", False)),
            uses_abbreviations=(h.payload or {}).get("uses_abbreviations", []),
            sentence_structure=(h.payload or {}).get("sentence_structure"),
        )
        for h in hits
    ]
