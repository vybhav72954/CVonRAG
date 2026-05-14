"""
CVonRAG — db.py
Async SQLite layer for invite-code auth + per-batchmate usage tracking.

The `invites` table is the entire persistence story: one row per batchmate,
keyed by the invite code (which doubles as their identity since OAuth is
explicitly out of scope — see plan, May 2026).

Usage counters are written on every gated request; daily counters reset
lazily (on next access) when the stored `today_date` no longer matches
today (UTC). This avoids a midnight cron and keeps state self-healing.
"""

from __future__ import annotations
import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

logger = logging.getLogger("cvonrag")
settings = get_settings()


class Base(DeclarativeBase):
    pass


class Invite(Base):
    """One row per batchmate. Code is the identity."""

    __tablename__ = "invites"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    parse_count: Mapped[int] = mapped_column(Integer, default=0)
    recommend_count: Mapped[int] = mapped_column(Integer, default=0)
    optimize_count: Mapped[int] = mapped_column(Integer, default=0)

    # Daily counters — reset lazily when today_date != today (UTC).
    optimize_today: Mapped[int] = mapped_column(Integer, default=0)
    bullets_today: Mapped[int] = mapped_column(Integer, default=0)
    today_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    def __repr__(self) -> str:  # B5 — readable debug output
        return (
            f"Invite(code={self.code!r}, optimize_today={self.optimize_today}, "
            f"optimize_count={self.optimize_count})"
        )


# ── Engine + session factory ──────────────────────────────────────────────────
# Built lazily on first call so tests that monkeypatch settings.sqlite_path
# (e.g. to an in-memory or tmp_path DB) see the override.

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
# B9: async lock guards cold-start factory creation so concurrent first-touch
# requests can't each build their own engine (the loser would be GC'd while
# its connection pool was still warming).
_factory_lock = asyncio.Lock()


def _engine_url() -> URL | str:
    """Return the async SQLite URL for the current settings.sqlite_path.

    Uses `URL.create()` so reserved URL characters in the path (`?`, `#`, `%`,
    `&`, …) are escaped properly. The previous string-format approach would
    have silently mis-parsed a `SQLITE_PATH` like `./cv?staging.db` because
    SQLAlchemy would have taken `staging.db` as a query string (B4).
    """
    path = settings.sqlite_path
    if path == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    return URL.create(drivername="sqlite+aiosqlite", database=path)


async def _ensure_factory_async() -> async_sessionmaker[AsyncSession]:
    """Async-safe lazy initializer for the engine + session factory.

    Double-checked locking pattern (B9): the fast path skips the lock when
    the factory is already built. On a cold start with truly concurrent
    requests, only one coroutine builds the engine while the others await
    the lock and then see the populated global.
    """
    global _engine, _session_factory
    if _session_factory is not None:
        return _session_factory
    async with _factory_lock:
        if _session_factory is None:
            _engine = create_async_engine(_engine_url(), echo=False, future=True)
            _session_factory = async_sessionmaker(
                _engine, class_=AsyncSession, expire_on_commit=False
            )
    return _session_factory


def _ensure_factory_sync() -> async_sessionmaker[AsyncSession]:
    """Sync fallback for callers that can't await — used by init_db() during
    lifespan startup, where the event loop is the lifespan's own and no
    concurrent request can reach the factory yet."""
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(_engine_url(), echo=False, future=True)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def init_db() -> None:
    """Create tables if missing. Called from the FastAPI lifespan startup hook.

    Runs during lifespan before any request can hit the app — no race window
    for the factory, so the sync builder is safe and avoids re-entering the
    lock that the lifespan-event-loop already holds nothing on.
    """
    _ensure_factory_sync()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite invite DB ready at %s", settings.sqlite_path)


async def close_db() -> None:
    """Dispose of the engine on shutdown. Lifespan teardown only runs after
    in-flight requests have drained (ASGI guarantee), so we don't need to
    explicitly wait on outstanding sessions."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a fresh AsyncSession per request."""
    factory = await _ensure_factory_async()
    async with factory() as session:
        yield session


def _reset_factory_for_tests() -> None:
    """Drop the cached engine/factory so a test fixture can re-init against a
    different sqlite_path. Safe to call from tests; not used in production."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
