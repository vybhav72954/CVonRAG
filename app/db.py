"""
CVonRAG — db.py
Async SQLite layer for Google-OAuth user identity + per-user usage tracking.

The `users` table is the entire persistence story: one row per user, keyed by
their verified Google Workspace email (the `email` claim from the ID token,
filtered through the `hd` claim and ADMIN_EMAILS allowlist).

Usage counters are written on every gated request; daily counters reset
lazily (on next access) when the stored `today_date` no longer matches
today (UTC). This avoids a midnight cron and keeps state self-healing.
"""

from __future__ import annotations
import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone
from pathlib import Path

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


class User(Base):
    """One row per OAuth-authenticated user. Email is the identity."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String, primary_key=True)
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

    def __repr__(self) -> str:
        return (
            f"User(email={self.email!r}, optimize_today={self.optimize_today}, "
            f"optimize_count={self.optimize_count})"
        )


# ── Engine + session factory ──────────────────────────────────────────────────
# Built lazily on first call so tests that monkeypatch settings.sqlite_path
# (e.g. to an in-memory or tmp_path DB) see the override.

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
# Async lock guards cold-start factory creation so concurrent first-touch
# requests can't each build their own engine (the loser would be GC'd while
# its connection pool was still warming).
_factory_lock = asyncio.Lock()


def _engine_url() -> URL | str:
    """
    Return the SQLAlchemy async SQLite connection target for the configured sqlite path.
    """
    path = settings.sqlite_path
    if path == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    path = str(Path(path).expanduser())
    return URL.create(drivername="sqlite+aiosqlite", database=path)


async def _ensure_factory_async() -> async_sessionmaker[AsyncSession]:
    """Lazy-initialise the session factory; safe under concurrent first-touch."""
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
    """Sync variant used during startup (before concurrent access begins)."""
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(_engine_url(), echo=False, future=True)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def init_db() -> None:
    """
    Initialise the users database and create any missing tables for application startup.
    """
    if settings.sqlite_path != ":memory:":
        parent = Path(settings.sqlite_path).expanduser().parent
        parent.mkdir(parents=True, exist_ok=True)
    _ensure_factory_sync()
    if _engine is None:
        raise RuntimeError(
            "init_db: _ensure_factory_sync() returned without populating _engine"
        )
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite users DB ready at %s", settings.sqlite_path)


async def close_db() -> None:
    """Dispose the engine and clear cached factories."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Request-scoped async session for FastAPI dependency injection."""
    factory = await _ensure_factory_async()
    async with factory() as session:
        yield session


def _reset_factory_for_tests() -> None:
    """Drop the cached engine/factory so a test fixture can re-init against a
    different sqlite_path. Safe to call from tests; not used in production."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
