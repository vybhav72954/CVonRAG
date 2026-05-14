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
import logging
from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Integer, String
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


# ── Engine + session factory ──────────────────────────────────────────────────
# Built lazily on first call so tests that monkeypatch settings.sqlite_path
# (e.g. to an in-memory or tmp_path DB) see the override.

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _engine_url() -> str:
    """Return the async sqlite URL for the current settings.sqlite_path."""
    # Special-case in-memory; everything else is a file URL.
    path = settings.sqlite_path
    if path == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    return f"sqlite+aiosqlite:///{path}"


def _ensure_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(_engine_url(), echo=False, future=True)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def init_db() -> None:
    """Create tables if missing. Called from the FastAPI lifespan startup hook."""
    _ensure_factory()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite invite DB ready at %s", settings.sqlite_path)


async def close_db() -> None:
    """Dispose of the engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a fresh AsyncSession per request."""
    factory = _ensure_factory()
    async with factory() as session:
        yield session


def _reset_factory_for_tests() -> None:
    """Drop the cached engine/factory so a test fixture can re-init against a
    different sqlite_path. Safe to call from tests; not used in production."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
