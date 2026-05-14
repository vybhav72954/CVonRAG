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
        """
        Return a debug-friendly string identifying the invite by code and optimization counters.
        
        Returns:
            A string in the form "Invite(code='...', optimize_today=<int>, optimize_count=<int>)".
        """
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
    """
    Return the SQLAlchemy async SQLite connection target for the configured sqlite path.
    
    If the configured path is the literal ":memory:", return the in-memory aiosqlite connection string. Otherwise return a SQLAlchemy `URL` created via `URL.create(...)` so filesystem characters reserved in URLs (for example ?, #, %, &) are escaped correctly.
    
    Returns:
        A `URL` for a file-backed SQLite database or the in-memory connection string `"sqlite+aiosqlite:///:memory:"`.
    """
    path = settings.sqlite_path
    if path == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    # Expand `~` the same way init_db() does for its mkdir step. Without
    # this, `SQLITE_PATH=~/cvonrag.db` would mkdir at $HOME/ (correct) but
    # the engine would try to open a literal `~/cvonrag.db` file because
    # SQLite itself doesn't expand the tilde.
    path = str(Path(path).expanduser())
    return URL.create(drivername="sqlite+aiosqlite", database=path)


async def _ensure_factory_async() -> async_sessionmaker[AsyncSession]:
    """
    Ensure the module-level async SQLAlchemy session factory is initialized for use.
    
    This function is safe to call concurrently from multiple coroutines; on the first invocation it initializes the engine and session factory and subsequent calls return the cached factory.
    
    Returns:
        async_sessionmaker[AsyncSession]: The session factory used to create `AsyncSession` instances.
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
    """
    Create and return the module-level SQLAlchemy async session factory if it does not already exist.
    
    Initializes the module-level async engine and corresponding async session factory when uninitialized; intended for use in startup code paths before concurrent access begins.
    
    Returns:
        The cached `async_sessionmaker` configured to produce `AsyncSession` instances.
    """
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(_engine_url(), echo=False, future=True)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def init_db() -> None:
    """
    Initialize the invite database and create any missing tables for application startup.
    
    Creates the parent directory for a file-backed SQLite database when needed, ensures the module-level engine and session factory are initialized, and applies the ORM metadata to create tables. Intended to be called during the application's startup (lifespan) before serving requests.
    
    Raises:
        RuntimeError: If the engine was not initialized by the factory initialization step.
    """
    if settings.sqlite_path != ":memory:":
        parent = Path(settings.sqlite_path).expanduser().parent
        parent.mkdir(parents=True, exist_ok=True)
    _ensure_factory_sync()
    # Explicit check rather than `assert` — Python -O strips asserts, and
    # a None engine here would surface as a confusing AttributeError on the
    # next line instead of a clear error.
    if _engine is None:
        raise RuntimeError(
            "init_db: _ensure_factory_sync() returned without populating _engine"
        )
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite invite DB ready at %s", settings.sqlite_path)


async def close_db() -> None:
    """
    Close and clean up the module's async database resources.
    
    If an async engine exists it is disposed and the cached engine and session factory are cleared so subsequent initialization can recreate them.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a request-scoped database session for dependency injection (e.g., FastAPI).
    
    Returns:
        AsyncSession: The session bound to the current request; closed automatically when the request scope ends.
    """
    factory = await _ensure_factory_async()
    async with factory() as session:
        yield session


def _reset_factory_for_tests() -> None:
    """Drop the cached engine/factory so a test fixture can re-init against a
    different sqlite_path. Safe to call from tests; not used in production."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
