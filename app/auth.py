"""
CVonRAG — auth.py
Google Workspace OAuth auth + per-user usage tracking.

One row per OAuth-authenticated user, keyed on the verified `email` claim.
The Authorization: Bearer <id_token> header is required on /parse, /recommend,
/optimize. Admin endpoints (/admin/usage) additionally require the user's
email to appear in settings.admin_emails.

TOKEN VERIFICATION
──────────────────
`_verify_id_token` uses google.oauth2.id_token.verify_oauth2_token, which:
  • fetches Google's JWKS (cached internally),
  • verifies the JWT signature,
  • checks the `aud` claim equals our client_id,
  • checks `iss` is accounts.google.com / https://accounts.google.com,
  • checks `exp` has not passed.
We then check `hd` (hosted domain) and `email_verified` ourselves before
trusting the email as identity. The verify call is sync and CPU/IO-bound on
JWKS fetch, so it's wrapped in asyncio.to_thread to keep the event loop free.

CONCURRENCY MODEL
─────────────────
All counter updates use atomic conditional UPDATE statements rather than
SELECT-then-mutate. SQLite serialises write transactions, so two concurrent
/optimize calls at `optimize_today = 19` cannot both bump to 20:

    UPDATE users
       SET optimize_today = optimize_today + 1, ...
     WHERE email = ?
       AND optimize_today < 20

The second request reads 20 inside its own transaction, the WHERE clause is
false, `rowcount == 0`, and we surface a 429.

PRE-INCREMENT
─────────────
A failing /optimize still counts toward the daily quota. Otherwise an abusive
caller could trigger many failures without exhausting their daily budget.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from math import ceil
from typing import Literal

import urllib3
from fastapi import Depends, Header, HTTPException, status
from google.auth.transport import urllib3 as google_urllib3
from google.oauth2 import id_token
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import User, get_session

logger = logging.getLogger("cvonrag")
settings = get_settings()

EndpointName = Literal["parse", "recommend", "optimize"]


def _seconds_until_midnight_utc(ref_dt: datetime | None = None) -> int:
    """Whole seconds until the next 00:00 UTC, ≥ 1."""
    now = ref_dt or datetime.now(UTC)
    tomorrow = now + timedelta(days=1)
    tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(int(ceil((tomorrow - now).total_seconds())), 1)


# ── Google ID token verification ──────────────────────────────────────────────

# Module-level transport handles JWKS fetching with internal caching (~6h
# TTL). Reused across every verify call. urllib3 is preferred over the
# requests-backed transport because urllib3 is already a transitive
# dependency (via httpx) — no need to add the heavier `requests` library.
_google_http = urllib3.PoolManager()
_google_request = google_urllib3.Request(_google_http)


def _verify_id_token_sync(token: str, client_id: str) -> dict:
    """Sync wrapper around google.oauth2.id_token.verify_oauth2_token.

    Returns the decoded claim payload on success; raises ValueError on any
    verification failure (bad signature, wrong aud/iss, expired, malformed).
    Kept as a separate function so tests can patch `_verify_id_token` (the
    async wrapper below) without having to mock Google's JWKS endpoint.
    """
    return id_token.verify_oauth2_token(token, _google_request, audience=client_id)


async def _verify_id_token(token: str, client_id: str) -> dict:
    """Async-friendly wrapper. JWKS fetch + JWT verify is sync, so offload to a thread."""
    return await asyncio.to_thread(_verify_id_token_sync, token, client_id)


# Google's published `iss` values. The library already checks this, but we
# assert here too so a future libversion change can't silently widen the set.
_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}


async def _authenticate_bearer(authorization: str | None) -> str:
    """Verify a Bearer token and return the authenticated email.

    Raises HTTPException 401 on any verification failure. The caller has
    already short-circuited the dev-bypass case (client_id == ""), so by the
    time we reach here we are committed to verifying.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header (expected 'Bearer <id_token>').",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = await _verify_id_token(token, settings.google_oauth_client_id)
    except ValueError as exc:
        # verify_oauth2_token raises ValueError for ANY failure: bad signature,
        # wrong aud, wrong iss, expired, malformed. Don't leak which one.
        logger.info("ID token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired ID token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # Defence-in-depth: verify_oauth2_token already validates iss, but pin it.
    if claims.get("iss") not in _VALID_ISSUERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ID token issuer is not Google.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # The hd claim is what restricts sign-in to the institutional Workspace.
    # An empty configured hd (dev only) skips this check.
    if settings.google_oauth_hd:
        if claims.get("hd") != settings.google_oauth_hd:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Sign-in restricted to the {settings.google_oauth_hd} "
                    "Google Workspace. Use your institutional account."
                ),
            )

    if not claims.get("email_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Google account email is not verified.",
        )

    email = (claims.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ID token has no email claim.",
        )
    return email


async def _upsert_user(session: AsyncSession, email: str) -> None:
    """Insert a row for `email` if absent; idempotent on repeat calls."""
    exists = await session.scalar(select(User.email).where(User.email == email))
    if exists is not None:
        return
    session.add(User(email=email))
    try:
        await session.commit()
    except IntegrityError:
        # Two concurrent first-sign-ins for the same email — the loser sees
        # PRIMARY KEY collision. Both should succeed (row exists either way).
        await session.rollback()


async def _lazy_reset_daily(session: AsyncSession, email: str, today: date) -> None:
    """Idempotent UTC-day rollover for a user's daily counters."""
    await session.execute(
        update(User)
        .where(
            User.email == email,
            or_(User.today_date.is_(None), User.today_date != today),
        )
        .values(today_date=today, optimize_today=0, bullets_today=0)
    )


def require_user(endpoint: EndpointName):
    """FastAPI dependency factory: verify ID token + atomically bump counters.

    On `endpoint == "optimize"` the atomic UPDATE also enforces the daily
    optimize cap and raises 429 with Retry-After on overage.
    Returns the updated User row, or None when OAuth is disabled
    (settings.google_oauth_client_id == "", dev/tests).
    """

    async def _dep(
        authorization: str | None = Header(default=None),
        session: AsyncSession = Depends(get_session),
    ) -> User | None:
        # Dev / test bypass: empty client_id means the gate is off.
        if not settings.google_oauth_client_id:
            return None

        email = await _authenticate_bearer(authorization)

        # First sign-in: create the row before any counter update can target it.
        await _upsert_user(session, email)

        now = datetime.now(UTC)
        today = now.date()

        # 1) Reset daily counters if the UTC date rolled over. Idempotent.
        await _lazy_reset_daily(session, email, today)

        # 2) Atomic counter increment, with cap check for /optimize.
        if endpoint == "optimize":
            stmt = (
                update(User)
                .where(
                    User.email == email,
                    User.optimize_today < settings.max_daily_optimizations,
                )
                .values(
                    optimize_today=User.optimize_today + 1,
                    optimize_count=User.optimize_count + 1,
                    last_seen=now,
                )
            )
        elif endpoint == "parse":
            stmt = (
                update(User)
                .where(User.email == email)
                .values(
                    parse_count=User.parse_count + 1,
                    last_seen=now,
                )
            )
        else:  # recommend
            stmt = (
                update(User)
                .where(User.email == email)
                .values(
                    recommend_count=User.recommend_count + 1,
                    last_seen=now,
                )
            )

        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount == 0:
            # User row exists (_upsert_user ensured it), so the only way an
            # UPDATE returns rowcount==0 is the cap predicate (optimize only).
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Daily optimize cap reached "
                    f"({settings.max_daily_optimizations}/day). "
                    f"Resets at 00:00 UTC."
                ),
                headers={"Retry-After": str(_seconds_until_midnight_utc(now))},
            )

        user = await session.scalar(select(User).where(User.email == email))
        return user

    return _dep


async def require_admin(
    user: User | None = Depends(require_user("recommend")),
) -> User:
    """Wraps require_user and additionally enforces the ADMIN_EMAILS allowlist.

    The "recommend" endpoint key is a deliberate choice — admin browsing of
    /admin/usage is a read-only action, so charging it against the cheap
    recommend counter rather than the gated optimize counter keeps the
    rate-limit math honest. (It still increments recommend_count for the
    admin's own row, which is correct: the admin is also a user.)

    Raises 401 if the dev bypass is active (client_id empty) — admin
    endpoints must never run unauthenticated; allowing them to would be a
    silent prod misconfig of the worst kind.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Admin endpoints require Google OAuth authentication. "
                "Set GOOGLE_OAUTH_CLIENT_ID in the server environment."
            ),
        )
    if user.email not in settings.admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is not authorised for admin endpoints.",
        )
    return user


async def check_and_reserve_bullets(
    user: User | None,
    requested: int,
    session: AsyncSession,
    quota_check_time: datetime,
) -> None:
    """Atomically reserve `requested` bullets against the user's per-day cap.

    No-op when user is None (dev bypass) or requested <= 0. Same atomic
    UPDATE-with-WHERE pattern as require_user: if reserving would exceed the
    cap, rowcount == 0 and we raise 429.

    No second _lazy_reset_daily here. The require_user("optimize") dep ran
    moments ago and already reset today_date for THIS request's UTC day.
    Recomputing today + resetting here would race against a UTC rollover
    that happens mid-request: dep at 23:59:59 increments optimize_today
    against day N, then this reset at 00:00:00.001 would see today=N+1 and
    zero everything — including the increment we just made.
    """
    if user is None or requested <= 0:
        return

    email = user.email

    stmt = (
        update(User)
        .where(
            User.email == email,
            User.bullets_today + requested <= settings.max_daily_bullets,
        )
        .values(bullets_today=User.bullets_today + requested)
    )
    result = await session.execute(stmt)
    await session.commit()

    if result.rowcount == 0:
        fresh = await session.scalar(select(User).where(User.email == email))
        if fresh is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User row vanished mid-request.",
            )
        remaining = max(settings.max_daily_bullets - fresh.bullets_today, 0)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily bullet cap reached "
                f"({settings.max_daily_bullets}/day; {remaining} remaining). "
                f"Resets at 00:00 UTC."
            ),
            headers={"Retry-After": str(_seconds_until_midnight_utc(quota_check_time))},
        )
