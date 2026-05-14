"""
CVonRAG — auth.py
Invite-code auth + per-code usage tracking.

One invite code per batchmate. The code IS the identity (no OAuth — see plan).
The X-Invite-Code request header is required on /parse, /recommend, /optimize.

CONCURRENCY MODEL (B2/B3 fix)
─────────────────────────────
All counter updates use atomic conditional UPDATE statements rather than
SELECT-then-mutate. SQLite serializes write transactions, so two concurrent
/optimize calls at `optimize_today = 19` cannot both bump to 20:

    UPDATE invites
       SET optimize_today = optimize_today + 1, ...
     WHERE code = ?
       AND optimize_today < 20

The second request reads 20 inside its own transaction, the WHERE clause is
false, `rowcount == 0`, and we surface a 429. The naive read-mutate-commit
pattern had a real race here because two readers could both see 19 before
either committed; this implementation closes that window.

PRE-INCREMENT (still)
─────────────────────
A failing /optimize still counts toward the daily quota. Otherwise an abusive
caller could trigger many failures without exhausting their daily budget,
which is exactly the behavior the cap is meant to prevent.

BULLET RESERVATION
──────────────────
`check_and_reserve_bullets` runs AFTER body parsing in /optimize (so the
caller knows `total_bullets_requested`). Same atomic-UPDATE pattern: a single
conditional statement either commits the reservation or rejects on cap.
"""

from __future__ import annotations
import logging
from datetime import date, datetime, timedelta, timezone
from math import ceil
from typing import Literal

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import Invite, get_session

logger = logging.getLogger("cvonrag")
settings = get_settings()

EndpointName = Literal["parse", "recommend", "optimize"]


def _seconds_until_midnight_utc() -> int:
    """Return integer seconds until the next 00:00 UTC. Used as Retry-After
    when an invite hits its daily cap — the cap resets at UTC midnight."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(int(ceil((tomorrow - now).total_seconds())), 1)


async def _lazy_reset_daily(session: AsyncSession, code: str, today: date) -> None:
    """Zero out daily counters when today's date has rolled over.

    Idempotent under concurrency: the WHERE clause makes this a no-op on every
    call after the first one each UTC day. Runs in its own statement so two
    racing requests crossing midnight UTC don't both write the reset (the
    second one's WHERE clause fails after the first one commits).

    B10: must explicitly cover `today_date IS NULL` (a fresh invite that has
    never had a request). In SQL, `NULL != today` evaluates to NULL (not
    TRUE), so a plain `!=` excludes those rows and the reset never fires —
    daily counters accumulate forever across UTC days. The previous Python
    implementation worked because `None != date(...)` is True in Python,
    but moving the comparison into SQL changes the semantic.
    """
    await session.execute(
        update(Invite)
        .where(
            Invite.code == code,
            or_(Invite.today_date.is_(None), Invite.today_date != today),
        )
        .values(today_date=today, optimize_today=0, bullets_today=0)
    )


async def _invite_exists(session: AsyncSession, code: str) -> bool:
    return (
        await session.scalar(select(Invite.code).where(Invite.code == code))
    ) is not None


def require_invite(endpoint: EndpointName):
    """FastAPI dependency factory: validate X-Invite-Code, bump the
    per-endpoint counter atomically, and enforce the daily-optimize cap.

    Returns the post-update `Invite` row so the route handler can chain into
    `check_and_reserve_bullets` for the bullet-level cap. Returns None when
    the gate is disabled (dev / tests).
    """

    async def _dep(
        x_invite_code: str | None = Header(default=None),
        session: AsyncSession = Depends(get_session),
    ) -> Invite | None:
        if not settings.invite_codes_required:
            return None

        # Strip whitespace defensively (B1) — InviteCreate.code rejects
        # whitespace at creation, so a stored code never contains it. The
        # strip here just absorbs copy-paste artifacts on the user side.
        code = (x_invite_code or "").strip()
        if not code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Invite-Code header.",
            )

        now = datetime.now(timezone.utc)
        today = now.date()

        # 1) Reset daily counters if the UTC date rolled over. Idempotent.
        await _lazy_reset_daily(session, code, today)

        # 2) Atomic counter increment, with cap check for /optimize.
        if endpoint == "optimize":
            stmt = (
                update(Invite)
                .where(
                    Invite.code == code,
                    Invite.optimize_today < settings.max_daily_optimizations,
                )
                .values(
                    optimize_today=Invite.optimize_today + 1,
                    optimize_count=Invite.optimize_count + 1,
                    last_seen=now,
                )
            )
        elif endpoint == "parse":
            stmt = (
                update(Invite)
                .where(Invite.code == code)
                .values(
                    parse_count=Invite.parse_count + 1,
                    last_seen=now,
                )
            )
        else:  # recommend
            stmt = (
                update(Invite)
                .where(Invite.code == code)
                .values(
                    recommend_count=Invite.recommend_count + 1,
                    last_seen=now,
                )
            )

        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount == 0:
            # Two possible reasons: code doesn't exist, OR (optimize-only) the
            # cap was already at the limit. Disambiguate with one extra read.
            # This read only fires on the rejection path; the happy path is
            # still one round-trip.
            if not await _invite_exists(session, code):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unknown invite code.",
                )
            # Code exists → must have been the cap check (only fires for optimize).
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Daily optimize cap reached "
                    f"({settings.max_daily_optimizations}/day). "
                    f"Resets at 00:00 UTC."
                ),
                headers={"Retry-After": str(_seconds_until_midnight_utc())},
            )

        # Fetch the updated row for the handler. check_and_reserve_bullets
        # only reads `invite.code`, but returning the full row keeps the API
        # surface honest with the type annotation.
        invite = await session.scalar(select(Invite).where(Invite.code == code))
        return invite

    return _dep


async def check_and_reserve_bullets(
    invite: Invite | None,
    requested: int,
    session: AsyncSession,
) -> None:
    """Pre-reserve `requested` bullets against the per-code daily bullet cap.

    Single atomic UPDATE — the WHERE clause guarantees that no two concurrent
    callers can both reserve when the cap would be exceeded together (B3).
    Called from /optimize after Pydantic body parsing, since `requested`
    isn't known until the request body is validated.
    """
    if invite is None or requested <= 0:
        return

    code = invite.code
    today = datetime.now(timezone.utc).date()

    # Idempotent daily reset — same pattern as the dep.
    await _lazy_reset_daily(session, code, today)

    # Atomic reservation: only commits if bullets_today + requested fits.
    stmt = (
        update(Invite)
        .where(
            Invite.code == code,
            Invite.bullets_today + requested <= settings.max_daily_bullets,
        )
        .values(bullets_today=Invite.bullets_today + requested)
    )
    result = await session.execute(stmt)
    await session.commit()

    if result.rowcount == 0:
        # Cap would be exceeded (or — vanishingly unlikely — the invite was
        # revoked between the dep and now). Re-read to compute remaining for
        # the error message.
        fresh = await session.scalar(select(Invite).where(Invite.code == code))
        if fresh is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invite revoked mid-request.",
            )
        remaining = max(settings.max_daily_bullets - fresh.bullets_today, 0)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily bullet cap reached "
                f"({settings.max_daily_bullets}/day; {remaining} remaining). "
                f"Resets at 00:00 UTC."
            ),
            headers={"Retry-After": str(_seconds_until_midnight_utc())},
        )
