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
    """
    Compute the number of whole seconds until the next 00:00 UTC.
    
    Returns:
        seconds (int): Seconds until the next UTC midnight, rounded up to the next integer and at least 1.
    """
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(int(ceil((tomorrow - now).total_seconds())), 1)


async def _lazy_reset_daily(session: AsyncSession, code: str, today: date) -> None:
    """
    Perform an idempotent UTC-day rollover for an invite's daily counters.
    
    If the invite's stored `today_date` is NULL or not equal to `today`, sets `today_date` to `today` and resets `optimize_today` and `bullets_today` to 0. The update is safe to call concurrently and has no effect when the stored date already matches `today`.
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
    """
    Check whether an invite with the given code exists.
    
    Parameters:
        code (str): The invite code to look up.
    
    Returns:
        bool: True if an invite row with the code exists, False otherwise.
    """
    return (
        await session.scalar(select(Invite.code).where(Invite.code == code))
    ) is not None


def require_invite(endpoint: EndpointName):
    """
    Create a FastAPI dependency that enforces invite-code authentication and atomically updates per-endpoint usage counters.
    
    This factory returns a dependency callable which:
    - Validates the `X-Invite-Code` header (raises 401 if missing or unknown when invite codes are required).
    - Performs an idempotent UTC-midnight rollover of daily counters before updating.
    - Atomically increments the appropriate per-endpoint counter and `last_seen`; for `endpoint == "optimize"` the update enforces the daily optimize cap and raises 429 with a `Retry-After` header when the cap is reached.
    - Returns the updated `Invite` row for downstream handlers, or `None` when invite-code checks are disabled by configuration.
    
    Parameters:
        endpoint (EndpointName): Which endpoint the dependency will protect; one of `"parse"`, `"recommend"`, or `"optimize"`.
    
    Returns:
        A FastAPI dependency callable that yields the updated `Invite` row when invite checks are active, or `None` when the invite gate is disabled.
    """

    async def _dep(
        x_invite_code: str | None = Header(default=None),
        session: AsyncSession = Depends(get_session),
    ) -> Invite | None:
        """
        Validate and atomically record per-invite usage for the specified endpoint and return the updated Invite row.
        
        Parameters:
            x_invite_code (str | None): Raw value of the `X-Invite-Code` header; leading/trailing whitespace is tolerated and will be stripped.
            session (AsyncSession): Database session used for atomic updates and reads.
        
        Returns:
            Invite | None: The updated Invite row after the increment, or `None` when invite enforcement is disabled via configuration.
        
        Raises:
            HTTPException: 401 Unauthorized if the header is missing/empty or the invite code does not exist.
            HTTPException: 429 Too Many Requests if the optimize endpoint has reached its per-code daily cap; response includes a `Retry-After` header indicating seconds until UTC midnight.
        """
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
    """
    Reserve the specified number of bullets from an invite's per-day quota, atomically updating the invite row.
    
    If `invite` is None or `requested` is less than or equal to 0, the function returns without action. The function performs a UTC-day rollover for the invite's daily counters before attempting an atomic reservation that ensures the per-day bullet cap is not exceeded.
    
    Parameters:
        invite (Invite | None): The invite row to charge bullets against; when `None` the function is a no-op.
        requested (int): Number of bullets to reserve for this request.
    
    Raises:
        HTTPException 401: If the invite was removed between the reservation attempt and the follow-up read ("Invite revoked mid-request.").
        HTTPException 429: If reserving `requested` bullets would exceed the invite's remaining daily bullets. The response includes a `Retry-After` header with seconds until 00:00 UTC and a detail message stating the daily cap and remaining bullets.
    """
    if invite is None or requested <= 0:
        return

    code = invite.code

    # No second `_lazy_reset_daily` here. The require_invite("optimize") dep
    # ran moments ago and already reset today_date for THIS request's UTC
    # day. Recomputing today + resetting here would race against a UTC
    # rollover that happens mid-request: dep at 23:59:59 increments
    # optimize_today against day N, then this reset at 00:00:00.001 would
    # see today=N+1 and zero everything — including the increment we just
    # made. Trusting the dep keeps both the increment and the reservation
    # on the same day.

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
