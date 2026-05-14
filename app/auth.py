"""
CVonRAG — auth.py
Invite-code auth + per-code usage tracking.

One invite code per batchmate. The code IS the identity (no OAuth — see plan).
The X-Invite-Code request header is required on /parse, /recommend, /optimize.
Each gated request:

  1. Looks up the code in the SQLite `invites` table.
  2. Updates last_seen.
  3. Resets the daily counters if a new UTC day has begun.
  4. For /optimize: checks the daily-optimize cap (429 if exceeded).
  5. Increments the per-endpoint counter (and optimize_today if applicable).

Why pre-increment rather than post: pre-incrementing means a failing request
still counts toward the quota, so an abusive caller cannot trigger many
failures without spending quota. A daily cap is approximate by design.

Bullet-level cap (max_daily_bullets) is checked AFTER body parsing in the
/optimize route, because the bullet count isn't known until Pydantic
validates the request — see `app/main.py` /optimize route for that path.
"""

from __future__ import annotations
import logging
from datetime import date, datetime, timedelta, timezone
from math import ceil
from typing import Literal

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
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


def _maybe_reset_daily(invite: Invite, today: date) -> None:
    """Zero out optimize_today/bullets_today if today_date != today.
    Mutates the invite row in place — caller is responsible for commit."""
    if invite.today_date != today:
        invite.today_date = today
        invite.optimize_today = 0
        invite.bullets_today = 0


def require_invite(endpoint: EndpointName):
    """Return a FastAPI dependency that validates the X-Invite-Code header,
    bumps counters for `endpoint`, and enforces the daily-optimize cap.

    The dependency yields the Invite row (or None when invite_codes_required
    is False) so the route handler can read remaining bullet quota.
    """

    async def _dep(
        x_invite_code: str | None = Header(default=None),
        session: AsyncSession = Depends(get_session),
    ) -> Invite | None:
        # Bypass entirely when the feature is disabled (tests + local dev).
        if not settings.invite_codes_required:
            return None

        if not x_invite_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Invite-Code header.",
            )

        invite = await session.scalar(
            select(Invite).where(Invite.code == x_invite_code)
        )
        if invite is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown invite code.",
            )

        now = datetime.now(timezone.utc)
        invite.last_seen = now
        _maybe_reset_daily(invite, now.date())

        # Daily-optimize cap fires BEFORE incrementing so the 21st call is
        # rejected, not allowed-then-counted.
        if endpoint == "optimize":
            if invite.optimize_today >= settings.max_daily_optimizations:
                retry_after = _seconds_until_midnight_utc()
                # Persist the lazy reset bookkeeping even on rejection so the
                # row reflects today's date (commit before raising).
                await session.commit()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Daily optimize cap reached "
                        f"({settings.max_daily_optimizations}/day). "
                        f"Resets at 00:00 UTC."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )
            invite.optimize_today += 1
            invite.optimize_count += 1
        elif endpoint == "parse":
            invite.parse_count += 1
        elif endpoint == "recommend":
            invite.recommend_count += 1

        await session.commit()
        # Refresh so subsequent reads in the same request see the committed
        # row (FastAPI handlers don't share the session with the dep).
        return invite

    return _dep


async def check_and_reserve_bullets(
    invite: Invite | None,
    requested: int,
    session: AsyncSession,
) -> None:
    """Called from the /optimize route after Pydantic body parsing.

    Pre-reserves `requested` bullets against the per-code daily bullet cap.
    Raises 429 if reservation would exceed the cap. No-op when the gate is
    disabled (invite is None) or when requested <= 0.
    """
    if invite is None or requested <= 0:
        return

    # The dep already reset daily counters today; re-read fresh row in this
    # session because the dep's session has been closed.
    fresh = await session.scalar(select(Invite).where(Invite.code == invite.code))
    if fresh is None:
        # Vanishingly unlikely — the row was just touched by the dep. Treat as
        # auth failure rather than silently allowing.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invite revoked mid-request.",
        )

    _maybe_reset_daily(fresh, datetime.now(timezone.utc).date())

    if fresh.bullets_today + requested > settings.max_daily_bullets:
        await session.commit()
        retry_after = _seconds_until_midnight_utc()
        remaining = max(settings.max_daily_bullets - fresh.bullets_today, 0)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily bullet cap reached "
                f"({settings.max_daily_bullets}/day; {remaining} remaining). "
                f"Resets at 00:00 UTC."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    fresh.bullets_today += requested
    await session.commit()
