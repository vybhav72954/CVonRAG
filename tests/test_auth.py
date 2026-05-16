"""
tests/test_auth.py
Coverage for the Google Workspace OAuth gate + per-user daily caps.

The autouse fixture in conftest.py disables the gate by default (so the
rest of the suite keeps passing). These tests opt back in by setting
`google_oauth_client_id` to a test value AND patching `_verify_id_token`
to bypass real JWKS verification.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, settings


SAMPLE_JD = (
    "We are hiring a Senior ML Engineer with strong Python, SARIMA forecasting, "
    "and production MLOps experience. Quantitative background required."
)

OPTIMIZE_BODY = {
    "job_description": SAMPLE_JD,
    "target_role_type": "ml_engineering",
    "total_bullets_requested": 1,
    "constraints": {
        "target_char_limit": 130,
        "tolerance": 2,
        "bullet_prefix": "•",
        "max_bullets_per_project": 1,
    },
    "projects": [
        {
            "project_id": "p-001",
            "title": "Forecasting",
            "core_facts": [
                {
                    "fact_id": "f-001",
                    "text": "Built SARIMA model reducing RMSE to 0.250",
                    "tools": ["SARIMA"],
                    "metrics": ["RMSE 0.250"],
                }
            ],
        }
    ],
}

RECOMMEND_BODY = {
    "job_description": SAMPLE_JD,
    "projects": OPTIMIZE_BODY["projects"],
    "top_k": 1,
}

# Minimal payload that passes /parse's magic-byte + 100-byte length check.
_PDF_MAGIC = b"%PDF-1.4\n"
_VALID_PDF_FILE = ("cv.pdf", _PDF_MAGIC + b"x" * 200, "application/pdf")

_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
_HD = "example.org"


def _claims_for(email: str, *, hd: str = _HD, verified: bool = True) -> dict[str, Any]:
    """Build a fake decoded ID-token claim payload for a given email."""
    return {
        "email": email,
        "email_verified": verified,
        "hd": hd,
        "iss": "accounts.google.com",
        "aud": _CLIENT_ID,
        "exp": 9_999_999_999,
    }


def _token_for(email: str) -> str:
    """Opaque test token. Real verification is mocked — content is irrelevant
    except as a key the fake verifier uses to route to the right claims."""
    return f"test-token-for-{email}"


async def _mock_parse_stream(file_bytes, filename):
    """Yield a single 'done' parse event for tests."""
    yield ("done", {"total_projects": 0, "total_facts": 0})


async def _empty_sse():
    """Stand-in for _sse_stream — yields nothing so StreamingResponse closes
    immediately and we can assert on status code alone."""
    if False:
        yield ""


@pytest.fixture
def client():
    """TestClient that runs lifespan (init_db + cleanup)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def gate_enabled(monkeypatch):
    """Turn the OAuth gate on AND install a fake token verifier.

    The verifier reads the token suffix to determine which email to
    authenticate as, so each test controls identity via the Bearer header.
    Tests that need wrong-hd / unverified-email cases install their own
    patch on top of this fixture.
    """
    monkeypatch.setattr(settings, "google_oauth_client_id", _CLIENT_ID)
    monkeypatch.setattr(settings, "google_oauth_hd", _HD)

    prefix = "test-token-for-"

    async def _fake_verify(token: str, client_id: str) -> dict[str, Any]:
        assert client_id == _CLIENT_ID
        if not token.startswith(prefix):
            raise ValueError("unrecognised test token")
        email = token[len(prefix):]
        return _claims_for(email)

    monkeypatch.setattr("app.auth._verify_id_token", _fake_verify)


def _auth(email: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token_for(email)}"}


# ── Gate-disabled (autouse default) ───────────────────────────────────────────

class TestOAuthGateDisabled:
    """When google_oauth_client_id="", no Authorization header is needed."""

    def test_recommend_works_without_header(self, client):
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post("/recommend", json=RECOMMEND_BODY)
        assert resp.status_code == 200


# ── Token verification ────────────────────────────────────────────────────────

class TestOAuthAuth:
    """When the gate is on, /parse, /recommend, /optimize require a valid Bearer."""

    def test_missing_authorization_header_returns_401(self, client, gate_enabled):
        resp = client.post("/recommend", json=RECOMMEND_BODY)
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"
        assert "Authorization" in resp.json()["detail"]

    def test_malformed_authorization_header_returns_401(self, client, gate_enabled):
        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"Authorization": "Token abc"},
        )
        assert resp.status_code == 401

    def test_empty_bearer_token_returns_401(self, client, gate_enabled):
        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client, gate_enabled):
        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"Authorization": "Bearer not-a-test-token"},
        )
        assert resp.status_code == 401
        assert "Invalid or expired" in resp.json()["detail"]

    def test_wrong_hd_returns_403(self, client, gate_enabled, monkeypatch):
        async def _fake(token, client_id):
            return _claims_for("intruder@other.com", hd="other.com")
        monkeypatch.setattr("app.auth._verify_id_token", _fake)

        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 403
        assert _HD in resp.json()["detail"]

    def test_unverified_email_returns_403(self, client, gate_enabled, monkeypatch):
        async def _fake(token, client_id):
            return _claims_for("alice@example.org", verified=False)
        monkeypatch.setattr("app.auth._verify_id_token", _fake)

        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 403

    def test_valid_token_reaches_handler(self, client, gate_enabled):
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers=_auth("alice@example.org"),
            )
        assert resp.status_code == 200

    # ── /parse parity ────────────────────────────────────────────────────────

    def test_parse_missing_header_returns_401(self, client, gate_enabled):
        resp = client.post("/parse", files={"file": _VALID_PDF_FILE})
        assert resp.status_code == 401

    def test_parse_valid_token_reaches_handler(self, client, gate_enabled):
        with patch("app.main.parse_and_stream", side_effect=_mock_parse_stream):
            resp = client.post(
                "/parse", files={"file": _VALID_PDF_FILE},
                headers=_auth("alice@example.org"),
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ── Admin endpoints ───────────────────────────────────────────────────────────

class TestAdminUsage:
    """GET /admin/usage gates on OAuth + ADMIN_EMAILS allowlist."""

    def test_dev_bypass_rejects_admin(self, client):
        """With the gate off (autouse default), /admin/usage must NOT be open —
        a missing admin gate would silently expose user counters in dev."""
        resp = client.get("/admin/usage")
        assert resp.status_code == 401
        assert "GOOGLE_OAUTH_CLIENT_ID" in resp.json()["detail"]

    def test_non_admin_returns_403(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        resp = client.get("/admin/usage", headers=_auth("alice@example.org"))
        assert resp.status_code == 403

    def test_admin_email_succeeds(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        resp = client.get("/admin/usage", headers=_auth("admin@example.org"))
        assert resp.status_code == 200
        # Admin also self-registered — should appear in the readout.
        emails = {u["email"] for u in resp.json()["users"]}
        assert "admin@example.org" in emails

    def test_unknown_admin_email_returns_403(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["someone-else@example.org"])
        resp = client.get("/admin/usage", headers=_auth("alice@example.org"))
        assert resp.status_code == 403

    def test_admin_email_match_is_case_insensitive(self, client, gate_enabled, monkeypatch):
        """The field_validator on admin_emails lower-cases entries at load
        time, so an operator who writes `ADMIN_EMAILS=["Admin@Example.org"]`
        in .env still gets a working allowlist. Locks in the invariant —
        if someone later strips the normaliser, this test fails before the
        admin sees a permanent silent 403 in prod.

        Use Settings.model_validate(...) to exercise the validator instead
        of monkeypatching, since direct setattr bypasses field_validators."""
        from app.config import Settings
        s = Settings.model_validate({"admin_emails": ["Admin@Example.org", "  OTHER@example.ORG  "]})
        assert s.admin_emails == ["admin@example.org", "other@example.org"]

        # End-to-end: a config value with mixed-case entries still matches a
        # lower-cased identity (which is what the verified email claim
        # always becomes at app/auth.py).
        monkeypatch.setattr(settings, "admin_emails", s.admin_emails)
        resp = client.get("/admin/usage", headers=_auth("admin@example.org"))
        assert resp.status_code == 200


# ── Counter increments ───────────────────────────────────────────────────────

class TestCounterIncrement:
    """A gated request bumps the corresponding counter on the user row."""

    def _usage_for(self, client: TestClient, admin_email: str) -> list[dict]:
        resp = client.get("/admin/usage", headers=_auth(admin_email))
        assert resp.status_code == 200, resp.text
        return resp.json()["users"]

    def test_recommend_increments_recommend_count(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers=_auth("alice@example.org"),
            )
        rows = self._usage_for(client, "admin@example.org")
        row = next(u for u in rows if u["email"] == "alice@example.org")
        # require_admin also went through require_user("recommend"), bumping
        # the admin's row by 1. Alice's row only gets the /recommend bump.
        assert row["recommend_count"] == 1
        assert row["parse_count"] == 0
        assert row["optimize_count"] == 0

    def test_parse_increments_parse_count(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        with patch("app.main.parse_and_stream", side_effect=_mock_parse_stream):
            client.post(
                "/parse", files={"file": _VALID_PDF_FILE},
                headers=_auth("alice@example.org"),
            )
        rows = self._usage_for(client, "admin@example.org")
        row = next(u for u in rows if u["email"] == "alice@example.org")
        assert row["parse_count"] == 1
        assert row["recommend_count"] == 0
        assert row["optimize_count"] == 0

    def test_optimize_increments_both_total_and_today(
        self, client, gate_enabled, monkeypatch
    ):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            resp = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers=_auth("alice@example.org"),
            )
        assert resp.status_code == 200
        rows = self._usage_for(client, "admin@example.org")
        row = next(u for u in rows if u["email"] == "alice@example.org")
        assert row["optimize_count"] == 1
        assert row["optimize_today"] == 1
        assert row["bullets_today"] == 1


# ── Daily caps ────────────────────────────────────────────────────────────────

class TestDailyOptimizeCap:
    def test_third_optimize_returns_429_when_cap_is_2(
        self, client, gate_enabled, monkeypatch
    ):
        monkeypatch.setattr(settings, "max_daily_optimizations", 2)
        monkeypatch.setattr(settings, "max_daily_bullets", 1000)

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            r1 = client.post("/optimize", json=OPTIMIZE_BODY, headers=_auth("a@example.org"))
            r2 = client.post("/optimize", json=OPTIMIZE_BODY, headers=_auth("a@example.org"))
            r3 = client.post("/optimize", json=OPTIMIZE_BODY, headers=_auth("a@example.org"))

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        assert int(r3.headers["Retry-After"]) > 0
        assert "Daily optimize cap" in r3.json()["detail"]


class TestDailyBulletsCap:
    def test_bullet_cap_blocks_second_call(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "max_daily_optimizations", 100)
        monkeypatch.setattr(settings, "max_daily_bullets", 1)

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            r1 = client.post("/optimize", json=OPTIMIZE_BODY, headers=_auth("b@example.org"))
            r2 = client.post("/optimize", json=OPTIMIZE_BODY, headers=_auth("b@example.org"))

        assert r1.status_code == 200
        assert r2.status_code == 429
        assert "Retry-After" in r2.headers
        assert int(r2.headers["Retry-After"]) > 0
        assert "Daily bullet cap" in r2.json()["detail"]


class TestAtomicCapEnforcement:
    """Atomic UPDATE-with-WHERE: two concurrent requests at the cap boundary
    cannot both pass."""

    @pytest.mark.anyio
    async def test_concurrent_optimize_at_cap_only_one_succeeds(
        self, gate_enabled, monkeypatch
    ):
        import asyncio
        from httpx import ASGITransport, AsyncClient
        from app.main import app as _app
        from app.db import close_db, init_db

        await init_db()
        try:
            monkeypatch.setattr(settings, "max_daily_optimizations", 2)
            monkeypatch.setattr(settings, "max_daily_bullets", 1000)

            with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
                transport = ASGITransport(app=_app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    # Burn one slot first (0→1).
                    r1 = await ac.post(
                        "/optimize", json=OPTIMIZE_BODY,
                        headers=_auth("race@example.org"),
                    )
                    assert r1.status_code == 200

                    # Fire two more concurrently — only one should slip into
                    # the second slot.
                    r2, r3 = await asyncio.gather(
                        ac.post(
                            "/optimize", json=OPTIMIZE_BODY,
                            headers=_auth("race@example.org"),
                        ),
                        ac.post(
                            "/optimize", json=OPTIMIZE_BODY,
                            headers=_auth("race@example.org"),
                        ),
                    )
                    statuses = sorted([r2.status_code, r3.status_code])
                    assert statuses == [200, 429], (
                        f"Expected one success + one cap rejection, got {statuses}. "
                        "Atomic UPDATE regressed."
                    )
        finally:
            await close_db()

    @pytest.mark.anyio
    async def test_concurrent_bullet_reservation_respects_cap(
        self, gate_enabled, monkeypatch
    ):
        import asyncio
        from httpx import ASGITransport, AsyncClient
        from app.main import app as _app
        from app.db import close_db, init_db

        await init_db()
        try:
            monkeypatch.setattr(settings, "max_daily_optimizations", 1000)
            monkeypatch.setattr(settings, "max_daily_bullets", 2)

            body = {**OPTIMIZE_BODY, "total_bullets_requested": 2}

            with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
                transport = ASGITransport(app=_app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    r1, r2 = await asyncio.gather(
                        ac.post(
                            "/optimize", json=body,
                            headers=_auth("bullet@example.org"),
                        ),
                        ac.post(
                            "/optimize", json=body,
                            headers=_auth("bullet@example.org"),
                        ),
                    )
                    statuses = sorted([r1.status_code, r2.status_code])
                    assert statuses == [200, 429], (
                        f"Expected one success + one bullet-cap rejection, got {statuses}."
                    )
        finally:
            await close_db()


# ── Day rollover ──────────────────────────────────────────────────────────────

class TestDailyResetEdgeCases:
    """The lazy reset must handle NULL today_date (fresh user) AND a stale
    today_date (UTC day rolled over since last access)."""

    def test_fresh_user_first_request_sets_today_date(
        self, client, gate_enabled, monkeypatch
    ):
        from datetime import datetime, timezone
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])

        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers=_auth("fresh@example.org"),
            )
        assert resp.status_code == 200

        rows = client.get("/admin/usage", headers=_auth("admin@example.org")).json()["users"]
        row = next(u for u in rows if u["email"] == "fresh@example.org")
        today_iso = datetime.now(timezone.utc).date().isoformat()
        assert row["today_date"] == today_iso

    def test_yesterday_today_date_triggers_reset(
        self, client, gate_enabled, monkeypatch
    ):
        from datetime import datetime, timedelta, timezone
        from app.db import User, _ensure_factory_sync
        import asyncio

        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])

        # First call sets today_date + bumps counters.
        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers=_auth("yesterday@example.org"),
            )

        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

        async def _rewind():
            from sqlalchemy import update
            factory = _ensure_factory_sync()
            async with factory() as s:
                await s.execute(
                    update(User).where(User.email == "yesterday@example.org").values(
                        today_date=yesterday, optimize_today=5, bullets_today=20,
                    )
                )
                await s.commit()
        asyncio.run(_rewind())

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            resp = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers=_auth("yesterday@example.org"),
            )
        assert resp.status_code == 200

        rows = client.get("/admin/usage", headers=_auth("admin@example.org")).json()["users"]
        row = next(u for u in rows if u["email"] == "yesterday@example.org")
        today_iso = datetime.now(timezone.utc).date().isoformat()
        assert row["today_date"] == today_iso
        # Reset to 0 then incremented this request.
        assert row["optimize_today"] == 1
        assert row["bullets_today"] == 1


# ── Pagination ────────────────────────────────────────────────────────────────

class TestUsagePagination:
    """/admin/usage supports ?limit and ?offset."""

    def _seed(self, client, gate_enabled, n: int, prefix: str):
        # Sign in each user once via /recommend so the row exists.
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            for i in range(n):
                client.post(
                    "/recommend", json=RECOMMEND_BODY,
                    headers=_auth(f"{prefix}-{i:03d}@example.org"),
                )

    def test_limit_clips_results(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        self._seed(client, gate_enabled, 5, "page")
        resp = client.get("/admin/usage?limit=2", headers=_auth("admin@example.org"))
        assert resp.status_code == 200
        # Plus the admin row itself, sorted alphabetically.
        users = resp.json()["users"]
        assert len(users) == 2

    def test_offset_skips_results(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        self._seed(client, gate_enabled, 5, "skip")
        resp = client.get(
            "/admin/usage?limit=2&offset=2",
            headers=_auth("admin@example.org"),
        )
        assert resp.status_code == 200
        assert len(resp.json()["users"]) == 2

    def test_rejects_limit_over_max(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        resp = client.get(
            "/admin/usage?limit=1000",
            headers=_auth("admin@example.org"),
        )
        assert resp.status_code == 422

    def test_rejects_negative_offset(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "admin_emails", ["admin@example.org"])
        resp = client.get(
            "/admin/usage?offset=-1",
            headers=_auth("admin@example.org"),
        )
        assert resp.status_code == 422
