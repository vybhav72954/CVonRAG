"""
tests/test_invites.py
Coverage for the per-batchmate invite-code gate + per-code daily caps.

The autouse fixture in conftest.py disables the gate by default (so the
rest of the suite keeps passing). These tests opt back in via
`patch.object(settings, "invite_codes_required", True)`.
"""
from __future__ import annotations

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

# A minimal payload that passes /parse's magic-byte + 100-byte length check.
# 200 bytes of filler keeps it well above the 100-byte floor.
_PDF_MAGIC = b"%PDF-1.4\n"
_VALID_PDF_FILE = ("cv.pdf", _PDF_MAGIC + b"x" * 200, "application/pdf")


async def _mock_parse_stream(file_bytes, filename):
    """
    Yield a single 'done' parse event for tests.
    
    Parameters:
        file_bytes (bytes): Input bytes from the caller (ignored).
        filename (str): Filename from the caller (ignored).
    
    Returns:
        A single tuple: `("done", {"total_projects": 0, "total_facts": 0})`, yielded once.
    """
    yield ("done", {"total_projects": 0, "total_facts": 0})


@pytest.fixture
def client():
    """
    Provide a TestClient that runs the application's lifespan so the test database and app startup/shutdown logic are executed for each test.
    
    Returns:
        TestClient: A context-managed TestClient configured with raise_server_exceptions=True for use in tests.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def gate_enabled(monkeypatch):
    """Turn the invite gate on for the duration of a test."""
    monkeypatch.setattr(settings, "invite_codes_required", True)


def _create_invite(client: TestClient, code: str, name: str = "Test") -> dict:
    """
    Create an invite via POST /admin/invites and return the created invite record.
    
    Asserts the HTTP response status is 200.
    
    Parameters:
        code (str): Invite code to create.
        name (str): Human-readable name for the invite; defaults to "Test".
    
    Returns:
        dict: The JSON-decoded response body representing the created invite.
    """
    resp = client.post("/admin/invites", json={"code": code, "name": name})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _empty_sse():
    """Stand-in for _sse_stream — yields nothing, so the StreamingResponse
    closes immediately and we can assert on the status code alone."""
    if False:
        yield ""


# ── Gate behaviour ────────────────────────────────────────────────────────────

class TestInviteGateDisabled:
    """When invite_codes_required=False (autouse default), no header needed."""

    def test_recommend_works_without_header(self, client):
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post("/recommend", json=RECOMMEND_BODY)
        assert resp.status_code == 200


class TestInviteAuth:
    """When the gate is on, /parse, /recommend, /optimize require a header."""

    def test_recommend_missing_header_returns_401(self, client, gate_enabled):
        resp = client.post("/recommend", json=RECOMMEND_BODY)
        assert resp.status_code == 401
        assert "X-Invite-Code" in resp.json()["detail"]

    def test_recommend_unknown_code_returns_401(self, client, gate_enabled):
        resp = client.post(
            "/recommend", json=RECOMMEND_BODY,
            headers={"X-Invite-Code": "BOGUS-CODE"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Unknown invite code."

    def test_recommend_valid_code_reaches_handler(self, client, gate_enabled):
        _create_invite(client, "AMAN-2K24")
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers={"X-Invite-Code": "AMAN-2K24"},
            )
        assert resp.status_code == 200

    def test_header_with_stray_whitespace_still_matches(self, client, gate_enabled):
        """
        Ensures an invite code header value with leading or trailing whitespace still matches an existing invite.
        
        Verifies that a request with an X-Invite-Code value containing surrounding whitespace is accepted (results in HTTP 200).
        """
        _create_invite(client, "TRIMTEST")
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers={"X-Invite-Code": " TRIMTEST "},
            )
        assert resp.status_code == 200

    # ── /parse parity with /recommend ────────────────────────────────────────
    # Mirrors the /recommend tests above so a regression in `require_invite("parse")`
    # wiring is caught here, not at runtime.

    def test_parse_missing_header_returns_401(self, client, gate_enabled):
        resp = client.post("/parse", files={"file": _VALID_PDF_FILE})
        assert resp.status_code == 401
        assert "X-Invite-Code" in resp.json()["detail"]

    def test_parse_unknown_code_returns_401(self, client, gate_enabled):
        resp = client.post(
            "/parse", files={"file": _VALID_PDF_FILE},
            headers={"X-Invite-Code": "BOGUS-CODE"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Unknown invite code."

    def test_parse_valid_code_reaches_handler(self, client, gate_enabled):
        _create_invite(client, "PARSE-OK")
        with patch("app.main.parse_and_stream", side_effect=_mock_parse_stream):
            resp = client.post(
                "/parse", files={"file": _VALID_PDF_FILE},
                headers={"X-Invite-Code": "PARSE-OK"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_parse_header_with_stray_whitespace_still_matches(self, client, gate_enabled):
        """
        Verifies that POST /parse accepts an invite code header with leading or trailing whitespace.
        
        Creates an invite "PTRIM", stubs the parser stream, sends a /parse request with header value " PTRIM ", and asserts the request succeeds (HTTP 200).
        """
        _create_invite(client, "PTRIM")
        with patch("app.main.parse_and_stream", side_effect=_mock_parse_stream):
            resp = client.post(
                "/parse", files={"file": _VALID_PDF_FILE},
                headers={"X-Invite-Code": " PTRIM "},
            )
        assert resp.status_code == 200


# ── Admin endpoints ───────────────────────────────────────────────────────────

class TestAdminInvites:
    """POST /admin/invites + GET /admin/usage are gated by INGEST_SECRET."""

    def test_create_invite_requires_secret_when_set(self, client):
        with patch.object(settings, "ingest_secret", "real-secret"):
            resp = client.post("/admin/invites", json={"code": "XYZ", "name": "n"})
        assert resp.status_code == 403

    def test_create_invite_with_correct_secret_returns_200(self, client):
        with patch.object(settings, "ingest_secret", "real-secret"):
            resp = client.post(
                "/admin/invites", json={"code": "XYZ", "name": "Amit"},
                headers={"X-Ingest-Secret": "real-secret"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == "XYZ"
        assert body["name"] == "Amit"
        assert body["optimize_count"] == 0

    def test_create_invite_duplicate_returns_409(self, client):
        _create_invite(client, "DUP")
        resp = client.post("/admin/invites", json={"code": "DUP", "name": "x"})
        assert resp.status_code == 409

    def test_create_invite_rejects_whitespace_in_code(self, client):
        """
        Ensure invite creation rejects codes containing whitespace.
        
        Asserts that POST /admin/invites with a code containing spaces (internal or trailing) returns HTTP 422.
        """
        resp = client.post("/admin/invites", json={"code": "BAD CODE", "name": "x"})
        assert resp.status_code == 422
        resp = client.post("/admin/invites", json={"code": "TRAIL ", "name": "x"})
        assert resp.status_code == 422

    def test_create_invite_rejects_control_chars_in_code(self, client):
        """B1: code regex `^[A-Za-z0-9_-]+$` should reject anything weird."""
        for bad in ["AB<C", "AB;C", "AB\nC", "AB/C", "AB.C"]:
            resp = client.post("/admin/invites", json={"code": bad, "name": "x"})
            assert resp.status_code == 422, f"{bad!r} should be rejected"

    def test_list_usage_returns_all_invites(self, client):
        _create_invite(client, "ALI", "Alice")
        _create_invite(client, "BOB", "Bob")
        resp = client.get("/admin/usage")
        assert resp.status_code == 200
        codes = {inv["code"] for inv in resp.json()["invites"]}
        assert codes == {"ALI", "BOB"}

    def test_list_usage_requires_secret_when_set(self, client):
        with patch.object(settings, "ingest_secret", "real-secret"):
            resp = client.get("/admin/usage")
        assert resp.status_code == 403


# ── Counter increments ────────────────────────────────────────────────────────

class TestCounterIncrement:
    """A gated request bumps the corresponding counter on the invite row."""

    def test_recommend_increments_recommend_count(self, client, gate_enabled):
        _create_invite(client, "CTR")
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers={"X-Invite-Code": "CTR"},
            )
        # Inspect via /admin/usage (autouse fixture leaves ingest_secret empty).
        invites = client.get("/admin/usage").json()["invites"]
        row = next(i for i in invites if i["code"] == "CTR")
        assert row["recommend_count"] == 1
        assert row["parse_count"] == 0
        assert row["optimize_count"] == 0

    def test_parse_increments_parse_count(self, client, gate_enabled):
        """A successful /parse must bump parse_count but not the other counters."""
        _create_invite(client, "PCNT")
        with patch("app.main.parse_and_stream", side_effect=_mock_parse_stream):
            client.post(
                "/parse", files={"file": _VALID_PDF_FILE},
                headers={"X-Invite-Code": "PCNT"},
            )
        row = next(
            i for i in client.get("/admin/usage").json()["invites"]
            if i["code"] == "PCNT"
        )
        assert row["parse_count"] == 1
        assert row["recommend_count"] == 0
        assert row["optimize_count"] == 0

    def test_optimize_increments_both_total_and_today(self, client, gate_enabled):
        _create_invite(client, "OPT")
        # Mock _sse_stream so we don't actually run the LLM pipeline.
        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            resp = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers={"X-Invite-Code": "OPT"},
            )
        assert resp.status_code == 200
        row = next(
            i for i in client.get("/admin/usage").json()["invites"]
            if i["code"] == "OPT"
        )
        assert row["optimize_count"] == 1
        assert row["optimize_today"] == 1
        # 1 bullet was requested (OPTIMIZE_BODY.total_bullets_requested=1)
        assert row["bullets_today"] == 1


# ── Daily caps ────────────────────────────────────────────────────────────────

class TestDailyOptimizeCap:
    """After max_daily_optimizations calls, the next call returns 429."""

    def test_third_optimize_returns_429_when_cap_is_2(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "max_daily_optimizations", 2)
        # Generous bullet budget so the optimize-count cap fires, not the bullet cap.
        monkeypatch.setattr(settings, "max_daily_bullets", 1000)
        _create_invite(client, "CAP1")

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            r1 = client.post("/optimize", json=OPTIMIZE_BODY, headers={"X-Invite-Code": "CAP1"})
            r2 = client.post("/optimize", json=OPTIMIZE_BODY, headers={"X-Invite-Code": "CAP1"})
            r3 = client.post("/optimize", json=OPTIMIZE_BODY, headers={"X-Invite-Code": "CAP1"})

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        assert int(r3.headers["Retry-After"]) > 0
        assert "Daily optimize cap" in r3.json()["detail"]


class TestDailyBulletsCap:
    """Bullets-per-day cap fires even when the optimize-count cap is fine."""

    def test_bullet_cap_blocks_second_call(self, client, gate_enabled, monkeypatch):
        monkeypatch.setattr(settings, "max_daily_optimizations", 100)
        monkeypatch.setattr(settings, "max_daily_bullets", 1)
        _create_invite(client, "BCAP")

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            r1 = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers={"X-Invite-Code": "BCAP"},
            )
            r2 = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers={"X-Invite-Code": "BCAP"},
            )

        assert r1.status_code == 200
        assert r2.status_code == 429
        # Mirror the optimize-cap test: bullet-cap 429s must also carry a
        # Retry-After header pointing at the next UTC midnight, so the
        # frontend can render a useful countdown.
        assert "Retry-After" in r2.headers
        assert int(r2.headers["Retry-After"]) > 0
        assert "Daily bullet cap" in r2.json()["detail"]


class TestAtomicCapEnforcement:
    """B2/B3: cap checks must be atomic. Two requests simultaneously at the
    cap boundary cannot both pass. With atomic UPDATE + WHERE, SQLite
    serializes the writes and the second one's WHERE clause fails."""

    @pytest.mark.anyio
    async def test_concurrent_optimize_at_cap_only_one_succeeds(
        self, gate_enabled, monkeypatch
    ):
        """Fire two concurrent /optimize calls at optimize_today=1 with cap=2.
        Both pass the cap (1 → 2 and 2 → 3 would have been the racy outcome
        of the old SELECT-then-mutate; with atomic UPDATE the second one is
        rejected). Validates B2 fix.
        """
        import asyncio
        from httpx import ASGITransport, AsyncClient
        from app.main import app
        from app.db import init_db

        # AsyncClient doesn't fire FastAPI's lifespan → init_db never runs.
        # Do it manually here (the autouse fixture already pointed sqlite_path
        # at a tmp tmp_path db).
        await init_db()

        monkeypatch.setattr(settings, "max_daily_optimizations", 2)
        monkeypatch.setattr(settings, "max_daily_bullets", 1000)

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                # Pre-create the invite (sync TestClient inside async test is
                # awkward, so we POST via the same client).
                r = await ac.post(
                    "/admin/invites",
                    json={"code": "RACECAP", "name": "Race"},
                )
                assert r.status_code == 200, r.text

                # First call burns one of the two slots (optimize_today goes 0→1).
                r1 = await ac.post(
                    "/optimize",
                    json=OPTIMIZE_BODY,
                    headers={"X-Invite-Code": "RACECAP"},
                )
                assert r1.status_code == 200

                # Now fire two MORE concurrent calls. Only one should slip
                # into the second slot (1→2); the other must 429.
                r2, r3 = await asyncio.gather(
                    ac.post(
                        "/optimize",
                        json=OPTIMIZE_BODY,
                        headers={"X-Invite-Code": "RACECAP"},
                    ),
                    ac.post(
                        "/optimize",
                        json=OPTIMIZE_BODY,
                        headers={"X-Invite-Code": "RACECAP"},
                    ),
                )
                statuses = sorted([r2.status_code, r3.status_code])
                # Exactly one 200, exactly one 429 — never two 200s.
                assert statuses == [200, 429], (
                    f"Expected one success + one cap rejection, got {statuses}. "
                    "If both are 200, B2's atomic UPDATE regressed."
                )

    @pytest.mark.anyio
    async def test_concurrent_bullet_reservation_respects_cap(
        self, gate_enabled, monkeypatch
    ):
        """B3 analogue: with bullet cap=2 and each request asking for 2
        bullets, two concurrent requests cannot both reserve (which would
        leave bullets_today=4)."""
        import asyncio
        from httpx import ASGITransport, AsyncClient
        from app.main import app
        from app.db import init_db

        await init_db()  # AsyncClient skips lifespan; init manually

        monkeypatch.setattr(settings, "max_daily_optimizations", 1000)
        monkeypatch.setattr(settings, "max_daily_bullets", 2)

        # Use a body that requests 2 bullets.
        body = {**OPTIMIZE_BODY, "total_bullets_requested": 2}

        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                r = await ac.post(
                    "/admin/invites",
                    json={"code": "BULLETRACE", "name": "BR"},
                )
                assert r.status_code == 200, r.text

                r1, r2 = await asyncio.gather(
                    ac.post(
                        "/optimize",
                        json=body,
                        headers={"X-Invite-Code": "BULLETRACE"},
                    ),
                    ac.post(
                        "/optimize",
                        json=body,
                        headers={"X-Invite-Code": "BULLETRACE"},
                    ),
                )
                statuses = sorted([r1.status_code, r2.status_code])
                assert statuses == [200, 429], (
                    f"Expected one success + one bullet-cap rejection, got {statuses}. "
                    "If both are 200, B3's atomic UPDATE regressed."
                )


class TestDailyResetEdgeCases:
    """B10 regression coverage: SQL `today_date != today` excludes rows where
    today_date IS NULL (a fresh invite). The reset must explicitly handle
    NULL, or daily counters accumulate across UTC days forever."""

    def test_fresh_invite_first_request_sets_today_date(self, client, gate_enabled):
        """
        Ensure a newly created invite has its today_date set after the first gated request.
        
        Verifies that an invite whose `today_date` is initially NULL is updated to the current UTC date after a successful gated request, causing the daily counters to initialize for that invite.
        """
        from datetime import datetime, timezone
        _create_invite(client, "FRESHX")
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers={"X-Invite-Code": "FRESHX"},
            )
        assert resp.status_code == 200
        row = next(
            i for i in client.get("/admin/usage").json()["invites"]
            if i["code"] == "FRESHX"
        )
        today_iso = datetime.now(timezone.utc).date().isoformat()
        assert row["today_date"] == today_iso, (
            f"Expected today_date={today_iso} after first request, got "
            f"{row['today_date']!r}. B10 regression: lazy_reset_daily's "
            f"WHERE clause is excluding NULL today_date rows."
        )

    def test_yesterday_today_date_triggers_reset(self, client, gate_enabled, monkeypatch):
        """If today_date is yesterday, the lazy reset must fire on the next
        request and zero out optimize_today / bullets_today."""
        from datetime import datetime, timedelta, timezone
        from app.db import Invite, _ensure_factory_sync
        import asyncio

        _create_invite(client, "YESTERDAY")
        # Burn one optimize to set today_date AND optimize_today=1.
        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers={"X-Invite-Code": "YESTERDAY"},
            )

        # Manually rewind today_date one day to simulate UTC rollover. MUST
        # use UTC here — production uses datetime.now(timezone.utc).date(), so
        # using local date.today() can mismatch in non-UTC timezones (e.g.
        # IST is +5:30 ahead, giving a ~5.5-hour window where local-yesterday
        # equals UTC-today and the test flakes).
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

        async def _rewind():
            """
            Set the invite with code "YESTERDAY" to a previous UTC date and preset daily counters.
            
            Updates that invite's `today_date` to the `yesterday` value and sets `optimize_today` to 5 and `bullets_today` to 20 so tests can simulate an invite whose daily counters belong to the previous UTC day.
            """
            from sqlalchemy import update
            factory = _ensure_factory_sync()
            async with factory() as s:
                await s.execute(
                    update(Invite).where(Invite.code == "YESTERDAY").values(
                        today_date=yesterday, optimize_today=5, bullets_today=20,
                    )
                )
                await s.commit()
        asyncio.run(_rewind())

        # Next request — reset should fire and zero the daily counters.
        with patch("app.main._sse_stream", new=lambda body: _empty_sse()):
            resp = client.post(
                "/optimize", json=OPTIMIZE_BODY,
                headers={"X-Invite-Code": "YESTERDAY"},
            )
        assert resp.status_code == 200
        row = next(
            i for i in client.get("/admin/usage").json()["invites"]
            if i["code"] == "YESTERDAY"
        )
        today_iso = datetime.now(timezone.utc).date().isoformat()
        assert row["today_date"] == today_iso
        # Reset to 0, then incremented to 1 for this request.
        assert row["optimize_today"] == 1
        # bullets_today resets to 0 then reserves 1 for this request (OPTIMIZE_BODY).
        assert row["bullets_today"] == 1


class TestAdminInviteIdempotency:
    """B11 regression coverage: concurrent /admin/invites with same code must
    return 409, not 500 from an unhandled IntegrityError."""

    def test_duplicate_code_returns_409_not_500(self, client):
        """Pre-existing test creates DUP via the happy path. Re-create to
        confirm IntegrityError is still translated to 409 after dropping the
        pre-check."""
        _create_invite(client, "RACE-DUP")
        resp = client.post(
            "/admin/invites",
            json={"code": "RACE-DUP", "name": "second"},
        )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]


class TestUsagePagination:
    """B8: /admin/usage supports ?limit and ?offset."""

    def test_default_returns_up_to_100(self, client):
        for i in range(5):
            _create_invite(client, f"BATCH-{i:03d}")
        resp = client.get("/admin/usage")
        assert resp.status_code == 200
        assert len(resp.json()["invites"]) == 5

    def test_limit_clips_results(self, client):
        for i in range(5):
            _create_invite(client, f"PAGE-{i:03d}")
        resp = client.get("/admin/usage?limit=2")
        assert resp.status_code == 200
        codes = [i["code"] for i in resp.json()["invites"]]
        assert codes == ["PAGE-000", "PAGE-001"]

    def test_offset_skips_results(self, client):
        for i in range(5):
            _create_invite(client, f"SKIP-{i:03d}")
        resp = client.get("/admin/usage?limit=2&offset=2")
        codes = [i["code"] for i in resp.json()["invites"]]
        assert codes == ["SKIP-002", "SKIP-003"]

    def test_rejects_limit_over_max(self, client):
        resp = client.get("/admin/usage?limit=1000")
        assert resp.status_code == 422

    def test_rejects_negative_offset(self, client):
        resp = client.get("/admin/usage?offset=-1")
        assert resp.status_code == 422
