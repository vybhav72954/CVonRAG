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


@pytest.fixture
def client():
    """Fresh TestClient (and lifespan) per test so the SQLite DB is rebuilt."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def gate_enabled(monkeypatch):
    """Turn the invite gate on for the duration of a test."""
    monkeypatch.setattr(settings, "invite_codes_required", True)


def _create_invite(client: TestClient, code: str, name: str = "Test") -> dict:
    """Create an invite via POST /admin/invites. ingest_secret is "" by
    default (autouse fixture), so no header is needed."""
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
        """B1 defense in depth: copy-paste often appends a trailing space.
        The header lookup strips before comparing so the user isn't 401'd
        for an invisible character."""
        _create_invite(client, "TRIMTEST")
        with patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            resp = client.post(
                "/recommend", json=RECOMMEND_BODY,
                headers={"X-Invite-Code": " TRIMTEST "},
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
        """B1: trailing whitespace in code would silently 401 the user later."""
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
