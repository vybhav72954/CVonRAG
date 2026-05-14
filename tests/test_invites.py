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
