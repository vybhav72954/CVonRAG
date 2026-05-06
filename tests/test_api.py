"""
tests/test_api.py
FastAPI endpoint tests via TestClient (no running services needed).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_JD = (
    "We are hiring a Senior ML Engineer with strong Python, SARIMA forecasting, "
    "and production MLOps experience. Quantitative background required."
)

MINIMAL_REQUEST = {
    "job_description": SAMPLE_JD,
    "target_role_type": "ml_engineering",
    "constraints": {
        "target_char_limit": 130,
        "tolerance": 2,
        "bullet_prefix": "•",
        "max_bullets_per_project": 2,
    },
    "projects": [
        {
            "project_id": "p-001",
            "title": "Forecasting",
            "core_facts": [
                {
                    "fact_id": "f-001",
                    "text": "Built SARIMA(2,0,0)(1,0,0)[12] model reducing RMSE to 0.250",
                    "tools": ["SARIMA"],
                    "metrics": ["RMSE 0.250"],
                }
            ],
        }
    ],
}

INGEST_BODY = {
    "bullets": [
        {
            "text": "• Enhanced forecast accuracy using ARIMAX | Reduced RMSE by 13.5%",
            "role_type": "data_science",
            "uses_separator": "|",
        }
    ]
}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── GET / ─────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_200(self, client):
        assert client.get("/").status_code == 200

    def test_service_name(self, client):
        assert client.get("/").json()["service"] == "CVonRAG"


# ── GET /health ───────────────────────────────────────────────────────────────

class MockAsyncClient:
    def __init__(self, *args, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, *args, **kwargs): raise Exception("Mocked connection error")

class TestHealth:
    def test_returns_200_and_required_fields(self, client):
        with patch("app.main.collection_info", new=AsyncMock(return_value={
            "qdrant_connected": False, "collection_exists": False, "vector_count": 0,
        })), patch("httpx.AsyncClient", new=MockAsyncClient):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        for field in ("status", "model", "llm_backend", "qdrant_connected", "ollama_ok", "groq_ok", "embed_ok"):
            assert field in body


# ── POST /ingest ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_valid_returns_200(self, client):
        with patch("app.main.ingest_gold_standard_bullets", new=AsyncMock(return_value=1)):
            assert client.post("/ingest", json=INGEST_BODY).status_code == 200

    def test_upserted_count_in_response(self, client):
        with patch("app.main.ingest_gold_standard_bullets", new=AsyncMock(return_value=1)):
            assert client.post("/ingest", json=INGEST_BODY).json()["upserted"] == 1

    def test_empty_bullets_422(self, client):
        assert client.post("/ingest", json={"bullets": []}).status_code == 422

    def test_missing_body_422(self, client):
        assert client.post("/ingest", json={}).status_code == 422

    def test_text_too_short_422(self, client):
        assert client.post("/ingest", json={"bullets": [{"text": "Hi"}]}).status_code == 422

    def test_qdrant_error_returns_500(self, client):
        with patch("app.main.ingest_gold_standard_bullets",
                   new=AsyncMock(side_effect=Exception("Qdrant down"))):
            resp = client.post("/ingest", json=INGEST_BODY)
        assert resp.status_code == 500
        assert "Qdrant down" in resp.json()["detail"]


class TestIngestAuth:
    """Cover the INGEST_SECRET enforcement path. The auth gate fires only when
    settings.ingest_secret is truthy — patch.object overrides the autouse
    fixture's empty default for the duration of each test."""

    def test_returns_403_when_secret_required_and_header_missing(self, client):
        from app.main import settings
        with patch.object(settings, "ingest_secret", "real-secret"):
            resp = client.post("/ingest", json=INGEST_BODY)
        assert resp.status_code == 403
        assert "X-Ingest-Secret" in resp.json()["detail"]

    def test_returns_403_when_secret_required_and_header_wrong(self, client):
        from app.main import settings
        with patch.object(settings, "ingest_secret", "real-secret"):
            resp = client.post(
                "/ingest", json=INGEST_BODY,
                headers={"X-Ingest-Secret": "wrong-secret"},
            )
        assert resp.status_code == 403

    def test_returns_200_when_secret_required_and_header_correct(self, client):
        from app.main import settings
        with patch.object(settings, "ingest_secret", "real-secret"), \
             patch("app.main.ingest_gold_standard_bullets", new=AsyncMock(return_value=1)):
            resp = client.post(
                "/ingest", json=INGEST_BODY,
                headers={"X-Ingest-Secret": "real-secret"},
            )
        assert resp.status_code == 200
        assert resp.json()["upserted"] == 1


# ── POST /optimize ────────────────────────────────────────────────────────────

class TestOptimize:
    def test_short_jd_422(self, client):
        assert client.post("/optimize", json={
            **MINIMAL_REQUEST, "job_description": "Too short"
        }).status_code == 422

    def test_empty_projects_422(self, client):
        assert client.post("/optimize", json={
            **MINIMAL_REQUEST, "projects": []
        }).status_code == 422

    def test_invalid_role_type_422(self, client):
        assert client.post("/optimize", json={
            **MINIMAL_REQUEST, "target_role_type": "bad_role"
        }).status_code == 422

    def test_char_limit_below_min_422(self, client):
        assert client.post("/optimize", json={
            **MINIMAL_REQUEST,
            "constraints": {**MINIMAL_REQUEST["constraints"], "target_char_limit": 50},
        }).status_code == 422

    def test_valid_request_returns_sse_content_type(self, client):
        async def fake_run(request):
            yield ("token", "• Built")
            yield ("done", None)

        with patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            resp = client.post("/optimize", json=MINIMAL_REQUEST)

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_sse_data_lines_are_valid_json(self, client):
        async def fake_run(request):
            yield ("token", "hello")
            yield ("done", None)

        with patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            resp = client.post("/optimize", json=MINIMAL_REQUEST)

        for line in resp.text.splitlines():
            if line.startswith("data:"):
                parsed = json.loads(line[len("data:"):].strip())
                assert "event_type" in parsed

    def test_sse_contains_done_event(self, client):
        async def fake_run(request):
            yield ("done", None)

        with patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            resp = client.post("/optimize", json=MINIMAL_REQUEST)

        assert "event: done" in resp.text


# ── Docs ──────────────────────────────────────────────────────────────────────

class TestDocs:
    def test_swagger_ui(self, client):
        assert client.get("/docs").status_code == 200

    def test_redoc(self, client):
        assert client.get("/redoc").status_code == 200

    def test_openapi_schema_has_endpoints(self, client):
        schema = client.get("/openapi.json").json()
        assert "/optimize" in schema["paths"]
        assert "/ingest" in schema["paths"]
        assert "/health" in schema["paths"]


# ── POST /parse ───────────────────────────────────────────────────────────────

class TestParse:
    def test_no_file_returns_422(self, client):
        resp = client.post("/parse")
        assert resp.status_code == 422

    def test_unsupported_extension_returns_415(self, client):
        resp = client.post(
            "/parse",
            files={"file": ("resume.txt", b"some text", "text/plain")},
        )
        assert resp.status_code == 415

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/parse",
            files={"file": ("resume.docx", b"", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self, client):
        big = b"x" * (11 * 1024 * 1024)
        resp = client.post(
            "/parse",
            files={"file": ("resume.docx", big, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert resp.status_code == 413

    def test_valid_docx_streams_sse(self, client):
        from unittest.mock import AsyncMock, patch, MagicMock
        import json as _json

        async def _mock_stream(file_bytes, filename, http_client):
            yield ("progress", {"message": "Found 1 project.", "current": 0, "total": 1})
            yield ("project",  {
                "project": {
                    "project_id": "p-000-my-proj",
                    "title": "My Project",
                    "core_facts": [{
                        "fact_id":  "my-proj-1",
                        "text":     "Built model with 87% accuracy",
                        "tools":    ["XGBoost"],
                        "metrics":  ["87%"],
                        "outcome":  "",
                    }],
                },
                "index": 0,
                "total": 1,
            })
            yield ("done", {"total_projects": 1, "total_facts": 1})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("resume.docx", b"fake-docx-content-" + b"x" * 200, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "event: progress" in body
        assert "event: project"  in body
        assert "event: done"     in body

    def test_valid_pdf_streams_sse(self, client):
        from unittest.mock import patch

        async def _mock_stream(file_bytes, filename, http_client):
            yield ("done", {"total_projects": 0, "total_facts": 0})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("cv.pdf", b"fake-pdf-content-" + b"x" * 200, "application/pdf")},
            )

        assert resp.status_code == 200

    def test_parse_error_yields_error_event(self, client):
        from unittest.mock import patch

        async def _mock_stream(file_bytes, filename, http_client):
            yield ("error", {"error_message": "corrupt file"})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("resume.docx", b"bad-bytes-" + b"x" * 200, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert resp.status_code == 200
        assert "event: error" in resp.text
