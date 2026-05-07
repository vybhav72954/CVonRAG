"""
tests/test_api.py
FastAPI endpoint tests via TestClient (no running services needed).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
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

# Magic-byte prefixes used to build realistic test file payloads (H7)
_DOCX_MAGIC = b"PK\x03\x04"           # ZIP / OOXML container
_PDF_MAGIC  = b"%PDF-1.4\n"           # PDF header (any version starts with %PDF)

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

    def test_invalid_role_type_rejected_at_ingest(self, client):
        """N6: a typo in role_type fails fast with 422 — never poisons Qdrant."""
        body = {"bullets": [{"text": "Built SARIMA forecasting model", "role_type": "data_sciecne"}]}
        resp = client.post("/ingest", json=body)
        assert resp.status_code == 422

    def test_valid_enum_role_type_accepted(self, client):
        """N6: the six enum values are accepted; default ('general') still works."""
        body = {"bullets": [{"text": "Built SARIMA forecasting model", "role_type": "data_science"}]}
        with patch("app.main.ingest_gold_standard_bullets", new=AsyncMock(return_value=1)):
            assert client.post("/ingest", json=body).status_code == 200


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

    def test_too_many_bullets_for_groq_returns_422(self, client):
        """H3: /optimize rejects requests that exceed groq_max_bullets_per_request."""
        from app.main import settings
        with patch.object(settings, "groq_api_key", "test-key"), \
             patch.object(settings, "groq_max_bullets_per_request", 2):
            resp = client.post("/optimize", json={
                **MINIMAL_REQUEST,
                "total_bullets_requested": 3,
            })
        assert resp.status_code == 422
        detail = resp.json()["detail"].lower()
        assert "groq" in detail or "bullets" in detail

    def test_groq_bullet_cap_not_triggered_when_within_limit(self, client):
        """H3: requests within the Groq cap proceed normally."""
        async def fake_run(request):
            yield ("done", None)

        from app.main import settings
        with patch.object(settings, "groq_api_key", "test-key"), \
             patch.object(settings, "groq_max_bullets_per_request", 5), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            resp = client.post("/optimize", json={
                **MINIMAL_REQUEST,
                "total_bullets_requested": 1,
            })
        assert resp.status_code == 200

    def test_groq_bullet_cap_skipped_when_no_api_key(self, client):
        """H3: cap does not fire when GROQ_API_KEY is unset (Ollama mode)."""
        async def fake_run(request):
            yield ("done", None)

        from app.main import settings
        with patch.object(settings, "groq_api_key", ""), \
             patch.object(settings, "groq_max_bullets_per_request", 2), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            resp = client.post("/optimize", json={
                **MINIMAL_REQUEST,
                "total_bullets_requested": 3,
            })
        assert resp.status_code == 200



# ── Rate limiting (H1) ────────────────────────────────────────────────────────

_RECOMMEND_BODY = {
    "job_description": SAMPLE_JD,
    "projects": MINIMAL_REQUEST["projects"],
    "top_k": 3,
}


class TestRateLimit:
    """H1: per-IP sliding-window rate limiting on /parse, /recommend, /optimize."""

    def test_optimize_429_after_limit_exceeded(self, client):
        async def fake_run(_r):
            yield ("done", None)

        from app.main import settings, _limiter
        with patch.object(settings, "rate_limit_enabled", True), \
             patch.object(settings, "rate_limit_optimize", 2), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            _limiter._windows.clear()
            client.post("/optimize", json=MINIMAL_REQUEST)
            client.post("/optimize", json=MINIMAL_REQUEST)
            resp = client.post("/optimize", json=MINIMAL_REQUEST)  # 3rd exceeds cap of 2
        assert resp.status_code == 429

    def test_optimize_retry_after_header_present(self, client):
        async def fake_run(_r):
            yield ("done", None)

        from app.main import settings, _limiter
        with patch.object(settings, "rate_limit_enabled", True), \
             patch.object(settings, "rate_limit_optimize", 1), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            _limiter._windows.clear()
            client.post("/optimize", json=MINIMAL_REQUEST)
            resp = client.post("/optimize", json=MINIMAL_REQUEST)
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) > 0

    def test_parse_429_after_limit_exceeded(self, client):
        async def _mock_stream(file_bytes, filename, http_client):
            yield ("done", {"total_projects": 0, "total_facts": 0})

        from app.main import settings, _limiter
        _file = ("cv.pdf", _PDF_MAGIC + b"x" * 200, "application/pdf")
        with patch.object(settings, "rate_limit_enabled", True), \
             patch.object(settings, "rate_limit_parse", 1), \
             patch("app.main.parse_and_stream", side_effect=_mock_stream):
            _limiter._windows.clear()
            client.post("/parse", files={"file": _file})
            resp = client.post("/parse", files={"file": _file})
        assert resp.status_code == 429

    def test_recommend_429_after_limit_exceeded(self, client):
        from app.main import settings, _limiter
        with patch.object(settings, "rate_limit_enabled", True), \
             patch.object(settings, "rate_limit_recommend", 1), \
             patch("app.main._do_recommend", new=AsyncMock(return_value=[])):
            _limiter._windows.clear()
            client.post("/recommend", json=_RECOMMEND_BODY)
            resp = client.post("/recommend", json=_RECOMMEND_BODY)
        assert resp.status_code == 429

    def test_rate_limit_disabled_allows_unlimited_calls(self, client):
        """When RATE_LIMIT_ENABLED=false, no 429 is returned regardless of call count."""
        async def fake_run(_r):
            yield ("done", None)

        from app.main import settings, _limiter
        with patch.object(settings, "rate_limit_enabled", False), \
             patch.object(settings, "rate_limit_optimize", 1), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            _limiter._windows.clear()
            for _ in range(3):
                resp = client.post("/optimize", json=MINIMAL_REQUEST)
        assert resp.status_code == 200

    def test_different_keys_tracked_independently(self, client):
        """Rate limit state for /parse and /optimize are independent buckets."""
        async def _mock_stream(file_bytes, filename, http_client):
            yield ("done", {"total_projects": 0, "total_facts": 0})

        async def fake_run(_r):
            yield ("done", None)

        from app.main import settings, _limiter
        _file = ("cv.pdf", _PDF_MAGIC + b"x" * 200, "application/pdf")
        with patch.object(settings, "rate_limit_enabled", True), \
             patch.object(settings, "rate_limit_parse", 1), \
             patch.object(settings, "rate_limit_optimize", 1), \
             patch("app.main.parse_and_stream", side_effect=_mock_stream), \
             patch("app.main.CVonRAGOrchestrator") as M:
            M.return_value.run = fake_run
            _limiter._windows.clear()
            # Exhaust the /parse bucket
            client.post("/parse", files={"file": _file})
            parse_resp = client.post("/parse", files={"file": _file})
            # /optimize bucket is still fresh — first call should succeed
            opt_resp = client.post("/optimize", json=MINIMAL_REQUEST)
        assert parse_resp.status_code == 429
        assert opt_resp.status_code == 200

    @pytest.mark.anyio
    async def test_gc_drops_expired_entries(self):
        """N2 regression: expired (ip, key) entries get swept so the dict can't grow forever.

        Direct unit test against _RateLimiter — no FastAPI test client needed.
        """
        from app.main import _RateLimiter, settings
        with patch.object(settings, "rate_limit_enabled", True):
            limiter = _RateLimiter()
            limiter._GC_EVERY_N_CHECKS = 3  # force GC quickly

            # Two distinct IPs each register one call
            await limiter.check("1.1.1.1", "k", max_calls=10, window_seconds=60)
            await limiter.check("2.2.2.2", "k", max_calls=10, window_seconds=60)
            assert len(limiter._windows) == 2

            # Make the timestamps look ancient (past the cutoff window)
            for dq in limiter._windows.values():
                dq[0] = -1e9

            # Trigger GC by hitting _GC_EVERY_N_CHECKS calls
            await limiter.check("3.3.3.3", "k", max_calls=10, window_seconds=60)

            # Old IPs swept; only the live entry remains
            assert len(limiter._windows) == 1
            assert ("3.3.3.3", "k") in limiter._windows


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
                files={"file": ("resume.docx", _DOCX_MAGIC + b"x" * 200, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
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
                files={"file": ("cv.pdf", _PDF_MAGIC + b"x" * 200, "application/pdf")},
            )

        assert resp.status_code == 200

    def test_parse_error_yields_error_event(self, client):
        from unittest.mock import patch

        async def _mock_stream(file_bytes, filename, http_client):
            yield ("error", {"error_message": "corrupt file"})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("resume.docx", _DOCX_MAGIC + b"x" * 200, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert resp.status_code == 200
        assert "event: error" in resp.text


# ── Magic-byte validation (H7) ────────────────────────────────────────────────

class TestFileMagic:
    """H7: file-type is verified by magic bytes, not just the filename extension."""

    _DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    _PDF_MIME  = "application/pdf"

    def test_docx_with_pdf_magic_returns_415(self, client):
        """A .docx file that starts with %PDF bytes is rejected."""
        resp = client.post(
            "/parse",
            files={"file": ("resume.docx", _PDF_MAGIC + b"x" * 200, self._DOCX_MIME)},
        )
        assert resp.status_code == 415

    def test_pdf_with_docx_magic_returns_415(self, client):
        """A .pdf file that starts with PK bytes is rejected."""
        resp = client.post(
            "/parse",
            files={"file": ("cv.pdf", _DOCX_MAGIC + b"x" * 200, self._PDF_MIME)},
        )
        assert resp.status_code == 415

    def test_renamed_exe_as_docx_returns_415(self, client):
        """A Windows executable (MZ header) renamed to .docx is rejected."""
        resp = client.post(
            "/parse",
            files={"file": ("cv.docx", b"MZ\x90\x00" + b"x" * 200, self._DOCX_MIME)},
        )
        assert resp.status_code == 415

    def test_valid_docx_magic_passes_check(self, client):
        """A .docx file with correct PK magic proceeds past validation."""
        async def _mock_stream(fb, fn, hc):
            yield ("done", {"total_projects": 0, "total_facts": 0})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("cv.docx", _DOCX_MAGIC + b"x" * 200, self._DOCX_MIME)},
            )
        assert resp.status_code == 200

    def test_valid_pdf_magic_passes_check(self, client):
        """A .pdf file with correct %PDF magic proceeds past validation."""
        async def _mock_stream(fb, fn, hc):
            yield ("done", {"total_projects": 0, "total_facts": 0})

        with patch("app.main.parse_and_stream", side_effect=_mock_stream):
            resp = client.post(
                "/parse",
                files={"file": ("cv.pdf", _PDF_MAGIC + b"x" * 200, self._PDF_MIME)},
            )
        assert resp.status_code == 200

    def test_magic_check_runs_after_size_check(self, client):
        """Files that are too small fail with 400 before reaching the magic check."""
        resp = client.post(
            "/parse",
            files={"file": ("cv.docx", b"PK", self._DOCX_MIME)},  # valid magic, too short
        )
        assert resp.status_code == 400  # not 415


# ── Streaming upload cap (N3) ─────────────────────────────────────────────────

class TestUploadSizeCap:
    """N3: oversized uploads are rejected without materialising the full file in RAM."""

    _DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def test_oversized_body_returns_413(self, client):
        """An 11 MB upload is rejected with 413 (cap is 10 MB)."""
        oversized = _DOCX_MAGIC + b"x" * (11 * 1024 * 1024)
        resp = client.post(
            "/parse",
            files={"file": ("big.docx", oversized, self._DOCX_MIME)},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_size_check_runs_before_full_read(self):
        """The chunked reader bails after exceeding max_bytes — it does not read past the cap."""
        import asyncio
        from app.main import _read_upload_capped

        class _StubUpload:
            """Mimics UploadFile.read(size) over a pre-buffered stream."""
            def __init__(self, data: bytes) -> None:
                self._buf = data
                self._pos = 0
                self.read_calls = 0

            async def read(self, size: int) -> bytes:
                self.read_calls += 1
                chunk = self._buf[self._pos:self._pos + size]
                self._pos += len(chunk)
                return chunk

        # 200 KB of payload, cap at 100 KB → expect 413 before second-half is read.
        stub = _StubUpload(b"x" * (200 * 1024))
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_read_upload_capped(stub, max_bytes=100 * 1024))
        assert exc_info.value.status_code == 413
        # Should have stopped reading well before consuming the whole buffer.
        assert stub._pos <= 100 * 1024 + 64 * 1024  # cap + one chunk overshoot
