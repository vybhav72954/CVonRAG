# tests/conftest.py
"""Shared fixtures and pytest configuration."""
import pytest


# all async tests use anyio automatically (configured via pyproject.toml asyncio_mode=auto)


@pytest.fixture(autouse=True)
def _isolate_settings_from_local_env(monkeypatch, tmp_path):
    """
    Isolate every test from the developer's local environment.

    • ingest_secret      → "" (a real INGEST_SECRET in .env would otherwise
                              produce spurious 403s on /ingest)
    • rate_limit_enabled → False
    • OAuth gate         → disabled (google_oauth_client_id="") so
                              require_user() returns None and tests that
                              don't send a Bearer token still pass. The
                              dedicated auth tests opt back in via
                              patch.object(settings, "google_oauth_client_id", ...)
                              plus a monkeypatched _verify_id_token.
    • sqlite_path        → per-test tmp_path file (no dev-DB pollution)
    • _limiter state     → cleared
    • Groq quota breaker → reset
    """
    from app.main import settings as main_settings, _limiter
    from app.chains import settings as chains_settings, _reset_quota_circuit
    from app.db import _reset_factory_for_tests

    monkeypatch.setattr(main_settings,   "ingest_secret", "")
    monkeypatch.setattr(chains_settings, "ingest_secret", "")

    monkeypatch.setattr(main_settings, "rate_limit_enabled", False)

    # OAuth gate off by default. Auth tests opt in by patching client_id back
    # on AND providing a fake _verify_id_token.
    monkeypatch.setattr(main_settings, "google_oauth_client_id", "")
    monkeypatch.setattr(main_settings, "google_oauth_hd", "")
    monkeypatch.setattr(main_settings, "admin_emails", [])

    # Per-test SQLite file.
    monkeypatch.setattr(main_settings, "sqlite_path", str(tmp_path / "test_users.db"))
    _reset_factory_for_tests()

    _limiter._windows.clear()
    _limiter._checks_since_gc = 0
    _limiter._window_seconds = None

    _reset_quota_circuit()
