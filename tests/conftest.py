# tests/conftest.py
"""Shared fixtures and pytest configuration."""
import pytest


# all async tests use anyio automatically (configured via pyproject.toml asyncio_mode=auto)


@pytest.fixture(autouse=True)
def _isolate_settings_from_local_env(monkeypatch, tmp_path):
    """
    Ensure tests run with standardized, non-environment-dependent application settings.
    
    Patches cached settings to clear ingest secrets, disable rate limiting and invite-code checks, point SQLite to a per-test file under `tmp_path`, reset the DB engine/session factory, clear in-memory rate-limiter state, and reset the Groq quota circuit breaker.
    """
    from app.main import settings as main_settings, _limiter
    from app.chains import settings as chains_settings, _reset_quota_circuit
    from app.db import _reset_factory_for_tests

    # Both modules import the same cached Settings instance; patch once on each
    # binding to keep both views in sync.
    monkeypatch.setattr(main_settings,   "ingest_secret", "")
    monkeypatch.setattr(chains_settings, "ingest_secret", "")

    # Disable rate limiting globally; individual rate-limit tests opt in via patch.object.
    monkeypatch.setattr(main_settings, "rate_limit_enabled", False)

    # Disable the invite-code gate by default. The require_invite dep short-
    # circuits to None when this is False, so existing tests that hit
    # /parse, /recommend, /optimize without an X-Invite-Code header still pass.
    monkeypatch.setattr(main_settings, "invite_codes_required", False)
    # Per-test SQLite file. Without this, every test would create/share the
    # dev DB at ./cvonrag.db.
    monkeypatch.setattr(main_settings, "sqlite_path", str(tmp_path / "test_invites.db"))
    # Drop the cached engine/session-factory so the next init_db() picks up
    # the new sqlite_path (db.py memoises both module-level).
    _reset_factory_for_tests()

    # Clear any accumulated window state so tests start clean.
    # _checks_since_gc and _window_seconds also reset (P8) so an unrelated test
    # can't trip the GC threshold or P7's mixed-window guard on the next call.
    _limiter._windows.clear()
    _limiter._checks_since_gc = 0
    _limiter._window_seconds = None

    # Reset the Groq quota-exhaustion circuit breaker. If a prior test
    # tripped it (or a developer's prior session left it tripped during an
    # interactive run), subsequent _groq_chat calls would fail fast and
    # produce confusing test failures unrelated to the test under test.
    _reset_quota_circuit()
