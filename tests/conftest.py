# tests/conftest.py
"""Shared fixtures and pytest configuration."""
import pytest


# all async tests use anyio automatically (configured via pyproject.toml asyncio_mode=auto)


@pytest.fixture(autouse=True)
def _isolate_settings_from_local_env(monkeypatch):
    """Reset environment-sensitive settings to known defaults for every test.

    Why: settings are loaded once via @lru_cache on first import. If the
    developer's local .env has INGEST_SECRET set, /ingest tests that don't
    send the matching X-Ingest-Secret header silently fail with 403.
    Rate limiting is disabled so repeated endpoint calls in a test suite
    don't trigger 429s. Tests that exercise auth or rate-limiting explicitly
    re-enable the relevant setting via patch.object.
    """
    from app.main import settings as main_settings, _limiter
    from app.chains import settings as chains_settings

    # Both modules import the same cached Settings instance; patch once on each
    # binding to keep both views in sync.
    monkeypatch.setattr(main_settings,   "ingest_secret", "")
    monkeypatch.setattr(chains_settings, "ingest_secret", "")

    # Disable rate limiting globally; individual rate-limit tests opt in via patch.object.
    monkeypatch.setattr(main_settings, "rate_limit_enabled", False)

    # Clear any accumulated window state so tests start clean.
    # _checks_since_gc and _window_seconds also reset (P8) so an unrelated test
    # can't trip the GC threshold or P7's mixed-window guard on the next call.
    _limiter._windows.clear()
    _limiter._checks_since_gc = 0
    _limiter._window_seconds = None
