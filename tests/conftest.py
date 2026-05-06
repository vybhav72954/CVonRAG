# tests/conftest.py
"""Shared fixtures and pytest configuration."""
import pytest


# all async tests use anyio automatically (configured via pyproject.toml asyncio_mode=auto)


@pytest.fixture(autouse=True)
def _isolate_settings_from_local_env(monkeypatch):
    """Reset environment-sensitive settings to known defaults for every test.

    Why: settings are loaded once via @lru_cache on first import. If the
    developer's local .env has INGEST_SECRET set, /ingest tests that don't
    send the matching X-Ingest-Secret header silently fail with 403 — even
    though the test logic is correct. This fixture neutralizes those values
    so tests are reproducible regardless of who runs them. Tests that want
    to exercise the auth path can re-enable the secret via patch.object.
    """
    from app.main import settings as main_settings
    from app.chains import settings as chains_settings

    # Both modules import the same cached Settings instance; patch once on each
    # binding to keep both views in sync.
    monkeypatch.setattr(main_settings,   "ingest_secret", "")
    monkeypatch.setattr(chains_settings, "ingest_secret", "")
