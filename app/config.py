"""
CVonRAG — config.py
Centralised settings via pydantic-settings.
All values come from environment variables or the .env file.
Zero paid API keys required.
"""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # silently ignore unknown env vars
    )

    # ── Hosted LLM provider selector ────────────────────────────────────────
    # "groq" (default) or "openrouter". Picks which hosted-LLM slot the
    # runtime reads. Empty key on the selected slot falls back to Ollama.
    # Embeddings stay on Ollama regardless.
    llm_provider: Literal["groq", "openrouter"] = "groq"

    # ── Groq (primary hosted LLM) ───────────────────────────────────────────
    # Set GROQ_API_KEY + LLM_PROVIDER=groq to route LLM calls through Groq.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Hard cap on bullets per /optimize request when using Groq.
    # Groq free tier is 30 req/min; 15 bullets × 4 correction iterations = 62 calls (~2 min max).
    # Set higher only if you have a paid Groq plan.
    groq_max_bullets_per_request: int = 15
    # Max seconds we'll honor the Retry-After header on a 429. Per-minute rate
    # limits cite ≤60s; values far larger (often thousands of seconds) signal
    # daily/monthly quota exhaustion, where blocking the request for an hour
    # just hangs the frontend and ties up a backend slot. Above this, we raise
    # immediately so the user sees a "quota exhausted" message instead of a
    # generic timeout, and the slot frees up for any non-Groq fallbacks.
    groq_max_retry_wait_seconds: int = 30

    # ── OpenRouter (fallback hosted LLM, OpenAI-compatible) ─────────────────
    # Set OPENROUTER_API_KEY + LLM_PROVIDER=openrouter to route through
    # OpenRouter. The free Llama 3.3 70B model is the default. Get a key at
    # https://openrouter.ai → Keys.
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Ollama (local fallback) ──────────────────────────────────────────────
    # Used when GROQ_API_KEY is not set, and always used for embeddings.
    ollama_base_url: str = "http://localhost:11434"

    # LLM — only used when Groq is not configured:
    #   qwen2.5:3b   ~2 GB RAM  (fast, fits 4 GB GPU — default)
    #   qwen2.5:7b   ~5 GB RAM  (good quality — development)
    #   qwen2.5:14b  ~10 GB RAM (recommended for quality — see DEVELOPER.md)
    #   qwen2.5:32b  ~20 GB RAM (best quality — strong GPU only)
    ollama_llm_model: str = "qwen2.5:3b"

    # Embedding model — always via Ollama (nomic-embed-text, 768-dim)
    ollama_embed_model: str = "nomic-embed-text"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None       # only needed for Qdrant Cloud
    qdrant_collection: str = "gold_standard_cvs"
    qdrant_vector_size: int = 768           # must match embed model output dim

    # ── RAG ───────────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5

    # ── Recommender / Scorer ──────────────────────────────────────────────────
    jd_snippet_max_chars: int = 800        # JD truncation for reason prompt
    style_query_jd_chars: int = 500        # JD chars in style-retrieval query
    jd_top_keywords: int = 10             # max required_skills shown to LLM
    top_n_facts_for_score: int = 3        # top-N fact scores averaged per project
    max_skills_per_project: int = 6       # matched JD keywords cap per project
    max_metrics_per_project: int = 4      # metric chips cap per project

    # ── Char-limit loop ───────────────────────────────────────────────────────
    # ge=1: at least one iteration so the initial draft is always produced.
    # _correction_loop returns None if range(1, n+1) is empty, which would
    # crash the orchestrator on `draft.text` — fail at boot instead.
    char_loop_max_iterations: Annotated[int, Field(ge=1)] = 4
    char_tolerance: int = 2

    # ── LLM generation ────────────────────────────────────────────────────────
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_context_window: int = 8192

    # ── Bullet typewriter stream ──────────────────────────────────────────────
    # Delay (seconds) between word chunks when streaming the finalized bullet
    # to the browser. Higher = slower, more deliberate typewriter feel.
    # Set to 0 to emit all chunks at once (e.g., in tests/CI).
    bullet_stream_chunk_delay: float = 0.025

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    port: int = 8000
    log_level: str = "INFO"

    # CORS: restrict to your frontend origin in production.
    cors_origins: list[str] = ["http://localhost:5173"]

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Per-IP sliding-window limits for the three expensive endpoints.
    # Set RATE_LIMIT_ENABLED=false to disable (useful in local dev / testing).
    rate_limit_enabled: bool = True
    rate_limit_window: int = 60       # window in seconds
    rate_limit_parse: int = 10        # max /parse calls per IP per window
    rate_limit_recommend: int = 20    # max /recommend calls per IP per window
    rate_limit_optimize: int = 5      # max /optimize calls per IP per window

    # ── Admin secret (ingest only) ────────────────────────────────────────────
    # Used ONLY by /ingest, which is hit by scripts/ingest_pdfs.py (a CLI tool
    # with no browser, so OAuth doesn't fit). Human-facing admin endpoints
    # (/admin/usage) authenticate via Google OAuth + ADMIN_EMAILS allowlist
    # instead. Leave blank to allow unauthenticated /ingest in dev.
    ingest_secret: str = ""

    # ── Google Workspace OAuth ────────────────────────────────────────────────
    # Replaces the prior invite-code gate. Frontend uses Google Identity
    # Services to obtain a 1-hour ID token and sends it as
    # `Authorization: Bearer <id_token>` on every gated request.
    # • google_oauth_client_id — Google Cloud OAuth 2.0 Web Client ID. Empty
    #   string disables the gate entirely (dev / tests), so require_user()
    #   returns None and the existing "no auth in tests" pattern keeps working.
    # • google_oauth_hd — institutional Google Workspace domain to restrict
    #   sign-in to (the `hd` claim in the verified ID token). Empty string
    #   accepts any Google account (dev only; never leave empty in prod).
    # • admin_emails — comma-separated emails allowlisted for /admin/*.
    google_oauth_client_id: str = ""
    google_oauth_hd: str = ""
    admin_emails: list[str] = []

    @field_validator("admin_emails")
    @classmethod
    def _normalise_admin_emails(cls, v: list[str]) -> list[str]:
        """Lower-case + strip each entry so the comparison in
        `require_admin` is case-insensitive.

        `user.email` is stored lower-cased (see app/auth.py — the email
        claim from the ID token is normalised before being trusted as
        identity). If an operator writes `ADMIN_EMAILS=["Admin@Foo.org"]`
        in .env, without this validator the membership check silently
        fails and the admin sees a permanent 403 with no diagnostic.
        Normalising at config-load time fixes the bug once, not per call.
        """
        return [e.strip().lower() for e in v if e and e.strip()]

    # Per-user daily cap on /optimize calls. Resets at 00:00 UTC.
    # ge=1 catches a misconfig (MAX_DAILY_OPTIMIZATIONS=0 would make every
    # /optimize call hit `0 < 0` → false → 429 immediately, surfacing as a
    # mysterious runtime issue instead of a clear boot-time failure).
    max_daily_optimizations: Annotated[int, Field(ge=1)] = 20
    # Per-user daily cap on cumulative bullets generated.
    # 60/day = 3 optimize × 20 bullets, or 20 optimize × 3 bullets — both
    # reasonable for a single user iterating on one resume.
    max_daily_bullets: Annotated[int, Field(ge=1)] = 60
    # SQLite path for the users table. Default keeps the DB next to the app
    # for local dev; deployments should mount this to a persistent volume.
    sqlite_path: str = "./cvonrag.db"

    def _is_production_env(self) -> bool:
        """`app_env == "production"` after normalization.

        Pydantic doesn't auto-trim/lowercase string env vars, so values like
        `APP_ENV=Production` or `APP_ENV=" production"` would silently skip
        both prod warnings below. Comparing a normalized copy makes the check
        robust to common casing/whitespace mistakes in the .env file.
        """
        return (self.app_env or "").strip().lower() == "production"

    @model_validator(mode="after")
    def _warn_production_cors(self) -> "Settings":
        """
        Log a warning when running in production while CORS origins are still the development default.
        
        This validator emits a startup warning if `app_env` is "production" and `cors_origins` equals ["http://localhost:5173"], advising to set `CORS_ORIGINS` to the production frontend origin.
        
        Returns:
            Settings: The same Settings instance.
        """
        if (
            self._is_production_env()
            and self.cors_origins == ["http://localhost:5173"]
        ):
            logging.getLogger("cvonrag").warning(
                "CORS_ORIGINS is still the dev default in production — "
                "set CORS_ORIGINS=[\"https://your-frontend.com\"] in .env before deploying."
            )
        return self

    @model_validator(mode="after")
    def _warn_production_ingest_secret(self) -> "Settings":
        """Warn if /ingest is unauthenticated in production."""
        if self._is_production_env() and not self.ingest_secret:
            logging.getLogger("cvonrag").warning(
                "INGEST_SECRET is empty in production — /ingest is "
                "unauthenticated. Set a long random INGEST_SECRET in .env "
                "before exposing the API to the internet."
            )
        return self

    @model_validator(mode="after")
    def _warn_production_oauth(self) -> "Settings":
        """Warn if Google OAuth is misconfigured in production.

        In prod, `google_oauth_client_id` and `google_oauth_hd` must both be
        set. An empty client_id leaves the API wide open (require_user returns
        None for everyone); an empty hd accepts any Google account, defeating
        the institutional restriction. Admin endpoints additionally need at
        least one email in `admin_emails`, otherwise nobody can hit /admin/*.
        """
        if not self._is_production_env():
            return self
        log = logging.getLogger("cvonrag")
        if not self.google_oauth_client_id:
            log.warning(
                "GOOGLE_OAUTH_CLIENT_ID is empty in production — /parse, "
                "/recommend, /optimize are UNAUTHENTICATED. Set it before "
                "exposing the API."
            )
        if not self.google_oauth_hd:
            log.warning(
                "GOOGLE_OAUTH_HD is empty in production — any Google account "
                "can sign in. Set it to your institutional Workspace domain."
            )
        if not self.admin_emails:
            log.warning(
                "ADMIN_EMAILS is empty in production — /admin/usage will "
                "reject every request (no allowlisted emails). Add at least "
                "one admin email."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Provide the application's Settings instance loaded from environment variables and the optional .env file.
    
    Returns:
        settings (Settings): The Settings instance populated from environment and .env configuration.
    """
    return Settings()