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

from pydantic import Field, model_validator
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

    # ── Admin auth ────────────────────────────────────────────────────────────
    # Set INGEST_SECRET in .env to protect the /ingest and /admin/* endpoints.
    # Leave blank to allow unauthenticated access (dev only).
    ingest_secret: str = ""

    # ── Invite-code auth ──────────────────────────────────────────────────────
    # Per-batchmate identity gate on /parse, /recommend, /optimize. Each
    # batchmate gets a unique code (admin creates via POST /admin/invites).
    # Code is sent in the X-Invite-Code request header. Per-code daily caps
    # below stop a single batchmate from exhausting the hosted-LLM quota.
    # Set INVITE_CODES_REQUIRED=false to disable (dev / tests).
    invite_codes_required: bool = True
    # Hard daily cap: after this many /optimize calls in one UTC day, the
    # invite code returns 429 until 00:00 UTC. 20 is comfortable for normal
    # iteration, low enough that an accidental refresh loop cannot drain a
    # day's free-tier hosted-LLM quota.
    max_daily_optimizations: int = 20
    # Bullet-level cap (each /optimize may produce up to total_bullets_requested
    # bullets, hard-capped to groq_max_bullets_per_request=15 per call).
    # 60/day = 3 optimize × 20 bullets, or 20 optimize × 3 bullets — both
    # reasonable for a batchmate iterating on a single resume.
    max_daily_bullets: int = 60
    # SQLite path for the invite table. Default keeps the DB next to the app
    # for local dev; deployments should mount this to a persistent volume.
    sqlite_path: str = "./cvonrag.db"

    @model_validator(mode="after")
    def _warn_production_cors(self) -> "Settings":
        if (
            self.app_env == "production"
            and self.cors_origins == ["http://localhost:5173"]
        ):
            logging.getLogger("cvonrag").warning(
                "CORS_ORIGINS is still the dev default in production — "
                "set CORS_ORIGINS=[\"https://your-frontend.com\"] in .env before deploying."
            )
        return self

    @model_validator(mode="after")
    def _warn_production_ingest_secret(self) -> "Settings":
        """Loud warning when INGEST_SECRET is empty in prod.

        Empty-secret = open access is intentional for local dev (documented in
        .env.example), but in production it leaves `/ingest` and `/admin/*`
        unauthenticated. Failing closed here would break the documented dev
        flow; warning at boot catches the real concern (misconfigured deploy)
        without forcing every dev to juggle headers on curl calls.
        """
        if self.app_env == "production" and not self.ingest_secret:
            logging.getLogger("cvonrag").warning(
                "INGEST_SECRET is empty in production — /ingest and /admin/* "
                "are unauthenticated. Set a long random INGEST_SECRET in .env "
                "before exposing the API to the internet."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()