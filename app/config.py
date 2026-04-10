"""
CVonRAG — config.py
Centralised settings via pydantic-settings.
All values come from environment variables or the .env file.
Zero paid API keys required.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # silently ignore unknown env vars
    )

    # ── Groq (preferred — fast, free tier available) ────────────────────────
    # Set GROQ_API_KEY to enable Groq; leave blank to fall back to Ollama.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Ollama (local fallback) ──────────────────────────────────────────────
    # Used when GROQ_API_KEY is not set, and always used for embeddings.
    ollama_base_url: str = "http://localhost:11434"

    # LLM — only used when Groq is not configured:
    #   qwen2.5:3b   ~2 GB RAM  (fast, okay quality — CI/testing)
    #   qwen2.5:7b   ~5 GB RAM  (good quality — development)
    #   qwen2.5:14b  ~10 GB RAM (great quality — recommended production)
    #   qwen2.5:32b  ~20 GB RAM (best quality — strong GPU only)
    ollama_llm_model: str = "qwen2.5:14b"

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
    char_loop_max_iterations: int = 4
    char_tolerance: int = 2

    # ── LLM generation ────────────────────────────────────────────────────────
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_context_window: int = 8192

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    port: int = 8000
    log_level: str = "INFO"

    # CORS: restrict to your frontend origin in production.
    cors_origins: list[str] = ["http://localhost:5173"]

    # ── Admin auth ────────────────────────────────────────────────────────────
    # Set INGEST_SECRET in .env to protect the /ingest endpoint.
    # Leave blank to allow unauthenticated access (dev only).
    ingest_secret: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()