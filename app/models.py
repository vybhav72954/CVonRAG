"""
CVonRAG — models.py
All Pydantic v2 schemas used across the API and internal pipeline.

The Content/Style Firewall is enforced at the data layer:
  • CoreFact  = CONTENT  (user-supplied, immutable)
  • StyleExemplar = STYLE  (RAG-retrieved, never used as content)
"""

from __future__ import annotations
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RoleType(StrEnum):
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE         = "data_science"
    ML_ENGINEERING       = "ml_engineering"
    PRODUCT_MANAGEMENT   = "product_management"
    QUANT_FINANCE        = "quant_finance"
    GENERAL              = "general"


class JDTone(StrEnum):
    HIGHLY_QUANTITATIVE = "highly_quantitative"
    ENGINEERING_FOCUSED = "engineering_focused"
    LEADERSHIP_FOCUSED  = "leadership_focused"
    RESEARCH_FOCUSED    = "research_focused"
    BALANCED            = "balanced"


class StreamEventType(StrEnum):
    # /optimize events (bullet event includes its metadata under .metadata)
    TOKEN    = "token"
    BULLET   = "bullet"
    # /parse events
    PROGRESS = "progress"
    PROJECT  = "project"
    # shared
    ERROR    = "error"
    DONE     = "done"


# ── User content (Phase 1 input) ──────────────────────────────────────────────

class CoreFact(BaseModel):
    """One atomic fact. All fields are CONTENT — never altered by the pipeline."""

    fact_id: str = Field(..., min_length=1, description="Stable ID for traceability")
    text: str    = Field(..., min_length=5)
    tools: list[str]   = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    outcome: str       = Field(default="")

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ProjectData(BaseModel):
    project_id: str
    title: str
    core_facts: Annotated[list[CoreFact], Field(min_length=1, max_length=12)]


class FormattingConstraints(BaseModel):
    target_char_limit: Annotated[int, Field(ge=60, le=300)] = 130
    tolerance: Annotated[int, Field(ge=1, le=5)]            = 2
    bullet_prefix: str                                       = "•"
    max_bullets_per_project: Annotated[int, Field(ge=1, le=8)] = 3

    @property
    def lower_bound(self) -> int:
        return self.target_char_limit - self.tolerance

    @property
    def upper_bound(self) -> int:
        return self.target_char_limit + self.tolerance


# ── Top-level API request ─────────────────────────────────────────────────────

_MAX_TOTAL_BULLETS = 50


class OptimizationRequest(BaseModel):
    job_description: Annotated[str, Field(min_length=50, max_length=10_000)]
    projects: Annotated[list[ProjectData], Field(min_length=1, max_length=20)]
    constraints: FormattingConstraints = Field(default_factory=FormattingConstraints)
    target_role_type: RoleType = RoleType.GENERAL
    total_bullets_requested: Annotated[int, Field(ge=1, le=_MAX_TOTAL_BULLETS)] | None = None

    @model_validator(mode="after")
    def cap_total_bullets(self) -> "OptimizationRequest":
        if self.total_bullets_requested is None:
            # Clamp to the same upper bound the Field enforces so the validator
            # can't produce a value the field declaration would have rejected.
            self.total_bullets_requested = min(
                _MAX_TOTAL_BULLETS,
                len(self.projects) * self.constraints.max_bullets_per_project,
            )
        return self


# ── Internal pipeline models ──────────────────────────────────────────────────

class ScoredFact(BaseModel):
    fact: CoreFact
    project_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    matched_jd_keywords: list[str] = Field(default_factory=list)


class StyleExemplar(BaseModel):
    """STYLE ONLY — content from this object must never appear in output."""
    exemplar_id: str
    text: str
    role_type: RoleType
    similarity_score: float = Field(ge=0.0, le=1.0)
    uses_separator: str | None = None
    uses_arrow: bool = False
    uses_abbreviations: list[str] = Field(default_factory=list)
    sentence_structure: str | None = None


class BulletDraft(BaseModel):
    text: str
    char_count: int
    iteration: int
    within_tolerance: bool
    source_fact_ids: list[str]


class BulletMetadata(BaseModel):
    bullet_index: int
    project_id: str
    source_fact_ids: list[str]
    char_count: int
    char_target: int
    iterations_taken: int
    exemplar_ids_used: list[str]
    jd_tone: JDTone
    within_tolerance: bool


class GeneratedBullet(BaseModel):
    text: str
    metadata: BulletMetadata


# ── SSE envelope ─────────────────────────────────────────────────────────────

class StreamChunk(BaseModel):
    event_type: StreamEventType
    data: Any = None
    error_message: str | None = None


# ── Project recommendation ───────────────────────────────────────────────────

class ProjectRecommendation(BaseModel):
    """One project scored and ranked against a JD."""
    project_id:     str
    title:          str
    score:          float = Field(ge=0.0, le=1.0)
    rank:           int
    reason:         str
    matched_skills: list[str] = Field(default_factory=list)
    top_metrics:    list[str] = Field(default_factory=list)
    recommended:    bool      = True
    core_facts:     list[Any] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    job_description: Annotated[str, Field(min_length=50, max_length=10_000)]
    projects:        Annotated[list[ProjectData], Field(min_length=1, max_length=20)]
    top_k:           Annotated[int, Field(ge=1, le=6)] = 3


class RecommendResponse(BaseModel):
    recommendations: list[ProjectRecommendation]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    llm_backend: str                  # "groq" | "openrouter" | "ollama"
    llm_provider: str = "ollama"      # mirrors llm_backend; named for the LLM_PROVIDER env var
    model: str                        # active LLM model name
    qdrant_connected: bool
    collection_exists: bool
    vector_count: int = 0
    llm_ok: bool = False              # active hosted LLM provider reachable (groq or openrouter)
    groq_ok: bool = False              # Back-compat: True only when active provider is groq AND reachable
    ollama_ok: bool = False            # Ollama LLM model loaded (only relevant when using Ollama)
    embed_ok: bool = False             # Ollama embed model loaded (always required)
