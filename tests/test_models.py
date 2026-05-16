"""
tests/test_models.py
Unit tests for all Pydantic schemas in models.py

Tests:
  • Field validation (types, lengths, ranges)
  • Computed properties (lower_bound / upper_bound)
  • CoreFact whitespace stripping
  • OptimizationRequest auto-cap logic
  • Enum round-trips
"""

import pytest
from pydantic import ValidationError
from typing import Any

from app.models import (
    CoreFact,
    ProjectData,
    FormattingConstraints,
    OptimizationRequest,
    RoleType,
    JDTone,
    StyleExemplar,
    BulletDraft,
    StreamChunk,
    StreamEventType,
)


# ── CoreFact ──────────────────────────────────────────────────────────────────

class TestCoreFact:
    def test_valid_minimal(self):
        f = CoreFact(fact_id="f-001", text="Built a model")
        assert f.fact_id == "f-001"
        assert f.text == "Built a model"
        assert f.tools == []
        assert f.metrics == []
        assert f.outcome == ""

    def test_strips_whitespace(self):
        f = CoreFact(fact_id="f-002", text="  Trimmed fact  ")
        assert f.text == "Trimmed fact"

    def test_text_too_short_raises(self):
        with pytest.raises(ValidationError):
            CoreFact(fact_id="f-003", text="Hi")   # min_length=5

    def test_full_fields(self):
        f = CoreFact(
            fact_id="f-004",
            text="Reduced RMSE to 0.250 using SARIMA model",
            tools=["SARIMA", "statsmodels"],
            metrics=["0.250", "RMSE"],
            outcome="Best forecast accuracy",
        )
        assert "SARIMA" in f.tools
        assert "0.250" in f.metrics


# ── FormattingConstraints ─────────────────────────────────────────────────────

class TestFormattingConstraints:
    def test_defaults(self):
        c = FormattingConstraints()
        assert c.target_char_limit == 130
        assert c.tolerance == 2
        assert c.bullet_prefix == "•"

    def test_bounds_computed_correctly(self):
        c = FormattingConstraints(target_char_limit=120, tolerance=3)
        assert c.lower_bound == 117
        assert c.upper_bound == 123

    def test_tolerance_limits(self):
        # tolerance must be 1–5
        with pytest.raises(ValidationError):
            FormattingConstraints(tolerance=0)
        with pytest.raises(ValidationError):
            FormattingConstraints(tolerance=6)

    def test_char_limit_boundaries(self):
        with pytest.raises(ValidationError):
            FormattingConstraints(target_char_limit=59)    # below min
        with pytest.raises(ValidationError):
            FormattingConstraints(target_char_limit=301)   # above max

    def test_exact_boundaries_valid(self):
        assert FormattingConstraints(target_char_limit=60).target_char_limit == 60
        assert FormattingConstraints(target_char_limit=300).target_char_limit == 300


# ── ProjectData ───────────────────────────────────────────────────────────────

class TestProjectData:
    def _make_fact(self, fid: str) -> CoreFact:
        return CoreFact(fact_id=fid, text="Some valid fact text here")

    def test_valid_project(self):
        p = ProjectData(
            project_id="p-001",
            title="ML Project",
            core_facts=[self._make_fact("f-001"), self._make_fact("f-002")],
        )
        assert len(p.core_facts) == 2

    def test_empty_core_facts_raises(self):
        with pytest.raises(ValidationError):
            ProjectData(project_id="p-002", title="Empty", core_facts=[])

    def test_too_many_facts_raises(self):
        with pytest.raises(ValidationError):
            ProjectData(
                project_id="p-003",
                title="Bloated",
                core_facts=[self._make_fact(f"f-{i:03d}") for i in range(13)],  # max=12
            )


# ── OptimizationRequest ───────────────────────────────────────────────────────

def _make_request(**kwargs: Any) -> OptimizationRequest:
    defaults: dict[str, Any] = dict(
        job_description="We are looking for a Senior ML Engineer with experience in Python and time-series forecasting.",
        projects=[
            ProjectData(
                project_id="p-001",
                title="Forecasting",
                core_facts=[CoreFact(fact_id="f-001", text="Built SARIMA model reducing RMSE to 0.250")],
            )
        ],
    )
    defaults.update(kwargs)
    return OptimizationRequest(**defaults)  # type: ignore


class TestOptimizationRequest:
    def test_valid_minimal(self):
        req = _make_request()
        assert req.target_role_type == RoleType.GENERAL

    def test_total_bullets_auto_calculated(self):
        req = _make_request()
        # 1 project × 3 bullets per project (default) = 3
        assert req.total_bullets_requested == 3

    def test_total_bullets_explicit(self):
        req = _make_request(total_bullets_requested=10)
        assert req.total_bullets_requested == 10

    def test_jd_too_short_raises(self):
        with pytest.raises(ValidationError):
            _make_request(job_description="Short")

    def test_empty_projects_raises(self):
        with pytest.raises(ValidationError):
            _make_request(projects=[])

    def test_role_type_enum(self):
        req = _make_request(target_role_type=RoleType.DATA_SCIENCE)
        assert req.target_role_type == RoleType.DATA_SCIENCE


# ── StyleExemplar ─────────────────────────────────────────────────────────────

class TestStyleExemplar:
    def test_valid(self):
        ex = StyleExemplar(
            exemplar_id="e-001",
            text="• Enhanced accuracy via ARIMAX | Reduced RMSE by 13.5%",
            role_type=RoleType.DATA_SCIENCE,
            similarity_score=0.87,
            uses_separator="|",
            uses_arrow=False,
        )
        assert ex.similarity_score == 0.87

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            StyleExemplar(
                exemplar_id="e-002",
                text="Some bullet text here",
                role_type=RoleType.GENERAL,
                similarity_score=1.5,   # > 1.0
            )


# ── BulletDraft ───────────────────────────────────────────────────────────────

class TestBulletDraft:
    def test_within_tolerance(self):
        draft = BulletDraft(
            text="• Built a model",
            char_count=15,
            iteration=1,
            within_tolerance=True,
            source_fact_ids=["f-001"],
        )
        assert draft.within_tolerance is True

    def test_not_within_tolerance(self):
        draft = BulletDraft(
            text="• Short",
            char_count=7,
            iteration=3,
            within_tolerance=False,
            source_fact_ids=["f-001", "f-002"],
        )
        assert draft.iteration == 3


# ── StreamChunk ───────────────────────────────────────────────────────────────

class TestStreamChunk:
    def test_token_event(self):
        chunk = StreamChunk(event_type=StreamEventType.TOKEN, data="Hello")
        payload = chunk.model_dump_json()
        assert "token" in payload
        assert "Hello" in payload

    def test_error_event(self):
        chunk = StreamChunk(
            event_type=StreamEventType.ERROR,
            error_message="Something went wrong",
        )
        assert chunk.error_message == "Something went wrong"
        assert chunk.data is None

    def test_done_event_serializes(self):
        chunk = StreamChunk(
            event_type=StreamEventType.DONE,
            data={"elapsed_seconds": 4.2},
        )
        dumped = chunk.model_dump()
        assert dumped["data"]["elapsed_seconds"] == 4.2


# ── Enum round-trips ──────────────────────────────────────────────────────────

class TestEnums:
    def test_role_type_values(self):
        assert RoleType("data_science") == RoleType.DATA_SCIENCE
        assert RoleType("ml_engineering") == RoleType.ML_ENGINEERING

    def test_jd_tone_values(self):
        assert JDTone("highly_quantitative") == JDTone.HIGHLY_QUANTITATIVE
        assert JDTone("balanced") == JDTone.BALANCED

    def test_stream_event_type_values(self):
        assert StreamEventType("token") == StreamEventType.TOKEN
        assert StreamEventType("bullet") == StreamEventType.BULLET
        assert StreamEventType("done") == StreamEventType.DONE
