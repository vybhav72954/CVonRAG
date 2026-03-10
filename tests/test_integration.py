"""
tests/test_integration.py
Integration tests — Ollama and Qdrant are fully mocked.
No running services needed.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import (
    CoreFact,
    FormattingConstraints,
    JDTone,
    OptimizationRequest,
    ProjectData,
    RoleType,
    ScoredFact,
    StyleExemplar,
)
from app.chains import SemanticMatcher, BulletAlchemist, CVonRAGOrchestrator

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_JD = (
    "We are looking for a Senior ML Engineer with expertise in Python, "
    "SARIMA forecasting, and production MLOps pipelines."
)

JD_ANALYSIS = {
    "required_skills": ["Python", "SARIMA", "MLOps"],
    "preferred_skills": ["Docker"],
    "key_action_verbs": ["build", "deploy", "optimize"],
    "tone": "highly_quantitative",
    "seniority": "senior",
    "domain_keywords": ["forecasting", "RMSE"],
}

FACT_SCORES = [
    {"fact_id": "f-001", "relevance_score": 0.95, "matched_jd_keywords": ["SARIMA"]},
    {"fact_id": "f-002", "relevance_score": 0.80, "matched_jd_keywords": ["MLOps"]},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_project() -> ProjectData:
    return ProjectData(
        project_id="p-001",
        title="Forecasting System",
        core_facts=[
            CoreFact(
                fact_id="f-001",
                text="Built SARIMA(2,0,0)(1,0,0)[12] model reducing RMSE to 0.250",
                tools=["SARIMA"],
                metrics=["RMSE 0.250"],
            ),
            CoreFact(
                fact_id="f-002",
                text="Optimized ensemble weights via SLSQP",
                tools=["SLSQP"],
            ),
        ],
    )


def make_request() -> OptimizationRequest:
    return OptimizationRequest(
        job_description=SAMPLE_JD,
        projects=[make_project()],
        constraints=FormattingConstraints(target_char_limit=130, tolerance=2),
        target_role_type=RoleType.ML_ENGINEERING,
        total_bullets_requested=1,  # limit to 1 so side_effect list is predictable
    )


def make_exemplars() -> list[StyleExemplar]:
    return [
        StyleExemplar(
            exemplar_id="e-001",
            text="• Enhanced accuracy using ARIMAX | Reduced RMSE by 13.5%",
            role_type=RoleType.DATA_SCIENCE,
            similarity_score=0.88,
            sentence_structure="verb → method → metric",
        )
    ]


def make_scored_facts() -> list[ScoredFact]:
    return [
        ScoredFact(
            fact=CoreFact(
                fact_id="f-001",
                text="Built SARIMA model reducing RMSE to 0.250",
                tools=["SARIMA"],
                metrics=["RMSE 0.250"],
            ),
            project_id="p-001",
            relevance_score=0.95,
            matched_jd_keywords=["SARIMA"],
        )
    ]


def make_async_gen(*tokens):
    """
    Factory: returns a coroutine-like callable that produces a fresh
    async generator each time it is called. This correctly mocks an
    async generator function such as _ollama_stream.
    """
    async def _gen(*args, **kwargs):
        for t in tokens:
            yield t
    return _gen


# ═════════════════════════════════════════════════════════════════════════════
# SemanticMatcher
# ═════════════════════════════════════════════════════════════════════════════

class TestSemanticMatcher:

    @pytest.mark.asyncio
    async def test_analyze_jd_parses_valid_json(self):
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=json.dumps(JD_ANALYSIS))):
            result = await SemanticMatcher().analyze_jd(SAMPLE_JD)
        assert result["tone"] == "highly_quantitative"
        assert "SARIMA" in result["required_skills"]

    @pytest.mark.asyncio
    async def test_analyze_jd_handles_json_fence(self):
        fenced = f"```json\n{json.dumps(JD_ANALYSIS)}\n```"
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=fenced)):
            result = await SemanticMatcher().analyze_jd(SAMPLE_JD)
        assert "required_skills" in result

    @pytest.mark.asyncio
    async def test_analyze_jd_handles_malformed_json(self):
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value="not json")):
            result = await SemanticMatcher().analyze_jd(SAMPLE_JD)
        assert result == {}

    @pytest.mark.asyncio
    async def test_score_facts_assigns_and_sorts_scores(self):
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=json.dumps(FACT_SCORES))):
            scored = await SemanticMatcher().score_facts(JD_ANALYSIS, [make_project()])
        assert len(scored) == 2
        assert scored[0].relevance_score >= scored[1].relevance_score

    @pytest.mark.asyncio
    async def test_score_facts_fallback_on_bad_json(self):
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value="broken")):
            scored = await SemanticMatcher().score_facts(JD_ANALYSIS, [make_project()])
        assert all(s.relevance_score == 0.5 for s in scored)

    @pytest.mark.asyncio
    async def test_score_facts_preserves_project_id(self):
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=json.dumps(FACT_SCORES))):
            scored = await SemanticMatcher().score_facts(JD_ANALYSIS, [make_project()])
        assert all(s.project_id == "p-001" for s in scored)


# ═════════════════════════════════════════════════════════════════════════════
# BulletAlchemist
# ═════════════════════════════════════════════════════════════════════════════

class TestBulletAlchemist:

    @pytest.mark.asyncio
    async def test_converges_first_iteration(self):
        """Bullet within tolerance → returns on iteration 1, no correction call."""
        bullet_130 = "• " + "x" * 128   # exactly 130 chars
        assert len(bullet_130) == 130

        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=bullet_130)):
            draft = await BulletAlchemist().generate_bullet(
                scored_facts=make_scored_facts(),
                exemplars=make_exemplars(),
                jd_analysis=JD_ANALYSIS,
                jd_tone=JDTone.HIGHLY_QUANTITATIVE,
                constraints=FormattingConstraints(target_char_limit=130, tolerance=2),
                role_type=RoleType.ML_ENGINEERING,
            )
        assert draft.within_tolerance is True
        assert draft.iteration == 1
        assert draft.char_count == 130

    @pytest.mark.asyncio
    async def test_correction_loop_runs_for_short_bullet(self):
        """Loop calls LLM again when bullet is too short, converges on later attempt."""
        short  = "• " + "x" * 80   # 82 chars — too short
        exact  = "• " + "x" * 128  # 130 chars — hit

        call_count = 0

        async def mock_chat(messages=None, system=None, temperature=None, max_tokens=None):
            nonlocal call_count
            call_count += 1
            return short if call_count < 3 else exact

        with patch("app.chains._ollama_chat", new=mock_chat):
            draft = await BulletAlchemist().generate_bullet(
                scored_facts=make_scored_facts(),
                exemplars=make_exemplars(),
                jd_analysis=JD_ANALYSIS,
                jd_tone=JDTone.HIGHLY_QUANTITATIVE,
                constraints=FormattingConstraints(target_char_limit=130, tolerance=2),
                role_type=RoleType.ML_ENGINEERING,
            )
        assert draft.within_tolerance is True
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_failsafe_returns_closest_on_max_iterations(self):
        """After max_iterations, returns closest draft — never hangs."""
        too_long = "• " + "x" * 200  # 202 chars — always too long

        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=too_long)), \
             patch("app.chains.settings") as mock_settings:
            mock_settings.char_loop_max_iterations = 2
            mock_settings.llm_temperature          = 0.3
            mock_settings.llm_max_tokens           = 512
            mock_settings.llm_context_window       = 8192
            mock_settings.ollama_base_url          = "http://localhost:11434"
            mock_settings.ollama_llm_model         = "qwen2.5:14b"
            mock_settings.retrieval_top_k          = 5

            draft = await BulletAlchemist().generate_bullet(
                scored_facts=make_scored_facts(),
                exemplars=make_exemplars(),
                jd_analysis=JD_ANALYSIS,
                jd_tone=JDTone.HIGHLY_QUANTITATIVE,
                constraints=FormattingConstraints(target_char_limit=130, tolerance=2),
                role_type=RoleType.ML_ENGINEERING,
            )
        assert draft is not None
        assert draft.within_tolerance is False

    @pytest.mark.asyncio
    async def test_source_fact_ids_populated(self):
        exact = "• " + "x" * 128
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=exact)):
            draft = await BulletAlchemist().generate_bullet(
                scored_facts=make_scored_facts(),
                exemplars=make_exemplars(),
                jd_analysis=JD_ANALYSIS,
                jd_tone=JDTone.HIGHLY_QUANTITATIVE,
                constraints=FormattingConstraints(target_char_limit=130, tolerance=2),
                role_type=RoleType.ML_ENGINEERING,
            )
        assert "f-001" in draft.source_fact_ids


# ═════════════════════════════════════════════════════════════════════════════
# CVonRAGOrchestrator — full pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestOrchestrator:

    @pytest.mark.asyncio
    async def test_emits_bullet_and_done_events(self):
        """Full pipeline with all external calls mocked — must emit bullet + done."""
        exact_bullet = "• " + "x" * 128

        # _ollama_stream is an async generator function; mock must also be one
        stream_mock = make_async_gen("• Built")

        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=[
            json.dumps(JD_ANALYSIS),   # analyze_jd
            json.dumps(FACT_SCORES),   # score_facts
            exact_bullet,              # generate_bullet iteration 1
        ])), patch("app.chains._ollama_stream", new=stream_mock), \
             patch("app.chains.retrieve_style_exemplars",
                   new=AsyncMock(return_value=make_exemplars())):

            events = []
            async for ev_type, _payload in CVonRAGOrchestrator().run(make_request()):
                events.append(ev_type)

        assert "bullet" in events
        assert "done" in events

    @pytest.mark.asyncio
    async def test_respects_total_bullets_requested(self):
        """total_bullets_requested=1 → exactly one bullet event emitted."""
        req = make_request()
        req.total_bullets_requested = 1
        exact_bullet = "• " + "x" * 128

        stream_mock = make_async_gen("token")

        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=[
            json.dumps(JD_ANALYSIS),
            json.dumps(FACT_SCORES),
            exact_bullet,
        ])), patch("app.chains._ollama_stream", new=stream_mock), \
             patch("app.chains.retrieve_style_exemplars",
                   new=AsyncMock(return_value=make_exemplars())):

            bullets = [
                payload
                async for ev_type, payload in CVonRAGOrchestrator().run(req)
                if ev_type == "bullet"
            ]

        assert len(bullets) == 1

    @pytest.mark.asyncio
    async def test_token_events_emitted_before_bullet(self):
        """Token stream events should arrive before the final bullet event."""
        exact_bullet = "• " + "x" * 128
        stream_mock  = make_async_gen("• ", "Built", " model")

        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=[
            json.dumps(JD_ANALYSIS),
            json.dumps(FACT_SCORES),
            exact_bullet,
        ])), patch("app.chains._ollama_stream", new=stream_mock), \
             patch("app.chains.retrieve_style_exemplars",
                   new=AsyncMock(return_value=make_exemplars())):

            events = [
                ev_type
                async for ev_type, _ in CVonRAGOrchestrator().run(make_request())
            ]

        # token events come before bullet
        first_bullet = events.index("bullet")
        assert any(events[i] == "token" for i in range(first_bullet))

    @pytest.mark.asyncio
    async def test_generated_bullet_has_correct_project_id(self):
        """GeneratedBullet metadata must reference the correct project_id."""
        exact_bullet = "• " + "x" * 128
        stream_mock  = make_async_gen("token")

        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=[
            json.dumps(JD_ANALYSIS),
            json.dumps(FACT_SCORES),
            exact_bullet,
        ])), patch("app.chains._ollama_stream", new=stream_mock), \
             patch("app.chains.retrieve_style_exemplars",
                   new=AsyncMock(return_value=make_exemplars())):

            bullets = [
                payload
                async for ev_type, payload in CVonRAGOrchestrator().run(make_request())
                if ev_type == "bullet"
            ]

        assert bullets[0].metadata.project_id == "p-001"
