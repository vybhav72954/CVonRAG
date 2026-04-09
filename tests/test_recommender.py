"""
tests/test_recommender.py
Unit tests for app/recommender.py — all LLM calls mocked.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import CoreFact, ProjectData
from app.recommender import (
    ProjectRecommendation,
    _project_score,
    _top_metrics,
    recommend_projects,
)
from app.models import ScoredFact


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fact(fact_id: str, text: str, metrics=None, tools=None) -> CoreFact:
    return CoreFact(
        fact_id=fact_id,
        text=text if len(text) >= 5 else text + " " * (5 - len(text)),
        metrics=metrics or [],
        tools=tools or [],
        outcome="",
    )


def _project(pid: str, title: str, facts: list[CoreFact]) -> ProjectData:
    return ProjectData(project_id=pid, title=title, core_facts=facts)


def _scored(fact: CoreFact, pid: str, score: float, keywords=None) -> ScoredFact:
    return ScoredFact(
        fact=fact,
        project_id=pid,
        relevance_score=score,
        matched_jd_keywords=keywords or [],
    )


SAMPLE_JD = (
    "We are seeking a Senior ML Engineer with Python, SARIMA forecasting, "
    "production MLOps, and experience with large-scale data pipelines. "
    "Quantitative background required. Experience with NLP a plus."
)

F1 = _fact("f1", "Built SARIMA(2,0,0)(1,0,0)[12] model reducing RMSE to 0.250",
           metrics=["RMSE=0.250"], tools=["SARIMA", "SLSQP"])
F2 = _fact("f2", "Optimised ensemble weights via constrained SLSQP",
           metrics=["84.5% weight"], tools=["SLSQP"])
F3 = _fact("f3", "Built multi-agent LLM system using LangChain and GPT-4",
           metrics=["87% speed improvement"], tools=["LangChain", "GPT-4"])
F4 = _fact("f4", "Implemented WGCNA clustering on GSE54564 gene expression data",
           metrics=["Cophenetic=0.941"], tools=["WGCNA", "Cytoscape"])
F5 = _fact("f5", "Reduced infra cost from $2500 to $800 per eval run",
           metrics=["$2500→$800"], tools=["FAISS", "OpenAI"])

P_TIME_SERIES = _project("p-ts",   "Time Series – Hourly Wages",     [F1, F2])
P_CUCKOO      = _project("p-ck",   "Cuckoo.ai",                       [F3, F5])
P_DEPRESSION  = _project("p-dep",  "Decoding Depression Networks",     [F4])
ALL_PROJECTS  = [P_TIME_SERIES, P_CUCKOO, P_DEPRESSION]


# ── Unit: _project_score ──────────────────────────────────────────────────────

class TestProjectScore:
    def test_empty_scored_facts_returns_zero(self):
        score, kws = _project_score("p-ts", [])
        assert score == 0.0
        assert kws == []

    def test_single_fact_score(self):
        scored = [_scored(F1, "p-ts", 0.9, ["SARIMA", "Python"])]
        score, kws = _project_score("p-ts", scored)
        assert score == pytest.approx(0.9)
        assert "SARIMA" in kws

    def test_averages_top_3_not_all(self):
        scored = [
            _scored(F1, "p-ts", 1.0, ["SARIMA"]),
            _scored(F2, "p-ts", 0.8, ["Python"]),
            _scored(F3, "p-ts", 0.6, ["MLOps"]),
            _scored(F4, "p-ts", 0.1, []),   # weak fact — should not drag down average of top-3
        ]
        score, _ = _project_score("p-ts", scored)
        # Expected: mean(1.0, 0.8, 0.6) = 0.8
        assert score == pytest.approx(0.8)

    def test_only_scores_correct_project(self):
        scored = [
            _scored(F1, "p-ts",  0.9, ["SARIMA"]),
            _scored(F3, "p-ck",  0.5, ["LLM"]),
        ]
        score_ts, _ = _project_score("p-ts", scored)
        score_ck, _ = _project_score("p-ck", scored)
        assert score_ts == pytest.approx(0.9)
        assert score_ck == pytest.approx(0.5)

    def test_deduplicates_keywords(self):
        scored = [
            _scored(F1, "p-ts", 0.9, ["Python", "SARIMA"]),
            _scored(F2, "p-ts", 0.8, ["Python", "MLOps"]),   # Python already in F1
        ]
        _, kws = _project_score("p-ts", scored)
        assert kws.count("Python") == 1


class TestTopMetrics:
    def test_extracts_metrics_from_all_facts(self):
        metrics = _top_metrics(P_TIME_SERIES)
        assert "RMSE=0.250" in metrics
        assert "84.5% weight" in metrics

    def test_deduplicates_metrics(self):
        f = _fact("f", "fact text here", metrics=["87%", "87%", "90%"])
        p = _project("p", "Project", [f])
        metrics = _top_metrics(p)
        assert metrics.count("87%") == 1

    def test_capped_at_four(self):
        facts = [_fact(f"f{i}", f"fact text {i}", metrics=[f"{i}%"]) for i in range(10)]
        p = _project("p", "Many Metrics", facts)
        assert len(_top_metrics(p)) <= 4

    def test_empty_project(self):
        p = _project("p", "Empty", [_fact("f", "no metrics")])
        assert _top_metrics(p) == []


# ── Integration: recommend_projects ───────────────────────────────────────────

def _mock_matcher(scored_facts):
    """Patch SemanticMatcher to return controlled scored facts."""
    mock_matcher = MagicMock()
    mock_matcher.analyze_jd = AsyncMock(return_value={
        "required_skills": ["Python", "SARIMA", "MLOps"],
        "tone": "highly_quantitative",
        "seniority": "senior",
    })
    mock_matcher.score_facts = AsyncMock(return_value=scored_facts)
    return mock_matcher


def _mock_chat(reasons_dict):
    """Patch _ollama_chat to return controlled reasons JSON."""
    return AsyncMock(return_value=json.dumps(reasons_dict))


class TestRecommendProjects:
    @pytest.mark.asyncio
    async def test_returns_all_projects_ranked(self):
        scored = [
            _scored(F1, "p-ts",  0.9, ["SARIMA", "Python"]),
            _scored(F2, "p-ts",  0.7, ["MLOps"]),
            _scored(F3, "p-ck",  0.6, ["LLM"]),
            _scored(F4, "p-dep", 0.2, []),
        ]
        reasons = {"p-ts": "Directly shows SARIMA and Python forecasting."}
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat(reasons)):
            result = await recommend_projects(ALL_PROJECTS, SAMPLE_JD, top_k=2)

        assert len(result) == 3   # all projects returned
        # Top project should be p-ts
        assert result[0].project_id == "p-ts"

    @pytest.mark.asyncio
    async def test_recommended_flag_set_for_top_k(self):
        scored = [
            _scored(F1, "p-ts",  0.9, ["SARIMA"]),
            _scored(F3, "p-ck",  0.6, ["LLM"]),
            _scored(F4, "p-dep", 0.2, []),
        ]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects(ALL_PROJECTS, SAMPLE_JD, top_k=2)

        recommended = [r for r in result if r.recommended]
        not_recommended = [r for r in result if not r.recommended]
        assert len(recommended)     == 2
        assert len(not_recommended) == 1

    @pytest.mark.asyncio
    async def test_ranks_are_sequential(self):
        scored = [_scored(F1, "p-ts", 0.9, []),
                  _scored(F3, "p-ck", 0.5, []),
                  _scored(F4, "p-dep", 0.1, [])]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects(ALL_PROJECTS, SAMPLE_JD)

        ranks = [r.rank for r in result]
        assert ranks == list(range(1, len(result) + 1))

    @pytest.mark.asyncio
    async def test_reason_from_llm_used_when_available(self):
        scored = [_scored(F1, "p-ts", 0.9, ["SARIMA"])]
        reasons = {"p-ts": "Directly demonstrates SARIMA forecasting — core JD requirement."}
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat(reasons)):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        assert "SARIMA" in result[0].reason

    @pytest.mark.asyncio
    async def test_fallback_reason_when_llm_fails(self):
        scored = [_scored(F1, "p-ts", 0.9, ["SARIMA", "Python"])]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", AsyncMock(side_effect=Exception("Ollama down"))):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        # Should not raise — fallback reason should mention matched skills
        assert len(result) == 1
        assert result[0].reason != ""

    @pytest.mark.asyncio
    async def test_empty_projects_returns_empty(self):
        result = await recommend_projects([], SAMPLE_JD, top_k=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_core_facts_passed_through(self):
        """core_facts must be present so /optimize can use them directly."""
        scored = [_scored(F1, "p-ts", 0.9, []), _scored(F2, "p-ts", 0.7, [])]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        assert len(result[0].core_facts) == 2
        assert result[0].core_facts[0]["fact_id"] == "f1"

    @pytest.mark.asyncio
    async def test_matched_skills_present_in_top_projects(self):
        scored = [
            _scored(F1, "p-ts", 0.9, ["SARIMA", "Python"]),
            _scored(F2, "p-ts", 0.7, ["MLOps"]),
        ]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        assert "SARIMA" in result[0].matched_skills or "Python" in result[0].matched_skills

    @pytest.mark.asyncio
    async def test_top_metrics_in_recommendation(self):
        scored = [_scored(F1, "p-ts", 0.9, [])]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        assert "RMSE=0.250" in result[0].top_metrics

    @pytest.mark.asyncio
    async def test_scores_sorted_descending(self):
        scored = [
            _scored(F1, "p-ts",  0.9, []),
            _scored(F3, "p-ck",  0.5, []),
            _scored(F4, "p-dep", 0.2, []),
        ]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", _mock_chat({})):
            result = await recommend_projects(ALL_PROJECTS, SAMPLE_JD)

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_bad_llm_json_uses_fallback(self):
        scored = [_scored(F1, "p-ts", 0.9, ["SARIMA"])]
        with patch("app.recommender.SemanticMatcher", return_value=_mock_matcher(scored)), \
             patch("app.recommender._ollama_chat", AsyncMock(return_value="not valid json {{")):
            result = await recommend_projects([P_TIME_SERIES], SAMPLE_JD, top_k=1)

        assert len(result) == 1
        assert result[0].reason != ""


# ── API endpoint test ─────────────────────────────────────────────────────────

class TestRecommendEndpoint:
    def test_recommend_returns_200(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.recommender import ProjectRecommendation as PR

        async def _mock_recommend(projects, job_description, top_k):
            return [
                PR(
                    project_id="p-ts", title="Time Series", score=0.9, rank=1,
                    reason="Strong SARIMA match.", matched_skills=["SARIMA"],
                    top_metrics=["RMSE=0.250"], recommended=True,
                    core_facts=[{"fact_id": "f1", "text": "Built model", "tools": [], "metrics": [], "outcome": ""}],
                ),
            ]

        with patch("app.main._do_recommend", side_effect=_mock_recommend):
            with TestClient(app) as client:
                resp = client.post("/recommend", json={
                    "job_description": SAMPLE_JD,
                    "projects": [{
                        "project_id": "p-ts",
                        "title": "Time Series",
                        "core_facts": [{
                            "fact_id": "f1",
                            "text": "Built SARIMA model reducing RMSE to 0.250",
                            "tools": ["SARIMA"],
                            "metrics": ["RMSE=0.250"],
                            "outcome": "",
                        }],
                    }],
                    "top_k": 1,
                })

        assert resp.status_code == 200
        body = resp.json()
        assert "recommendations" in body
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["rank"] == 1

    def test_recommend_rejects_short_jd(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            resp = client.post("/recommend", json={
                "job_description": "Too short",
                "projects": [{"project_id": "p", "title": "P", "core_facts": [
                    {"fact_id": "f1", "text": "Some fact with detail", "tools": [], "metrics": [], "outcome": ""},
                ]}],
            })

        assert resp.status_code == 422

    def test_recommend_rejects_empty_projects(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            resp = client.post("/recommend", json={
                "job_description": SAMPLE_JD,
                "projects": [],
            })

        assert resp.status_code == 422
