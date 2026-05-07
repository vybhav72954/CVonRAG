"""
app/recommender.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given a list of all parsed projects and a JD, recommends which 2–3
projects to include and explains why.

Pipeline:
  1. Use SemanticMatcher to score every fact in every project
  2. Roll up fact scores → project-level score (top-N mean)
  3. Rank projects, take top K
  4. Call LLM once to write a one-line reason per recommended project
     ("Strong match: directly demonstrates Python + SARIMA + MLOps")

Returns a list of ProjectRecommendation objects sorted by score desc.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging

from app.chains import SemanticMatcher, _ollama_chat, _strip_json_fences
from app.config import get_settings
from app.models import ProjectData, ProjectRecommendation

logger   = logging.getLogger(__name__)
settings = get_settings()


# ── Prompt ────────────────────────────────────────────────────────────────────

_REASON_SYSTEM = """\
You are a career coach helping a student pick the best projects for a job application.

Given a list of projects with their relevance scores and matched JD skills,
write ONE short sentence (≤ 15 words) explaining why each top project is a strong match.
Be specific — mention actual skills or tools from the project.

Return ONLY a valid JSON object — no fences, no preamble.
Schema: {"project_id": "reason string", ...}

Good reason: "Directly demonstrates SARIMA forecasting and Python MLOps — core JD requirements."
Bad reason:  "This project is relevant to the job description."
"""


# ── Scorer ────────────────────────────────────────────────────────────────────

def _project_score(project_id: str, scored_facts: list) -> tuple[float, list[str]]:
    """
    Roll up individual fact scores to a project-level score.
    Strategy: mean of the top-3 fact scores for this project.
    This rewards depth (multiple strong facts) without penalising projects
    that have a few weak bullets alongside great ones.
    """
    facts = [sf for sf in scored_facts if sf.project_id == project_id]
    if not facts:
        return 0.0, []

    facts_sorted = sorted(facts, key=lambda x: x.relevance_score, reverse=True)
    top_n        = facts_sorted[:settings.top_n_facts_for_score]
    score        = sum(f.relevance_score for f in top_n) / len(top_n)
    kws          = list(dict.fromkeys(
        kw for f in top_n for kw in f.matched_jd_keywords
    ))[:settings.max_skills_per_project]
    return round(score, 3), kws


def _top_metrics(project: ProjectData) -> list[str]:
    """Extract key numeric metrics from all facts in a project."""
    metrics: list[str] = []
    for fact in project.core_facts:
        metrics.extend(fact.metrics)
    return list(dict.fromkeys(metrics))[:settings.max_metrics_per_project]


# ── Main recommendation function ──────────────────────────────────────────────

async def recommend_projects(
    projects:        list[ProjectData],
    job_description: str,
    top_k:           int = 3,
) -> list[ProjectRecommendation]:
    """
    Score all projects against the JD and return recommendations sorted by score.

    top_k: how many projects to mark as recommended (rest are returned as
           non-recommended so the frontend can show them as "also available").
    """
    if not projects:
        return []

    # ── Phase 1: analyse JD + score all facts ─────────────────────────────────
    matcher    = SemanticMatcher()
    jd_analysis = await matcher.analyze_jd(job_description)
    scored      = await matcher.score_facts(jd_analysis, projects)

    # ── Phase 2: roll up to project level ────────────────────────────────────
    project_map = {p.project_id: p for p in projects}
    project_scores: list[tuple[str, float, list[str]]] = []

    for project in projects:
        score, matched_skills = _project_score(project.project_id, scored)
        project_scores.append((project.project_id, score, matched_skills))

    project_scores.sort(key=lambda x: x[1], reverse=True)

    # ── Phase 3: get LLM reasons for top-K ───────────────────────────────────
    top_ids = [pid for pid, _, _ in project_scores[:top_k]]

    reasons: dict[str, str] = {}
    if top_ids:
        top_summary = [
            {
                "project_id":     pid,
                "title":          project_map[pid].title,
                "score":          round(score, 2),
                "matched_skills": skills[:5],
                "sample_facts":   [f.text for f in project_map[pid].core_facts[:3]],
            }
            for pid, score, skills in project_scores[:top_k]
        ]
        jd_snippet = job_description[:settings.jd_snippet_max_chars]
        prompt = (
            f"Job Description (excerpt):\n{jd_snippet}\n\n"
            f"Top projects:\n{json.dumps(top_summary, indent=2)}"
        )
        # Mirror the H5 pattern from chains.py: try plain first, retry with
        # json_mode=True on JSONDecodeError, fall through to the skill-list
        # fallback if both fail. Keeps recommender consistent with score_facts/
        # analyze_jd so reasoning models don't degrade the UX silently (N19).
        msgs = [{"role": "user", "content": prompt}]
        try:
            raw = await _ollama_chat(system=_REASON_SYSTEM, messages=msgs, temperature=0.2)
            try:
                reasons = json.loads(_strip_json_fences(raw))
            except json.JSONDecodeError:
                logger.warning(
                    "Reason JSON parse failed — retrying with json_mode. Raw: %.200s", raw,
                )
                raw = await _ollama_chat(
                    system=_REASON_SYSTEM, messages=msgs, temperature=0.2, json_mode=True,
                )
                reasons = json.loads(_strip_json_fences(raw))
            if not isinstance(reasons, dict):
                reasons = {}
        except Exception as exc:
            logger.warning("Reason generation failed: %s — using fallback.", exc)
            reasons = {}

    # ── Phase 4: assemble recommendations ────────────────────────────────────
    recommendations: list[ProjectRecommendation] = []

    for rank, (pid, score, skills) in enumerate(project_scores, start=1):
        project   = project_map[pid]
        is_top    = rank <= top_k
        reason    = reasons.get(pid, "")
        if not reason:
            # Fallback reason from matched skills
            if skills:
                reason = f"Demonstrates {', '.join(skills[:3])} — relevant to this role."
            else:
                reason = "Potentially relevant — review the facts."

        recommendations.append(ProjectRecommendation(
            project_id     = pid,
            title          = project.title,
            score          = score,
            rank           = rank,
            reason         = reason,
            matched_skills = skills,
            top_metrics    = _top_metrics(project),
            recommended    = is_top,
            core_facts     = [f.model_dump() for f in project.core_facts],
        ))

    return recommendations
