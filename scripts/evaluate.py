#!/usr/bin/env python3
"""
scripts/evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CVonRAG pipeline benchmark — produces resume-citable metrics across the
five guarantees the architecture claims:

  1. Char accuracy        — % of bullets landing within ±tolerance of target
  2. Fact preservation    — % of source numerics that survive into output
  3. Retrieval quality    — Qdrant cosine-similarity stats on style exemplars
  4. Recommender hit-rate — % of cases where a labelled correct project is
                            in the top-2 recommended projects
  5. End-to-end latency   — P50 / P95 per-bullet (amortised: total per-case
                            time ÷ bullets generated; setup overhead spread
                            across bullets) + per-case context

Input:   tests/eval_set.json  (list of CV+JD cases with labelled correct_projects)
Output:  stdout report + docs/EVAL_REPORT.md + eval_results.json (raw per-case data)

Usage:
  # Run all five phases on Groq + Ollama, all cases in tests/eval_set.json:
  python scripts/evaluate.py

  # Subset / smoke run:
  python scripts/evaluate.py --backend groq --limit 3 --phase char,latency

  # Different eval set:
  python scripts/evaluate.py --eval-set tests/eval_set_v2.json

Requires:  the same services /optimize needs — Ollama for embeddings, Qdrant
           seeded with the gold_standard_cvs collection, and either a Groq /
           OpenRouter key in .env (for --backend groq) or `ollama serve`
           running with the model in OLLAMA_LLM_MODEL (for --backend ollama).

The script imports app/ directly — no FastAPI process needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root so GROQ_API_KEY / OPENROUTER_API_KEY pick up
# before app.config.get_settings() caches them.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Make `app` importable when run as `python scripts/evaluate.py`.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.chains import CVonRAGOrchestrator, HostedLLMQuotaExhausted, _hosted_llm_config
from app.config import get_settings
from app.models import (
    FormattingConstraints,
    GeneratedBullet,
    OptimizationRequest,
    ProjectData,
    RoleType,
)
from app.parser import extract_facts, parse_document_bytes
from app.recommender import recommend_projects
from app.vector_store import collection_info, retrieve_style_exemplars

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger   = logging.getLogger("cvonrag.evaluate")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Numeric extraction — Phase 2 (fact preservation)
# ─────────────────────────────────────────────────────────────────────────────

# Matches: 0.250, 92.3%, 14B, 1.2M, 87, 0.944, 10k, 25b, 3x
# Allows internal `.` or `,` and an optional magnitude/percent suffix.
# Lowercase b/m/k included alongside uppercase B/M/K so that "10k users" or
# "25b rows" survive the integrity check. Trailing word-boundary lookahead
# keeps "10kg" from matching (the `g` would be a word char, blocking the
# match) so we don't pick up arbitrary suffixed identifiers.
_NUMERIC_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*[%BMKbmkx]?(?!\w)")


def extract_numerics(text: str) -> list[str]:
    """Return numeric tokens preserved verbatim — order kept for debugging."""
    if not text:
        return []
    return _NUMERIC_RE.findall(text)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    backend: str
    # Phase 1
    bullets: list[dict] = field(default_factory=list)
    # Phase 2 (one entry per bullet generated for this case)
    fact_preservation: list[dict] = field(default_factory=list)
    # Phase 3
    retrieval_scores: list[float] = field(default_factory=list)
    # Phase 4
    recommended_titles: list[str] = field(default_factory=list)
    correct_titles: list[str] = field(default_factory=list)
    hit_at_2: bool | None = None
    # Phase 5 — both are recorded so the report can show per-bullet (headline)
    # AND per-case (context). Per-case includes one-time JD analysis + scoring
    # + retrieval setup amortised across the case's bullets; per-bullet divides
    # that total by the bullet count, giving the comparable "what does it cost
    # to produce one optimised bullet" number.
    latency_s: float | None = None              # whole orchestrator run (per case)
    latency_per_bullet_s: float | None = None   # latency_s / len(bullets)
    bullets_generated: int = 0
    # Error capture
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Backend switching — toggle settings live so a single process can run
# multiple backends sequentially without subprocess spawning.
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def with_backend(backend: str):
    """Mutate settings to force the desired LLM backend, restore on exit.

    backend == "groq" / "openrouter": leave hosted-LLM settings as configured
        in .env. The script aborts in main() if neither key is present.
    backend == "ollama": blank both hosted keys so _hosted_llm_config() returns
        None, forcing every LLM call through the local Ollama path. The model
        used is settings.ollama_llm_model.
    """
    saved = {
        "llm_provider":       settings.llm_provider,
        "groq_api_key":       settings.groq_api_key,
        "openrouter_api_key": settings.openrouter_api_key,
    }
    try:
        if backend == "ollama":
            settings.groq_api_key       = ""
            settings.openrouter_api_key = ""
        elif backend in ("groq", "openrouter"):
            settings.llm_provider = backend  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


def active_model_label() -> str:
    """Human-friendly label for the report header — provider + model."""
    cfg = _hosted_llm_config()
    if cfg is not None:
        _, _, model = cfg
        return f"{settings.llm_provider}:{model}"
    return f"ollama:{settings.ollama_llm_model}"


# ─────────────────────────────────────────────────────────────────────────────
# Eval-set loading
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_set(path: Path) -> list[dict]:
    """Load tests/eval_set.json, strip the comment header, and validate.

    The stub file ships with a {"_comment": "..."} header — drop it so callers
    can treat the file as a uniform list[case].

    Three validations beyond schema-presence:
      • target_role_type must be a valid RoleType (otherwise the orchestrator
        crashes mid-run at RoleType(case["target_role_type"]) with a less
        helpful traceback)
      • case_id must be unique (the run-cache is keyed on
        f"{case_id}::{backend}", so collisions silently reuse parsed projects
        from a different case)
      • case_id must be non-empty (cache-key correctness)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of cases, got {type(raw).__name__}")
    cases = [c for c in raw if not (isinstance(c, dict) and c.get("_comment"))]

    valid_roles    = {r.value for r in RoleType}
    seen_case_ids: set[str] = set()
    for c in cases:
        for required in ("case_id", "cv_path", "jd_text", "target_role_type"):
            if required not in c:
                raise ValueError(f"{path}: case {c.get('case_id', '<?>')} missing field {required!r}")
        cid = str(c["case_id"]).strip()
        if not cid:
            raise ValueError(f"{path}: case has empty case_id")
        if cid in seen_case_ids:
            raise ValueError(f"{path}: duplicate case_id {cid!r}")
        seen_case_ids.add(cid)
        if c["target_role_type"] not in valid_roles:
            raise ValueError(
                f"{path}: case {cid} has invalid target_role_type "
                f"{c['target_role_type']!r}; expected one of {sorted(valid_roles)}"
            )
    return cases


async def parse_cv_to_projects(cv_path: Path) -> list[ProjectData]:
    """Run the full parser pipeline (extract bullets → LLM facts) on one CV."""
    file_bytes = cv_path.read_bytes()
    raw_projects = parse_document_bytes(file_bytes, cv_path.name)
    if not raw_projects:
        raise RuntimeError(f"Parser produced no projects from {cv_path.name}")

    out: list[ProjectData] = []
    for i, rp in enumerate(raw_projects):
        facts = await extract_facts(rp)
        if not facts:
            continue
        # Same slug strategy as parser.parse_and_stream, just inline.
        from app.parser import _make_slug
        try:
            out.append(ProjectData(
                project_id=f"p-{i:03d}-{_make_slug(rp.title)}",
                title=rp.title,
                core_facts=facts,
            ))
        except Exception as exc:
            logger.warning("Skipping '%s' in %s: %s", rp.title, cv_path.name, exc)
            continue
    if not out:
        raise RuntimeError(f"No valid projects after fact-extraction for {cv_path.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-case runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_one_case(
    case: dict,
    phases: set[str],
    backend: str,
    char_target: int,
    fact_lookup_cache: dict[str, list[ProjectData]],
) -> CaseResult:
    """Execute the requested phases for one case under the active backend."""
    cid = case["case_id"]
    result = CaseResult(case_id=cid, backend=backend, correct_titles=case.get("correct_projects", []))

    cv_path = (_PROJECT_ROOT / case["cv_path"]).resolve()
    if not cv_path.exists():
        result.error = f"CV not found: {cv_path}"
        return result

    # ── Parse CV (shared across phases for the same case+backend) ────────────
    cache_key = f"{cid}::{backend}"
    if cache_key in fact_lookup_cache:
        projects = fact_lookup_cache[cache_key]
    else:
        try:
            projects = await parse_cv_to_projects(cv_path)
        except HostedLLMQuotaExhausted as exc:
            result.error = f"quota_exhausted: {exc}"
            return result
        except Exception as exc:
            result.error = f"parse_failed: {exc}"
            return result
        fact_lookup_cache[cache_key] = projects

    fact_by_id = {f.fact_id: f for p in projects for f in p.core_facts}

    # ── Phase 3: Retrieval quality (cheap — one embedding call) ──────────────
    if "retrieval" in phases:
        try:
            exemplars = await retrieve_style_exemplars(
                query_text=case["jd_text"], top_k=settings.retrieval_top_k,
            )
            result.retrieval_scores = [e.similarity_score for e in exemplars]
        except Exception as exc:
            logger.warning("retrieval failed for %s: %s", cid, exc)

    # ── Phase 4: Recommender hit-rate ────────────────────────────────────────
    if "recommender" in phases:
        try:
            recs = await recommend_projects(projects, case["jd_text"], top_k=3)
            top2 = [r.title for r in recs[:2]]
            result.recommended_titles = top2
            correct = case.get("correct_projects", [])
            if correct:
                # Case-insensitive substring match — labelling stays tolerant of
                # how the LLM names the same project across parses.
                norm_correct = [c.lower() for c in correct]
                hit = any(
                    any(nc in t.lower() or t.lower() in nc for nc in norm_correct)
                    for t in top2
                )
                result.hit_at_2 = hit
        except HostedLLMQuotaExhausted as exc:
            result.error = f"quota_exhausted: {exc}"
            return result
        except Exception as exc:
            logger.warning("recommender failed for %s: %s", cid, exc)

    # ── Phases 1, 2, 5: drive the orchestrator end-to-end ────────────────────
    needs_optimize = phases & {"char", "facts", "latency"}
    if needs_optimize:
        # Per-case `target_char_limit` overrides the CLI default — lets the
        # eval set carry mixed targets (e.g. 110-char bullets for one role,
        # 140 for another) so Phase 1 always measures against the correct
        # target instead of the global flag.
        case_char_target = int(case.get("target_char_limit", char_target))
        req = OptimizationRequest(
            job_description=case["jd_text"],
            projects=projects[:3],  # top 3 projects max per case
            constraints=FormattingConstraints(
                target_char_limit=case_char_target,
                tolerance=settings.char_tolerance,
                max_bullets_per_project=2,
            ),
            target_role_type=RoleType(case.get("target_role_type", "general")),
            total_bullets_requested=case.get("max_bullets", 4),
        )

        orchestrator = CVonRAGOrchestrator()
        bullets: list[GeneratedBullet] = []
        t0 = time.perf_counter()
        try:
            async for event_type, payload in orchestrator.run(req):
                if event_type == "bullet" and isinstance(payload, GeneratedBullet):
                    bullets.append(payload)
        except HostedLLMQuotaExhausted as exc:
            result.error = f"quota_exhausted: {exc}"
            return result
        except Exception as exc:
            logger.warning("orchestrator failed for %s: %s", cid, exc)
            result.error = f"optimize_failed: {exc}"
            return result
        result.latency_s = time.perf_counter() - t0
        result.bullets_generated = len(bullets)
        # Per-bullet amortised cost — includes JD analysis + scoring +
        # retrieval setup spread across the bullets generated. This is the
        # apples-to-apples number to cite for "how long to produce one
        # bullet"; raw latency_s is reported alongside as case-level context.
        result.latency_per_bullet_s = (
            result.latency_s / len(bullets) if bullets else None
        )

        # Phase 1: char accuracy per bullet
        if "char" in phases:
            for b in bullets:
                delta = abs(b.metadata.char_count - b.metadata.char_target)
                result.bullets.append({
                    "text":              b.text,
                    "char_count":        b.metadata.char_count,
                    "char_target":       b.metadata.char_target,
                    "delta":             delta,
                    "iterations":        b.metadata.iterations_taken,
                    "within_tolerance":  b.metadata.within_tolerance,
                    "first_pass":        b.metadata.iterations_taken == 1 and b.metadata.within_tolerance,
                })

        # Phase 2: fact preservation per bullet
        if "facts" in phases:
            for b in bullets:
                # Source numerics: union over each source fact's text + metrics.
                source_nums: set[str] = set()
                for fid in b.metadata.source_fact_ids:
                    fact = fact_by_id.get(fid)
                    if fact is None:
                        continue
                    source_nums |= set(extract_numerics(fact.text))
                    for m in fact.metrics:
                        source_nums |= set(extract_numerics(m))
                output_nums = set(extract_numerics(b.text))
                # Two distinct quality measures (the architectural One Rule
                # is the first one — the second is reported as context only):
                #
                #   integrity = of numerics that appear in output, how many
                #               match a source numeric verbatim. This is
                #               what the architectural claim guarantees:
                #               0.250 must stay 0.250, never silently
                #               become 0.25 or appear as a fabricated 87%.
                #               Violations are the `fabricated` set —
                #               output numerics with no source match. The
                #               target is 100% integrity / 0 fabrications.
                #
                #   coverage  = of source numerics passed in, how many made
                #               it into the output. The bullet has ~130
                #               chars and must select from 3 facts × ~3
                #               numerics each, so low coverage is normal,
                #               not a bug. Reported for completeness.
                fabricated          = output_nums - source_nums
                preserved_in_output = output_nums & source_nums
                # If the bullet has zero output numerics, per-bullet integrity
                # is undefined — there's nothing to score. Report None instead
                # of 1.0 so a reader scanning the JSON doesn't misread an
                # "everything-was-dropped" bullet as a perfect preservation.
                # The aggregate is unaffected: bullets with no output numerics
                # contribute 0 to both numerator (fabrications) and denominator
                # (total output numerics) in the totals.
                integrity_rate = (
                    None if not output_nums
                    else len(preserved_in_output) / len(output_nums)
                )
                coverage_rate = (
                    1.0 if not source_nums
                    else len(source_nums & output_nums) / len(source_nums)
                )
                result.fact_preservation.append({
                    "bullet_index":    b.metadata.bullet_index,
                    "source_numerics": sorted(source_nums),
                    "output_numerics": sorted(output_nums),
                    "fabricated":      sorted(fabricated),
                    "integrity_rate":  integrity_rate,
                    "coverage_rate":   coverage_rate,
                })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation — per-backend metric rollups
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[CaseResult], phases: set[str]) -> dict[str, Any]:
    """Compute report-ready metrics from a list of per-case results.

    All metrics are robust to missing data: if no bullets were generated for
    any case, the char section returns "n/a" instead of dividing by zero.
    """
    section: dict[str, Any] = {}

    all_bullets = [b for r in results for b in r.bullets]
    if "char" in phases and all_bullets:
        first_pass = sum(1 for b in all_bullets if b["first_pass"])
        within     = sum(1 for b in all_bullets if b["within_tolerance"])
        deltas     = [b["delta"] for b in all_bullets]
        iters      = [b["iterations"] for b in all_bullets]
        section["char"] = {
            "n":                  len(all_bullets),
            "first_pass_rate":    first_pass / len(all_bullets),
            "overall_rate":       within / len(all_bullets),
            "mean_delta":         statistics.mean(deltas),
            "max_delta":          max(deltas),
            "mean_iterations":    statistics.mean(iters),
            "max_iterations":     max(iters),
            "iteration_budget":   settings.char_loop_max_iterations,
            "tolerance":          settings.char_tolerance,
        }

    if "facts" in phases:
        per_bullet = [fp for r in results for fp in r.fact_preservation]
        if per_bullet:
            total_output_nums = sum(len(fp["output_numerics"]) for fp in per_bullet)
            total_fabricated  = sum(len(fp["fabricated"]) for fp in per_bullet)
            total_source_nums = sum(len(fp["source_numerics"]) for fp in per_bullet)
            total_covered     = sum(
                len(set(fp["source_numerics"]) & set(fp["output_numerics"]))
                for fp in per_bullet
            )
            section["facts"] = {
                "bullets_tested":  len(per_bullet),
                # Headline — the architectural claim
                "output_numerics": total_output_nums,
                "fabricated":      total_fabricated,
                "integrity_rate":  (
                    1.0 if total_output_nums == 0
                    else (total_output_nums - total_fabricated) / total_output_nums
                ),
                # Context only — not the architectural claim
                "source_numerics": total_source_nums,
                "coverage_rate":   (
                    1.0 if total_source_nums == 0
                    else total_covered / total_source_nums
                ),
            }

    if "retrieval" in phases:
        all_scores = [s for r in results for s in r.retrieval_scores]
        if all_scores:
            section["retrieval"] = {
                "queries":     len(results),
                "exemplars":   len(all_scores),
                "mean_score":  statistics.mean(all_scores),
                "min_score":   min(all_scores),
                "max_score":   max(all_scores),
                "score_stdev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            }

    if "recommender" in phases:
        hits = [r for r in results if r.hit_at_2 is not None]
        if hits:
            section["recommender"] = {
                "cases":      len(hits),
                "hit_at_2":   sum(1 for r in hits if r.hit_at_2) / len(hits),
            }

    if "latency" in phases:
        per_bullet = sorted(
            r.latency_per_bullet_s for r in results
            if r.latency_per_bullet_s is not None
        )
        per_case = sorted(r.latency_s for r in results if r.latency_s is not None)
        # Emit the section whenever either dimension has data, so a run that
        # records case-level wall-clock but produces zero bullets across every
        # case (rare — most orchestrator failures early-return before
        # latency_s is set, but possible) still surfaces the per-case numbers
        # instead of vanishing from the report.
        if per_bullet or per_case:
            section["latency"] = {
                "bullets":          sum(r.bullets_generated for r in results),
                "cases":            len(per_case),
                "p50_per_bullet":   None,
                "p95_per_bullet":   None,
                "min_per_bullet":   None,
                "max_per_bullet":   None,
                "p50_per_case":     None,
                "p95_per_case":     None,
                "min_per_case":     None,
                "max_per_case":     None,
            }
            # Nearest-rank percentile, 0-indexed. For N<20 this lands on max
            # regardless (mathematically correct — small samples can't resolve
            # P95 below max), but the ceil form is right for N≥20 too.
            if per_bullet:
                p95_idx = max(0, min(len(per_bullet) - 1, math.ceil(0.95 * len(per_bullet)) - 1))
                section["latency"].update({
                    "p50_per_bullet": statistics.median(per_bullet),
                    "p95_per_bullet": per_bullet[p95_idx],
                    "min_per_bullet": min(per_bullet),
                    "max_per_bullet": max(per_bullet),
                })
            if per_case:
                case_p95_idx = max(0, min(len(per_case) - 1, math.ceil(0.95 * len(per_case)) - 1))
                section["latency"].update({
                    "p50_per_case": statistics.median(per_case),
                    "p95_per_case": per_case[case_p95_idx],
                    "min_per_case": min(per_case),
                    "max_per_case": max(per_case),
                })

    return section


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_report(
    aggregates_by_backend: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    """Human-readable plain-text report (also written to docs/EVAL_REPORT.md)."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("CVonRAG Evaluation Report")
    lines.append("=" * 64)
    lines.append(f"Eval set:           {meta['eval_set']}")
    lines.append(f"Cases run:          {meta['cases_run']}")
    lines.append(f"Phases:             {', '.join(sorted(meta['phases']))}")
    lines.append(f"Qdrant vectors:     {meta['vector_count']}")
    lines.append(f"Embedding model:    {settings.ollama_embed_model} (dim {settings.qdrant_vector_size})")
    if meta.get("data_leakage_warning"):
        lines.append("")
        lines.append("WARNING: eval CVs are drawn from docs/good_cvs/, which is the")
        lines.append("         same corpus seeded into Qdrant. Retrieval-quality scores")
        lines.append("         have data leakage and should be reported with that caveat.")
    lines.append("")

    for backend, agg in aggregates_by_backend.items():
        lines.append("─" * 64)
        lines.append(f"Backend: {backend}    ({meta['backend_models'].get(backend, '?')})")
        lines.append("─" * 64)

        if "char" in agg:
            c = agg["char"]
            lines.append("[Char Validation]")
            lines.append(f"  Bullets generated:    {c['n']}")
            lines.append(f"  First-pass accuracy:  {c['first_pass_rate']:.1%}  "
                         f"(≤{c['tolerance']} chars, 1 iteration)")
            lines.append(f"  Overall accuracy:     {c['overall_rate']:.1%}  "
                         f"(≤{c['tolerance']} chars, up to {c['iteration_budget']} iter)")
            lines.append(f"  Mean char delta:      {c['mean_delta']:.2f}")
            lines.append(f"  Mean iterations:      {c['mean_iterations']:.2f} / max {c['iteration_budget']}")
            lines.append("")

        if "facts" in agg:
            f = agg["facts"]
            matched = f["output_numerics"] - f["fabricated"]
            lines.append("[Numeric Integrity]   (the architectural One Rule)")
            lines.append(f"  Bullets tested:       {f['bullets_tested']}")
            lines.append(f"  Integrity rate:       {f['integrity_rate']:.1%}  "
                         f"({matched}/{f['output_numerics']} output numerics matched source)")
            lines.append(f"  Fabrications:         {f['fabricated']}  "
                         f"(target: 0 — these are claim violations)")
            lines.append(f"  Source coverage:      {f['coverage_rate']:.1%}  "
                         f"(of {f['source_numerics']} source numerics; low OK — ~130-char budget)")
            lines.append("")

        if "retrieval" in agg:
            r = agg["retrieval"]
            lines.append("[Qdrant Retrieval]")
            lines.append(f"  Queries:              {r['queries']}")
            lines.append(f"  Mean cosine sim:      {r['mean_score']:.3f}")
            lines.append(f"  Stdev:                {r['score_stdev']:.3f}")
            lines.append(f"  Range:                {r['min_score']:.3f} – {r['max_score']:.3f}")
            lines.append("")

        if "recommender" in agg:
            r = agg["recommender"]
            lines.append("[Recommender]")
            lines.append(f"  Cases scored:         {r['cases']}")
            lines.append(f"  Hit Rate @2:          {r['hit_at_2']:.1%}")
            lines.append("")

        if "latency" in agg:
            l = agg["latency"]
            def _s(v: float | None) -> str:
                return f"{v:.2f}s" if v is not None else "n/a"
            lines.append(f"[Latency — {meta['backend_models'].get(backend, backend)}]")
            lines.append(f"  Per bullet (amortised, {l['bullets']} bullets) — resume-citable")
            lines.append(f"    P50:                {_s(l['p50_per_bullet'])}")
            lines.append(f"    P95:                {_s(l['p95_per_bullet'])}")
            lines.append(f"    Range:              {_s(l['min_per_bullet'])} – {_s(l['max_per_bullet'])}")
            lines.append(f"  Per case ({l['cases']} cases, full orchestrator including JD analysis + scoring + retrieval)")
            lines.append(f"    P50:                {_s(l['p50_per_case'])}")
            lines.append(f"    P95:                {_s(l['p95_per_case'])}")
            lines.append(f"    Range:              {_s(l['min_per_case'])} – {_s(l['max_per_case'])}")
            lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_PHASES = {"char", "facts", "retrieval", "recommender", "latency"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVonRAG pipeline benchmark.")
    p.add_argument("--eval-set", type=Path, default=_PROJECT_ROOT / "tests" / "eval_set.json",
                   help="Path to eval_set.json (default: tests/eval_set.json).")
    p.add_argument("--backend", choices=["groq", "openrouter", "ollama", "both"],
                   default="both",
                   help="Which LLM backend(s) to benchmark. 'both' = groq then ollama.")
    p.add_argument("--phase", default="all",
                   help="Comma-separated phases to run: char,facts,retrieval,recommender,latency,all")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N cases (smoke testing).")
    p.add_argument("--char-target", type=int, default=130,
                   help="Bullet character target for phase 1.")
    p.add_argument("--inter-case-sleep", type=float, default=1.5,
                   help="Seconds to sleep between cases on a hosted backend (Groq rate-limit cushion).")
    p.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT,
                   help="Where to write EVAL_REPORT.md / eval_results.json.")
    return p.parse_args()


async def run_backend(
    backend: str,
    cases: list[dict],
    phases: set[str],
    args: argparse.Namespace,
) -> tuple[list[CaseResult], dict[str, Any]]:
    """Run all cases under one backend; returns (results, aggregate)."""
    cache: dict[str, list[ProjectData]] = {}
    results: list[CaseResult] = []

    print(f"\n>>> Running {len(cases)} case(s) on backend={backend} ({active_model_label()})", flush=True)
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['case_id']} … ", end="", flush=True)
        result = await run_one_case(
            case, phases, backend, args.char_target, cache,
        )
        results.append(result)
        if result.error:
            print(f"ERROR ({result.error})")
            if result.error.startswith("quota_exhausted"):
                print("  ↳ quota exhausted — stopping this backend.")
                break
        else:
            bits = []
            if result.bullets:
                bits.append(f"{len(result.bullets)} bullets")
            if result.latency_s is not None:
                bits.append(f"{result.latency_s:.1f}s")
            if result.hit_at_2 is not None:
                bits.append("HIT" if result.hit_at_2 else "miss")
            print(", ".join(bits) if bits else "ok")

        # Rate-limit cushion on hosted backends only.
        if backend != "ollama" and args.inter_case_sleep > 0 and i < len(cases):
            await asyncio.sleep(args.inter_case_sleep)

    return results, aggregate(results, phases)


def _persist_results(
    aggregates: dict[str, dict[str, Any]],
    all_raw: dict[str, list[dict]],
    meta: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write the markdown report + raw JSON to disk.

    Called once per completed backend (not just at the end) so a Ctrl+C or
    failure during --backend both leaves the prior backend's results
    durable on disk — without this, killing the run mid-Ollama would have
    nuked an already-completed Groq leg.
    """
    report = format_report(aggregates, meta)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docs").mkdir(parents=True, exist_ok=True)
    (output_dir / "docs" / "EVAL_REPORT.md").write_text(
        "```\n" + report + "\n```\n", encoding="utf-8",
    )
    (output_dir / "eval_results.json").write_text(
        json.dumps(
            {"meta": {**meta, "phases": sorted(meta["phases"])}, "by_backend": all_raw, "aggregates": aggregates},
            indent=2,
        ),
        encoding="utf-8",
    )


async def main_async(args: argparse.Namespace) -> int:
    # ── Resolve phases ───────────────────────────────────────────────────────
    if args.phase.strip().lower() == "all":
        phases = set(ALL_PHASES)
    else:
        phases = {p.strip() for p in args.phase.split(",") if p.strip()}
        unknown = phases - ALL_PHASES
        if unknown:
            print(f"Unknown phase(s): {unknown}. Choose from {ALL_PHASES}", file=sys.stderr)
            return 2

    # ── Load cases ───────────────────────────────────────────────────────────
    if not args.eval_set.exists():
        print(f"Eval set not found: {args.eval_set}", file=sys.stderr)
        print("Hint: run `python scripts/build_eval_set.py` first to generate it.", file=sys.stderr)
        return 2
    cases = load_eval_set(args.eval_set)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print("Eval set is empty — nothing to benchmark.", file=sys.stderr)
        return 2

    # ── Resolve backends ─────────────────────────────────────────────────────
    if args.backend == "both":
        backends = ["groq", "ollama"]
    else:
        backends = [args.backend]

    # ── Sanity-check Qdrant before burning LLM time ──────────────────────────
    info = await collection_info()
    vector_count = info.get("vector_count", 0)
    if vector_count == 0 and "retrieval" in phases:
        print("WARN: Qdrant collection is empty — retrieval scores will be empty.", file=sys.stderr)

    # ── Detect data leakage ──────────────────────────────────────────────────
    leakage = any("docs/good_cvs/" in c["cv_path"].replace("\\", "/") for c in cases)

    # ── Run each backend ─────────────────────────────────────────────────────
    aggregates: dict[str, dict[str, Any]] = {}
    all_raw: dict[str, list[dict]] = {}
    backend_models: dict[str, str] = {}
    # Meta references the mutable dicts above — they fill in as backends complete.
    meta: dict[str, Any] = {
        "eval_set":             str(args.eval_set.relative_to(_PROJECT_ROOT) if args.eval_set.is_relative_to(_PROJECT_ROOT) else args.eval_set),
        "cases_run":            len(cases),
        "phases":               phases,
        "vector_count":         vector_count,
        "backend_models":       backend_models,
        "data_leakage_warning": leakage and "retrieval" in phases,
    }

    for backend in backends:
        with with_backend(backend):
            # Fail fast if a hosted backend is selected but no key is set.
            if backend != "ollama" and _hosted_llm_config() is None:
                print(f"\nSkipping backend={backend}: no API key configured "
                      f"(set {backend.upper()}_API_KEY in .env).", file=sys.stderr)
                continue
            backend_models[backend] = active_model_label()
            results, agg = await run_backend(backend, cases, phases, args)
            aggregates[backend] = agg
            all_raw[backend] = [
                {
                    "case_id":            r.case_id,
                    "backend":            r.backend,
                    "bullets":            r.bullets,
                    "fact_preservation":  r.fact_preservation,
                    "retrieval_scores":   r.retrieval_scores,
                    "recommended_titles": r.recommended_titles,
                    "correct_titles":     r.correct_titles,
                    "hit_at_2":           r.hit_at_2,
                    "latency_s":            r.latency_s,
                    "latency_per_bullet_s": r.latency_per_bullet_s,
                    "bullets_generated":    r.bullets_generated,
                    "error":                r.error,
                }
                for r in results
            ]
            # Persist after each backend so a Ctrl+C or later-backend failure
            # can't take the already-completed backends' results down with it.
            _persist_results(aggregates, all_raw, meta, args.output_dir)

    if not aggregates:
        print("No backend produced results — see warnings above.", file=sys.stderr)
        return 1

    print("\n" + format_report(aggregates, meta))
    print(f"\nWrote {args.output_dir / 'docs' / 'EVAL_REPORT.md'}")
    print(f"Wrote {args.output_dir / 'eval_results.json'}")
    return 0


def main() -> int:
    args = parse_args()
    # Windows default console is cp1252 — Unicode separators in the report
    # crash print(). Reconfigure if available; fall back silently otherwise.
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
