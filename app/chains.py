"""
CVonRAG — chains.py
The full 5-phase pipeline, powered by Groq (preferred) or Ollama (fallback).

Phase 1 → FastAPI (main.py)
Phase 2 → SemanticMatcher   — JD analysis + fact scoring
Phase 3 → vector_store      — style exemplar retrieval
Phase 4 → BulletAlchemist   — generation + ±2 char-limit loop
Phase 5 → CVonRAGOrchestrator.run() — async generator of SSE events
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator

import httpx

from app.config import get_settings
from app.models import (
    BulletDraft,
    BulletMetadata,
    CoreFact,
    FormattingConstraints,
    GeneratedBullet,
    JDTone,
    OptimizationRequest,
    ProjectData,
    RoleType,
    ScoredFact,
    StyleExemplar,
)
from app.vector_store import retrieve_style_exemplars

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Shared async HTTP client ──────────────────────────────────────────────────

_http: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=180.0)
    return _http


async def close_http() -> None:
    """Gracefully close the shared HTTP client (call on shutdown)."""
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


# ═════════════════════════════════════════════════════════════════════════════
# LLM helpers — route to Groq (if GROQ_API_KEY set) or Ollama (fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _using_groq() -> bool:
    """True when a Groq API key is configured."""
    return bool(settings.groq_api_key)


def _build_messages(messages: list[dict], system: str | None) -> list[dict]:
    """Prepend system message for OpenAI-compatible APIs (Groq)."""
    if system:
        return [{"role": "system", "content": system}] + messages
    return messages


# ── Groq rate-limit retry constants ──────────────────────────────────────────

_GROQ_MAX_RETRIES = 3
_GROQ_RETRY_AFTER_DEFAULT = 10.0  # seconds when Retry-After header is absent


def _groq_retry_wait(response: httpx.Response) -> float:
    """Return Retry-After seconds from a 429 response, or the default."""
    try:
        return float(response.headers.get("retry-after", _GROQ_RETRY_AFTER_DEFAULT))
    except (TypeError, ValueError):
        return _GROQ_RETRY_AFTER_DEFAULT


async def _groq_chat(
    messages: list[dict],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Non-streaming Groq call (OpenAI-compatible) — retries up to 3× on 429."""
    payload = {
        "model": settings.groq_model,
        "messages": _build_messages(messages, system),
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
        "stream": False,
    }
    for attempt in range(_GROQ_MAX_RETRIES):
        try:
            r = await get_http().post(
                f"{settings.groq_base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=60.0,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            # Strip Qwen/Llama extended-reasoning blocks if present
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < _GROQ_MAX_RETRIES - 1:
                wait = _groq_retry_wait(exc.response)
                logger.warning(
                    "Groq 429 (chat) — retry %d/%d after %.1fs",
                    attempt + 1, _GROQ_MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")  # loop always returns or re-raises


async def _groq_stream(
    messages: list[dict],
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming Groq call (OpenAI-compatible SSE) — retries up to 3× on 429."""
    payload = {
        "model": settings.groq_model,
        "messages": _build_messages(messages, system),
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }
    for attempt in range(_GROQ_MAX_RETRIES):
        try:
            async with get_http().stream(
                "POST",
                f"{settings.groq_base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                return  # stream ended normally
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < _GROQ_MAX_RETRIES - 1:
                wait = _groq_retry_wait(exc.response)
                logger.warning(
                    "Groq 429 (stream) — retry %d/%d after %.1fs",
                    attempt + 1, _GROQ_MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise


async def _ollama_chat_inner(
    messages: list[dict],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Non-streaming Ollama /api/chat call."""
    payload: dict = {
        "model":  settings.ollama_llm_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "num_predict": max_tokens or settings.llm_max_tokens,
            "num_ctx":     settings.llm_context_window,
        },
    }
    if system:
        payload["system"] = system

    r = await get_http().post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


async def _ollama_stream_inner(
    messages: list[dict],
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming Ollama /api/chat — yields text tokens."""
    payload: dict = {
        "model":  settings.ollama_llm_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
            "num_ctx":     settings.llm_context_window,
        },
    }
    if system:
        payload["system"] = system

    async with get_http().stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue


# ── Public interface (rest of the code calls these only) ─────────────────────

async def _ollama_chat(
    messages: list[dict],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """LLM call — routes to Groq if GROQ_API_KEY is set, else Ollama."""
    if _using_groq():
        return await _groq_chat(messages, system, temperature, max_tokens)
    return await _ollama_chat_inner(messages, system, temperature, max_tokens)


async def _ollama_stream(
    messages: list[dict],
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming LLM call — routes to Groq if GROQ_API_KEY is set, else Ollama."""
    if _using_groq():
        async for token in _groq_stream(messages, system):
            yield token
    else:
        async for token in _ollama_stream_inner(messages, system):
            yield token


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Semantic Matcher
# ═════════════════════════════════════════════════════════════════════════════

_JD_ANALYSIS_SYSTEM = """\
You are a senior technical recruiter. Analyse the job description and return ONLY valid JSON.
No markdown fences, no preamble, no explanation — raw JSON only.

Schema:
{
  "required_skills": ["string"],
  "preferred_skills": ["string"],
  "key_action_verbs": ["string"],
  "tone": "highly_quantitative|engineering_focused|leadership_focused|research_focused|balanced",
  "seniority": "junior|mid|senior|staff|principal",
  "domain_keywords": ["string"]
}"""

_FACT_SCORING_SYSTEM = """\
You are a resume-fit scoring engine. Score each fact's relevance to the JD on 0.0–1.0.
Return ONLY a valid JSON array — no fences, no preamble.

Schema:
[{"fact_id": "string", "relevance_score": float, "matched_jd_keywords": ["string"]}]"""


class SemanticMatcher:
    """Phase 2: analyse the JD and score every CoreFact for relevance."""

    async def analyze_jd(self, jd: str) -> dict:
        raw = await _ollama_chat(
            system=_JD_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": f"Job Description:\n\n{jd}"}],
            temperature=0.05,
        )
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            logger.warning("JD analysis JSON parse failed. Raw: %.200s", raw)
            return {}

    async def score_facts(
        self,
        jd_analysis: dict,
        projects: list[ProjectData],
    ) -> list[ScoredFact]:
        flat: list[tuple[str, CoreFact]] = [
            (p.project_id, f)
            for p in projects
            for f in p.core_facts
        ]
        payload = [{"fact_id": f.fact_id, "text": f.text} for _, f in flat]
        prompt  = (
            f"JD Analysis:\n{json.dumps(jd_analysis, indent=2)}\n\n"
            f"Facts:\n{json.dumps(payload, indent=2)}"
        )
        raw = await _ollama_chat(
            system=_FACT_SCORING_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
        )
        try:
            scores: list[dict] = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            logger.warning("Fact scoring JSON parse failed — assigning 0.5 to all.")
            scores = [
                {"fact_id": f.fact_id, "relevance_score": 0.5, "matched_jd_keywords": []}
                for _, f in flat
            ]

        score_map = {s["fact_id"]: s for s in scores}
        result = [
            ScoredFact(
                fact=fact,
                project_id=project_id,
                relevance_score=float(score_map.get(fact.fact_id, {}).get("relevance_score", 0.5)),
                matched_jd_keywords=score_map.get(fact.fact_id, {}).get("matched_jd_keywords", []),
            )
            for project_id, fact in flat
        ]
        result.sort(key=lambda x: x.relevance_score, reverse=True)
        return result

    def infer_tone(self, jd_analysis: dict) -> JDTone:
        try:
            return JDTone(jd_analysis.get("tone", "balanced"))
        except ValueError:
            return JDTone.BALANCED


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — The Alchemist
# ═════════════════════════════════════════════════════════════════════════════

_ALCHEMIST_SYSTEM = """\
You are CVonRAG's Bullet Alchemist — the world's most precise resume bullet-point writer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT / STYLE FIREWALL  (ZERO TOLERANCE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT  (User JSON ONLY):
  • Every number, %, financial figure — EXACT, never rounded or altered
  • Every named tool, algorithm, library, framework
  • The actual outcome the user achieved
  • Specific action verbs for what the user actually did

STYLE  (Exemplars ONLY):
  • Visual separators: | • ; and special chars ↑ ↓ → & w/
  • Sentence architecture: VERB → TOOL → METRIC → IMPACT
  • Abbreviation conventions: vs, w/, ~, approx

ABSOLUTE PROHIBITIONS:
  ✗ NEVER copy company names or domain jargon from an Exemplar into output
  ✗ NEVER attach an Exemplar's metric to the user's tool (or vice versa)
  ✗ NEVER invent steps, deployments, or outcomes not in the User JSON
  ✗ NEVER alter any number. 87% stays 87%. 0.250 stays 0.250. RMSE stays RMSE.

OUTPUT: ONE plain-text bullet line. No markdown, no quotes, no explanation."""

_GENERATION_TEMPLATE = """\
Role Type   : {role_type}
JD Tone     : {jd_tone}
JD Keywords : {jd_keywords}

━━ USER JSON FACTS (CONTENT — nouns, numbers, tools from here ONLY) ━━
{facts}

━━ GOLD STANDARD EXEMPLARS (STYLE REFERENCE ONLY — do NOT copy content) ━━
{exemplars}

━━ FEW-SHOT TRANSFORMATION EXAMPLES ━━
Fact   : "Built SARIMA(2,0,0)(1,0,0)[12] model using ACF/PACF... reduced RMSE to 0.250"
Style  : "Enhanced forecast accuracy using ARIMAX | Reduced RMSE by 13.5%"
Output : • Built SARIMA(2,0,0)(1,0,0)[12] model via ACF/PACF analysis | Optimized ensemble weights via constrained SLSQP, ↑ predictive accuracy by reducing RMSE to 0.250

Fact   : "Architected multi-agent LLM system using LangChain + GPT-4... 87% faster (8-12 weeks to 5-10 days)"
Style  : "Handled class imbalance via undersampling; ↑ F1 score from 0.39 to 0.49"
Output : • Architected multi-agent LLM system using LangChain & GPT-4 | Orchestrated 6 specialized agents, ↑ evaluation speed by 87% (8-12 weeks to 5-10 days)

━━ TASK ━━
Bullet prefix : "{prefix}"
Target length : {target} chars  (acceptable range: {lower}–{upper} chars)

Write the bullet now (single line, no explanation):"""

_EXPAND_TEMPLATE = """\
Current bullet ({cur} chars):
{bullet}

{delta} chars SHORT of target {target} (range: {lower}–{upper}).

Expand by:
  • Replacing short words with longer technical synonyms already in the JSON
  • Adding a missing tool, step, or metric from the JSON that was omitted
  • Spelling out an abbreviation you used

FIREWALL: all new content must come from the original JSON facts only.
Output only the revised bullet (single line, no explanation):"""

_COMPRESS_TEMPLATE = """\
Current bullet ({cur} chars):
{bullet}

{delta} chars OVER target {target} (range: {lower}–{upper}).

Compress by:
  • & instead of "and", w/ instead of "with"
  • ↑ for "increased", ↓ for "decreased", → for "to"
  • ~ for "approximately"; remove "successfully", "effectively", "the"

FIREWALL: do NOT remove or alter any number, percentage, or named tool.
Output only the revised bullet (single line, no explanation):"""


class BulletAlchemist:
    """Phase 4: generate one bullet point with the ±tolerance char-limit loop."""

    def _build_initial_prompt(
        self,
        scored_facts: list[ScoredFact],
        exemplars: list[StyleExemplar],
        jd_analysis: dict,
        jd_tone: JDTone,
        constraints: FormattingConstraints,
        role_type: RoleType,
    ) -> str:
        """Build the initial generation prompt (shared by stream + correction)."""
        return _GENERATION_TEMPLATE.format(
            role_type=role_type.value,
            jd_tone=jd_tone.value,
            jd_keywords=", ".join(jd_analysis.get("required_skills", [])[:settings.jd_top_keywords]),
            facts=_format_facts(scored_facts),
            exemplars=_format_exemplars(exemplars),
            prefix=constraints.bullet_prefix,
            target=constraints.target_char_limit,
            lower=constraints.lower_bound,
            upper=constraints.upper_bound,
        )

    async def generate_bullet(
        self,
        scored_facts: list[ScoredFact],
        exemplars: list[StyleExemplar],
        jd_analysis: dict,
        jd_tone: JDTone,
        constraints: FormattingConstraints,
        role_type: RoleType,
    ) -> BulletDraft:
        """Run the full correction loop (cold start — no streamed draft)."""
        initial_prompt = self._build_initial_prompt(
            scored_facts, exemplars, jd_analysis, jd_tone, constraints, role_type,
        )

        history: list[dict] = [{"role": "user", "content": initial_prompt}]
        bullet  = _clean_bullet(
            await _ollama_chat(system=_ALCHEMIST_SYSTEM, messages=history),
            constraints.bullet_prefix,
        )
        history.append({"role": "assistant", "content": bullet})

        return await self._correction_loop(bullet, history, scored_facts, constraints)

    async def _correction_loop(
        self,
        bullet: str,
        history: list[dict],
        scored_facts: list[ScoredFact],
        constraints: FormattingConstraints,
    ) -> BulletDraft:
        """Shared ±tolerance char-limit correction loop."""
        best: BulletDraft | None = None

        for iteration in range(1, settings.char_loop_max_iterations + 1):
            length = len(bullet)
            within = constraints.lower_bound <= length <= constraints.upper_bound

            draft = BulletDraft(
                text=bullet,
                char_count=length,
                iteration=iteration,
                within_tolerance=within,
                source_fact_ids=[sf.fact.fact_id for sf in scored_facts],
            )
            if best is None or abs(length - constraints.target_char_limit) < abs(
                best.char_count - constraints.target_char_limit
            ):
                best = draft

            if within:
                logger.debug("Bullet converged on iteration %d (%d chars).", iteration, length)
                return draft

            if iteration == settings.char_loop_max_iterations:
                logger.warning(
                    "Loop exhausted after %d iterations — returning closest (%d chars, target %d).",
                    iteration, best.char_count, constraints.target_char_limit,
                )
                return best  # type: ignore[return-value]

            delta = abs(length - constraints.target_char_limit)
            correction = (
                _EXPAND_TEMPLATE if length < constraints.lower_bound else _COMPRESS_TEMPLATE
            ).format(
                cur=length, bullet=bullet, delta=delta,
                target=constraints.target_char_limit,
                lower=constraints.lower_bound,
                upper=constraints.upper_bound,
            )
            history.append({"role": "user", "content": correction})
            bullet = _clean_bullet(
                await _ollama_chat(system=_ALCHEMIST_SYSTEM, messages=history),
                constraints.bullet_prefix,
            )
            history.append({"role": "assistant", "content": bullet})

        return best  # type: ignore[return-value]


# ═════════════════════════════════════════════════════════════════════════════
# Typewriter stream — chunk the FINAL bullet to the browser
# ═════════════════════════════════════════════════════════════════════════════

async def _chunk_stream(text: str) -> AsyncGenerator[str, None]:
    """Yield the finalized bullet word-by-word so the browser typewriter shows
    exactly the text that will be kept (no replace-on-correction blink).

    Why: the previous design streamed tokens of an *initial* LLM draft, then
    silently rewrote it through the char-limit correction loop — the user saw
    one sentence type out and a different one snap into place. Now we run the
    correction loop silently, then chunk-stream the final draft.
    """
    if not text:
        yield ""
        return
    parts = re.findall(r"\S+\s*", text) or [text]
    delay = settings.bullet_stream_chunk_delay
    for part in parts:
        yield part
        if delay:
            await asyncio.sleep(delay)


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — wires all phases
# ═════════════════════════════════════════════════════════════════════════════

class CVonRAGOrchestrator:
    def __init__(self) -> None:
        self.matcher  = SemanticMatcher()
        self.alchemist = BulletAlchemist()

    async def run(
        self,
        request: OptimizationRequest,
    ) -> AsyncGenerator[tuple[str, object], None]:
        """
        Yields (event_type, payload) tuples.
        event_type: "token" | "bullet" | "done" | "error"
        """
        # Phase 2
        jd_analysis = await self.matcher.analyze_jd(request.job_description)
        jd_tone     = self.matcher.infer_tone(jd_analysis)
        all_scored  = await self.matcher.score_facts(jd_analysis, request.projects)

        project_fact_map: dict[str, list[ScoredFact]] = {}
        for sf in all_scored:
            project_fact_map.setdefault(sf.project_id, []).append(sf)

        # Phase 3
        style_query = (
            request.job_description[:settings.style_query_jd_chars]
            + " "
            + " ".join(sf.fact.text for sf in all_scored[:8])
        )
        exemplars = await retrieve_style_exemplars(
            query_text=style_query,
            role_type=request.target_role_type,
            top_k=settings.retrieval_top_k,
        )

        bullet_index      = 0
        bullets_remaining = request.total_bullets_requested or 999

        for project in request.projects:
            if bullets_remaining <= 0:
                break

            scored = project_fact_map.get(project.project_id, [])
            if not scored:
                continue

            n = min(
                request.constraints.max_bullets_per_project,
                len(scored),
                bullets_remaining,
            )

            for i in range(n):
                primary    = scored[i]
                supporting = _pick_supporting_facts(primary, scored, exclude_idx=i)
                facts      = [primary] + supporting

                # Phase 4: run the full correction loop silently — we want the
                # *final* bullet, not an initial draft that may get rewritten.
                draft = await self.alchemist.generate_bullet(
                    scored_facts=facts,
                    exemplars=exemplars,
                    jd_analysis=jd_analysis,
                    jd_tone=jd_tone,
                    constraints=request.constraints,
                    role_type=request.target_role_type,
                )

                # Phase 5: chunk-stream the *final* bullet so the typewriter
                # in the browser shows exactly what the user is going to keep.
                async for token in _chunk_stream(draft.text):
                    yield ("token", token)

                metadata = BulletMetadata(
                    bullet_index=bullet_index,
                    project_id=project.project_id,
                    source_fact_ids=draft.source_fact_ids,
                    char_count=draft.char_count,
                    char_target=request.constraints.target_char_limit,
                    iterations_taken=draft.iteration,
                    exemplar_ids_used=[e.exemplar_id for e in exemplars],
                    jd_tone=jd_tone,
                    within_tolerance=draft.within_tolerance,
                )
                yield ("bullet", GeneratedBullet(text=draft.text, metadata=metadata))

                bullet_index      += 1
                bullets_remaining -= 1

        yield ("done", None)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _format_facts(scored_facts: list[ScoredFact]) -> str:
    lines = []
    for sf in scored_facts:
        f    = sf.fact
        line = f"• [{f.fact_id}] {f.text}"
        if f.metrics:
            line += f"\n  └─ metrics : {', '.join(f.metrics)}"
        if f.tools:
            line += f"\n  └─ tools   : {', '.join(f.tools)}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no facts provided)"


def _format_exemplars(exemplars: list[StyleExemplar]) -> str:
    if not exemplars:
        return "(none — use standard: action-verb → tool → metric → impact)"
    lines = []
    for i, ex in enumerate(exemplars, 1):
        lines.append(f"{i}. {ex.text}")
        if ex.sentence_structure:
            lines.append(f"   [structure: {ex.sentence_structure}]")
    return "\n".join(lines)


def _clean_bullet(text: str, prefix: str) -> str:
    """Strip LLM artefacts and enforce the bullet prefix."""
    # Remove Qwen2.5 extended-reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown bold/italic
    text = re.sub(r"\*+", "", text)
    # Remove any leading dash/asterisk the LLM might have added
    text = re.sub(r"^[-–—*]\s*", "", text.strip())
    text = text.strip()
    # Enforce correct prefix
    if prefix and not text.startswith(prefix):
        text = f"{prefix} {text.lstrip('•·-– ')}"
    return text.strip()


def _pick_supporting_facts(
    primary: ScoredFact,
    all_facts: list[ScoredFact],
    exclude_idx: int,
    max_supporting: int = 2,
) -> list[ScoredFact]:
    """Return facts that share JD keywords with the primary (up to max_supporting)."""
    primary_kw = set(primary.matched_jd_keywords)
    return [
        sf for j, sf in enumerate(all_facts)
        if j != exclude_idx and set(sf.matched_jd_keywords) & primary_kw
    ][:max_supporting]


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences that Qwen sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()