"""
app/parser.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Document parsing pipeline for POST /parse.

Input:  .docx or .pdf file bytes + filename
Output: async generator of SSE events:
  ('progress', {message, current, total})
  ('project',  {project: ProjectData, index, total})
  ('done',     {total_projects, total_facts})
  ('error',    {error_message})

Architecture:
  1. File bytes → RawProject list  (heading + bullets)
  2. RawProject → Ollama LLM      → CoreFact list
  3. Yield one 'project' event per project as it completes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.config import get_settings
from app.models import CoreFact, ProjectData

logger   = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Raw document types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawProject:
    title:   str
    bullets: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# .docx parsing
# ─────────────────────────────────────────────────────────────────────────────

def _is_project_heading(para) -> bool:
    """
    True if the paragraph is a project-level heading.
    Accepts:  Heading 1/2 style  OR  entirely-bold short line
    Rejects:  anything ending in ':'  (subsection labels like 'Data & Preprocessing:')
    """
    text = para.text.strip()
    if not text or text.endswith(":"):
        return False
    if "Heading" in para.style.name:
        return True
    # Manually-bolded headings: all runs are bold, line is short
    if para.runs and all(r.bold for r in para.runs if r.text.strip()):
        return len(text) < 120
    return False


def _is_label(text: str) -> bool:
    """Short line ending in ':' — a subsection label, not a fact."""
    s = text.strip()
    return s.endswith(":") and len(s) < 60


def parse_docx_bytes(file_bytes: bytes) -> list[RawProject]:
    """Parse a .docx file into a list of RawProject objects."""
    from docx import Document  # lazy import — not a runtime dep unless /parse is called

    doc = Document(io.BytesIO(file_bytes))
    projects: list[RawProject] = []
    current: RawProject | None = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if _is_project_heading(para):
            if current and current.bullets:
                projects.append(current)
            current = RawProject(title=text)
        else:
            if _is_label(text) or current is None:
                continue
            clean = re.sub(r"^[▪•\-–→*]\s*", "", text).strip()
            if len(clean) > 15:
                current.bullets.append(clean)

    if current and current.bullets:
        projects.append(current)
    return projects


# ─────────────────────────────────────────────────────────────────────────────
# .pdf parsing
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*[▪•\-–→*]\s+(.+)$")


def parse_pdf_bytes(file_bytes: bytes) -> list[RawProject]:
    """
    Extract bullets from a PDF.
    PDFs lose heading-level style info, so all bullets go into one project.
    The LLM groups them into facts during extraction.
    """
    import pdfplumber  # lazy import

    bullets: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            for line in text.splitlines():
                m = _BULLET_RE.match(line)
                if m:
                    b = m.group(1).strip()
                    if len(b) > 30:
                        bullets.append(b)

    return [RawProject("CV Bullets", bullets)] if bullets else []


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def parse_document_bytes(file_bytes: bytes, filename: str) -> list[RawProject]:
    """Route to the correct parser based on file extension."""
    fname = filename.lower()
    if fname.endswith(".docx"):
        return parse_docx_bytes(file_bytes)
    if fname.endswith(".pdf"):
        return parse_pdf_bytes(file_bytes)
    raise ValueError(f"Unsupported file type '{filename}'. Use .docx or .pdf")


# ─────────────────────────────────────────────────────────────────────────────
# LLM fact extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a CV data extractor. Given a project title and raw bullet points from a resume,
extract 3-6 structured facts suitable for resume optimization.

Return ONLY a valid JSON array — no markdown fences, no preamble.

Schema for each element:
{
  "fact_id": "slug-N",
  "text": "one clear sentence preserving ALL numbers, model names, and technical terms EXACTLY",
  "tools": ["tool1", "tool2"],
  "metrics": ["RMSE=0.250", "87%", "AUC 0.944"],
  "outcome": "the key result or business impact"
}

Critical rules:
- NEVER alter any number (0.250 stays 0.250, not 0.25)
- Include full model names (SARIMA(2,0,0)(1,0,0)[12] not just SARIMA)
- fact_id: kebab-slug from title + index (e.g. cuckoo-ai-1, time-series-2)
- text: ONE sentence that can stand alone as a resume fact
- Prefer facts with measurable outcomes over process-only descriptions
"""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences that some models add."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _make_slug(title: str) -> str:
    """'Time Series Analysis – Hourly Wages' → 'time-series-analysi'"""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:20]


async def extract_facts(project: RawProject, http_client=None) -> list[CoreFact]:
    """
    Extract structured CoreFacts from a project's raw bullets via LLM.
    Uses the shared _ollama_chat() from chains.py (routes to Groq or Ollama).
    On any failure, falls back to one fact per bullet (up to 4).

    Note: http_client param is kept for backward compatibility but no longer used.
    """
    from app.chains import _ollama_chat  # deferred import to avoid circular dep

    slug         = _make_slug(project.title)
    bullets_text = "\n".join(f"- {b}" for b in project.bullets[:20])

    try:
        raw = await _ollama_chat(
            system=_EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": f'Project: "{project.title}"\n\nBullets:\n{bullets_text}'}],
            temperature=0.1,
            max_tokens=1500,
        )
        raw       = _strip_fences(raw)
        raw_facts = json.loads(raw)
        if not isinstance(raw_facts, list):
            raise ValueError("LLM returned non-list")
    except Exception as exc:
        logger.warning("Fact extraction failed for '%s': %s — falling back.", project.title, exc)
        # Fallback: first 4 bullets become facts verbatim
        return [
            CoreFact(fact_id=f"{slug}-{i + 1}", text=b, tools=[], metrics=[], outcome="")
            for i, b in enumerate(project.bullets[:4])
        ]

    facts: list[CoreFact] = []
    for i, item in enumerate(raw_facts):
        try:
            text = item.get("text", "")
            if not text:
                logger.debug("Skipping fact %d: missing 'text' field.", i)
                continue
            facts.append(CoreFact(
                fact_id=item.get("fact_id", f"{slug}-{i + 1}"),
                text=text,
                tools=item.get("tools", []),
                metrics=item.get("metrics", []),
                outcome=item.get("outcome", ""),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed fact %d: %s", i, exc)
            continue

    return facts


# ─────────────────────────────────────────────────────────────────────────────
# Streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def parse_and_stream(
    file_bytes: bytes,
    filename:   str,
    http_client=None,        # DEPRECATED — kept for backward compat, unused
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Full parse pipeline as an async generator.

    Yields (event_type, data) tuples consumed by POST /parse.
    """
    # ── Step 1: parse document into raw projects ──────────────────────────────
    try:
        raw_projects = parse_document_bytes(file_bytes, filename)
    except Exception as exc:
        yield ("error", {"error_message": str(exc)})
        return

    if not raw_projects:
        yield ("error", {
            "error_message": (
                "No projects found in the document. "
                "For .docx files, ensure project titles use Heading 1 style or are bold. "
                "For .pdf files, ensure bullets start with ▪ or •."
            ),
        })
        return

    total = len(raw_projects)
    yield ("progress", {
        "message": f"Found {total} project{'s' if total != 1 else ''}. Extracting facts…",
        "current": 0,
        "total":   total,
    })

    # ── Step 2: extract facts for each project ────────────────────────────────
    total_facts = 0

    for i, raw_proj in enumerate(raw_projects):
        yield ("progress", {
            "message": f"Extracting: {raw_proj.title}",
            "current": i,
            "total":   total,
        })

        facts = await extract_facts(raw_proj, http_client)

        # Guard: every project must have at least one fact
        if not facts:
            facts = [CoreFact(
                fact_id=f"{_make_slug(raw_proj.title)}-1",
                text=raw_proj.bullets[0] if raw_proj.bullets else "Add your fact here",
                tools=[],
                metrics=[],
                outcome="",
            )]

        slug    = _make_slug(raw_proj.title)
        project = ProjectData(
            project_id=f"p-{i:03d}-{slug}",
            title=raw_proj.title,
            core_facts=facts,
        )

        total_facts += len(facts)
        yield ("project", {
            "project": project.model_dump(),
            "index":   i,
            "total":   total,
        })

    yield ("done", {"total_projects": total, "total_facts": total_facts})