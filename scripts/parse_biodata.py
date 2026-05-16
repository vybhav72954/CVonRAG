#!/usr/bin/env python3
"""
scripts/parse_biodata.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PER-SESSION script. Run this each time you want to generate bullets.

What it does:
  1. Reads your biodata .docx file
  2. Groups paragraphs into projects (by Heading 1 / bold headings)
  3. Uses Ollama to extract structured core_facts for each project:
       - fact_id, text, tools, metrics, outcome
  4. Merges with the job description you paste at the prompt (or pass via --jd_file)
  5. Writes the full OptimizationRequest JSON to --output (default: request.json)
  6. Optionally POSTs it straight to POST /optimize and streams the bullets

Usage:
  # Interactive — paste JD when prompted, get JSON
  python scripts/parse_biodata.py --docx Vybhav_Chaturvedi_Biodata.docx

  # With a saved JD file
  python scripts/parse_biodata.py \\
      --docx    Vybhav_Chaturvedi_Biodata.docx \\
      --jd_file job_description.txt \\
      --output  request.json

  # Parse AND immediately call the API
  python scripts/parse_biodata.py \\
      --docx     Vybhav_Chaturvedi_Biodata.docx \\
      --jd_file  job_description.txt \\
      --stream                         # POSTs to /optimize and prints bullets live

  # Control which projects to include (comma-separated indices, 0-based)
  python scripts/parse_biodata.py --docx ... --projects 0,1,3

Dependencies:
  pip install python-docx httpx rich

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import httpx
from docx import Document

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    console = Console()
    log = console.print
except ImportError:
    console = None
    log = print


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Parse the docx into raw projects
# ─────────────────────────────────────────────────────────────────────────────

class RawProject(NamedTuple):
    title:   str
    bullets: list[str]   # raw text of every bullet/paragraph under this heading


def _is_project_heading(para) -> bool:
    """
    True if this paragraph is a project-level heading.
    Excludes subsection labels like "Agentic AI Architecture:" even if bold.
    """
    text = para.text.strip()
    if not text:
        return False
    # Subsection labels always end with ":" — never treat as project heading
    if text.endswith(":"):
        return False
    # Check explicit Heading style (Heading 1, Heading 2, etc.)
    if "Heading" in para.style.name:
        return True
    # Check if entire paragraph is bold (manually bolded heading)
    # Only treat as heading if it's reasonably short (project titles < 120 chars)
    if para.runs and all(r.bold for r in para.runs if r.text.strip()):
        if len(text) < 120:
            return True
    return False


def _is_subsection_label(text: str) -> bool:
    """True for lines like 'Data & Preprocessing:' — labels, not facts."""
    stripped = text.strip()
    return (
        stripped.endswith(":")
        and len(stripped) < 60
        and not any(c.isdigit() for c in stripped[:20])  # labels rarely start with numbers
    )


def parse_docx(docx_path: Path) -> list[RawProject]:
    """
    Parse the .docx into a list of RawProject objects.
    Each project contains its title and all bullet/paragraph text under it.
    """
    doc = Document(str(docx_path))
    projects: list[RawProject] = []
    current_title  = None
    current_bullets: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if _is_project_heading(para):
            # Save previous project
            if current_title and current_bullets:
                projects.append(RawProject(current_title, list(current_bullets)))
            current_title   = text
            current_bullets = []
        else:
            # Skip pure subsection labels — they're noise, not facts
            if _is_subsection_label(text):
                continue
            # Everything else is a bullet/fact
            # Remove leading bullet markers
            clean = re.sub(r"^[▪•\-–→*]\s*", "", text).strip()
            if clean and len(clean) > 15:   # skip very short fragments
                current_bullets.append(clean)

    # Don't forget the last project
    if current_title and current_bullets:
        projects.append(RawProject(current_title, list(current_bullets)))

    return projects


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Ollama — extract core_facts from each project
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are a CV data extractor. Given a project title and its raw bullet points,
extract 3-6 structured facts suitable for resume optimization.

Return ONLY a valid JSON array — no markdown fences, no preamble.

Each fact must follow this exact schema:
{
  "fact_id": "short-unique-id",
  "text": "one clear descriptive sentence preserving ALL numbers, model names, and technical terms EXACTLY",
  "tools": ["list", "of", "specific", "tools/algorithms/libraries"],
  "metrics": ["list", "of", "exact", "numeric", "results like RMSE=0.250, 87%, AUC 0.944"],
  "outcome": "the key result or impact achieved"
}

CRITICAL rules:
- Never round, alter, or omit any number (0.250 stays 0.250, not 0.25)
- Include ALL model names (SARIMA(2,0,0)(1,0,0)[12] not just SARIMA)
- fact_id should be: project-slug-N (e.g. time-series-1, cuckoo-ai-2)
- text should be ONE sentence that could stand alone as a resume bullet
- If the same metric appears in multiple bullets, include it ONCE in the best fact
- Prefer facts with measurable outcomes over process-only descriptions
"""


def extract_facts_for_project(
    project: RawProject,
    ollama_url: str,
    model: str,
) -> list[dict]:
    """Call Ollama to extract structured core_facts from a project's raw bullets."""
    # Build the prompt — show title + all bullets
    bullets_text = "\n".join(f"  - {b}" for b in project.bullets)
    prompt = f'Project: "{project.title}"\n\nRaw bullets:\n{bullets_text}'

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "system": _EXTRACTION_SYSTEM,
        "options": {"temperature": 0.1, "num_predict": 1500, "num_ctx": 4096},
    }

    r = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json()["message"]["content"].strip()

    # Strip fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        facts = json.loads(raw)
        if isinstance(facts, list):
            return facts
        log(f"  [yellow]Warning: LLM returned non-list for '{project.title}' — skipping[/yellow]")
        return []
    except json.JSONDecodeError as exc:
        log(f"  [yellow]JSON parse failed for '{project.title}': {exc}[/yellow]")
        log(f"  [dim]Raw output: {raw[:300]}[/dim]")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build the OptimizationRequest JSON
# ─────────────────────────────────────────────────────────────────────────────

def build_request(
    projects_with_facts: list[tuple[str, list[dict]]],
    job_description: str,
    char_limit: int,
    role_type: str,
    max_bullets: int,
    tolerance: int = 2,
) -> dict:
    """Assemble the full OptimizationRequest payload."""
    projects = []
    for i, (title, facts) in enumerate(projects_with_facts):
        if not facts:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        projects.append({
            "project_id":  f"p-{i:03d}-{slug[:20]}",
            "title":       title,
            "core_facts":  facts,
        })

    return {
        "job_description": job_description,
        "target_role_type": role_type,
        "constraints": {
            "target_char_limit": char_limit,
            "tolerance":         tolerance,
            "bullet_prefix":     "•",
            "max_bullets_per_project": max_bullets,
        },
        "projects": projects,
        "total_bullets_requested": len(projects) * max_bullets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 (optional): Stream bullets from /optimize
# ─────────────────────────────────────────────────────────────────────────────

def stream_optimize(request_payload: dict, api_url: str) -> None:
    """POST to /optimize and print bullets as they stream in."""
    log(f"\n[bold green]Streaming from {api_url}/optimize …[/bold green]\n")

    with httpx.stream(
        "POST",
        f"{api_url}/optimize",
        json=request_payload,
        timeout=600,
        headers={"Content-Type": "application/json"},
    ) as resp:
        resp.raise_for_status()
        buffer = ""

        for chunk in resp.iter_text():
            buffer += chunk
            frames = buffer.split("\n\n")
            buffer = frames.pop()

            for frame in frames:
                if not frame.strip():
                    continue
                event_type = "message"
                data_line  = ""
                for line in frame.split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    if line.startswith("data:"):
                        data_line  = line[5:].strip()

                if not data_line:
                    continue
                try:
                    parsed = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                if event_type == "token":
                    print(parsed.get("data", ""), end="", flush=True)

                elif event_type == "bullet":
                    bullet_data = parsed.get("data", {})
                    text = bullet_data.get("text", "")
                    meta = bullet_data.get("metadata", {})
                    print()   # newline after token stream
                    log(f"\n[bold cyan]━━ FINAL BULLET ━━[/bold cyan]")
                    log(f"[white]{text}[/white]")
                    log(
                        f"[dim]  {meta.get('char_count')} chars | "
                        f"{meta.get('iterations_taken')} iter | "
                        f"{'✓ within tol' if meta.get('within_tolerance') else '⚠ outside tol'}[/dim]"
                    )

                elif event_type == "error":
                    log(f"[red]Error: {parsed.get('error_message')}[/red]")

                elif event_type == "done":
                    elapsed = parsed.get("data", {}).get("elapsed_seconds", "?")
                    log(f"\n[bold green]Done in {elapsed}s[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ROLE_CHOICES = [
    "ml_engineering", "data_science", "data_science_consultant",
    "quant_finance", "product_management", "general",
]


def main():
    parser = argparse.ArgumentParser(
        description="Parse a biodata .docx and produce an OptimizationRequest JSON."
    )
    parser.add_argument("--docx",        required=True,   help="Path to your biodata .docx file")
    parser.add_argument("--jd_file",     default=None,    help="Path to job description .txt (or paste interactively)")
    parser.add_argument("--output",      default="request.json", help="Output JSON file path")
    parser.add_argument("--ollama",      default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--model",       default="qwen2.5:14b", help="Ollama model for extraction")
    parser.add_argument("--role_type",   default="ml_engineering", choices=ROLE_CHOICES)
    parser.add_argument("--char_limit",  type=int, default=130, help="Target bullet char count")
    parser.add_argument("--max_bullets", type=int, default=2,   help="Max bullets per project")
    parser.add_argument("--projects",    default=None,
                        help="Comma-separated indices of projects to include (e.g. 0,1,3). Default: all")
    parser.add_argument("--stream",      action="store_true", help="POST to /optimize and stream bullets")
    parser.add_argument("--api_url",     default="http://localhost:8000", help="CVonRAG API base URL (for --stream)")
    parser.add_argument("--list_projects", action="store_true", help="List detected projects and exit")
    args = parser.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.exists():
        log(f"[red]File not found: {docx_path}[/red]")
        sys.exit(1)

    # ── Parse docx ────────────────────────────────────────────────────────────
    log(f"\n[bold]CVonRAG — Biodata Parser[/bold]")
    log(f"  Input : {docx_path.name}")

    raw_projects = parse_docx(docx_path)
    if not raw_projects:
        log("[red]No projects found. Check your docx has Heading 1 style project names.[/red]")
        sys.exit(1)

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list_projects:
        log(f"\n[bold]Detected {len(raw_projects)} projects:[/bold]\n")
        for i, p in enumerate(raw_projects):
            log(f"  [{i}] [cyan]{p.title}[/cyan] ({len(p.bullets)} bullets)")
        log("\nUse --projects 0,1,3 to select which to include.\n")
        return

    # ── Filter projects ───────────────────────────────────────────────────────
    if args.projects is not None:
        indices = [int(x.strip()) for x in args.projects.split(",")]
        selected = [raw_projects[i] for i in indices if i < len(raw_projects)]
        log(f"  Selected {len(selected)} of {len(raw_projects)} projects via --projects flag")
    else:
        selected = raw_projects
        log(f"  Found {len(selected)} projects in docx")

    # ── Get job description ───────────────────────────────────────────────────
    if args.jd_file:
        jd_text = Path(args.jd_file).read_text(encoding="utf-8").strip()
        log(f"  JD   : loaded from {args.jd_file} ({len(jd_text)} chars)")
    else:
        log("\n[bold yellow]Paste the job description below.[/bold yellow]")
        log("[dim]Press Enter twice (blank line) when done:[/dim]\n")
        lines = []
        while True:
            try:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            except EOFError:
                break
        jd_text = "\n".join(lines).strip()

    if len(jd_text) < 50:
        log("[red]Job description too short (< 50 chars). Please provide the full JD.[/red]")
        sys.exit(1)

    # ── Extract facts via LLM ─────────────────────────────────────────────────
    log(f"\nExtracting core_facts for {len(selected)} projects via Ollama ({args.model}) …\n")
    projects_with_facts: list[tuple[str, list[dict]]] = []

    for proj in selected:
        log(f"  [cyan]→ {proj.title}[/cyan] ({len(proj.bullets)} raw bullets)")
        facts = extract_facts_for_project(proj, args.ollama, args.model)
        log(f"    [green]✓ {len(facts)} facts extracted[/green]")
        for f in facts:
            log(f"    [dim]{f.get('fact_id', '?'):25s} | metrics={f.get('metrics', [])}[/dim]")
        projects_with_facts.append((proj.title, facts))

    # ── Build request JSON ─────────────────────────────────────────────────────
    request = build_request(
        projects_with_facts=projects_with_facts,
        job_description=jd_text,
        char_limit=args.char_limit,
        role_type=args.role_type,
        max_bullets=args.max_bullets,
    )

    # ── Write to file ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n[bold green]OptimizationRequest written → {output_path}[/bold green]")
    log(f"  Projects : {len(request['projects'])}")
    total_facts = sum(len(p["core_facts"]) for p in request["projects"])
    log(f"  Facts    : {total_facts}")
    log(f"  Bullets  : up to {request['total_bullets_requested']} requested\n")

    # ── Preview first project ──────────────────────────────────────────────────
    if request["projects"]:
        first = request["projects"][0]
        log(f"[bold]Preview — first project: '{first['title']}'[/bold]")
        for f in first["core_facts"][:3]:
            log(f"  [cyan]{f['fact_id']}[/cyan]: {f['text'][:90]}")
            if f.get("metrics"):
                log(f"    metrics: {f['metrics']}")

    # ── Stream mode ────────────────────────────────────────────────────────────
    if args.stream:
        stream_optimize(request, args.api_url)
    else:
        log(f"\n[bold]Next step — generate bullets:[/bold]")
        log(f"  Option A (stream in terminal):")
        log(f"    python scripts/parse_biodata.py --docx {args.docx} --jd_file <jd.txt> --stream\n")
        log(f"  Option B (curl):")
        log(f"    curl -X POST {args.api_url}/optimize \\")
        log(f"         -H 'Content-Type: application/json' \\")
        log(f"         -d @{output_path} \\")
        log(f"         --no-buffer\n")
        log(f"  Option C (Python one-liner):")
        log(f"    python -c \"")
        log(f"      import json, httpx, sys")
        log(f"      r = httpx.post('{args.api_url}/optimize', json=json.load(open('{output_path}')), timeout=600)")
        log(f"      [print(l) for l in r.text.splitlines() if l.startswith('data:')]")
        log(f"    \"")


if __name__ == "__main__":
    main()
