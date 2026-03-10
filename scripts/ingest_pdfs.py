#!/usr/bin/env python3
"""
scripts/ingest_pdfs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE-TIME script. Run it once to seed your Qdrant vector store with
Gold Standard CV bullets extracted from your curated PDF collection.

What it does:
  1. Reads every PDF in --pdf_dir
  2. Extracts all bullet points (lines beginning with ▪, •, -, or –)
  3. Calls Ollama LLM once per bullet to tag:
       - role_type       (ml_engineering / data_science / software_engineering / etc.)
       - uses_separator  ("|" / ";" / None)
       - uses_arrow      (True if bullet contains ↑ ↓ →)
       - sentence_structure  (e.g. "verb → tool → metric → impact")
  4. POSTs the complete list to POST /ingest on the running CVonRAG backend

Usage:
  python scripts/ingest_pdfs.py \\
      --pdf_dir  /path/to/your/good_cvs/ \\
      --api_url  http://localhost:8000 \\
      --ollama   http://localhost:11434 \\
      --model    qwen2.5:7b \\
      --dry_run          # optional: print bullets without posting

Dependencies (add to your venv):
  pip install pdfplumber httpx rich

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import re
import sys
from pathlib import Path

import httpx
import pdfplumber

# Try rich for pretty output; fall back to plain print
try:
    from rich.console import Console
    from rich.progress import track
    console = Console()
    log = console.print
except ImportError:
    console = None
    log = print
    def track(it, description=""):
        return it


# ── Bullet detection ──────────────────────────────────────────────────────────

# Bullet markers used in the PGDBA CV template (▪, •, -, –, →, *)
_BULLET_RE = re.compile(r"^\s*[▪•\-–→*]\s+(.+)$")


def _looks_like_header(text: str) -> bool:
    """True if the line is a section header, not a bullet."""
    stripped = text.strip()
    # Short lines that are all-caps or end with ":" are headers
    return (
        len(stripped) < 50
        and (stripped.isupper() or stripped.endswith(":"))
    )


def extract_bullets_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract all bullet-point lines from a PDF.
    Returns cleaned bullet text (without the leading marker character).
    """
    bullets = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # Extract with layout preservation so columns don't bleed together
            text = page.extract_text(layout=True) or ""
            for line in text.splitlines():
                m = _BULLET_RE.match(line)
                if m:
                    bullet_text = m.group(1).strip()
                    # Skip very short lines (likely artefacts) and headers
                    if len(bullet_text) > 30 and not _looks_like_header(bullet_text):
                        bullets.append(bullet_text)
    return bullets


# ── LLM tagging ───────────────────────────────────────────────────────────────

_TAGGING_SYSTEM = """\
You are a CV analyst. Given ONE resume bullet point, return ONLY a valid JSON object — no fences, no explanation.

Schema:
{
  "role_type": "ml_engineering|data_science|software_engineering|quant_finance|product_management|general",
  "uses_separator": "|" or ";" or null,
  "uses_arrow": true or false,
  "sentence_structure": "short phrase like: verb → tool → metric → impact"
}

Rules:
- role_type must be exactly one of the six enum values
- uses_separator: the visual separator char actually present in the bullet, or null
- uses_arrow: true if the bullet contains any of ↑ ↓ → ↗ ↘
- sentence_structure: 3-6 words describing the grammatical architecture
"""


def tag_bullet(bullet_text: str, ollama_url: str, model: str) -> dict:
    """Call Ollama to tag a single bullet with style metadata."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f'Bullet: "{bullet_text}"'}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200, "num_ctx": 2048},
        "system": _TAGGING_SYSTEM,
    }
    try:
        r = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        # Strip potential ```json fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        log(f"  [yellow]Tag failed for bullet (will use defaults): {exc}[/yellow]")
        return {
            "role_type": "general",
            "uses_separator": None,
            "uses_arrow": False,
            "sentence_structure": None,
        }


# ── Ingest call ───────────────────────────────────────────────────────────────

def post_to_ingest(bullets: list[dict], api_url: str) -> dict:
    """POST the fully-tagged bullets to POST /ingest."""
    r = httpx.post(
        f"{api_url}/ingest",
        json={"bullets": bullets},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed CVonRAG Qdrant store with Gold Standard bullets from PDFs."
    )
    parser.add_argument("--pdf_dir",  required=True,  help="Folder containing your good CV PDFs")
    parser.add_argument("--api_url",  default="http://localhost:8000", help="CVonRAG API base URL")
    parser.add_argument("--ollama",   default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--model",    default="qwen2.5:7b",  help="Ollama model for tagging (7b is fine)")
    parser.add_argument("--batch_size", type=int, default=50, help="Bullets per /ingest call")
    parser.add_argument("--dry_run",  action="store_true", help="Print bullets, do not POST to API")
    parser.add_argument("--skip_tag", action="store_true", help="Skip LLM tagging (faster, less rich metadata)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        log(f"[red]Error: {pdf_dir} is not a directory.[/red]")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        log(f"[red]No PDF files found in {pdf_dir}[/red]")
        sys.exit(1)

    log(f"\n[bold]CVonRAG — Gold Standard Ingestion[/bold]")
    log(f"  PDFs found     : {len(pdfs)}")
    log(f"  API target     : {args.api_url}")
    log(f"  Ollama model   : {args.model}")
    log(f"  Dry run        : {args.dry_run}\n")

    # ── Step 1: Extract all bullets ───────────────────────────────────────────
    all_bullets_raw: list[tuple[str, str]] = []   # (filename, bullet_text)

    for pdf_path in track(pdfs, description="Extracting bullets from PDFs…"):
        bullets = extract_bullets_from_pdf(pdf_path)
        log(f"  [cyan]{pdf_path.name}[/cyan]: {len(bullets)} bullets extracted")
        for b in bullets:
            all_bullets_raw.append((pdf_path.name, b))

    log(f"\n[green]Total bullets extracted: {len(all_bullets_raw)}[/green]")

    if not all_bullets_raw:
        log("[red]No bullets found. Check that your PDFs use ▪ or • bullet markers.[/red]")
        sys.exit(1)

    # ── Step 2: Tag with LLM (optional but recommended) ───────────────────────
    tagged_bullets: list[dict] = []

    if args.skip_tag:
        log("[yellow]Skipping LLM tagging — all bullets will have role_type=general[/yellow]\n")
        for _, text in all_bullets_raw:
            tagged_bullets.append({
                "text":               text,
                "role_type":          "general",
                "uses_separator":     "|" if "|" in text else (";" if ";" in text else None),
                "uses_arrow":         any(c in text for c in "↑↓→↗↘"),
                "uses_abbreviations": [],
                "sentence_structure": None,
            })
    else:
        log(f"Tagging {len(all_bullets_raw)} bullets via Ollama ({args.model})…")
        log("(This is the slow step — ~2-3 seconds per bullet. Get a coffee.)\n")

        for filename, text in track(all_bullets_raw, description="Tagging bullets…"):
            tag = tag_bullet(text, args.ollama, args.model)
            tagged_bullets.append({
                "text":               text,
                "role_type":          tag.get("role_type", "general"),
                "uses_separator":     tag.get("uses_separator"),
                "uses_arrow":         tag.get("uses_arrow", False),
                "uses_abbreviations": [],
                "sentence_structure": tag.get("sentence_structure"),
            })

    # ── Step 3: Dry run preview ───────────────────────────────────────────────
    if args.dry_run:
        log("\n[bold yellow]DRY RUN — printing first 10 tagged bullets:[/bold yellow]\n")
        for i, b in enumerate(tagged_bullets[:10]):
            log(f"  [{i+1}] role={b['role_type']:20s} sep={str(b['uses_separator']):4s} arrow={b['uses_arrow']}")
            log(f"       text: {b['text'][:100]}")
            log(f"       struct: {b['sentence_structure']}\n")
        log(f"[dim]... and {len(tagged_bullets)-10} more bullets (not posted, dry run)[/dim]")
        return

    # ── Step 4: POST to /ingest in batches ───────────────────────────────────
    total_upserted = 0
    batch_size = args.batch_size
    batches = [tagged_bullets[i:i+batch_size] for i in range(0, len(tagged_bullets), batch_size)]

    log(f"\nPosting {len(tagged_bullets)} bullets in {len(batches)} batches to {args.api_url}/ingest …\n")

    for i, batch in enumerate(batches, 1):
        try:
            result = post_to_ingest(batch, args.api_url)
            upserted = result.get("upserted", len(batch))
            total_upserted += upserted
            log(f"  Batch {i}/{len(batches)}: [green]✓ {upserted} bullets upserted[/green]")
        except httpx.HTTPStatusError as exc:
            log(f"  Batch {i}/{len(batches)}: [red]✗ HTTP {exc.response.status_code} — {exc.response.text[:200]}[/red]")
        except Exception as exc:
            log(f"  Batch {i}/{len(batches)}: [red]✗ {exc}[/red]")

    log(f"\n[bold green]Done. Total upserted: {total_upserted} / {len(tagged_bullets)} bullets[/bold green]")
    log(f"Check: curl {args.api_url}/health\n")


if __name__ == "__main__":
    main()
