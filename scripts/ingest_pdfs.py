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
      --secret   your_secret_here  # or set INGEST_SECRET env var
      --dry_run          # optional: print bullets without posting


Dependencies (add to your venv):
  pip install pdfplumber httpx rich


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


import argparse
import json
import os
import re
import sys
from pathlib import Path


import httpx
import pdfplumber


# Try rich for pretty output; fall back to plain print
try:
    from rich.console import Console
    from rich.progress import track          # ← removed: from tqdm import tqdm
    console = Console()
    log = console.print
except ImportError:
    console = None
    log = print
    def track(it, description=""):
        return it



# ── Bullet detection ──────────────────────────────────────────────────────────


# Bullet markers used in the PGDBA CV template (▪, •, -, –, →, *)
_BULLET_RE = re.compile(r"(?:^|\s{3,})(?:[^\w\s]\s+)?([A-Z0-9].{40,})")
def _looks_like_header(text: str) -> bool:
    """True if the line is a section header, not a bullet."""
    stripped = text.strip()
    # Short lines that are all-caps or end with ":" are headers
    return (
        len(stripped) < 50
        and (stripped.isupper() or stripped.endswith(":"))
    )


# PGDBA CV bullet marker characters (private-use / non-standard Unicode).
# pdfplumber sees these in page.chars but NOT in page.extract_words().
#   \uf0a7 — Wingdings private-use glyph that renders as ▪ (most PGDBA CVs)
#   §       — Section sign repurposed as a bullet (e.g. JP50, Sayali's CV)
_BULLET_MARKER_CHARS = {'\uf0a7', '§'}



def _find_right_column_threshold(words: list, page_chars: list, page_width: float) -> float:
    """
    Determine the x0 threshold (fraction of page_width) separating the narrow
    left column (project titles) from the wide right column (bullet content).


    Strategy:
      1. If the page contains known bullet-marker chars (\uf0a7 or §), set the
         threshold 3pt below the leftmost marker — reliable for all PGDBA templates
         regardless of which marker variant they use.
      2. If no known markers found (plain-text CVs like Soham/Harsh), fall back to
         gap detection on word x0 positions — unchanged from previous behaviour.
    """
    markers = [c for c in page_chars if c['text'] in _BULLET_MARKER_CHARS]
    if markers:
        min_x0 = min(c['x0'] for c in markers)
        return max(0.05, (min_x0 - 3) / page_width)


    # Gap-detection fallback (works for plain-text CVs)
    if not words:
        return 0.18
    x0_fracs = sorted(set(round(w['x0'] / page_width, 2) for w in words))
    candidates = [x for x in x0_fracs if 0.05 <= x <= 0.40]
    if len(candidates) < 2:
        return 0.18
    best_gap_start, best_gap_size = 0.18, 0
    for i in range(len(candidates) - 1):
        gap = candidates[i+1] - candidates[i]
        if gap > best_gap_size:
            best_gap_size = gap
            best_gap_start = (candidates[i] + candidates[i+1]) / 2
    return best_gap_start


def _is_bleed_word(s: str) -> bool:
    """True if trailing text looks like a left-column title bleed, not real content."""
    s = s.strip()
    # Acronyms (OLS, PCA, SMOTE) → real content, don't strip
    if s.replace(' ', '').isupper():
        return False
    # Contains digits or brackets → real content (R2=0.87, GridSearchCV, etc.)
    if any(c.isdigit() or c in '()[]{}=.' for c in s):
        return False
    # Short common English title words → likely bleed
    return True


def extract_bullets_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extracts bullets using Word-Level Coordinate Filtering.
    Groups words into physical lines and drops words that start in the left column.
    """
    bullets = []
    in_target_section = False
    
    project_headers = ["ACADEMIC PROJECT", "ADDITIONAL PROJECT"]
    # Shortened keywords to make it impossible to miss (e.g. "RESPONSIBILITY")
    stop_headers = [
        "WORK EXPERIENCE", "AWARDS", "RESPONSIBILITY", 
        "EXTRA CURRICULAR", "EXTRACURRICULAR", "ELECTIVES", "INTERESTS"
    ]


    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # 1. Extract every individual word with its exact bounding box
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            # 2. Group words into physical horizontal lines based on their 'top' coordinate
            lines = []
            words.sort(key=lambda w: (w['top'], w['x0']))
            
            current_line = []
            current_top = None
            
            for w in words:
                if current_top is None:
                    current_top = w['top']
                    current_line.append(w)
                elif abs(w['top'] - current_top) <= 4:  # 4px vertical tolerance to group a line
                    current_line.append(w)
                else:
                    lines.append(current_line)
                    current_line = [w]
                    current_top = w['top']
            if current_line:
                lines.append(current_line)
                
            page_width = float(page.width)
            
            # 3. Process each physical line
            for line in lines:
                # Construct the full text of the line to check for headers
                full_line_text = " ".join([w['text'] for w in line]).upper()
                
                if any(ph in full_line_text for ph in project_headers):
                    in_target_section = True
                    continue
                if any(sh in full_line_text for sh in stop_headers):
                    in_target_section = False
                    continue
                    
                if not in_target_section:
                    continue
                    
                # 4. THE MAGIC FILTER: Keep ONLY words that start right to the Threshold
                threshold = _find_right_column_threshold(words, page.chars, page_width)
                right_words = [w['text'] for w in line if w['x0'] > threshold * page_width]
                
                if not right_words:
                    continue
                    
                bullet_text = " ".join(right_words).strip()


                # Split on mid-line bullet markers (▪ • ■), take the last segment
                # Handles: "LMs] ▪ Enhanced..." → "Enhanced..."
                #          "AI ▪ Employed..." → "Employed..."
                if any(marker in bullet_text for marker in ['▪', '•', '■']):
                    parts = re.split(r'\s*[▪•■]\s*', bullet_text)
                    # Take the longest part — the actual bullet content
                    bullet_text = max(parts, key=len).strip()
                
                bullet_text = re.sub(
                    r'\s+[A-Z][a-zA-Z\s]{2,20}$',
                    lambda m: '' if _is_bleed_word(str(m.group(0)).strip()) else str(m.group(0)),
                    bullet_text
                ).strip()
                
                # 5. Clean up any lingering punctuation at the very start
                bullet_text = re.sub(r'^[^a-zA-Z0-9]+', '', bullet_text).strip()
                
                if not bullet_text:
                    continue
                if bullet_text.upper().startswith(("KEY SKILLS", "ACADEMIC", "WORK", "AWARDS")):
                    continue
                
                # 6. Append if it's an actual bullet point
                if len(bullet_text) > 40 and not _looks_like_header(bullet_text):
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
  "uses_abbreviations": ["w/", "vs", "~"],
  "sentence_structure": "short phrase like: verb → tool → metric → impact"
}


Rules:
- role_type must be exactly one of the six enum values
- uses_separator: the visual separator char actually present in the bullet, or null
- uses_arrow: true if the bullet contains any of ↑ ↓ → ↗ ↘
- uses_abbreviations: list of shorthand tokens found (e.g. "w/", "vs", "~", "approx", "&"); empty list if none
- sentence_structure: 3-6 words describing the grammatical architecture
"""



def tag_bullet(bullet_text: str, ollama_url: str, model: str, timeout: int = 120) -> dict:
    """Call Ollama to tag a single bullet with style metadata."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f'Bullet: "{bullet_text}"'}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512, "num_ctx": 2048},  # ← 200→512
        "system": _TAGGING_SYSTEM,
    }
    _DEFAULTS = {
        "role_type": "general",
        "uses_separator": None,
        "uses_arrow": False,
        "sentence_structure": None,
    }
    for attempt in range(3):                                                    # ← retry up to 3×
        try:
            r = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json()["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            if not raw:                                                         # ← guard empty string
                raise ValueError("Model returned empty content")
            # Extract first {...} block in case model adds preamble text
            match = re.search(r'\{.*\}', raw, re.DOTALL)                       # ← extract JSON blob
            if not match:
                raise ValueError(f"No JSON object found in response: {raw[:80]}")
            return json.loads(match.group(0))
        except Exception as exc:
            if attempt == 2:                                                    # ← only warn on final failure
                log(f"  [yellow]Tag failed for bullet (will use defaults): {exc}[/yellow]")
    return _DEFAULTS



# ── Ingest call ───────────────────────────────────────────────────────────────


def post_to_ingest(bullets: list[dict], api_url: str, secret: str | None = None) -> dict:
    """POST the fully-tagged bullets to POST /ingest."""
    headers = {}
    if secret:
        headers["X-Ingest-Secret"] = secret
    r = httpx.post(
        f"{api_url}/ingest",
        json={"bullets": bullets},
        headers=headers,
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
    parser.add_argument("--debug_pdf", help="Print ALL extracted lines from a single PDF (use with filename)")
    parser.add_argument("--skip_tag", action="store_true", help="Skip LLM tagging (faster, less rich metadata)")
    parser.add_argument("--secret",   default=None, help="X-Ingest-Secret value (or set INGEST_SECRET env var)")
    args = parser.parse_args()

    # Resolve ingest secret: CLI flag > env var > None
    ingest_secret = args.secret or os.environ.get("INGEST_SECRET") or None


    # Debug mode: print what extraction logic sees from a single PDF
    if args.debug_pdf:
        pdf_dir = Path(args.pdf_dir)
        debug_pdf_path = pdf_dir / args.debug_pdf
        
        if not debug_pdf_path.exists():
            log(f"[red]Error: {debug_pdf_path} not found in {pdf_dir}[/red]")
            sys.exit(1)
        
        log(f"\n[bold]DEBUG MODE — Extracted bullets from {args.debug_pdf}:[/bold]\n")
        
        # Call the actual extraction logic instead of just dumping raw text
        bullets = extract_bullets_from_pdf(debug_pdf_path)
        
        if not bullets:
            log("[yellow]0 bullets extracted. The parser might have missed the target section headers or bullet markers.[/yellow]\n")
        else:
            for i, bullet in enumerate(bullets, 1):
                log(f"[cyan]{i:2d}.[/cyan] {bullet}")
            log(f"\n[bold green]Total bullets extracted: {len(bullets)}[/bold green]\n")
            
        return


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
    log(f"  Ingest secret  : {'✓ set' if ingest_secret else '✗ not set (will fail if server requires it)'}")
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


        for filename, text in track(all_bullets_raw, description="Tagging bullets…"):  # ← was tqdm(..., description=)
            tag = tag_bullet(text, args.ollama, args.model)
            tagged_bullets.append({
                "text":               text,
                "role_type":          tag.get("role_type", "general"),
                "uses_separator":     tag.get("uses_separator"),
                "uses_arrow":         tag.get("uses_arrow", False),
                "uses_abbreviations": tag.get("uses_abbreviations", []),
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


    for i, batch in enumerate(track(batches, description="Posting batches…")):  # ← was tqdm(..., desc=)
        try:
            result = post_to_ingest(batch, args.api_url, secret=ingest_secret)
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