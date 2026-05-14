#!/usr/bin/env python3
"""
scripts/build_eval_set.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate the *skeleton* of tests/eval_set.json from PDFs in docs/good_cvs/.

For each selected CV the script:
  1. Runs the full parser (PDF → bullets → LLM-extracted CoreFacts)
  2. Writes a case entry with:
       - cv_path
       - jd_text          ← template JD per role_type (EDIT BEFORE BENCHMARK)
       - target_role_type ← rotated through the role types
       - target_char_limit = 130
       - correct_projects = []   ← YOU FILL THIS IN
       - _parsed_project_titles  ← reference list of titles the parser found,
                                    so you know which strings to put in
                                    correct_projects

After running, hand-edit tests/eval_set.json:
  • Replace each placeholder JD with a realistic one for that role
  • Fill correct_projects from the _parsed_project_titles list (1–3 titles)
  • Drop or duplicate cases so you end up with ~15 across role types

Then run:
  python scripts/evaluate.py

Cost note: this script calls the LLM once per CV for fact extraction. With
~15 CVs and Groq it takes 2–5 min and consumes ~15 LLM calls.

Usage:
  python scripts/build_eval_set.py
  python scripts/build_eval_set.py --pdf-dir docs/good_cvs --count 15
  python scripts/build_eval_set.py --output tests/eval_set.json --force
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.parser import extract_facts, parse_document_bytes

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cvonrag.build_eval_set")


# Role-type rotation + matching template JDs the user will hand-edit.
# Keep these short and obvious so the placeholder is clearly a placeholder.
ROLE_ROTATION = [
    "data_science",
    "ml_engineering",
    "software_engineering",
    "quant_finance",
    "product_management",
]

JD_TEMPLATES = {
    "data_science": (
        "[EDIT ME] We are hiring a Data Scientist to build predictive models for "
        "customer retention. Stack: Python, scikit-learn, SQL. Strong statistics "
        "and experimentation background required. Communicate insights to "
        "non-technical stakeholders."
    ),
    "ml_engineering": (
        "[EDIT ME] We are hiring an ML Engineer to productionise deep-learning "
        "models. Stack: PyTorch, FastAPI, Docker, Kubernetes. Experience with "
        "model serving, monitoring, and CI/CD for ML systems."
    ),
    "software_engineering": (
        "[EDIT ME] We are hiring a Backend Engineer for a high-throughput trading "
        "system. Stack: Python, PostgreSQL, Redis, async I/O. Strong systems "
        "fundamentals and a track record shipping production services."
    ),
    "quant_finance": (
        "[EDIT ME] We are hiring a Quantitative Researcher to develop alpha "
        "signals for equity strategies. Strong statistics, time-series, and "
        "Python required. Experience with backtesting and risk modelling."
    ),
    "product_management": (
        "[EDIT ME] We are hiring an Associate Product Manager for a B2B SaaS "
        "analytics product. Strong analytical skills, comfort with SQL, and "
        "experience translating customer problems into roadmap priorities."
    ),
}


async def build_case(cv_path: Path, case_idx: int) -> dict:
    """Parse one CV, return a skeleton case dict ready for human editing."""
    file_bytes = cv_path.read_bytes()
    raw_projects = parse_document_bytes(file_bytes, cv_path.name)
    if not raw_projects:
        raise RuntimeError(f"No projects extracted from {cv_path.name}")

    titles: list[str] = []
    for rp in raw_projects:
        try:
            facts = await extract_facts(rp)
            if facts:
                titles.append(rp.title)
        except Exception as exc:
            logger.warning("extract_facts failed for '%s' in %s: %s", rp.title, cv_path.name, exc)

    if not titles:
        raise RuntimeError(f"No project produced any facts for {cv_path.name}")

    role = ROLE_ROTATION[case_idx % len(ROLE_ROTATION)]
    return {
        "case_id":                 f"case-{case_idx:02d}-{role}",
        "cv_path":                 str(cv_path.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
        "jd_text":                 JD_TEMPLATES[role],
        "target_role_type":        role,
        "target_char_limit":       130,
        "max_bullets":             4,
        "correct_projects":        [],
        "_parsed_project_titles":  titles,
    }


def _positive_int(value: str) -> int:
    """argparse type for "must be a positive integer" CLI flags."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {n})")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tests/eval_set.json from PDFs.")
    p.add_argument("--pdf-dir", type=Path, default=_PROJECT_ROOT / "docs" / "good_cvs",
                   help="Directory of CV PDFs to draw cases from.")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "tests" / "eval_set.json",
                   help="Where to write the eval set.")
    p.add_argument("--count", type=_positive_int, default=15,
                   help="How many cases to generate (will pick the first N PDFs alphabetically).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file without prompting.")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    if not args.pdf_dir.exists():
        print(f"PDF dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    # cv_path.relative_to(_PROJECT_ROOT) inside build_case() will ValueError
    # if the PDF lives outside the repo. Without this up-front check that
    # raise gets caught by the per-PDF try/except and every PDF turns into a
    # SKIP — the script writes an empty eval set and exits 0, which silently
    # produces useless output. Fail fast with a clear message instead.
    try:
        args.pdf_dir.resolve().relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        print(
            f"--pdf-dir must be inside repo root ({_PROJECT_ROOT}); got {args.pdf_dir}",
            file=sys.stderr,
        )
        return 2

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.pdf_dir}", file=sys.stderr)
        return 2

    pdfs = pdfs[:args.count]
    print(f"Building {len(pdfs)} cases from {args.pdf_dir} …", flush=True)

    if args.output.exists() and not args.force:
        print(f"{args.output} exists. Pass --force to overwrite.", file=sys.stderr)
        return 2

    cases: list[dict] = []
    header = {
        "_comment": (
            "Auto-generated by scripts/build_eval_set.py. "
            "EDIT before benchmarking: replace JDs marked [EDIT ME], fill correct_projects "
            "from _parsed_project_titles, drop any cases you don't want. "
            "NOTE: CVs from docs/good_cvs/ are also seeded into Qdrant — Phase 3 retrieval "
            "scores have data leakage. Use held-out CVs for an honest Phase 3 number."
        )
    }
    cases.append(header)

    for i, pdf in enumerate(pdfs):
        print(f"  [{i+1}/{len(pdfs)}] {pdf.name} … ", end="", flush=True)
        try:
            case = await build_case(pdf, i)
            cases.append(case)
            print(f"{len(case['_parsed_project_titles'])} projects")
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output} with {len(cases) - 1} cases.")
    print("Next: hand-edit JDs and correct_projects, then run scripts/evaluate.py.")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
