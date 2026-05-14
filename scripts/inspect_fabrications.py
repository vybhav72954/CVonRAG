#!/usr/bin/env python3
"""
scripts/inspect_fabrications.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Diagnostic helper: print every bullet in eval_results.json that has at
least one fabricated numeric (an output token that has no exact match in
the source CoreFacts passed to the alchemist).

Usage:
  python scripts/inspect_fabrications.py
  python scripts/inspect_fabrications.py --backend ollama
  python scripts/inspect_fabrications.py --file other_results.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, default=_PROJECT_ROOT / "eval_results.json")
    p.add_argument("--backend", default="groq",
                   help="Which backend's results to inspect (default: groq).")
    args = p.parse_args()

    if not args.file.exists():
        print(f"Not found: {args.file}", file=sys.stderr)
        print("Run `scripts/evaluate.py` first to produce eval_results.json.", file=sys.stderr)
        return 2

    data = json.loads(args.file.read_text(encoding="utf-8"))
    backend_data = data.get("by_backend", {}).get(args.backend, [])
    if not backend_data:
        print(f"No results for backend={args.backend!r} in {args.file}", file=sys.stderr)
        print(f"Available backends: {list(data.get('by_backend', {}).keys())}", file=sys.stderr)
        return 2

    found = 0
    for case in backend_data:
        for bullet, fp in zip(case.get("bullets", []), case.get("fact_preservation", [])):
            if not fp.get("fabricated"):
                continue
            found += 1
            print("=" * 64)
            print(f"CASE:        {case['case_id']}  (bullet #{fp['bullet_index']})")
            print(f"FABRICATED:  {fp['fabricated']}")
            print(f"TEXT:        {bullet['text']}")
            print(f"SOURCE:      {fp['source_numerics']}")
            print(f"OUTPUT:      {fp['output_numerics']}")
            print(f"CHARS:       {bullet['char_count']} (target {bullet['char_target']}, iter {bullet['iterations']})")
    print("=" * 64)
    print(f"Total fabrications: {found} across {sum(len(c['bullets']) for c in backend_data)} bullets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
