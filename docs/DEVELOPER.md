# CVonRAG — Developer Guide

Complete reference for setup, running, testing, architecture, and troubleshooting.

---

## Table of Contents

1. [System overview](#1-system-overview)
2. [Choose your inference path](#2-choose-your-inference-path)
3. [Prerequisites](#3-prerequisites)
4. [Setup — step by step](#4-setup--step-by-step)
5. [Running the app (every session)](#5-running-the-app-every-session)
6. [Testing](#6-testing)
7. [User journey — what batchmates see](#7-user-journey--what-batchmates-see)
8. [Architecture deep dive](#8-architecture-deep-dive)
9. [Project structure](#9-project-structure)
10. [API endpoints](#10-api-endpoints)
11. [Environment variables](#11-environment-variables)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick reference cheatsheet](#13-quick-reference-cheatsheet)

---

## 1. System overview

CVonRAG has two roles:

**Admin (you, Vybhav):** Clone repo, seed Qdrant with Gold Standard CVs, run the server. One-time setup plus a few start commands each session.

**Users (batchmates):** Upload their biodata, paste a JD, get polished bullets. They never touch Qdrant, Ollama, or any backend config.

The Gold Standard CVs in Qdrant provide style only (sentence structure, separator patterns, verb placement). A user's metrics, tools, and outcomes are never drawn from these CVs. Ever.

```
Admin (you) → ~/good_cvs/*.pdf → scripts/ingest_pdfs.py → Qdrant (style exemplars)
                                                                │
User → biodata.docx → POST /parse → projects + CoreFacts        │ retrieved at query time
User → JD text      → POST /recommend → ranked projects         │
User → confirm      → POST /optimize → SSE bullets ←────────────┘
                                         │
                                         ▼
                              Browser: bullets grouped by project
                              char badge, copy button, Copy All
```

---

## 2. Choose your inference path

The Groq + Ollama hybrid is on `master`. You have two options for the LLM:

### Option A: Groq API (recommended)

Set `GROQ_API_KEY` in `.env`. All LLM calls route through Groq's free API (Llama 3.3 70B at ~500 tok/sec). Ollama is still needed for embeddings only (`nomic-embed-text` is tiny, runs fine on any hardware).

**Pros:** Free, fast (~2-3 sec/bullet), 70B model quality, no GPU needed for LLM.
**Cons:** Requires internet, rate-limited (Groq free tier: 30 req/min). The backend caps `total_bullets_requested` at `GROQ_MAX_BULLETS_PER_REQUEST` (default 15) so a single `/optimize` call stays inside that quota even with a 4-iteration correction loop.

### Option B: Ollama local (fallback)

Leave `GROQ_API_KEY` blank. All LLM calls go through local Ollama. Pick a model that fits your RAM.

**Pros:** Fully offline, no API dependency.
**Cons:** On a GTX 1650 (4GB VRAM), only `qwen2.5:3b` runs on GPU. Anything larger falls back to CPU (~60 sec/bullet).

**Bottom line:** Use Groq. Switch to Ollama only if you need to work offline.

> **Already set up?** Skip ahead to [Section 5](#5-running-the-app-every-session). Section 4 is one-time only — re-running it (especially the seeding step, 4G) can corrupt your Qdrant collection. See [Section 12 — Troubleshooting](#12-troubleshooting) for safe re-seeding.

---

## 3. Prerequisites

| Tool | Min version | Install | Verify |
|------|-------------|---------|--------|
| Python | 3.12 | [python.org](https://python.org) | `python --version` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| Docker | 24+ | [docker.com](https://docker.com) | `docker --version` |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) | `ollama --version` |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) | `node --version` |

---

## 4. Setup — step by step

Do these once. After this, every session is just 4 start commands (see Section 5).

> **Have you done this before?** Run `curl http://localhost:8000/health` after starting the backend (Section 5). If you see `"vector_count": 288` (or whatever you seeded), **skip 4G** — re-running the seeder appends new vectors and creates duplicates (see Section 12).

### 4A. Clone and install

```bash
git clone https://github.com/vybhav72954/CVonRAG.git
cd CVonRAG
# `master` is the working branch (Groq + Ollama hybrid). No need to switch.

uv venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

uv pip install -e ".[dev]"
cp .env.example .env

# Verify imports:
python -c "import fastapi, qdrant_client, httpx; print('OK')"
```

### 4B. Configure your LLM path

**If using Groq (recommended):**

1. Go to [console.groq.com](https://console.groq.com)
2. Create a project (or use existing)
3. Go to API Keys → Create API Key
4. Copy the key (starts with `gsk_...`)
5. Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile   # default, no need to change
```

**If using Ollama local:**

Leave `GROQ_API_KEY` blank in `.env` and pull an LLM model:

```bash
ollama pull qwen2.5:3b     # fits 4GB VRAM
# OR: ollama pull qwen2.5:14b  # needs 12GB+ free RAM
```

Then set in `.env`:
```env
OLLAMA_LLM_MODEL=qwen2.5:3b
```

### 4C. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4
```

Verify: `curl http://localhost:6333/healthz`

### 4D. Pull embedding model

Always needed regardless of Groq/Ollama choice:

```bash
ollama serve              # skip if already running as daemon
ollama pull nomic-embed-text
```

### 4E. Start the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4F. Verify health

```bash
curl http://localhost:8000/health | python -m json.tool
```

The response distinguishes three independent backends:
- `"qdrant_connected": true` — Qdrant is reachable
- `"embed_ok": true` — Ollama has `nomic-embed-text` pulled (always required, even with Groq)
- `"groq_ok": true` — Groq API reachable (only when `GROQ_API_KEY` is set)
- `"ollama_ok": true` — Ollama LLM model is pulled (only relevant when `GROQ_API_KEY` is empty)

`"status": "ok"` requires *the LLM you're using* + `embed_ok` + `qdrant_connected` to all be true. So with Groq configured, `ollama_ok` being false is expected and doesn't degrade status. With Groq blank, `groq_ok` being false is expected.

`"vector_count"` shows how many style exemplars are seeded — `0` means you haven't run 4G yet.

### 4G. Seed Qdrant with Gold Standard CVs

> **Run this exactly once.** `scripts/ingest_pdfs.py` mints a fresh UUID per bullet, so re-running it appends a second copy of every bullet — your retrieval will return duplicate exemplars to the LLM. If `vector_count` is already `> 0`, **stop and read [Section 12 — Troubleshooting](#12-troubleshooting) → "I want to re-seed Qdrant"**.

```bash
# Preview what will be extracted (no writes — safe to re-run):
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run

# Actually seed (--skip_tag uses heuristic tagging, no LLM needed):
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag

# Verify:
curl http://localhost:8000/health | python -m json.tool
# → "vector_count": 288  (or however many bullets you have)
```

The script POSTs in batches (default 50 per call). The backend caps any single `/ingest` request at `_MAX_BULLETS_PER_INGEST = 100` — so the seeder works fine but a hand-crafted call larger than 100 bullets will 422.

**To upgrade tagging later (once Groq is configured):**
The first run with `--skip_tag` gives every bullet `role_type=general`. To get proper per-bullet `role_type` (data_science, ml_engineering, etc.), you must wipe the collection first and re-seed without `--skip_tag` — see Section 12.

**Protecting `/ingest` if exposed publicly:** set `INGEST_SECRET` in `.env` and pass it via `--ingest_secret` or the `X-Ingest-Secret` header. Constant-time comparison is enforced server-side.

### 4H. Install and start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 5. Running the app (every session)

After one-time setup, each coding/demo session needs just these commands. Run each in a separate terminal:

```bash
# Terminal 1 — Qdrant (may already be running)
docker start qdrant

# Terminal 2 — Ollama (needed for embeddings, even with Groq)
ollama serve

# Terminal 3 — Backend
cd CVonRAG
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Terminal 4 — Frontend
cd CVonRAG/frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

**You do NOT need to re-run on a regular session:**
- `uv pip install` (deps already installed)
- `ollama pull nomic-embed-text` (model is on disk)
- `python scripts/ingest_pdfs.py` (Qdrant volume `qdrant_storage` persists across `docker stop`/`start` — re-running will *duplicate* every bullet)
- `cp .env.example .env` (would overwrite your secrets)
- `npm install` (only after a `git pull` that changed `package.json`)

If `curl http://localhost:8000/health` shows the expected `vector_count` and `status: ok`, you're good.

---

## 6. Testing

The full suite (~318 tests across 8 files) is fully mocked — no live Ollama, Qdrant, or Groq needed. Test counts grow with every audit, so verify the live count rather than trusting this number:

```bash
pytest --collect-only -q | tail -1     # → "318 tests collected" (or whatever today's count is)
```

```bash
# Run everything:
pytest

# Useful variants:
pytest -x                                   # stop on first failure
pytest tests/test_parser.py -v              # one file, verbose
pytest tests/ -k "test_score"               # filter by name
pytest --cov=app --cov-report=term-missing  # with coverage
```

`tests/conftest.py` autouses a fixture that resets `INGEST_SECRET` to empty and disables rate limiting for the duration of each test. Tests that exercise auth or rate-limit explicitly opt in via `patch.object(settings, ...)`.

### What tests cover

| File | What it tests |
|------|---------------|
| `test_models.py` | Pydantic schema validation, field constraints, edge cases |
| `test_chains.py` | LLM helpers, `_clean_bullet`, `_strip_json_fences`, fact-scoring, char-limit loop, Groq retry |
| `test_integration.py` | Full pipeline: JD analysis → scoring → retrieval → generation |
| `test_api.py` | All FastAPI endpoints, SSE format, file upload validation, rate limit, ingest auth |
| `test_char_limit_stress.py` | Char-limit loop convergence, Unicode counting, boundary conditions |
| `test_parser.py` | PDF/DOCX extraction, LLM fact extraction, streaming pipeline |
| `test_recommender.py` | Project scoring (mean-of-top-3), recommendation ranking, reason generation |
| `test_vector_store.py` | Qdrant retrieval, embedding retry/backoff, `role_type` payload coercion |

---

## 7. User journey — what batchmates see

Users open http://localhost:5173 and go through a 3-screen wizard:

**Screen 1 — Upload CV:** Drag-and-drop .docx or .pdf. Projects are parsed and displayed with tool/metric chips as they stream in. An optional "edit facts" toggle lets users fix anything the parser missed.

**Screen 2 — Paste JD:** Paste the full job description. Optionally adjust role type, target char count (default 130), bullets per project (default 2), and projects to recommend (default 3). Click "Analyse JD" — the AI ranks all projects by relevance and pre-selects the best ones with match scores and reasons.

**Screen 3 — Review bullets:** Bullets generate with a typewriter animation, then appear as cards grouped by project. Each card shows char count, tolerance badge (green = within ±2), iterations taken, and a copy button. "Copy all" grabs everything formatted by project.

---

## 8. Architecture deep dive

### The 5-phase pipeline

Every call to `POST /optimize` runs:

```
Phase 1: Pydantic validation (OptimizationRequest)
Phase 2: SemanticMatcher — JD analysis + fact scoring (0–1 per CoreFact)
Phase 3: Qdrant RAG retrieval — embed(JD + top facts) → top-5 StyleExemplars
Phase 4: BulletAlchemist — generate bullet + ±2 char-limit correction loop
Phase 5: CVonRAGOrchestrator — stream tokens via SSE, yield final bullets
```

### LLM routing

All LLM calls go through `_ollama_chat()` and `_ollama_stream()` in `chains.py`. These check `settings.groq_api_key` (the `_using_groq()` helper):

- **Key set:** Routes to `_groq_chat()` / `_groq_stream()` which hit Groq's OpenAI-compatible API. 429s honour `Retry-After` (3 attempts).
- **Key empty:** Falls back to `_ollama_chat_inner()` / `_ollama_stream_inner()` which hit local Ollama.

Embeddings always go through Ollama (`nomic-embed-text` in `vector_store.py`) — Groq doesn't expose a 768-dim-compatible embedding endpoint.

`parser.py` and `recommender.py` both reuse `_ollama_chat()` from `chains.py` (deferred import in `parser.py` to avoid circular deps), so CV parsing and recommendation reasons also route through Groq when configured.

### The content/style firewall

The most important architectural constraint. Prevents hallucinated metrics.

**CONTENT (from user's CoreFacts only):** Every number, tool, algorithm, outcome — preserved exactly.
**STYLE (from Qdrant exemplars only):** Separators (| ; ↑ ↓ →), sentence architecture (VERB → TOOL → METRIC → IMPACT), abbreviation style.

The Alchemist prompt explicitly forbids copying content from exemplars or altering any number.

### The char-limit loop

Each bullet iterates up to `CHAR_LOOP_MAX_ITERATIONS` (default 4) times:
1. Generate initial draft
2. If within `target ± tolerance` → done
3. If too long → compress prompt (use & instead of "and", ↑ instead of "increased", etc.)
4. If too short → expand prompt (add omitted tools/metrics from facts)
5. Return closest draft if loop exhausts

With Groq (70B): ~90%+ of bullets converge within 2 iterations.

### Project recommendation engine

`POST /recommend` scores each project by the mean of its top-3 fact relevance scores (rewards depth without penalising projects with weak side-bullets). An LLM generates one-sentence reasons naming specific matched skills.

---

## 9. Project structure

```
cvonrag/
├── app/
│   ├── config.py          ← Settings (Groq + Ollama + Qdrant + app config)
│   ├── models.py          ← All Pydantic v2 schemas
│   ├── chains.py          ← 5-phase pipeline + Groq/Ollama routing
│   ├── vector_store.py    ← Qdrant + nomic-embed-text embeddings
│   ├── parser.py          ← .docx/.pdf → CoreFacts via LLM (SSE)
│   ├── recommender.py     ← Project scoring + ranking
│   └── main.py            ← FastAPI app: /parse, /recommend, /optimize, /ingest, /health
│
├── tests/                 ← 243 tests, fully mocked
├── frontend/              ← SvelteKit 3-screen wizard
│   ├── src/lib/api.js     ← fetch + SSE stream handling (with error recovery)
│   ├── src/lib/stores.js  ← Svelte stores (wizard state machine)
│   └── src/routes/+page.svelte  ← The UI (602 lines)
│
├── scripts/
│   ├── ingest_pdfs.py     ← Admin: seed Qdrant from PDFs
│   └── parse_biodata.py   ← Terminal: parse .docx → request.json
│
├── docs/DEVELOPER.md      ← This file
├── kaggle_pipeline_test.py ← Standalone pipeline for Kaggle H100
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md
```

---

## 10. API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — Qdrant + Ollama status |
| `POST` | `/parse` | Upload .docx/.pdf → SSE stream of projects + facts |
| `POST` | `/recommend` | Score all projects vs JD → ranked recommendations |
| `POST` | `/optimize` | Generate bullets → SSE stream of tokens + bullets |
| `POST` | `/ingest` | Admin: seed Qdrant with Gold Standard bullets |

Interactive docs: http://localhost:8000/docs

---

## 11. Environment variables

`.env.example` is the canonical reference. The tables below list the most-touched settings; everything else lives in `app/config.py`.

### Groq (recommended)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(empty)* | Set to enable Groq. Get at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Override only if proxying |
| `GROQ_MAX_BULLETS_PER_REQUEST` | `15` | Hard cap per `/optimize` so a single request can't blow the 30 req/min free-tier quota (15 bullets × 4 iterations ≈ 62 calls) |

### Ollama (LLM fallback, always used for embeddings)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `qwen2.5:3b` | Only used when `GROQ_API_KEY` is empty. `3b` fits 4 GB VRAM. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Always used for embeddings — don't change unless the new model is also 768-dim |

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | *(none)* | Only for Qdrant Cloud |
| `QDRANT_COLLECTION` | `gold_standard_cvs` | Collection name |
| `QDRANT_VECTOR_SIZE` | `768` | Must match embedding model dim |

### Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `5` | Number of style exemplars to retrieve |
| `CHAR_LOOP_MAX_ITERATIONS` | `4` | Max correction loop iterations |
| `CHAR_TOLERANCE` | `2` | Acceptable chars from target (±) — don't lower below 2 |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature — keep ≤ 0.5 |
| `LLM_MAX_TOKENS` | `512` | Max output tokens |
| `BULLET_STREAM_CHUNK_DELAY` | `0.025` | Inter-word delay (s) for the typewriter. Set `0` in CI/tests. |

### Rate limiting & admin

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Disable for local dev / load tests |
| `RATE_LIMIT_WINDOW` | `60` | Window in seconds. **All routes must share one window** (enforced) |
| `RATE_LIMIT_PARSE` | `10` | `/parse` calls per IP per window |
| `RATE_LIMIT_RECOMMEND` | `20` | `/recommend` calls per IP per window |
| `RATE_LIMIT_OPTIMIZE` | `5` | `/optimize` calls per IP per window |
| `INGEST_SECRET` | *(empty)* | Required `X-Ingest-Secret` header for `/ingest` when set. Constant-time comparison. **Set this before exposing the API publicly.** |

### App

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Set to `production` on deploy |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | JSON-encoded list. **Replace with your real frontend URL before deploying** — a warning fires at startup if you leave the dev default in production |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose traces |
| `PORT` | `8000` | Backend port |

---

## 12. Troubleshooting

### I want to re-seed Qdrant (or I think it's already seeded)

`scripts/ingest_pdfs.py` is **not idempotent** — every run mints fresh UUIDs and *appends*, so re-running with the same PDFs gives you 2× the bullets. Symptoms: retrieval returns near-duplicate exemplars, `vector_count` jumps to a multiple of 288, or the LLM starts copying-pasting the same phrasing patterns.

**Check first** — is the collection already seeded?
```bash
curl http://localhost:8000/health | python -m json.tool
# "vector_count": 288 → already seeded, do nothing
# "vector_count": 576 → seeded twice, see "wipe and re-seed" below
# "vector_count": 0   → run the seeder once (Section 4G)
```

**Wipe and re-seed** (the only safe way to re-run):
```bash
# 1. Drop the collection (irreversible — you'll have to re-seed)
curl -X DELETE http://localhost:6333/collections/gold_standard_cvs

# 2. Verify it's gone
curl http://localhost:8000/health | python -m json.tool
# → "collection_exists": false, "vector_count": 0

# 3. Seed cleanly. The collection is auto-recreated on first ingest.
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag
```

If you want LLM-tagged `role_type` instead of all-`general`, drop `--skip_tag` (Groq must be configured; takes ~10 min for 288 bullets at the seeding script's 2.2 s/req throttle).

### Health check shows "groq_ok: false"

You set `GROQ_API_KEY` but the key is wrong, expired, or your network can't reach `api.groq.com`. Check the key at console.groq.com and try `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`.

### "httpx.ReadTimeout" or very slow generation

You're hitting local Ollama with a model too large for your hardware. Either set `GROQ_API_KEY` in `.env` (recommended) or switch to a smaller model: `OLLAMA_LLM_MODEL=qwen2.5:3b`.

### "JD analysis timed out" in the browser

The `/recommend` endpoint has a 90-second browser timeout. If using local Ollama with a large model, this is expected. Switch to Groq.

### Health check shows "ollama_ok: false"

If using Groq, this only flags the *Ollama LLM model* — and Groq is handling LLM, so this field is irrelevant when `groq_ok: true`. The one that *does* matter regardless of LLM choice is `embed_ok` (Ollama must have `nomic-embed-text` pulled for embeddings). If `embed_ok` is false: `ollama pull nomic-embed-text` and ensure `ollama serve` is running.

### "collection_exists: false" or "vector_count: 0"

Run the seeding script: `python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag`

### PDF shows 0 bullets

PGDBA CVs use private-use Unicode bullet markers (\uf0a7 Wingdings glyph) that `pdfplumber.extract_words()` silently discards. The ingestion script uses `page.chars` to detect them. If a specific PDF still shows 0 bullets, check its bullet character with: `python scripts/ingest_pdfs.py --debug_pdf path/to/file.pdf`

### Qdrant container won't start

```bash
docker ps -a | grep qdrant          # check if it exists but stopped
docker start qdrant                  # restart it
# OR nuke and recreate:
docker rm qdrant && docker volume rm qdrant_storage
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4
```

### Frontend can't reach backend

Check that `VITE_API_URL` in `frontend/.env` matches your backend port (default: `http://localhost:8000`). Check browser console for CORS errors — if deploying, set `CORS_ORIGINS` in `.env`.

### 429 "Rate limit exceeded" from `/parse`, `/recommend`, or `/optimize`

The backend's per-IP sliding-window limiter kicked in. Wait the seconds shown in the response, or set `RATE_LIMIT_ENABLED=false` in `.env` for local dev. Defaults: `/parse` 10/min, `/recommend` 20/min, `/optimize` 5/min.

### 422 "total_bullets_requested exceeds the server cap"

You're using Groq and asked for more bullets than `GROQ_MAX_BULLETS_PER_REQUEST` (default 15). Either reduce `max_bullets_per_project` × number of selected projects, or raise the cap in `.env` (only safe if you're on a paid Groq plan).

### 422 from `/ingest` with "Input should be 'general' | ..."

Your bullet's `role_type` doesn't match the `RoleType` enum. The seeding script handles this; if you're calling `/ingest` by hand, use one of: `software_engineering`, `data_science`, `ml_engineering`, `product_management`, `quant_finance`, `general`.

### 403 "Invalid or missing X-Ingest-Secret header"

You set `INGEST_SECRET` in `.env`. The seeding script reads it from `.env` automatically; for hand-rolled calls, add `-H "X-Ingest-Secret: <your-secret>"`.

### Tests fail

Run `pytest -x` to see the first failure. If a test references a setting your local `.env` overrides, the autouse fixture in `tests/conftest.py` should be resetting it — check that the test isn't bypassing the fixture with its own `patch.object`. You should be on `master` (or a feature branch off it).

---

## 13. Quick reference cheatsheet

```bash
# ── Every session (4 terminals) ──────────────────────────────
docker start qdrant               # Terminal 1
ollama serve                      # Terminal 2
uvicorn app.main:app --reload     # Terminal 3 (venv activated)
cd frontend && npm run dev        # Terminal 4

# ── One-time admin (DO NOT re-run on a primed DB!) ───────────
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag

# ── Re-seeding (only when you actually need to) ──────────────
curl -X DELETE http://localhost:6333/collections/gold_standard_cvs
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag

# ── Testing ──────────────────────────────────────────────────
pytest                            # full suite (~318 tests)
pytest -x                         # stop on first failure
pytest --cov=app                  # with coverage
pytest --collect-only -q | tail -1   # verify live test count

# ── Health check ─────────────────────────────────────────────
curl http://localhost:8000/health | python -m json.tool

# ── Useful URLs ──────────────────────────────────────────────
# App:      http://localhost:5173
# API docs: http://localhost:8000/docs
# Qdrant:   http://localhost:6333/dashboard
```


```
# 1. Validate extraction quality on one PDF first (no writes):
python scripts/ingest_pdfs.py --debug_pdf "Soham Chatterjee.pdf"

# 2. Dry-run across all PDFs (still no writes):
python scripts/ingest_pdfs.py --pdf_dir "Z:/PGDBA Content/Competition/CVonRAG/docs/good_cvs" --dry_run

# 3. Re-seed with Groq tagging for proper role_type filtering:
python scripts/ingest_pdfs.py --pdf_dir "Z:/PGDBA Content/Competition/CVonRAG/docs/good_cvs"

# The script upserts (new UUIDs each run), so you'll want to drop the collection first if you don't want stale
# duplicates:
curl -X DELETE http://localhost:6333/collections/gold_standard_cvs

Then /ingest recreates it fresh on the first call.
```