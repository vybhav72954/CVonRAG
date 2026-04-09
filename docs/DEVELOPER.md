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

As of the `spike/groq` branch, you have two options:

### Option A: Groq API (recommended)

Set `GROQ_API_KEY` in `.env`. All LLM calls route through Groq's free API (Llama 3.3 70B at ~500 tok/sec). Ollama is still needed for embeddings only (nomic-embed-text is tiny, runs fine on any hardware).

**Pros:** Free, fast (~2-3 sec/bullet), 70B model quality, no GPU needed for LLM.
**Cons:** Requires internet, rate-limited (30 req/min on free tier — fine for 60 batchmates used one-at-a-time).

### Option B: Ollama local (fallback)

Leave `GROQ_API_KEY` blank. All LLM calls go through local Ollama. Pick a model that fits your RAM.

**Pros:** Fully offline, no API dependency.
**Cons:** On your GTX 1650 (4GB VRAM), only qwen2.5:3b runs on GPU. Anything larger falls back to CPU (~60 sec/bullet).

**Bottom line:** Use Groq. Switch to Ollama only if you need to work offline.

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

Do these once. After this, every session is just 3 start commands (see Section 5).

### 4A. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cvonrag.git
cd cvonrag
git checkout spike/groq          # use the Groq-enabled branch

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
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4
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

You should see `"status": "ok"`, `"qdrant_connected": true`. If using Groq, `"ollama_ok"` may be false for the LLM model — that's fine, only nomic-embed-text matters.

### 4G. Seed Qdrant with Gold Standard CVs

```bash
# Preview what will be extracted (no writes):
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run

# Actually seed (--skip_tag uses heuristic tagging, no LLM needed):
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag

# Verify:
curl http://localhost:8000/health | python -m json.tool
# → "vector_count": 288  (or however many bullets you have)
```

To re-tag bullets with proper LLM-powered role types later (once Groq is configured):
```bash
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs
# This time without --skip_tag, the LLM tags each bullet with role_type
```

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
cd cvonrag
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 4 — Frontend
cd cvonrag/frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

---

## 6. Testing

All 243 tests are fully mocked — no live Ollama, Qdrant, or Groq needed.

```bash
# Run everything:
pytest

# Useful variants:
pytest -x                                   # stop on first failure
pytest tests/test_parser.py -v              # one file, verbose
pytest tests/ -k "test_score"               # filter by name
pytest --cov=app --cov-report=term-missing  # with coverage
```

Expected output:
```
tests/test_models.py             35 passed
tests/test_chains.py             16 passed
tests/test_integration.py        14 passed
tests/test_api.py                18 passed
tests/test_char_limit_stress.py  40 passed
tests/test_parser.py             36 passed
tests/test_recommender.py        23 passed
                                 ────────
                                 243 passed
```

### What tests cover

| File | What it tests |
|------|---------------|
| `test_models.py` | Pydantic schema validation, field constraints, edge cases |
| `test_chains.py` | `_clean_bullet`, `_strip_json_fences`, metric preservation, tone inference |
| `test_integration.py` | Full pipeline: JD analysis → scoring → retrieval → generation |
| `test_api.py` | All FastAPI endpoints, SSE format, file upload validation |
| `test_char_limit_stress.py` | Char-limit loop convergence, Unicode counting, boundary conditions |
| `test_parser.py` | PDF/DOCX extraction, LLM fact extraction, streaming pipeline |
| `test_recommender.py` | Project scoring (mean-of-top-3), recommendation ranking, reason generation |

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

### LLM routing (spike/groq branch)

All LLM calls go through `_ollama_chat()` and `_ollama_stream()` in `chains.py`. These check `settings.groq_api_key`:

- **Key set:** Routes to `_groq_chat()` / `_groq_stream()` which hit Groq's OpenAI-compatible API.
- **Key empty:** Falls back to `_ollama_chat_inner()` / `_ollama_stream_inner()` which hit local Ollama.

Embeddings always go through Ollama (nomic-embed-text in `vector_store.py`).

`parser.py` also uses `_ollama_chat()` from chains (deferred import to avoid circular deps), so CV parsing also routes through Groq when configured.

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

### Groq (recommended)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(empty)* | Set to enable Groq. Get at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |

### Ollama (fallback for LLM, always used for embeddings)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `qwen2.5:14b` | Only used when GROQ_API_KEY is empty |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Always used for embeddings |

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
| `CHAR_TOLERANCE` | `2` | Acceptable chars from target (±) |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `LLM_MAX_TOKENS` | `512` | Max output tokens |

### App

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment name |
| `CORS_ORIGINS` | `["*"]` | Restrict before deploying |
| `PORT` | `8000` | Backend port |

---

## 12. Troubleshooting

### "httpx.ReadTimeout" or very slow generation

You're hitting local Ollama with a model too large for your hardware. Either set `GROQ_API_KEY` in `.env` (recommended) or switch to a smaller model: `OLLAMA_LLM_MODEL=qwen2.5:3b`.

### "JD analysis timed out" in the browser

The `/recommend` endpoint has a 90-second browser timeout. If using local Ollama with a large model, this is expected. Switch to Groq.

### Health check shows "ollama_ok: false"

If using Groq, this only matters for embedding. Make sure `ollama serve` is running and `nomic-embed-text` is pulled. The LLM model doesn't need to be pulled when using Groq.

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

### Tests fail after Groq swap

Make sure you're on the `spike/groq` branch. The test mocks were updated to match the new `_ollama_chat` routing in `parser.py`. Run `pytest -x` to see the first failure.

---

## 13. Quick reference cheatsheet

```bash
# ── Every session (4 terminals) ──────────────────────────────
docker start qdrant               # Terminal 1
ollama serve                      # Terminal 2
uvicorn app.main:app --reload     # Terminal 3 (venv activated)
cd frontend && npm run dev        # Terminal 4

# ── One-time admin ───────────────────────────────────────────
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --skip_tag

# ── Testing ──────────────────────────────────────────────────
pytest                            # all 243 tests
pytest -x                         # stop on first failure
pytest --cov=app                  # with coverage

# ── Health check ─────────────────────────────────────────────
curl http://localhost:8000/health | python -m json.tool

# ── Useful URLs ──────────────────────────────────────────────
# App:      http://localhost:5173
# API docs: http://localhost:8000/docs
# Qdrant:   http://localhost:6333/dashboard
```
