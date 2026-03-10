# CVonRAG — Developer Guide
### From zero to deployed, step by step.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Backend Setup with uv](#3-backend-setup-with-uv)
4. [Environment Variables](#4-environment-variables)
5. [Running Locally (without Docker)](#5-running-locally-without-docker)
6. [Running with Docker Compose](#6-running-with-docker-compose)
7. [Seeding the Vector Store](#7-seeding-the-vector-store)
8. [Running the Tests](#8-running-the-tests)
9. [Frontend Setup (SvelteKit)](#9-frontend-setup-sveltekit)
10. [GitHub Setup](#10-github-setup)
11. [Deploying the Backend (Railway)](#11-deploying-the-backend-railway)
12. [Deploying the Frontend (Vercel)](#12-deploying-the-frontend-vercel)
13. [Kaggle H100 Workflow](#13-kaggle-h100-workflow)
14. [pyproject.toml Reference](#14-pyprojecttoml-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

Install these before anything else:

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.12 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker + Docker Compose | latest | [docker.com](https://docker.com) |
| Node.js | ≥ 20 | [nodejs.org](https://nodejs.org) |
| Git | any | pre-installed on most systems |

Verify:
```bash
python --version      # Python 3.12+
uv --version          # uv 0.x.x
docker --version      # Docker 24+
node --version        # v20+
git --version
```

---

## 2. Project Structure

```
CVonRAG/
│
├── app/                        ← FastAPI backend (Python)
│   ├── __init__.py
│   ├── config.py               ← all settings via env vars
│   ├── models.py               ← Pydantic v2 schemas
│   ├── chains.py               ← 5-phase pipeline (the core engine)
│   ├── vector_store.py         ← Qdrant + Ollama embeddings
│   └── main.py                 ← FastAPI app + SSE endpoints
│
├── tests/                      ← pytest test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py          ← Pydantic schema tests
│   ├── test_chains.py          ← pipeline helper unit tests
│   ├── test_integration.py     ← mocked Ollama + Qdrant
│   ├── test_api.py             ← FastAPI endpoint tests
│   └── test_char_limit_stress.py ← char-limit loop edge cases
│
├── frontend/                   ← SvelteKit UI
│   ├── src/
│   │   ├── routes/
│   │   │   ├── +layout.svelte
│   │   │   └── +page.svelte    ← main UI
│   │   ├── lib/
│   │   │   ├── api.js          ← SSE client + fetch helpers
│   │   │   └── stores.js       ← Svelte reactive state
│   │   └── app.css
│   ├── package.json
│   ├── svelte.config.js        ← adapter-node for Railway/Render
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── .env.example
│
├── kaggle_pipeline_test.py     ← H100 smoke test + ingestion notebook
├── pyproject.toml              ← single source of truth for deps + tools
├── Dockerfile                  ← backend container (Render/Railway)
├── docker-compose.yml          ← local stack: app + Ollama + Qdrant
├── .env.example
├── .gitignore
└── DEVELOPER.md                ← this file
```

---

## 3. Backend Setup with uv

`uv` is a fast Rust-based pip/venv replacement. It reads `pyproject.toml` directly.

```bash
# Clone the repo (or unzip the project)
cd CVonRAG

# Create a virtual environment in .venv/
uv venv

# Activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell

# Install all production dependencies
uv pip install -e .

# Install dev + test dependencies
uv pip install -e ".[dev]"

# Verify
python -c "import fastapi, qdrant_client, httpx; print('All imports OK')"
```

> **Why uv over pip?**
> 10–100× faster installs, built-in venv management, reads `pyproject.toml` natively.
> Drop-in replacement — all pip commands work with `uv pip`.

---

## 4. Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit with your actual values (no paid keys needed!)
nano .env    # or code .env / vim .env
```

**Minimum required settings for local dev:**
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_EMBED_MODEL=nomic-embed-text
QDRANT_URL=http://localhost:6333
```

Everything else has safe defaults. You **do not** need API keys for Anthropic or OpenAI.

---

## 5. Running Locally (without Docker)

You'll need Ollama and Qdrant running separately.

### 5a. Start Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4
```

### 5b. Start Ollama and pull models
```bash
# Install Ollama: https://ollama.com/download
ollama serve &             # starts the Ollama daemon

# Pull models (one-time, ~8-10 GB)
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### 5c. Start the FastAPI app
```bash
# With uv-managed venv activated:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# OR directly:
python app/main.py
```

API docs available at: **http://localhost:8000/docs**

---

## 6. Running with Docker Compose

The recommended approach — everything starts in one command.

```bash
# First time (downloads models — ~8-10 GB, takes 10-20 min):
docker compose up --build

# Subsequent runs (models cached in volume):
docker compose up

# Background mode:
docker compose up -d

# View logs:
docker compose logs -f app
docker compose logs -f ollama

# Stop everything:
docker compose down

# Stop AND remove volumes (wipes Qdrant data + Ollama models):
docker compose down -v
```

**Services started:**
| Service | URL | Purpose |
|---|---|---|
| `app` | http://localhost:8000 | FastAPI backend |
| `ollama` | http://localhost:11434 | LLM + embeddings |
| `qdrant` | http://localhost:6333 | Vector store |
| `model-puller` | — | Pulls models once, then exits |

> **Note:** `model-puller` showing `Exited (0)` is **normal** — it pulled the models and stopped.

---

## 7. Seeding the Vector Store

Before generating bullets, you must load Gold Standard CV examples into Qdrant.
These are your style references — the RAG retrieves them to learn sentence structure.

### Option A — via the API (recommended)
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "bullets": [
      {
        "text": "• Enhanced forecast accuracy using ARIMAX and VECM | Reduced RMSE by 13.5% for price prediction",
        "role_type": "data_science",
        "uses_separator": "|",
        "uses_arrow": false,
        "sentence_structure": "verb → method → metric → domain"
      },
      {
        "text": "• Handled class imbalance via undersampling; ↑ F1 score from 0.39 to 0.49",
        "role_type": "ml_engineering",
        "uses_arrow": true,
        "sentence_structure": "verb → technique → metric delta"
      },
      {
        "text": "• Architected Kafka + Spark pipeline | ↓ data latency by 340ms (p99) at 12M daily events",
        "role_type": "software_engineering",
        "uses_separator": "|",
        "uses_arrow": true,
        "sentence_structure": "verb → tool stack → metric → scale"
      }
    ]
  }'
```

### Option B — via the Kaggle notebook
Run `kaggle_pipeline_test.py` on the H100 notebook (see §13).
It seeds Qdrant with 10 high-quality bullets automatically.

### Verify ingestion:
```bash
curl http://localhost:8000/health
# → "vector_count": 3  (or however many you ingested)
```

---

## 8. Running the Tests

```bash
# All tests (no live services needed — everything is mocked)
pytest

# Specific test file
pytest tests/test_models.py -v
pytest tests/test_chains.py -v
pytest tests/test_api.py -v
pytest tests/test_integration.py -v
pytest tests/test_char_limit_stress.py -v

# Run only fast tests (skip any marked slow/requires_ollama)
pytest -m "not slow and not requires_ollama and not requires_qdrant"

# With coverage report
pip install pytest-cov
pytest --cov=app --cov-report=term-missing

# Run a single test by name
pytest tests/test_chains.py::TestCleanBullet::test_strips_qwen_think_block -v
```

**What each test file covers:**
| File | Tests | What it guards |
|---|---|---|
| `test_models.py` | 35 | Pydantic validation, bounds, enum safety |
| `test_chains.py` | 28 | `_clean_bullet`, JSON fences, metric fidelity, `<think>` stripping |
| `test_integration.py` | 12 | Full pipeline with mocked Ollama — JD analysis, scoring, char-limit loop |
| `test_api.py` | 18 | Every endpoint — 422 validation, SSE content-type, JSON data lines |
| `test_char_limit_stress.py` | 40 | Every tolerance boundary, Unicode chars, convergence simulation |

**Expected output:**
```
================================ 133 passed in 4.2s ================================
```

---

## 9. Frontend Setup (SvelteKit)

### Why SvelteKit?
- Simplest full-stack framework — no JSX, no virtual DOM
- File-based routing — one file per page, zero boilerplate
- First-class SSE support via native `fetch()` + `ReadableStream`
- Deploys to Vercel for free (or Railway alongside the backend)
- "Vibe coding" friendly — you read and understand every line

### Local dev:
```bash
cd frontend

# Copy env
cp .env.example .env
# Set PUBLIC_API_URL=http://localhost:8000

# Install dependencies
npm install

# Start dev server (hot reload)
npm run dev
# → http://localhost:5173
```

### Build for production:
```bash
npm run build
# Output in frontend/build/

# Preview the production build
npm run preview
```

### Key files to understand:
| File | What it does |
|---|---|
| `src/routes/+page.svelte` | Entire UI — form, SSE stream display |
| `src/lib/api.js` | `optimizeResume()` — the SSE fetch client |
| `src/lib/stores.js` | Reactive state (`bullets`, `tokenBuffer`, `status`) |
| `svelte.config.js` | Uses `adapter-node` for Railway/Render deployment |

### Connecting to the backend SSE stream:
```javascript
// The core pattern in src/lib/api.js
const resp = await fetch(`${BASE}/optimize`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
// Read the stream frame by frame
const reader = resp.body.getReader();
```

> **Why not EventSource?**
> `EventSource` only supports GET requests. Our `/optimize` endpoint is POST (needs a body).
> We use `fetch()` + `ReadableStream` instead — same streaming, works with POST.

---

## 10. GitHub Setup

```bash
# In the project root:
git init
git add .
git commit -m "feat: initial CVonRAG implementation"

# Create a new repo on github.com (don't initialise with README)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/cvonrag.git
git branch -M main
git push -u origin main
```

**Branch strategy:**
```
main          ← stable, deployed to production
dev           ← active development
feature/*     ← individual features
```

**.env is in .gitignore** — never committed. Set secrets via the platform dashboard.

---

## 11. Deploying the Backend (Railway)

Railway is the easiest cloud deployment for this stack.

### Option A — Railway (recommended, free tier available)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project (or create new)
railway init

# Deploy
railway up
```

**Set environment variables in Railway dashboard:**
```
OLLAMA_BASE_URL    = https://your-ollama-service.railway.app
QDRANT_URL         = https://your-cluster.cloud.qdrant.io   ← Qdrant Cloud free tier
QDRANT_API_KEY     = (from Qdrant Cloud dashboard)
OLLAMA_LLM_MODEL   = qwen2.5:14b
OLLAMA_EMBED_MODEL = nomic-embed-text
APP_ENV            = production
CORS_ORIGINS       = ["https://your-frontend.vercel.app"]
PORT               = 8000
```

> **Important for production:** Ollama needs a GPU VM (Railway GPU workers or a separate VPS).
> For a fully managed free alternative, swap to **Groq API** (see §15 Troubleshooting).

### Option B — Render

1. New Web Service → Connect GitHub repo
2. Runtime: **Docker**
3. Set env vars in Render dashboard (same as above)
4. Render auto-detects the `Dockerfile` and binds to `$PORT`

---

## 12. Deploying the Frontend (Vercel)

```bash
cd frontend

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Set env var in Vercel dashboard:
# PUBLIC_API_URL = https://your-backend.railway.app
```

Or connect via GitHub:
1. vercel.com → New Project → Import from GitHub
2. Root directory: `frontend`
3. Framework: SvelteKit
4. Add environment variable: `PUBLIC_API_URL`

---

## 13. Kaggle H100 Workflow

The Kaggle notebook (`kaggle_pipeline_test.py`) is a **standalone smoke test**.
It does not connect to your local machine.

```
What it does:
  1. Detects your GPU (auto-selects 72B/32B/14B based on VRAM)
  2. Loads Qwen2.5 via transformers (no Ollama daemon needed)
  3. Loads nomic-embed-text for 768-dim embeddings
  4. Creates an in-memory Qdrant instance
  5. Runs the full 5-phase pipeline end-to-end
  6. Exports a CSV of generated bullets to /kaggle/working/
```

**Setup:**
1. kaggle.com → Code → New Notebook
2. Settings → Accelerator → **GPU T4 x2** (or H100 if available)
3. Settings → Internet → **ON** (needed for model download)
4. Upload `kaggle_pipeline_test.py` or paste contents
5. Run All

**For 80 GB H100 (single):**
The notebook auto-detects and loads `Qwen2.5-72B` in **4-bit NF4** quantisation.
- Uses ~40 GB of your 80 GB VRAM
- Nearly identical quality to full bfloat16
- ~120 tokens/second

**First run:** Downloads ~40 GB of weights (cached in session storage after that).

---

## 14. pyproject.toml Reference

`pyproject.toml` is the single source of truth for the entire project.

```toml
# Install production deps:
uv pip install -e .

# Install dev deps (includes pytest, ruff, mypy):
uv pip install -e ".[dev]"

# Run linter:
ruff check app/ tests/

# Run formatter:
ruff format app/ tests/

# Run type checker:
mypy app/

# Run all tests:
pytest
```

**Dependency groups:**
| Group | Command | Contents |
|---|---|---|
| base | `uv pip install -e .` | FastAPI, Qdrant, httpx, Pydantic |
| dev | `uv pip install -e ".[dev]"` | pytest, ruff, mypy |

---

## 15. Troubleshooting

### "Ollama not ready" on /health
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If models aren't pulled yet:
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### "Qdrant connection refused"
```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:v1.12.4

# Or via compose:
docker compose up qdrant
```

### Bullets are too short/long and not converging
Increase `CHAR_LOOP_MAX_ITERATIONS` to 5 in `.env`.
Or use a larger model (`qwen2.5:32b`) for better instruction-following.

### model-puller keeps restarting
This is normal if the models are already downloaded — Docker restarts it briefly then it exits 0.
Set `restart: "no"` in `docker-compose.yml` if it bothers you (already set).

### Swapping to Groq (cloud, free tier) for production
Replace `_ollama_chat` and `_ollama_stream` in `chains.py` with Groq's
OpenAI-compatible endpoint:
```python
# Groq endpoint (OpenAI-compatible)
BASE = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {settings.groq_api_key}"}
MODEL = "qwen-qwq-32b"   # or "llama-3.3-70b-versatile"
```
Groq's free tier: 30 requests/minute, 1M tokens/day — enough for development.

### Tests failing with ImportError
```bash
# Make sure you're in the project root with venv activated
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

### Docker build slow on Apple Silicon
Add `--platform linux/amd64` to Docker build commands, or add to `docker-compose.yml`:
```yaml
services:
  app:
    platform: linux/amd64
```

---

## Quick Reference Card

```bash
# ── Setup (once) ──────────────────────────────────────────────────────────────
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env

# ── Local dev ─────────────────────────────────────────────────────────────────
docker compose up --build          # full stack (first time, downloads models)
docker compose up                  # subsequent runs

# ── Tests (no services needed) ────────────────────────────────────────────────
pytest                             # all 133 tests
pytest tests/test_api.py -v        # just API tests
pytest -x                          # stop on first failure

# ── Frontend ──────────────────────────────────────────────────────────────────
cd frontend && npm install && npm run dev

# ── Deploy ────────────────────────────────────────────────────────────────────
railway up                         # backend
cd frontend && vercel              # frontend

# ── Lint + format ─────────────────────────────────────────────────────────────
ruff check app/ tests/
ruff format app/ tests/
```

---

## 16. The Two Local Scripts

### `scripts/ingest_pdfs.py` — Seed Qdrant (run once)

```bash
# Install extra deps
pip install pdfplumber rich  # already in your venv if you did uv pip install -e ".[dev]"

# Preview what will be extracted (no API call)
python scripts/ingest_pdfs.py \
    --pdf_dir  /path/to/your/good_cvs/ \
    --dry_run

# Full run — extract + tag + seed Qdrant
python scripts/ingest_pdfs.py \
    --pdf_dir  /path/to/your/good_cvs/ \
    --api_url  http://localhost:8000 \
    --ollama   http://localhost:11434 \
    --model    qwen2.5:7b    # 7b is fine for tagging — faster than 14b
```

**Flags:**
| Flag | Default | Description |
|---|---|---|
| `--pdf_dir` | required | Folder with your good CV PDFs |
| `--api_url` | `localhost:8000` | CVonRAG API URL |
| `--ollama` | `localhost:11434` | Ollama URL |
| `--model` | `qwen2.5:7b` | Model for bullet tagging (7b is fast enough) |
| `--batch_size` | 50 | Bullets per `/ingest` call |
| `--dry_run` | off | Print bullets without posting |
| `--skip_tag` | off | Skip LLM tagging, use heuristics instead (fastest) |

**Expected output:**
```
Adwaith_CV.pdf : 17 bullets extracted
Praveen_CV.pdf : 22 bullets extracted
...
Total bullets extracted: 280
Tagging 280 bullets via Ollama (qwen2.5:7b)…
  Batch 1/6: ✓ 50 bullets upserted
  ...
Done. Total upserted: 280 / 280 bullets
```

---

### `scripts/parse_biodata.py` — Parse your docx (run per session)

```bash
# Step 1: See what projects were detected in your docx
python scripts/parse_biodata.py \
    --docx  Vybhav_Chaturvedi_Biodata.docx \
    --list_projects

# Output:
#   [0] 'Decoding Depression Networks (GSE54564)' (25 bullets)
#   [1] 'Music Playlist Generation using Real Time Emotions' (4 bullets)
#   [2] 'Time Series Analysis – Hourly Wages' (28 bullets)
#   [3] 'Automated Image Captioning (Flickr-8k)' (4 bullets)
#   [4] 'Cuckoo.ai' (8 bullets)

# Step 2: Parse selected projects + paste JD interactively → write request.json
python scripts/parse_biodata.py \
    --docx      Vybhav_Chaturvedi_Biodata.docx \
    --projects  2,4             \  # select Time Series + Cuckoo.ai
    --role_type ml_engineering  \
    --char_limit 130            \
    --max_bullets 2             \
    --output    request.json

# Step 3a: POST request.json and stream bullets in terminal
python scripts/parse_biodata.py \
    --docx     Vybhav_Chaturvedi_Biodata.docx \
    --jd_file  job_description.txt \
    --projects 2,4 \
    --stream

# Step 3b: Or use curl
curl -X POST http://localhost:8000/optimize \
     -H 'Content-Type: application/json' \
     -d @request.json \
     --no-buffer
```

**Flags:**
| Flag | Default | Description |
|---|---|---|
| `--docx` | required | Path to your biodata .docx |
| `--jd_file` | prompt | Path to JD text file (or paste interactively) |
| `--output` | `request.json` | Where to write the JSON |
| `--ollama` | `localhost:11434` | Ollama URL |
| `--model` | `qwen2.5:14b` | Model for fact extraction (use 14b for best quality) |
| `--role_type` | `ml_engineering` | Target role |
| `--char_limit` | 130 | Target bullet character count |
| `--max_bullets` | 2 | Max bullets per project |
| `--projects` | all | Comma-separated indices from `--list_projects` |
| `--stream` | off | POST to /optimize and stream bullets live |
| `--list_projects` | off | Print detected projects and exit |

---

### Complete end-to-end workflow

```bash
# 1. Start the full stack
docker compose up

# 2. Seed Qdrant ONCE with your good CVs
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs/ --dry_run    # preview
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs/              # seed

# 3. Check it worked
curl http://localhost:8000/health
# → "vector_count": 280

# 4. For each job application:
python scripts/parse_biodata.py \
    --docx     Vybhav_Chaturvedi_Biodata.docx \
    --jd_file  ~/jobs/amazon_mle.txt \
    --projects 2,4 \
    --role_type ml_engineering \
    --stream
```
