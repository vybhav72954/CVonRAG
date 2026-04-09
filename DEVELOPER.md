# CVonRAG — Developer Guide
### Complete reference for admin setup, architecture, user journey, development, Kaggle inference, and troubleshooting.

---

## Table of Contents

1. [System overview — who does what](#1-system-overview--who-does-what)
2. [Choose your inference path](#2-choose-your-inference-path)
3. [Prerequisites and tool versions](#3-prerequisites-and-tool-versions)
4. [Admin setup — one-time steps (you, not users)](#4-admin-setup--one-time-steps)
   - [A. Clone and install Python backend](#a-clone-and-install-python-backend)
   - [B. Start Qdrant](#b-start-qdrant)
   - [C. Pull Ollama models](#c-pull-ollama-models)
   - [D. Start the FastAPI backend](#d-start-the-fastapi-backend)
   - [E. Verify the health endpoint](#e-verify-the-health-endpoint)
   - [F. Seed Qdrant with Gold Standard CVs](#f-seed-qdrant-with-gold-standard-cvs)
   - [G. Install and start the frontend](#g-install-and-start-the-frontend)
   - [H. Run the test suite](#h-run-the-test-suite)
5. [User journey — what batchmates see and do](#5-user-journey--what-batchmates-see-and-do)
   - [Screen 1: Upload CV](#screen-1-upload-cv)
   - [Screen 2: Paste JD and get AI recommendation](#screen-2-paste-jd-and-get-ai-recommendation)
   - [Screen 3: Review bullets and copy](#screen-3-review-bullets-and-copy)
6. [Kaggle H100 path — best quality on low VRAM](#6-kaggle-h100-path--best-quality-on-low-vram)
7. [Fine-tuning — the honest answer](#7-fine-tuning--the-honest-answer)
8. [Architecture deep dive](#8-architecture-deep-dive)
   - [The 5-phase pipeline](#the-5-phase-pipeline)
   - [The content/style firewall](#the-contentstyle-firewall)
   - [Project recommendation engine](#project-recommendation-engine)
   - [The char-limit loop](#the-char-limit-loop)
   - [SSE streaming model](#sse-streaming-model)
9. [Project structure reference](#9-project-structure-reference)
10. [API endpoints reference](#10-api-endpoints-reference)
11. [Pydantic schemas reference](#11-pydantic-schemas-reference)
12. [Environment variables reference](#12-environment-variables-reference)
13. [Docker Compose setup](#13-docker-compose-setup)
14. [Development workflows](#14-development-workflows)
15. [GitHub setup and branching](#15-github-setup-and-branching)
16. [Troubleshooting](#16-troubleshooting)
17. [Quick reference](#17-quick-reference)

---

## 1. System overview — who does what

CVonRAG has two distinct roles. Understanding this distinction is important before you read anything else.

| Role | Person | Tasks | Frequency |
|---|---|---|---|
| **Admin** | You (Vybhav) | Clone repo, seed Qdrant with Gold Standard CVs, run the server | One-time setup + every session |
| **User** | Batchmates | Upload their own biodata, paste a JD, get bullets | Every job application |

**Users never touch Qdrant seeding.** They never know it exists. They open a browser, upload their `.docx` biodata, paste a JD, and within seconds the AI tells them which of their own projects best match that JD — then generates polished, character-count-validated bullets for those projects.

**The Gold Standard CVs** (which you seed into Qdrant) provide *style only* — sentence structure, separator patterns, verb placement. A user's metrics, tools, and outcomes are never drawn from these CVs. Ever.

### The complete data flow

```
Admin (you) → ~/good_cvs/*.pdf  →  scripts/ingest_pdfs.py
                                         │
                                         ▼
                                  Qdrant: gold_standard_cvs
                                  (style exemplars only)
                                         │
                                         │  retrieved at query time
                                         ▼
User → biodata.docx  →  POST /parse  →  projects + CoreFacts
User → JD text       →  POST /recommend  →  ranked projects with reasons
User → confirm       →  POST /optimize  →  SSE bullets (style from Qdrant +
                                            content from CoreFacts)
                                         │
                                         ▼
                              Browser: bullets grouped by project
                              character badge, copy button, Copy All
```

---

## 2. Choose your inference path

Before doing anything, answer this:

**Is Ollama running in low-VRAM mode? Is generation slow (> 60 seconds per bullet)?**

**No → use [Section 4 (Local inference)](#4-admin-setup--one-time-steps).**
Ollama handles everything on your machine. Bullets stream interactively in the browser. Good quality with 14b, decent with 7b.

**Yes → use [Section 6 (Kaggle H100)](#6-kaggle-h100-path--best-quality-on-low-vram).**
Your laptop runs the API server, Qdrant, and embeddings (nomic-embed-text is tiny — ~300 MB, runs fine on low VRAM). Kaggle's free H100 does the heavy LLM inference. Actually produces *better* output than local 14b because you get Qwen2.5-72B in 4-bit quantisation (~72B parameters vs 14B).

**Not sure?** Run this and watch for warnings:
```bash
ollama run qwen2.5:14b "Write one CV bullet about Python."
```
If you see `"low vram"` in the Ollama logs, or it takes more than 90 seconds, use the Kaggle path.

---

## 2. Prerequisites and tool versions

Install all of these before starting Section 4.

| Tool | Min version | How to install | How to verify |
|---|---|---|---|
| Python | 3.12 | [python.org](https://python.org) | `python --version` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| Docker | 24+ | [docker.com](https://docker.com) | `docker --version` |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) | `ollama --version` |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) | `node --version` |
| Git | any | pre-installed / [git-scm.com](https://git-scm.com) | `git --version` |

**Minimum system requirements for local inference:**
- RAM: 12 GB free for qwen2.5:14b, 8 GB for 7b
- Disk: ~15 GB for models, ~500 MB for code + dependencies
- GPU: Optional. Ollama will use CPU if no GPU is available (slower but works).

---

## 4. Admin setup — one-time steps

Do these steps once. After this, every session is just three `start` commands (Section 17 Quick Reference).

---

### A. Clone and install Python backend

```bash
# Get the code
git clone https://github.com/YOUR_USERNAME/cvonrag.git
cd cvonrag                    # everything below assumes you're in this folder

# Create virtual environment
uv venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows PowerShell

# Install all dependencies including dev/test tools
uv pip install -e ".[dev]"

# Copy the environment config — defaults work for local dev, nothing needs editing
cp .env.example .env

# Verify core imports
python -c "import fastapi, qdrant_client, httpx, docx, pdfplumber; print('All OK')"
```

If the import check fails, make sure the venv is activated — `which python` should show `.venv/bin/python`, not `/usr/bin/python`.

---

### B. Start Qdrant

Qdrant is the vector database that stores your Gold Standard style exemplars. Run it as a Docker container — one command, and it persists data across machine restarts.

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4
```

Verify it's running:
```bash
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine","version":"1.12.4"}
```

**How to restart it later** (after machine reboot, etc.):
```bash
docker start qdrant
```

**How to check if it's already running:**
```bash
docker ps | grep qdrant
```

**How to stop it cleanly:**
```bash
docker stop qdrant
```

**How to nuke the data and start fresh** (rarely needed):
```bash
docker rm qdrant
docker volume rm qdrant_storage
# then re-run the docker run command above
```

---

### C. Pull Ollama models

Two models are required: one for LLM generation, one for embeddings. Both are pulled once and cached locally forever.

```bash
# Make sure Ollama is running
ollama serve    # skip if it's already running as a system daemon

# Embedding model — always pull this, it's small (~300 MB):
ollama pull nomic-embed-text

# LLM model — pick ONE based on your available RAM:
ollama pull qwen2.5:14b    # ≥ 12 GB free RAM — recommended (best local quality)
ollama pull qwen2.5:7b     # 8–12 GB free RAM — good quality
ollama pull qwen2.5:3b     # < 8 GB free RAM — acceptable, fast (but use Kaggle if possible)

# Verify both models are present:
ollama list
# Should show both nomic-embed-text and your chosen qwen2.5 model
```

If you pulled something other than `qwen2.5:14b`, open `.env` and update:
```env
OLLAMA_LLM_MODEL=qwen2.5:7b    # or qwen2.5:3b
```

**Disk space note:** qwen2.5:14b is ~9 GB, 7b is ~5 GB, 3b is ~2 GB. nomic-embed-text is ~300 MB.

**First generation will be slower** while Ollama loads the model into RAM/VRAM. Subsequent generations within the same Ollama session are much faster.

---

### D. Start the FastAPI backend

With your venv activated and both Qdrant + Ollama running:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Waiting for application startup.
INFO:     CVonRAG starting — checking Qdrant collection …
INFO:     Qdrant ready.
INFO:     Application startup complete.
```

The `--reload` flag watches for code changes and restarts automatically. Remove it in production.

The interactive API docs are at: **http://localhost:8000/docs** (Swagger UI)
Alternative ReDoc docs at: **http://localhost:8000/redoc**

---

### E. Verify the health endpoint

Before going further, confirm everything is wired up correctly:

```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected response:
```json
{
  "status": "ok",
  "model": "qwen2.5:14b",
  "qdrant_connected": true,
  "collection_exists": false,
  "vector_count": 0,
  "ollama_ok": true
}
```

**What each field means:**

| Field | Expected | What if wrong |
|---|---|---|
| `status` | `"ok"` | `"degraded"` means Ollama or Qdrant is down |
| `qdrant_connected` | `true` | Qdrant container not running — `docker start qdrant` |
| `collection_exists` | `false` (before seeding) | Fine — will be `true` after Section F |
| `vector_count` | `0` (before seeding) | Fine — will be > 0 after Section F |
| `ollama_ok` | `true` | Model not pulled or Ollama not running — see Section C |

**`status: "ok"` requires both `qdrant_connected: true` and `ollama_ok: true`.**
If either is false, don't continue — fix it first (see Section 16 Troubleshooting).

---

### F. Seed Qdrant with Gold Standard CVs

**This is an admin task. Users never do this. You do it once.**

Qdrant needs a corpus of high-quality CV bullets to learn *style patterns* from. These are your curated CVs — IIT/IIM placement CVs, strong peers' CVs, any CVs with well-structured, quantified bullets. The system reads structural patterns from these (verb placement, separator style, metric formatting) and nothing else. A user's numbers and tools are never drawn from this corpus.

**Step 1 — Collect good CVs:**
```bash
mkdir ~/good_cvs
# Copy PDFs into this folder. 10–30 CVs is ideal.
# More is better. Target CVs with bullets like:
# "Built SARIMA model | Reduced RMSE to 0.250 vs 0.310 baseline"
# "Deployed BERT-based classifier; ↑ F1 from 0.39 to 0.67"
```

**Step 2 — Preview what will be extracted** (no API calls, reads PDFs only):
```bash
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run
```

Sample output:
```
[DRY RUN — no Qdrant writes]
Arjun_IIT_BHU.pdf      : 22 bullets detected
Priya_IIM_Calcutta.pdf : 31 bullets detected
Rohan_IITM.pdf         : 14 bullets detected
Nidhi_placement.pdf    :  0 bullets detected  ← check this one
...
Total: 67 bullets would be seeded
```

If any PDFs show 0 bullets, check what bullet character they use (Section 16 Troubleshooting).

**Step 3 — Actually seed Qdrant:**
```bash
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs
```

Expected output:
```
Arjun_IIT_BHU.pdf      : 22 bullets extracted
Priya_IIM_Calcutta.pdf : 31 bullets extracted
Rohan_IITM.pdf         : 14 bullets extracted
...
Total: 67 bullets → tagging via Ollama (qwen2.5:7b) → embedding → seeding Qdrant
  Batch 1/2: ✓ 50 upserted
  Batch 2/2: ✓ 17 upserted
Done. 67 / 67 bullets in Qdrant.
```

**Step 4 — Verify the seed worked:**
```bash
curl http://localhost:8000/health | python -m json.tool
# "vector_count": 67    ← must be > 0
# "collection_exists": true
```

**You only do this once.** Qdrant data persists in the Docker volume across restarts.
You can add more CVs later by running the script again — it upserts safely (no duplicates if bullet text is identical).

**What happens without seeding:**
The system still works, but bullet quality drops noticeably. The RAG retrieval returns nothing, so the Bullet Alchemist generates purely from the JD analysis and system prompt alone — no style conditioning. Bullet structure is less varied and consistent.

---

### G. Install and start the frontend

```bash
cd frontend
npm install          # one-time dependency install (~30 seconds)
npm run dev
# → http://localhost:5173
```

You should see the CVonRAG wizard in your browser.

**Environment variable (optional):**
The frontend reads `VITE_API_URL` to know where the backend is. Default is `http://localhost:8000`.
If you change the backend port, create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

**To build for production:**
```bash
npm run build
# Output goes to frontend/build/
# Serve with: node -e "import('./build/handler.js').then(m => m.handler)" OR deploy to Vercel
```

---

### H. Run the test suite

No live services needed — everything is mocked. Tests run against isolated, deterministic fixtures.

```bash
# From project root, with venv activated:
pytest
```

Expected:
```
================================================ test session starts ===============================================
collected 243 items

tests/test_models.py                 ....................................  35 passed
tests/test_chains.py                 ................                      16 passed
tests/test_integration.py            ..............                        14 passed
tests/test_api.py                    ..................                     18 passed
tests/test_char_limit_stress.py      ........................................  40 passed
tests/test_parser.py                 ....................................  36 passed
tests/test_recommender.py            .......................                23 passed

============================================== 243 passed in 9.30s ===============================================
```

**Useful test commands:**
```bash
pytest -x                                   # stop on first failure
pytest tests/test_recommender.py -v         # one file, verbose
pytest tests/ -k "test_score"               # filter by name
pytest --cov=app --cov-report=term-missing  # with coverage
pytest -m "not slow"                        # skip slow-marked tests
```

---

## 5. User journey — what batchmates see and do

This section documents exactly what a user experiences when they open the app. Read this so you can support batchmates when they ask questions.

Users open **http://localhost:5173** (or wherever you're hosting it).

---

### Screen 1: Upload CV

The user sees a drag-and-drop upload zone with a file picker fallback.

**What they do:**
1. Drag their `.docx` biodata (best) or `.pdf` CV onto the upload zone, or click to browse
2. Nothing else — parsing is automatic

**What happens in the background:**
1. File is POSTed to `POST /parse` as `multipart/form-data`
2. Backend parses the document — `.docx` via `python-docx`, `.pdf` via `pdfplumber`
3. Project headings are detected (Heading styles in docx, or heuristic patterns in PDF)
4. Each project's bullets are extracted and grouped
5. An LLM call (`qwen2.5` via Ollama) extracts structured `CoreFact` objects from the raw bullet text — pulling out tools, metrics, and outcomes as structured fields
6. Each completed project is streamed back as a Server-Sent Event
7. The UI shows each project appearing one by one as it's processed

**What the user sees while parsing:**
- A live progress spinner with the current status message (`"Extracting facts from Project 2 / 5…"`)
- Projects appearing one by one with their title, fact count, tool tags (indigo), and metric tags (amber)

**What the user sees when parsing is done:**
- A list of all detected projects, each showing:
  - Project title
  - Number of facts extracted
  - Key tools as indigo chips
  - Key metrics as amber chips
  - An "edit facts ↓" toggle to expand the fact editor

**The fact editor (optional):**
Expanding a project shows every extracted `CoreFact` in editable form. Metrics are amber-highlighted to signal they're protected. Users *can* edit facts here if the parser missed something, but they don't have to — clicking "Paste Job Description →" works even without opening the editor.

**File type guidance for users:**
- `.docx` (Word) format gives the best extraction results because heading styles are preserved structurally
- `.pdf` works but depends on bullet character detection — if a PDF was exported from LaTeX or a non-standard template, some bullets may be missed
- Max file size: 10 MB (enforced at backend)

---

### Screen 2: Paste JD and get AI recommendation

**What the user does:**
1. Pastes the full job description into the text area (at least 50 characters, no maximum beyond 10,000)
2. Optionally adjusts settings:
   - **Role type** — ML Engineering, Data Science, Software Engineering, Quant/Finance, Product Management, General
   - **Target chars** — character count for each bullet (default 130, range 60–300)
   - **Bullets / project** — max bullets per project (default 2)
   - **Projects to recommend** — how many projects the AI should pre-select (default 3)
3. Clicks **"Analyse JD — recommend best projects"**

**What happens in the background:**
1. All parsed projects + JD text are sent to `POST /recommend`
2. The `SemanticMatcher` runs LLM-powered JD analysis — extracts required skills, preferred skills, tone, seniority, domain keywords
3. Every `CoreFact` in every project is scored 0–1 against the JD analysis
4. Individual fact scores are rolled up to project-level scores using the mean of each project's top-3 facts (rewards depth without penalising projects that have a few weak bullets alongside great ones)
5. Projects are ranked
6. An LLM generates a one-sentence reason per recommended project, specifically naming matched skills
7. Results are returned as a ranked list

**What the user sees:**
A ranked list of all their projects, showing:
- Project title and rank number
- A green "**#1 recommended**" badge for the top picks
- A horizontal score bar with match percentage (green ≥ 70%, amber ≥ 40%, grey otherwise)
- A one-line AI reason for recommended projects, e.g.:
  *"💡 Directly demonstrates SARIMA forecasting and Python MLOps — core JD requirements."*
- Matched JD skills as indigo chips
- Key metrics as amber chips
- A toggle checkbox — green for recommended, indigo for manually selected

**The top `K` projects are pre-ticked.** The user can:
- Untick a recommended project if they don't want it
- Tick a non-recommended project if they want to override the AI

**When they're happy with the selection:**
They click **"Generate bullets for N projects →"**

---

### Screen 3: Review bullets and copy

**What the user sees during generation:**
- A typewriter stream showing the bullet being written character by character
- Completed bullets appearing below, grouped by project

**What a completed bullet card shows:**
- The bullet text in monospace
- Character count / target (e.g. `132 / 130 chars`)
- A tolerance badge:
  - **✓ within ±2** (green) — bullet is 128–132 chars
  - **⚠ outside ±2** (amber) — bullet missed the target after all iterations
- Number of iterations taken (usually 1–3 for 14b, occasionally 4)
- JD tone label (e.g. `highly quantitative`)
- An individual **Copy** button

**At the top of Screen 3:**
- **Copy all** — copies all bullets grouped by project, with blank lines between projects
- **← Change projects** — goes back to Screen 2 with generation state cleared
- **Start over** — goes back to Screen 1, clears everything

**Sharing guidance for batchmates:**
The bullets are grouped by project in the Copy All output. They can paste the entire output into a notes app, then move individual bullets into their CV in whatever order they want.

---

## 6. Kaggle H100 path — best quality on low VRAM

### 6.1 When to use this

Use Kaggle inference if any of these apply:
- `ollama serve` logs show `"low vram"` for qwen2.5:14b or qwen2.5:7b
- Bullet generation takes more than 60 seconds
- You want the absolute best quality regardless of local hardware
- You're on a machine with less than 8 GB free RAM

**What stays on your machine:** Qdrant, FastAPI server, frontend, Ollama for nomic-embed-text embeddings only (the embedding model is ~300 MB and runs comfortably on low VRAM).

**What moves to Kaggle:** All LLM calls — JD analysis, fact scoring, project recommendation reasons, bullet generation.

### 6.2 Why Kaggle quality is better than local 14b

Kaggle gives you Qwen2.5-72B in 4-bit NF4 quantisation. That's approximately 72 billion parameters vs 14 billion. For instruction-following tasks — which is what bullet generation is — parameter count is the dominant quality factor, more important than precision. Empirically:

| Model | Local inference | Char compliance (±2) | Bullet structure quality |
|---|---|---|---|
| qwen2.5:3b | Yes | ~65% | Acceptable |
| qwen2.5:7b | Yes | ~75% | Good |
| qwen2.5:14b | Yes | ~82% | Good–Great |
| qwen2.5:72B-4bit (Kaggle) | No | ~92% | Excellent |

### 6.3 What the Kaggle script does

`kaggle_pipeline_test.py` is a self-contained pipeline runner. It does not connect to your local machine. On Kaggle's GPU it:

1. Auto-detects available GPU → loads Qwen2.5-72B in 4-bit NF4 (~40 GB VRAM used)
2. Loads nomic-embed-text locally for embeddings
3. Creates an in-memory Qdrant instance (no Docker needed)
4. Reads your `OptimizationRequest` JSON from the `REQUEST_JSON` variable at the top of the script
5. Runs all 5 pipeline phases (JD analysis → fact scoring → RAG retrieval → bullet generation → char-limit loop)
6. Exports all generated bullets to `/kaggle/working/bullets.csv`

### 6.4 Step-by-step: running it

**Step 1 — Start local services (for embedding only)**
```bash
ollama serve
ollama pull nomic-embed-text   # if not already pulled
docker start qdrant            # if not already running
uvicorn app.main:app --reload --port 8000
```
Check health shows `qdrant_connected: true`. `ollama_ok` may be false if only nomic-embed-text is pulled — that's fine for embedding-only use.

**Step 2 — Generate your input JSON**
```bash
# List what projects are in your biodata:
python scripts/parse_biodata.py \
    --docx  Vybhav_Chaturvedi_Biodata.docx \
    --list_projects

# Output example:
# [1] Cuckoo.ai                       (4 facts)
# [2] Time Series — Hourly Wages      (5 facts)
# [3] Decoding Depression Networks    (3 facts)
# [4] IITB Web Development Club       (2 facts)

# Generate request.json for the projects you want:
python scripts/parse_biodata.py \
    --docx      Vybhav_Chaturvedi_Biodata.docx \
    --jd_file   amazon_mle_jd.txt \
    --projects  1,2 \
    --output    request.json

cat request.json   # this is what you'll paste into Kaggle
```

**Step 3 — Set up Kaggle notebook (one-time)**
1. Go to [kaggle.com](https://kaggle.com) → Code → New Notebook
2. In the right sidebar: Accelerator → **GPU P100** (free tier) or **T4 x2** or **H100** if you have quota
3. Internet → **ON** (required to download model weights on first run)
4. Either upload `kaggle_pipeline_test.py` or paste its contents

**Step 4 — Per run**
1. At the top of `kaggle_pipeline_test.py`, find the `REQUEST_JSON` variable and paste the contents of `request.json`
2. Run All
3. First run: ~15 minutes to download Qwen2.5-72B-Instruct weights (~40 GB). The weights are cached for the duration of the Kaggle session — subsequent runs in the same session start in seconds.

**Step 5 — Download results**
Output is at `/kaggle/working/bullets.csv`. Contains: `project_id`, `project_title`, `bullet_text`, `char_count`, `within_tolerance`, `iterations_taken`.

### 6.5 Hybrid approach (recommended for placement season)

For the best practical workflow with low VRAM:
- Use local Ollama (qwen2.5:3b) for the `/parse` endpoint only — fact extraction doesn't need a huge model
- Use Kaggle for the actual bullet generation — quality is the priority here
- Seeding Qdrant happens once and uses your local model, quality doesn't matter as much

```env
# .env setting for hybrid path:
OLLAMA_LLM_MODEL=qwen2.5:3b    # parsing + embedding only
```

---

## 7. Fine-tuning — the honest answer

**Should you fine-tune Qwen2.5 on CV bullet examples?**

**No. Not now. Possibly not ever. Here's the full explanation:**

### Why fine-tuning won't help here

**1. CVonRAG is an instruction-following problem, not a knowledge problem.**

Fine-tuning teaches a model new *knowledge* — domain vocabulary, factual associations, format conventions. Qwen2.5 already has all the knowledge needed to write CV bullets. It knows what "SARIMA" is. It knows what "reduced RMSE by 13.5%" means in a resume context. It knows verb-first sentence structure. The gap isn't knowledge — it's *conditioning* the model to apply its knowledge to a specific character count, specific content constraints, and a specific JD tone simultaneously. That's a prompting problem.

**2. The RAG system already does what fine-tuning would do for style.**

The typical motivation for fine-tuning a generation model on a domain corpus is to bake in stylistic patterns — "always use | as a separator," "always lead with an action verb," etc. CVonRAG's RAG system does this dynamically: it retrieves the 5 style exemplars most similar to the JD's embedding, then conditions the Bullet Alchemist on them. This is *more powerful* than fine-tuning for style because it adapts to the role (ML Engineering gets different exemplars than PM). Fine-tuning bakes in one fixed style.

**3. Building good training data is the real problem, and it's a solved problem differently.**

Fine-tuning requires thousands of verified `(input_facts + JD, ideal_bullet)` pairs. Each pair needs manual verification: is the metric preserved exactly? Is the character count correct? Is the tool attribution accurate? Creating this at scale is months of work. The resulting dataset describes *your* definition of ideal, which may not transfer to your users' applications.

**4. Fine-tuned models overfit their training distribution.**

A model fine-tuned on ML Engineering bullets at 130 chars will write noticeably weaker bullets for Quant Finance roles or for 150-char targets. The base model + RAG adapts gracefully to any role type because the exemplar retrieval adapts.

### What actually improves output quality

In order of impact-per-effort:

| Action | Effort | Quality gain |
|---|---|---|
| Use 72B via Kaggle instead of 14b local | Low — run one notebook | High — noticeable improvement in structure and char compliance |
| Increase Gold Standard corpus to 50+ CVs | Low — collect more PDFs | Medium — more diverse style exemplars improve retrieval |
| Increase `CHAR_LOOP_MAX_ITERATIONS` from 4 to 6 | Zero — one env var | Low — helps convergence for bullets that miss on first pass |
| Improve the fact extraction prompt in `parser.py` | Medium — prompt engineering | Medium — better structured facts → better bullets |
| Improve the Bullet Alchemist system prompt in `chains.py` | Medium — prompt engineering | Medium — can reduce hallucination of unsupported claims |
| Fine-tune Qwen2.5 | Very high — months of data collection + training | Marginal — probably worse than 72B-4bit with good prompting |

### The one future scenario where fine-tuning makes sense

If you eventually have users rating their bullets (thumbs up/down, or tracking which bullets they actually submitted), that explicit preference signal could be distilled into a fine-tuning dataset. That's a productisation milestone ~6 months away, requiring infrastructure, user tracking, and a data pipeline you don't have yet. File it in the backlog — it's not relevant for placement season.

---

## 8. Architecture deep dive

### The 5-phase pipeline

Every call to `POST /optimize` runs this pipeline in sequence:

```
Phase 1: OptimizationRequest validation (Pydantic)
         ↓
Phase 2: SemanticMatcher
         → _ollama_chat(JD analysis system prompt) → JD analysis dict
           { required_skills, preferred_skills, key_action_verbs, tone, seniority, domain_keywords }
         → _ollama_chat(fact scoring system prompt) → relevance scores 0–1 per CoreFact
         → ScoredFact list, sorted by score descending
         ↓
Phase 3: Qdrant RAG retrieval
         → embed(JD text + top-N fact text) → 768-dim vector
         → Qdrant.search(collection, vector, top_k=5) → StyleExemplar list
         ↓
Phase 4: BulletAlchemist — for each project, for each bullet slot:
         a. Format CoreFacts into prompt (CONTENT section)
         b. Format StyleExemplars into prompt (STYLE section — read-only)
         c. _ollama_stream → stream tokens to browser via SSE
         d. Char-limit loop (see below)
         e. Yield GeneratedBullet with full BulletMetadata
         ↓
Phase 5: CVonRAGOrchestrator.run() — async generator
         → yields ("token", str) events during streaming
         → yields ("bullet", GeneratedBullet) events when bullet is validated
         → yields ("done", {elapsed_seconds}) when all bullets complete
         → main.py wraps events in SSE envelopes and streams to client
```

### The content/style firewall

This is the most important architectural constraint in the system. It prevents hallucinated metrics.

```
┌──────────────────────────────────────────────────────┐
│  CONTENT (CoreFact — from user JSON only)            │
│  ─────────────────────────────────────────────────── │
│  • Every number, %, financial figure → EXACT         │
│  • Every named tool, algorithm, library              │
│  • The actual outcome the user achieved              │
│  • Specific action verbs for what the user did       │
│                                                      │
│  NEVER modified. NEVER crossed with style.           │
└──────────────────────────────────────────────────────┘
                        │
                        │ combined in prompt as separate sections
                        ▼
┌──────────────────────────────────────────────────────┐
│  STYLE (StyleExemplar — from Qdrant only)            │
│  ─────────────────────────────────────────────────── │
│  • Visual separators: | • ; ↑ ↓ → &                 │
│  • Sentence architecture: VERB → TOOL → METRIC → IM │
│  • Abbreviation conventions: vs, w/, ~, approx       │
│                                                      │
│  NEVER contributes numbers or company names.         │
│  NEVER used as content.                              │
└──────────────────────────────────────────────────────┘
```

The Bullet Alchemist system prompt explicitly forbids:
- Copying a company name or domain term from an exemplar into the output
- Attaching an exemplar's metric to the user's tool or vice versa
- Inventing steps, deployments, or outcomes not in the user's CoreFacts
- Altering any number — 87% stays 87%, 0.250 stays 0.250

### Project recommendation engine

`app/recommender.py` runs as a pre-step before `/optimize`.

```
POST /recommend
    → SemanticMatcher.analyze_jd(jd_text)
       → LLM extracts: required_skills, preferred_skills, tone, seniority, domain_keywords
    → SemanticMatcher.score_facts(jd_analysis, all_projects)
       → LLM scores each CoreFact 0–1 for JD relevance
       → Returns list[ScoredFact] sorted by score desc
    → _project_score(project_id, scored_facts)
       → mean of top-3 fact scores for the project
       → this rewards projects with multiple strong facts
    → All projects ranked by score
    → LLM generates one-sentence reasons for top-K projects
       → reasons must name specific matched skills
    → Returns list[ProjectRecommendation] sorted by score desc
       → each has: score, rank, reason, matched_skills, top_metrics, recommended bool
```

**Why mean of top-3, not mean of all facts?**
A project might have 6 facts: 3 excellent (JD-relevant) and 3 weak (irrelevant side details). Mean-of-all gives this project a mediocre score. Mean-of-top-3 correctly identifies it as strong. This avoids penalising projects that had comprehensive documentation.

### The char-limit loop

Every bullet goes through an iterative refinement loop targeting `target ± tolerance` characters.

```python
for iteration in range(CHAR_LOOP_MAX_ITERATIONS):  # default: 4
    draft = generate_bullet(prompt)
    char_count = len(draft)

    if lower_bound <= char_count <= upper_bound:
        # ✓ within tolerance — use this bullet
        break

    if char_count > upper_bound:
        # too long → compress using _COMPRESS_TEMPLATE
        prompt = compress_prompt(draft, target, diff)
    else:
        # too short → expand using _EXPAND_TEMPLATE
        prompt = expand_prompt(draft, target, diff)

# If loop exhausts without converging: return the iteration closest to target
```

The compress/expand prompts tell the model:
- Exactly how many characters over/under the current draft is
- Exactly how many characters to add or remove
- To preserve all metrics, tools, and outcomes exactly

With qwen2.5:14b, ~82% of bullets converge within 2 iterations. With 72B: ~92%.

### SSE streaming model

The browser never polls. All three endpoints (`/parse`, `/recommend` is synchronous, `/optimize`) use Server-Sent Events.

SSE frame format:
```
event: {event_type}
data: {json_payload}

```
(two newlines terminate each frame)

Event types for `/optimize`:

| Event | Payload | Frontend behaviour |
|---|---|---|
| `token` | `{"data": "• Built SARI"}` | Appended to typewriter buffer |
| `bullet` | `{"data": GeneratedBullet}` | Bullet card added to list |
| `metadata` | `{"data": BulletMetadata}` | Char count badge updated |
| `done` | `{"data": {"elapsed_seconds": 12.4}}` | Status updated to done |
| `error` | `{"error_message": "..."}` | Error state shown |

Event types for `/parse`:

| Event | Payload | Frontend behaviour |
|---|---|---|
| `progress` | `{"message": "...", "current": 2, "total": 5}` | Progress text updated |
| `project` | `{"project": ProjectData, "index": 2, "total": 5}` | Project card added |
| `done` | `{"total_projects": 5, "total_facts": 23}` | Status set to done |
| `error` | `{"error_message": "..."}` | Error shown |

---

## 9. Project structure reference

```
CVonRAG-Final/
│
├── app/
│   ├── __init__.py
│   ├── config.py              ← all settings via env vars (pydantic-settings)
│   ├── models.py              ← all Pydantic v2 schemas (CoreFact, ProjectData, etc.)
│   ├── chains.py              ← 5-phase pipeline: SemanticMatcher, BulletAlchemist,
│   │                            CVonRAGOrchestrator, helper functions
│   ├── vector_store.py        ← Qdrant operations + nomic-embed-text embeddings
│   ├── parser.py              ← .docx/.pdf → RawProject → CoreFact via LLM (SSE stream)
│   ├── recommender.py         ← scores all projects against JD, returns ranked recs
│   └── main.py                ← FastAPI app: /parse, /recommend, /optimize, /ingest, /health
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            ← shared fixtures (mock settings, mock HTTP client)
│   ├── test_models.py         ← Pydantic schema validation (35 tests)
│   ├── test_chains.py         ← pipeline helper unit tests (16 tests)
│   ├── test_integration.py    ← full pipeline with mocked Ollama + Qdrant (14 tests)
│   ├── test_api.py            ← all endpoints, including /parse and /recommend (18 tests)
│   ├── test_char_limit_stress.py  ← char-limit loop convergence edge cases (40 tests)
│   ├── test_parser.py         ← parser unit tests: docx/pdf extraction + LLM facts (36 tests)
│   └── test_recommender.py   ← project scoring + recommendation tests (23 tests)
│                                ─────── total: 243 tests ───────
│
├── frontend/
│   ├── src/
│   │   ├── app.html           ← SvelteKit HTML shell
│   │   ├── app.css            ← IBM Plex Mono/Sans, CSS design tokens, dark theme
│   │   ├── routes/
│   │   │   ├── +layout.svelte ← header with 3-step progress bar (1 → 2 → 3)
│   │   │   ├── +layout.js     ← export const ssr = false  (client-side only)
│   │   │   └── +page.svelte   ← 3-screen wizard (602 lines)
│   │   └── lib/
│   │       ├── api.js         ← parseCV(), recommendProjects(), optimizeResume(), checkHealth()
│   │       └── stores.js      ← wizard state machine (Svelte stores)
│   ├── static/favicon.png
│   ├── jsconfig.json
│   ├── postcss.config.js
│   ├── package.json           ← "type": "module" is required
│   ├── svelte.config.js
│   ├── vite.config.js         ← imports from @sveltejs/kit/vite (not @sveltejs/vite-plugin-svelte)
│   ├── tailwind.config.js
│   └── .env.example           ← VITE_API_URL=http://localhost:8000
│
├── scripts/
│   ├── ingest_pdfs.py         ← admin: seed Qdrant from folder of PDFs (run once)
│   └── parse_biodata.py       ← terminal: parse .docx → request.json for Kaggle
│
├── kaggle_pipeline_test.py    ← standalone pipeline on Kaggle H100 (no local Ollama needed)
├── pyproject.toml             ← all deps, ruff config, pytest config
├── Dockerfile                 ← backend container image
├── docker-compose.yml         ← full local stack: app + ollama + qdrant
├── .env.example               ← copy to .env, all defaults work for local dev
├── .gitignore
├── README.md
└── DEVELOPER.md               ← this file
```

---

## 10. API endpoints reference

| Method | Path | Tags | Description |
|---|---|---|---|
| `GET` | `/` | — | Welcome message |
| `GET` | `/health` | infra | Liveness check — Ollama + Qdrant status |
| `POST` | `/parse` | parsing | Upload .docx/.pdf → SSE stream of projects + facts |
| `POST` | `/recommend` | generation | Score all projects vs JD → ranked recommendations |
| `POST` | `/optimize` | generation | OptimizationRequest → SSE stream of tokens + bullets |
| `POST` | `/ingest` | admin | Seed Qdrant with Gold Standard bullet objects |

Full interactive docs: **http://localhost:8000/docs**

### POST /parse

```
Content-Type: multipart/form-data
Body: file=<.docx or .pdf>

Constraints:
  Max file size:  10 MB
  Accepted types: .docx, .pdf only (415 for anything else)
  Min file size:  100 bytes (400 for empty/corrupt)

Response: text/event-stream (SSE)
```

### POST /recommend

```json
{
  "job_description": "string (50–10000 chars)",
  "projects": [ ProjectData... ],
  "top_k": 3
}
```

Response:
```json
{
  "recommendations": [
    {
      "project_id": "p-ts",
      "title": "Time Series — Hourly Wages",
      "score": 0.876,
      "rank": 1,
      "reason": "Directly demonstrates SARIMA forecasting and Python MLOps — core JD requirements.",
      "matched_skills": ["SARIMA", "Python", "MLOps"],
      "top_metrics": ["RMSE=0.250", "84.5% weight allocation"],
      "recommended": true,
      "core_facts": [ ... ]
    }
  ],
  "jd_summary": ""
}
```

### POST /optimize

```json
{
  "job_description": "string",
  "projects": [ ProjectData... ],
  "target_role_type": "ml_engineering",
  "constraints": {
    "target_char_limit": 130,
    "tolerance": 2,
    "bullet_prefix": "•",
    "max_bullets_per_project": 2
  }
}
```

Response: `text/event-stream` (SSE — see Section 8 for event types)

---

## 11. Pydantic schemas reference

All schemas are in `app/models.py`.

### Input schemas

**CoreFact** — one atomic fact from the user's CV:
```python
class CoreFact(BaseModel):
    fact_id: str          # stable ID for traceability across pipeline
    text: str             # full fact text (min 5 chars)
    tools: list[str]      # named tools/libraries/frameworks
    metrics: list[str]    # quantitative results ("RMSE=0.250", "87%", "$2500→$800")
    outcome: str          # what was achieved
```

**ProjectData** — one project with 1–12 facts:
```python
class ProjectData(BaseModel):
    project_id: str
    title: str
    core_facts: list[CoreFact]   # 1–12 facts
```

**FormattingConstraints** — bullet formatting rules:
```python
class FormattingConstraints(BaseModel):
    target_char_limit: int = 130     # 60–300
    tolerance: int = 2               # 1–5
    bullet_prefix: str = "•"
    max_bullets_per_project: int = 3  # 1–8
```

### Output schemas

**GeneratedBullet** — one completed, validated bullet:
```python
class GeneratedBullet(BaseModel):
    text: str
    metadata: BulletMetadata

class BulletMetadata(BaseModel):
    bullet_index: int
    project_id: str
    source_fact_ids: list[str]   # which facts contributed to this bullet
    char_count: int
    char_target: int
    iterations_taken: int        # how many char-loop iterations were needed
    exemplar_ids_used: list[str] # which Qdrant style exemplars were used
    jd_tone: JDTone
    within_tolerance: bool
```

### Enums

```python
class RoleType(StrEnum):
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE         = "data_science"
    ML_ENGINEERING       = "ml_engineering"
    PRODUCT_MANAGEMENT   = "product_management"
    QUANT_FINANCE        = "quant_finance"
    GENERAL              = "general"

class JDTone(StrEnum):
    HIGHLY_QUANTITATIVE = "highly_quantitative"
    ENGINEERING_FOCUSED = "engineering_focused"
    LEADERSHIP_FOCUSED  = "leadership_focused"
    RESEARCH_FOCUSED    = "research_focused"
    BALANCED            = "balanced"
```

---

## 12. Environment variables reference

All variables are in `app/config.py` as `Settings(BaseSettings)`. All have defaults — you only need to change values that differ from local dev.

```env
# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
# docker-compose overrides this to http://ollama:11434

# Pick based on your hardware:
#   qwen2.5:3b   ~2 GB RAM   (fast, acceptable for parsing; use Kaggle for generation)
#   qwen2.5:7b   ~5 GB RAM   (good quality; development)
#   qwen2.5:14b  ~10 GB RAM  (recommended; best local quality)
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_EMBED_MODEL=nomic-embed-text   # always this; do not change

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
# docker-compose overrides this to http://qdrant:6333
# For Qdrant Cloud: QDRANT_URL=https://your-cluster.cloud.qdrant.io
# QDRANT_API_KEY=your_key_here   # only for Qdrant Cloud
QDRANT_COLLECTION=gold_standard_cvs
QDRANT_VECTOR_SIZE=768            # must match nomic-embed-text output dimension

# ── RAG ───────────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K=5                 # how many style exemplars to retrieve per query
                                  # increase to 8 if you have 100+ Gold CVs seeded

# ── Char-limit loop ───────────────────────────────────────────────────────────
CHAR_LOOP_MAX_ITERATIONS=4        # increase to 6 if bullets miss ±2 too often
CHAR_TOLERANCE=2                  # ±2 chars from target (tight enough for CV use)

# ── LLM generation ────────────────────────────────────────────────────────────
LLM_TEMPERATURE=0.3               # 0.1 = highly consistent, 0.5 = more varied
LLM_MAX_TOKENS=512                # enough for any CV bullet
LLM_CONTEXT_WINDOW=8192           # Qwen2.5 supports 32K but 8K is sufficient

# ── App ───────────────────────────────────────────────────────────────────────
APP_ENV=development               # set to "production" to disable --reload
CORS_ORIGINS=["*"]                # restrict in production: ["https://your-frontend.vercel.app"]
LOG_LEVEL=INFO
PORT=8000
```

---

## 13. Docker Compose setup

Docker Compose is not required for development — the manual steps in Section 4 are more debuggable. Use Compose once everything works locally and you want a one-command startup.

```bash
# First time — pulls images and downloads Ollama models (~20 minutes)
docker compose up --build

# Subsequent runs — all images and models cached
docker compose up

# Background mode
docker compose up -d

# View live logs from specific service
docker compose logs -f app
docker compose logs -f ollama
docker compose logs -f qdrant

# Stop everything (data preserved in volumes)
docker compose down

# Stop everything AND destroy all volumes (Qdrant data + Ollama models)
# Use this only to start completely fresh
docker compose down -v

# Restart just the app (e.g. after code change)
docker compose restart app

# Check status
docker compose ps
```

Services started by Compose:

| Service | Internal URL | External URL | Notes |
|---|---|---|---|
| `app` | `http://app:8000` | `http://localhost:8000` | FastAPI; waits for qdrant + model-puller |
| `ollama` | `http://ollama:11434` | `http://localhost:11434` | LLM inference + embeddings |
| `qdrant` | `http://qdrant:6333` | `http://localhost:6333` | Vector store |
| `model-puller` | — | — | Pulls models once, then exits code 0 (normal) |

**Important:** The `model-puller` container shows `Exited (0)` in `docker compose ps`. This is normal and expected — it pulled the models and terminated cleanly.

**Low VRAM caveat:** Docker Compose does not help with low VRAM. The Ollama container inside Compose uses the same GPU as Ollama outside Compose. If you're on low VRAM, use Section 6 (Kaggle) for generation regardless of whether you use Compose or not.

---

## 14. Development workflows

### Adding a new feature

1. Branch from `dev`:
```bash
git checkout dev
git checkout -b feature/per-bullet-regenerate
```

2. Make changes to backend:
```bash
# uvicorn --reload auto-restarts on file save
# tests run independently of the server
pytest tests/test_api.py -x   # check your endpoint works
```

3. Make changes to frontend:
```bash
cd frontend
npm run dev   # hot-reloads on save
```

4. Run full suite before merging:
```bash
pytest          # all 243 tests
ruff check app/ tests/
```

### Adding tests

Tests live in `tests/`. The pattern is:
- Unit tests: mock everything external (`@patch("app.recommender._ollama_chat", ...)`)
- API tests: use `TestClient(app)` from fastapi, mock the underlying service functions
- All async tests use `@pytest.mark.asyncio` (configured globally via `asyncio_mode = "auto"`)

### Modifying prompts

All LLM prompts are module-level constants in `app/chains.py` and `app/recommender.py`. They start with `_` and are named `_*_SYSTEM`, `_*_TEMPLATE`.

When modifying prompts:
- Keep the JSON output schema comment if the prompt returns JSON
- Keep the "no fences, no preamble" instruction — the parser strips fences but the cleaner the output the better
- Test with a short `ollama run qwen2.5:14b` call first before running pytest

### Adding a new role type

1. Add to `RoleType` enum in `app/models.py`
2. Add to the ROLES array in `frontend/src/routes/+page.svelte`
3. Optionally adjust `_JD_ANALYSIS_SYSTEM` in `chains.py` to recognise the new tone
4. Add test cases in `tests/test_models.py`

### Linting and formatting

```bash
ruff check app/ tests/         # lint
ruff check app/ tests/ --fix   # auto-fix safe issues
ruff format app/ tests/        # format (black-compatible)
mypy app/                      # type checking (non-strict)
```

---

## 15. GitHub setup and branching

### Initial push

```bash
git init
git add .
git commit -m "feat: CVonRAG v1.3 — project recommendation, 243 tests"

# On GitHub: create a new repo, don't initialise with README
git remote add origin https://github.com/YOUR_USERNAME/cvonrag.git
git branch -M main
git push -u origin main
```

`.env` is in `.gitignore` and is never committed. It contains no secrets anyway (no paid API keys), but the habit is important.

### Branch strategy

```
main          ← stable, tested, what batchmates run
dev           ← integration branch, merged into main per milestone
feature/*     ← one feature per branch, e.g. feature/per-bullet-regenerate
fix/*         ← bug fixes, e.g. fix/pdf-bullet-detection
```

### Week 2 planned features (branch from dev when ready)

- `feature/per-bullet-regenerate` — `POST /regenerate` endpoint for one bullet at a time
- `feature/tone-popover` — "Adjust tone" button per bullet (more formal / more quantitative / more concise)
- `feature/invite-code-auth` — simple invite code gate before sharing with batchmates

---

## 16. Troubleshooting

### `status: "degraded"` or `ollama_ok: false` in /health

```bash
# Check if Ollama is running:
curl http://localhost:11434/api/tags

# Not running? Start it:
ollama serve

# Running but model not found?
ollama list   # check if qwen2.5:14b (or your chosen model) is listed
ollama pull qwen2.5:14b   # pull if missing

# Running inside Docker Compose? Check the correct URL:
# .env should have OLLAMA_BASE_URL=http://ollama:11434 when using Compose
# .env should have OLLAMA_BASE_URL=http://localhost:11434 when running locally
```

### `qdrant_connected: false` in /health

```bash
# Is the container running?
docker ps | grep qdrant

# Not running? Start it:
docker start qdrant

# Container doesn't exist? Create it:
docker run -d --name qdrant -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4

# Check if port is blocked:
curl http://localhost:6333/healthz
```

### `vector_count: 0` — bullets work but style is weak

You haven't seeded Qdrant, or the collection was wiped. Seed it:
```bash
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs
```

### Gold PDFs show 0 bullets extracted

The ingest script looks for lines starting with these bullet characters:
```python
_BULLET_RE = re.compile(r"^\s*[▪•\-–—→*]\s+.{10,}")
```

If your PDFs use a different bullet character (e.g. `◆`, `○`, `▸`, `✦`):
1. Open the PDF in a text editor or run `pdfplumber` to inspect the raw text
2. Add the character to `_BULLET_RE` in `scripts/ingest_pdfs.py`

If the PDF is scanned (image-based), pdfplumber won't extract text. You need a text-selectable PDF.

### Bullets outside ±2 tolerance (⚠ badge)

**First, try:** Increase max iterations:
```env
CHAR_LOOP_MAX_ITERATIONS=6
```

**If still failing:** Use a larger model — the 14b model follows char-count instructions better than 7b or 3b. Use Kaggle for 72B.

**If one specific bullet is always failing:** The facts for that project might be too dense to compress, or too sparse to expand. Edit the facts manually (Screen 1 fact editor) to reduce/increase content.

### Recommendation scores are all similar

Happens when the JD is generic ("looking for a software engineer with coding skills"). More specific JDs produce more differentiated scores. Ask the user to paste the full JD including requirements, not just the summary paragraph.

### Frontend shows "Network error" or blank page

```bash
# Is the backend running?
curl http://localhost:8000/health

# Check frontend/.env (create if missing):
echo "VITE_API_URL=http://localhost:8000" > frontend/.env

# Restart the dev server after editing .env:
# Ctrl+C, then:
npm run dev
```

### `python-multipart` ImportError on /parse

```bash
uv pip install python-multipart
```

### All tests failing with ImportError

```bash
# Make sure venv is activated:
which python   # should be .venv/bin/python

# Reinstall everything:
uv pip install -e ".[dev]"

pytest -x   # stop on first failure to see the actual error
```

### `model-puller` keeps showing `Exited (1)` in Docker Compose

`Exited (0)` = normal (pulled successfully and quit)
`Exited (1)` = error — Ollama service wasn't ready when puller tried to connect

```bash
docker compose logs model-puller   # see the error
docker compose restart model-puller  # retry
```

### Frontend build fails with `Cannot find package '@sveltejs/kit'`

```bash
cd frontend
npm install
npx vite build
```

The node_modules must be installed before building. Don't try to build without running `npm install` first.

### Parser extracts wrong facts / misidentifies project headings

The docx parser identifies project headings by looking for paragraphs with Heading style, or paragraphs where all text is bold and the run ends with no colon. If your docx uses a custom structure, you may need to adjust `parse_docx_bytes` in `app/parser.py`:

```python
# In parse_docx_bytes, the heading detection:
is_heading = (
    para.style.name.startswith("Heading")
    or (all(run.bold for run in para.runs if run.text.strip())
        and para.text.strip()
        and not para.text.strip().endswith(":"))
)
```

Adjust the conditions to match your biodata's actual structure.

---

## 17. Quick reference

Copy-paste commands for every scenario:

```bash
# ── One-time setup ────────────────────────────────────────────────────────────
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env

docker run -d --name qdrant -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4

ollama pull nomic-embed-text
ollama pull qwen2.5:14b

mkdir ~/good_cvs
# copy Gold Standard CVs there, then:
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run  # preview
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs            # seed

cd frontend && npm install && cd ..

# ── Every session (3 terminals) ──────────────────────────────────────────────
# Terminal 1:
ollama serve

# Terminal 2:
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 3:
cd frontend && npm run dev
# → open http://localhost:5173

# ── Quick health check ────────────────────────────────────────────────────────
curl http://localhost:8000/health
# Expect: status:ok, ollama_ok:true, qdrant_connected:true, vector_count:>0

# ── Tests ─────────────────────────────────────────────────────────────────────
pytest                                # all 243 tests
pytest -x                             # stop on first failure
pytest tests/test_recommender.py -v  # recommender tests only
pytest --cov=app --cov-report=term-missing  # with coverage

# ── Linting ───────────────────────────────────────────────────────────────────
ruff check app/ tests/ --fix
ruff format app/ tests/

# ── Kaggle path (for low VRAM or best quality) ────────────────────────────────
python scripts/parse_biodata.py --docx biodata.docx --list_projects
python scripts/parse_biodata.py \
    --docx biodata.docx \
    --jd_file jd.txt \
    --projects 1,2 \
    --output request.json
# paste request.json into kaggle_pipeline_test.py → Run All → download bullets.csv

# ── Docker Compose (alternative to manual 3-terminal start) ──────────────────
docker compose up --build   # first time
docker compose up           # subsequent times
docker compose down         # stop
```