# CVonRAG

**RAG-powered resume bullet optimizer. 100% local, 100% free - maybe, you never know.**

Upload your CV or biodata, paste a job description, get 3–5 polished, character-count-validated bullets ready to paste. Numbers and metrics are preserved exactly. Style comes from a Gold Standard corpus of strong CV bullets retrieved via vector search.

No OpenAI. No Anthropic. No paid APIs. Runs entirely on your machine.

---

## How it works

```
Your CV (.docx/.pdf)
    ↓  parser extracts projects + facts
    ↓  you select projects, optionally edit facts
Job Description (pasted in browser)
    ↓  LLM analyses JD tone + keywords
    ↓  RAG retrieves style exemplars from Qdrant
    ↓  5-phase pipeline: score → retrieve → generate → validate → stream
3–5 bullets, char-count validated (±2), streamed live
```

**Stack:**
| Layer | Tool |
|---|---|
| LLM + Embeddings | Ollama + Qwen2.5:14b + nomic-embed-text |
| Vector Store | Qdrant |
| Backend | FastAPI + SSE streaming |
| Frontend | SvelteKit + Tailwind |
| Package manager | uv |

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.12 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | latest | [docker.com](https://docker.com) |
| Ollama | latest | [ollama.com](https://ollama.com/download) |
| Node.js | ≥ 20 | [nodejs.org](https://nodejs.org) |

---

## Getting started

### Step 1 — Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cvonrag.git
cd cvonrag

uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

cp .env.example .env           # defaults work for local dev — no changes needed
```

### Step 2 — Start Ollama and pull models (one-time, ~8 GB)

```bash
ollama serve                   # in a separate terminal, or run as a daemon

ollama pull qwen2.5:14b        # the generation model (~8 GB)
ollama pull nomic-embed-text   # the embedding model (~300 MB)
```

If your machine has < 12 GB RAM, use `qwen2.5:7b` instead.
Edit `OLLAMA_LLM_MODEL=qwen2.5:7b` in `.env`.

### Step 3 — Start Qdrant

```bash
docker run -d -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4
```

### Step 4 — Start the backend

```bash
# with venv activated:
uvicorn app.main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/health`

### Step 5 — Seed the vector store

You need a folder of good CV PDFs — these are your style references.
Any CVs with strong, quantified bullets work (IIT/IIM placement CVs, your peers' CVs, etc.).

```bash
mkdir ~/good_cvs
# copy some PDF CVs there

# preview what will be extracted (dry run, no API calls)
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run

# seed Qdrant
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs
```

Verify: `curl http://localhost:8000/health` → `"vector_count"` should be > 0.

You only need to do this once. Qdrant persists data in the Docker volume.

### Step 6 — Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Using the app

1. **Upload your CV** — drag and drop your `.docx` biodata or a `.pdf`.
   Projects are extracted automatically. Numbers are preserved exactly (RMSE=0.250 stays 0.250).

2. **Review and edit facts** — each project shows extracted facts with metrics highlighted in amber.
   Editing is optional — generation doesn't wait for it.

3. **Paste the job description** — set role type, target character count (default 130 ±2), and max bullets per project.

4. **Get your bullets** — streamed live. Each bullet shows char count, tolerance badge, and a Copy button.

---

## Running via Docker Compose (optional, for convenience)

Once everything works locally, you can start the entire stack with one command:

```bash
docker compose up --build    # first time (~10–20 min to pull Ollama models)
docker compose up            # subsequent runs
```

Services: FastAPI on :8000, Ollama on :11434, Qdrant on :6333.

---

## Running the tests

No live services needed everything is mocked.

```bash
pytest                        # 220 tests, all should pass
pytest tests/test_parser.py   # just the parser tests
pytest tests/test_api.py      # just the API endpoint tests
pytest -x                     # stop on first failure
```

---

## Project structure

```
cvonrag/
├── app/
│   ├── config.py             ← all settings via env vars
│   ├── models.py             ← Pydantic v2 schemas
│   ├── chains.py             ← 5-phase RAG pipeline
│   ├── vector_store.py       ← Qdrant + embeddings
│   ├── parser.py             ← .docx/.pdf → structured facts
│   └── main.py               ← FastAPI + /parse + /optimize endpoints
│
├── tests/                    ← 220 tests, all mocked
│
├── frontend/
│   └── src/
│       ├── routes/+page.svelte  ← 3-screen wizard UI
│       ├── lib/api.js           ← SSE client for /parse and /optimize
│       └── lib/stores.js        ← wizard state machine
│
├── scripts/
│   ├── ingest_pdfs.py        ← seed Qdrant from a folder of PDFs (run once)
│   └── parse_biodata.py      ← parse your .docx → OptimizationRequest JSON
│
├── kaggle_pipeline_test.py   ← standalone smoke test on Kaggle H100
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── DEVELOPER.md              ← full developer reference
```

---

## The two local scripts

### `scripts/ingest_pdfs.py` — seed Qdrant (run once)

Reads a folder of PDF CVs, extracts bullet points, uses Ollama to tag their style, and posts them to Qdrant. These become your style exemplars.

```bash
# Preview extraction without posting anything
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run

# Seed Qdrant
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs

# Use a faster model for tagging (7b is fine here)
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --model qwen2.5:7b
```

### `scripts/parse_biodata.py` — parse your docx (terminal alternative to the UI)

For power users who prefer the terminal. Parses your biodata docx, extracts facts via LLM, and either writes `request.json` or streams bullets directly.

```bash
# See what projects were detected
python scripts/parse_biodata.py --docx Vybhav_Chaturvedi_Biodata.docx --list_projects

# Select projects and stream bullets directly
python scripts/parse_biodata.py \
    --docx     Vybhav_Chaturvedi_Biodata.docx \
    --jd_file  amazon_mle_jd.txt \
    --projects 2,4 \
    --stream
```

---

## About the Kaggle notebook

`kaggle_pipeline_test.py` is a standalone smoke test you can run on Kaggle's free H100.

**When to use it:** if your laptop doesn't have enough RAM to run `qwen2.5:14b` (~12 GB RAM needed), you can run the full pipeline on Kaggle instead. It loads Qwen2.5 via HuggingFace Transformers, runs the 5-phase pipeline end-to-end, and exports generated bullets to a CSV.

**No training involved.** CVonRAG is pure RAG — it does not fine-tune or modify any model weights. Qwen2.5 is used as-is, via prompting. The H100 helps only for inference speed when the local machine is underpowered.

**Setup:** kaggle.com → New Notebook → upload `kaggle_pipeline_test.py` → Accelerator: GPU T4 x2 or H100 → Internet: ON → Run All.

---

## Environment variables

Copy `.env.example` to `.env`. For local dev, the defaults work without changes.

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:14b          # change to qwen2.5:7b if low on RAM
OLLAMA_EMBED_MODEL=nomic-embed-text
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=gold_standard_cvs
```

---

## Troubleshooting

**`ollama_ok: false` in /health**
```bash
curl http://localhost:11434/api/tags   # is Ollama running?
ollama pull qwen2.5:14b               # are models pulled?
```

**`qdrant_connected: false` in /health**
```bash
# Is the Qdrant container running?
docker ps | grep qdrant
# If not:
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4
```

**Bullets outside ±2 character tolerance**
Add `CHAR_LOOP_MAX_ITERATIONS=6` to `.env`. Or switch to `qwen2.5:14b` if on 7b.

**Frontend can't reach backend (CORS or network error)**
Make sure `VITE_API_URL=http://localhost:8000` is set in `frontend/.env`.

**`python-multipart` error on /parse**
```bash
uv pip install python-multipart
```

**Tests failing with ImportError**
```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

---

## License

MIT