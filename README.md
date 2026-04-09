# CVonRAG

**RAG-powered resume bullet optimizer. 100% local, 100% free.**

Upload your biodata, paste a job description, and the AI recommends which of your projects to highlight, then generates 3–5 polished, character-count-validated bullets, grouped by project and ready to paste.

No OpenAI. No Anthropic. No paid APIs. Runs entirely on your machine.

---

## What it does

```
You upload:   your biodata (.docx or .pdf)
              a job description (pasted in browser)

System does:  extracts ALL your projects and facts automatically
              scores every project against the JD (0–100% match)
              recommends the best 2–3 with one-line reasoning
              lets you override the selection
              generates bullets grouped by project
              validates each bullet to ±2 characters of your target
```

Numbers are preserved exactly. RMSE=0.250 stays 0.250. The style comes from a curated Gold Standard corpus — never the content.

---

## Stack

| Layer | Tool |
|---|---|
| LLM + Embeddings | Ollama + Qwen2.5 + nomic-embed-text |
| Vector Store | Qdrant |
| Backend | FastAPI + SSE streaming |
| Frontend | SvelteKit + Tailwind |

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| Docker | 24+ | [docker.com](https://docker.com) |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) |

---

## Getting started (admin setup — run once)

These steps are done by whoever runs the server. Users just open the browser.

### 1. Install and configure

```bash
git clone https://github.com/YOUR_USERNAME/cvonrag.git
cd cvonrag
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env    # defaults work for local dev
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4
```

### 3. Pull Ollama models

```bash
ollama serve

# Pick based on your RAM:
ollama pull qwen2.5:14b       # >= 12 GB free RAM
# ollama pull qwen2.5:7b      # 8-12 GB free RAM

ollama pull nomic-embed-text   # always — for embeddings
```

**Low VRAM?** See [DEVELOPER.md §B](DEVELOPER.md#section-b--kaggle-h100-inference-recommended-for-low-vram) — run inference on a free Kaggle H100 instead.

### 4. Seed the style corpus (once)

The system needs a collection of Gold Standard CVs to learn bullet style from. These are your curated CVs — users never upload or access them. They teach the system sentence structure and style patterns only; user content is never drawn from them.

```bash
mkdir ~/good_cvs
# place 5–20 good CV PDFs here (IIT/IIM placement CVs, strong peers' CVs, etc.)

python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs --dry_run   # preview
python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs             # seed
```

Verify: `curl http://localhost:8000/health` → `"vector_count"` should be > 0. Do this once.

### 5. Start the backend and frontend

```bash
# Backend:
uvicorn app.main:app --reload --port 8000

# Frontend (in another terminal):
cd frontend && npm install && npm run dev
# -> http://localhost:5173
```

---

## User flow (what batchmates see)

Users visit `http://localhost:5173` and:

**Screen 1 — Upload CV**
Drag and drop a `.docx` biodata or `.pdf`. All projects are extracted automatically. Metrics are highlighted in amber — they're preserved exactly. Editing facts is optional and non-blocking.

**Screen 2 — Paste JD → AI Recommendation**
Paste the full job description, click "Analyse JD". The system scores every project against the JD and shows:
- A ranked list with match percentage and one-line reasoning per project
- The top 2–3 pre-selected, rest shown as "also available"
- A toggle to override the selection

Then click "Generate Bullets".

**Screen 3 — Results**
Bullets stream live, grouped by project. Each bullet shows:
- Character count and ±2 tolerance badge
- Number of iterations taken to hit the target
- Individual Copy button
- Copy All at the top

---

## Running the tests

```bash
pytest        # 243 tests, no live services needed, ~9 seconds
```

---

## Architecture: the 5-phase pipeline

```
POST /parse
  -> parser.py extracts projects + facts via LLM (SSE stream)

POST /recommend
  -> recommender.py scores projects vs JD
  -> returns ranked list with match % and reasons

POST /optimize
  Phase 1: OptimizationRequest validated
  Phase 2: SemanticMatcher — JD analysis + fact scoring
  Phase 3: Qdrant — retrieve style exemplars (top-K by embedding similarity)
  Phase 4: BulletAlchemist — generate + ±2 char-limit loop (up to 4 iterations)
  Phase 5: SSE stream — tokens + bullets + metadata -> browser
```

**Content/style firewall:** CoreFacts (your numbers, tools, outcomes) are immutable. StyleExemplars (from Qdrant) provide sentence patterns only. The two never mix.

---

## Low VRAM / best quality: Kaggle H100

If Ollama is slow or in low-VRAM mode, run the heavy inference on Kaggle's free H100 instead. Qwen2.5-72B in 4-bit quantisation gives noticeably better bullet quality than local 14b.

See [DEVELOPER.md §B](DEVELOPER.md#section-b--kaggle-h100-inference-recommended-for-low-vram) for step-by-step.

---

## Troubleshooting

**`ollama_ok: false` in `/health`** → run `ollama serve` and `ollama pull qwen2.5:14b`

**`vector_count: 0`** → seed Qdrant: `python scripts/ingest_pdfs.py --pdf_dir ~/good_cvs`

**Bullets outside ±2** → add `CHAR_LOOP_MAX_ITERATIONS=6` to `.env`

**Frontend network error** → check `frontend/.env` has `VITE_API_URL=http://localhost:8000`

Full troubleshooting: [DEVELOPER.md §H](DEVELOPER.md#section-h--troubleshooting)

---

## License

MIT