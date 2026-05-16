<div align="center">

# CVonRAG

### RAG-powered resume bullet optimizer for the PGDBA placement window
**IIM Calcutta · IIT Kharagpur · ISI Kolkata**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![SvelteKit](https://img.shields.io/badge/SvelteKit-2-FF3E00?style=flat-square&logo=svelte&logoColor=white)](https://kit.svelte.dev)
[![Tailwind](https://img.shields.io/badge/Tailwind-CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector_db-DC382D?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=flat-square&logo=groq&logoColor=white)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-embeddings-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![Tests](https://img.shields.io/badge/tests-395_passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](#tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--launch-orange?style=flat-square)](#)

*Upload a biodata `.docx`. Paste a job description. Get character-validated, JD-aligned bullets — with every number, tool, and outcome preserved verbatim.*

</div>

---

## Overview

A batchmate uploads a simple biodata `.docx` (project list with brief descriptions — see `docs/Vybhav Chaturvedi_Biodata.docx` for the canonical shape) and pastes a target job description. The system parses their projects, scores each against the JD, retrieves stylistic patterns from a curated Qdrant corpus, and generates polished resume-ready bullets at a target character budget — preserving every number, tool, and outcome from the biodata, while only the sentence structure is borrowed from the Gold Standard exemplars.

```
You upload:    your biodata (Word .docx)
               a job description (pasted in browser)

System does:   extracts your projects + facts (tools, metrics, outcome)
               scores every project against the JD (0–100% match)
               recommends the best 2–3 with one-line reasoning
               lets you override the selection
               generates bullets grouped by project
               validates each bullet within ±2 chars of your target (best-effort)
```

**The architectural One Rule:** numbers, tools, and outcomes come from your biodata only — never from the Gold Standard corpus. `RMSE=0.250` stays `0.250`. `0.87` is not rewritten as `87%`. See `CLAUDE.md` for the full architectural rationale.

---

## How this repo is used

| Audience | What you do |
|---|---|
| **Batchmate (end user)** | Open the deployed Vercel URL, enter your invite code, upload biodata, paste JD. No setup. |
| **Vybhav (admin)** | Curate Gold CVs, seed Qdrant, deploy via `docs/DEPLOYMENT.md`, issue invite codes. |
| **Self-hosting (anyone else)** | Clone → local dev per the steps below → adapt `docs/DEPLOYMENT.md` to your hosting choice. |

---

## Stack

| Layer | Tool |
|---|---|
| LLM (primary) | Groq (Llama 3.3 70B) — Developer paid tier |
| LLM (fallback) | OpenRouter, switchable via `LLM_PROVIDER` env var |
| Embeddings | Ollama `nomic-embed-text` (768-dim, self-hosted) |
| Vector store | Qdrant (local Docker for dev, Qdrant Cloud free tier for prod) |
| Identity / quotas | SQLite + invite codes + per-user daily caps |
| Backend | FastAPI + uvicorn (SSE streaming) |
| Frontend | SvelteKit + Tailwind |
| Tests | pytest (395 mocked tests, no live services) |

Architecture rationale and trade-offs: `docs/LLM_HOSTING.md`. Pipeline phases: `CLAUDE.md`. Per-file deep dive: `docs/DEVELOPER.md`.

---

## Local development (every contributor)

Need 4 things running. Once installed, each session is just `docker start qdrant`, `ollama serve`, `uvicorn`, `npm run dev`.

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | 24+ | [docker.com](https://docker.com) |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) |

### One-time setup

```bash
git clone https://github.com/vybhav72954/CVonRAG.git
cd CVonRAG
uv venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
cp .env.example .env

# Edit .env — set GROQ_API_KEY (required for the hosted LLM path)

# Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.4

# Embeddings — always required, even with Groq for the LLM
ollama serve
ollama pull nomic-embed-text

# Seed Qdrant with the Gold Standard corpus (admin only — see docs/DEVELOPER.md)
python scripts/ingest_pdfs.py --pdf_dir docs/good_cvs --skip_tag

# Frontend
cd frontend && npm install
```

### Each session

```bash
docker start qdrant
ollama serve
uvicorn app.main:app --reload --port 8000     # backend on :8000
cd frontend && npm run dev                    # frontend on :5173
```

Health check:
```bash
curl http://localhost:8000/health
# Expect: status=ok, vector_count>=288, llm_ok=true, embed_ok=true
```

---

## Tests

```bash
pytest                                         # 395 mocked tests, ~13s, no live services
pytest -x                                      # stop on first failure
pytest --cov=app --cov-report=term-missing     # with coverage
pytest --collect-only -q | tail -1             # verify current count
```

Test count is current as of the Eleventh-pass; verify with the last command if reading this in the future.

---

## Deployment

For the production deploy (Groq Developer + Qdrant Cloud + Railway + Vercel, ~$5–10/mo for the placement window): **see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)** for the step-by-step runbook.

Architectural decisions and trade-offs: [`docs/LLM_HOSTING.md`](docs/LLM_HOSTING.md).
End-to-end test runbook: [`docs/flow.md`](docs/flow.md).

---

## Documentation map

| File | Purpose |
|---|---|
| `README.md` (this file) | Project overview + local dev quickstart |
| `CLAUDE.md` (root) | Canonical architectural reference — read before editing any code |
| `docs/DEVELOPER.md` | Comprehensive operator guide (setup, troubleshooting, eval cycle) |
| `docs/DEPLOYMENT.md` | Step-by-step production deployment runbook |
| `docs/LLM_HOSTING.md` | Why Groq + Ollama embeddings; provider switch mechanics |
| `docs/flow.md` | Placement-day end-to-end test + admin runbook |
| `docs/AUDIT_HISTORY.md` | Full per-pass audit history (passes 1–10 archived) |

---

## License

MIT
