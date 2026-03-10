# =============================================================================
# CVonRAG — Kaggle H100 Pipeline Notebook
# =============================================================================
# Assumes: H100 GPU (80GB VRAM) or H100 x2 (160GB)
#
# What this notebook does (all 5 phases):
#   Phase 2 → JD analysis + fact scoring           (Qwen2.5-72B-Instruct)
#   Phase 3 → Style exemplar retrieval             (Qdrant in-memory)
#   Phase 4 → Bullet generation + char-limit loop  (Qwen2.5-72B-Instruct)
#   Phase 5 → Batch output + CSV export
#
# Setup:
#   Kaggle → New Notebook → Accelerator: GPU H100 (or P100 if H100 unavailable)
#   Internet: ON
#   Run all cells top to bottom.
# =============================================================================


# %% [markdown]
# ## Cell 1 — Install Dependencies

# %%
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("qdrant-client==1.12.1")
pip(
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "einops",           # required by Qwen2.5
    "tiktoken",         # required by Qwen2.5 tokenizer
    "sentence-transformers>=3.0.0",
)

print("✅ Dependencies installed")


# %% [markdown]
# ## Cell 2 — GPU Detection + Model Selection

# %%
import torch

assert torch.cuda.is_available(), "No GPU detected — check Kaggle accelerator settings"

n_gpus   = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
vram_per = torch.cuda.get_device_properties(0).total_memory / 1e9
vram_total = vram_per * n_gpus

print(f"✅ {n_gpus}× {gpu_name}  |  {vram_per:.0f} GB each  |  {vram_total:.0f} GB total")

# ── Model selection based on available VRAM ───────────────────────────────────
#   H100 80GB  → 72B in bfloat16 (~144GB across 2 GPUs, or 4-bit on 1 GPU)
#   H100 80GB  → 32B in bfloat16 (~65GB on 1 GPU — fits easily)
#   H100 80GB  → 14B in bfloat16 (~28GB on 1 GPU — trivial)

if vram_total >= 140:
    LLM_MODEL    = "Qwen/Qwen2.5-72B-Instruct"
    LOAD_IN_4BIT = False    # fits in bfloat16 across 2× H100
    print("→ 72B in bfloat16 (dual H100 mode)")
elif vram_total >= 70:
    LLM_MODEL    = "Qwen/Qwen2.5-72B-Instruct"
    LOAD_IN_4BIT = True     # 4-bit quantisation fits on single H100
    print("→ 72B in 4-bit (single H100 mode)")
elif vram_total >= 60:
    LLM_MODEL    = "Qwen/Qwen2.5-32B-Instruct"
    LOAD_IN_4BIT = False
    print("→ 32B in bfloat16")
else:
    LLM_MODEL    = "Qwen/Qwen2.5-14B-Instruct"
    LOAD_IN_4BIT = False
    print("→ 14B in bfloat16 (fallback)")

EMBED_MODEL  = "nomic-ai/nomic-embed-text-v1"
EMBED_DIM    = 768
print(f"\nLLM   : {LLM_MODEL}  (4-bit={LOAD_IN_4BIT})")
print(f"Embed : {EMBED_MODEL}  ({EMBED_DIM}-dim)")


# %% [markdown]
# ## Cell 3 — Load Qwen2.5 (LLM)

# %%
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch

print(f"Loading {LLM_MODEL} ...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL,
    trust_remote_code=True,
    padding_side="left",
)

if LOAD_IN_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16,   # bfloat16 is optimal on H100
        device_map="auto",            # distributes across all available GPUs
        trust_remote_code=True,
    )

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.05,
    return_full_text=False,
)

print(f"✅ {LLM_MODEL} loaded")
# Show actual memory usage
for i in range(n_gpus):
    used  = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"   GPU {i}: {used:.1f} / {total:.0f} GB used")


# %% [markdown]
# ## Cell 4 — Load Embedding Model (nomic-embed-text)

# %%
from sentence_transformers import SentenceTransformer

print(f"Loading {EMBED_MODEL} ...")

# trust_remote_code=True required for nomic-embed-text
embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cuda",
    trust_remote_code=True,
)

def embed(text: str) -> list[float]:
    return embedder.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    """H100 can handle large batches — use batch_size=64 comfortably."""
    return embedder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

# Sanity check
test_vec = embed("test embedding")
print(f"✅ {EMBED_MODEL} loaded — vector dim: {len(test_vec)}")
assert len(test_vec) == EMBED_DIM, f"Expected {EMBED_DIM}-dim, got {len(test_vec)}"


# %% [markdown]
# ## Cell 5 — LLM Helper (Handles Qwen2.5 Chat Template)

# %%
import json, re

def llm_chat(
    system: str,
    user_message: str,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
) -> str:
    """
    Single-turn Qwen2.5 chat call.
    Strips <think>...</think> reasoning blocks from output.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    out = llm_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]["generated_text"]

    # Remove extended reasoning blocks
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    return out.strip()


def llm_multiturn(
    system: str,
    history: list[dict],
    temperature: float = 0.3,
    max_new_tokens: int = 512,
) -> str:
    """
    Multi-turn chat — used by the char-limit correction loop.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    messages = [{"role": "system", "content": system}] + history
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    out = llm_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]["generated_text"]
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    return out.strip()


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# Sanity check
resp = llm_chat(
    system="Reply with exactly the word: READY",
    user_message="Are you ready?",
    max_new_tokens=10,
)
print(f"LLM sanity check → '{resp}'")
print("✅ LLM helper working")


# %% [markdown]
# ## Cell 6 — Qdrant In-Memory + Ingest Gold Standard Bullets

# %%
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "gold_standard_cvs"

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)

# ── Gold Standard bullets ─────────────────────────────────────────────────────
# These are STYLE exemplars only. Add your own high-quality bullets here.
# The more diverse and high-quality, the better the style retrieval.
GOLD_STANDARD_BULLETS = [
    # Data Science / Forecasting
    {
        "text": "• Enhanced forecast accuracy using ARIMAX and VECM | Reduced RMSE by 13.5% for aluminium price prediction",
        "role_type": "data_science",
        "uses_separator": "|",
        "uses_arrow": False,
        "sentence_structure": "verb → method → metric → domain",
    },
    {
        "text": "• Implemented WGCNA co-expression network on 500+ gene dataset | Identified 3 hub genes linked to disease pathway",
        "role_type": "research_focused",
        "uses_separator": "|",
        "sentence_structure": "verb → method → scale → outcome",
    },
    # ML Engineering
    {
        "text": "• Handled class imbalance via undersampling; ↑ F1 score from 0.39 to 0.49 on fraud detection task",
        "role_type": "ml_engineering",
        "uses_arrow": True,
        "sentence_structure": "verb → technique → metric delta → domain",
    },
    {
        "text": "• Fine-tuned Llama-3 8B on 50K domain samples | ↑ task accuracy by 22% vs GPT-4 baseline (zero-shot)",
        "role_type": "ml_engineering",
        "uses_separator": "|",
        "uses_arrow": True,
        "sentence_structure": "verb → model → scale → comparative metric",
    },
    {
        "text": "• Built RAG pipeline using LangChain & Pinecone; ↓ hallucination rate from 34% → 8% on QA benchmark",
        "role_type": "ml_engineering",
        "uses_arrow": True,
        "sentence_structure": "verb → tool stack → metric delta → benchmark",
    },
    # Software Engineering
    {
        "text": "• Architected distributed Kafka + Spark pipeline | ↓ data latency by 340ms (p99) across 12M daily events",
        "role_type": "software_engineering",
        "uses_separator": "|",
        "uses_arrow": True,
        "sentence_structure": "verb → tool stack → metric → scale",
    },
    {
        "text": "• Refactored monolith into 8 microservices w/ FastAPI | ↑ deploy frequency 3× & ↓ incident rate by 60%",
        "role_type": "software_engineering",
        "uses_separator": "|",
        "uses_arrow": True,
        "sentence_structure": "verb → outcome → tool → dual metric",
    },
    # Quant / Finance
    {
        "text": "• Developed SLSQP-optimized portfolio allocation; ↑ Sharpe ratio 1.2 → 1.8 vs equal-weight baseline",
        "role_type": "quant_finance",
        "uses_arrow": True,
        "sentence_structure": "verb → method → metric delta → comparison",
    },
    # Agentic / LLM
    {
        "text": "• Orchestrated 6-agent LLM pipeline using CrewAI | ↓ processing time by 73% vs sequential baseline",
        "role_type": "ml_engineering",
        "uses_separator": "|",
        "uses_arrow": True,
        "sentence_structure": "verb → scale → tool → comparative metric",
    },
    {
        "text": "• Deployed FastAPI inference service w/ async batching | ↓ p99 latency from 850ms → 210ms at 500 RPS",
        "role_type": "software_engineering",
        "uses_separator": "|",
        "uses_arrow": True,
        "sentence_structure": "verb → tool → technique → metric range → scale",
    },
]

# Embed and upsert using batch embedding (H100 handles this fast)
texts  = [b["text"] for b in GOLD_STANDARD_BULLETS]
vecs   = embed_batch(texts)
points = [
    PointStruct(id=str(uuid.uuid4()), vector=vec, payload=bullet)
    for bullet, vec in zip(GOLD_STANDARD_BULLETS, vecs)
]
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ Ingested {len(points)} Gold Standard bullets into Qdrant")


# %% [markdown]
# ## Cell 7 — Phase 2: JD Analysis

# %%
SAMPLE_JD = """
We are hiring a Senior ML Engineer to own production-grade forecasting systems.

Requirements:
- Expert Python + time-series modeling (SARIMA, ARIMA, Prophet, ensemble methods)
- Deep statistical background: RMSE, MAE, ACF/PACF diagnostics
- Constrained optimization (scipy SLSQP, L-BFGS-B)
- MLOps: versioning, deployment pipelines, drift monitoring
- Nice-to-have: LangChain, GPT-4, agentic system design
"""

JD_ANALYSIS_SYSTEM = """\
Analyse this job description. Return ONLY valid JSON — no fences, no preamble.
Schema:
{
  "required_skills": [...],
  "preferred_skills": [...],
  "key_action_verbs": [...],
  "tone": "highly_quantitative|engineering_focused|leadership_focused|research_focused|balanced",
  "seniority": "junior|mid|senior|staff|principal",
  "domain_keywords": [...]
}"""

raw = llm_chat(system=JD_ANALYSIS_SYSTEM, user_message=f"JD:\n{SAMPLE_JD}", temperature=0.05)
jd_analysis = json.loads(strip_json_fences(raw))

print("✅ JD Analysis:")
print(json.dumps(jd_analysis, indent=2))


# %% [markdown]
# ## Cell 8 — Phase 2: Fact Scoring

# %%
USER_FACTS = [
    {
        "fact_id": "f-001",
        "text": "Built SARIMA(2,0,0)(1,0,0)[12] model using ACF/PACF analysis for seasonal demand forecasting",
        "tools": ["SARIMA", "statsmodels", "ACF/PACF"],
        "metrics": ["RMSE 0.250"],
        "outcome": "Best forecast accuracy for seasonal demand",
    },
    {
        "fact_id": "f-002",
        "text": "Optimized ensemble weights via constrained SLSQP, reducing RMSE from 0.31 to 0.250 (19.4% improvement)",
        "tools": ["SLSQP", "scipy"],
        "metrics": ["RMSE 0.250", "RMSE 0.31", "19.4%"],
        "outcome": "Reduced forecast error below baseline",
    },
    {
        "fact_id": "f-003",
        "text": "Architected multi-agent LLM evaluation system using LangChain + GPT-4, orchestrating 6 specialized agents; reduced evaluation time from 8-12 weeks to 5-10 days (87% faster)",
        "tools": ["LangChain", "GPT-4"],
        "metrics": ["87%", "8-12 weeks", "5-10 days"],
        "outcome": "87% faster evaluation pipeline",
    },
    {
        "fact_id": "f-004",
        "text": "Deployed FastAPI microservice serving SARIMA predictions with Redis caching; achieved <50ms p99 latency at 1000 RPS",
        "tools": ["FastAPI", "Redis", "SARIMA"],
        "metrics": ["<50ms", "p99", "1000 RPS"],
        "outcome": "Production-grade low-latency inference service",
    },
]

SCORING_SYSTEM = """\
Score each fact's relevance to the JD on 0.0–1.0.
Return ONLY valid JSON array — no fences, no preamble:
[{"fact_id": "...", "relevance_score": float, "matched_jd_keywords": [...]}]"""

raw = llm_chat(
    system=SCORING_SYSTEM,
    user_message=f"JD Analysis:\n{json.dumps(jd_analysis)}\n\nFacts:\n{json.dumps(USER_FACTS)}",
    temperature=0.05,
)
scores = json.loads(strip_json_fences(raw))
score_map = {s["fact_id"]: s for s in scores}

print("✅ Fact Relevance Scores:")
for s in sorted(scores, key=lambda x: x["relevance_score"], reverse=True):
    bar = "█" * int(s["relevance_score"] * 20)
    print(f"  {s['fact_id']}  {s['relevance_score']:.2f}  {bar}  → {s['matched_jd_keywords']}")


# %% [markdown]
# ## Cell 9 — Phase 3: Style Exemplar Retrieval

# %%
# Build query from JD + top facts
top_facts = sorted(USER_FACTS, key=lambda f: score_map.get(f["fact_id"], {}).get("relevance_score", 0), reverse=True)
query_text = SAMPLE_JD[:400] + " " + " ".join(f["text"] for f in top_facts[:3])

exemplars = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=embed(query_text),
    limit=4,
    with_payload=True,
)

print("✅ Retrieved Style Exemplars (STYLE REFERENCE ONLY):")
for i, ex in enumerate(exemplars, 1):
    print(f"\n  {i}. [sim={ex.score:.3f}]  {ex.payload['text']}")
    print(f"     structure : {ex.payload.get('sentence_structure', 'N/A')}")
    print(f"     separator : {ex.payload.get('uses_separator', 'none')}  "
          f"arrows: {ex.payload.get('uses_arrow', False)}")


# %% [markdown]
# ## Cell 10 — Phase 4: The Alchemist (Generation + Char-Limit Loop)

# %%
def format_facts(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        line = f"• [{f['fact_id']}] {f['text']}"
        if f.get("metrics"):
            line += f"\n  └─ metrics : {', '.join(f['metrics'])}"
        if f.get("tools"):
            line += f"\n  └─ tools   : {', '.join(f['tools'])}"
        lines.append(line)
    return "\n".join(lines)


def format_exemplars(hits) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit.payload['text']}")
        if s := hit.payload.get("sentence_structure"):
            lines.append(f"   [structure: {s}]")
    return "\n".join(lines)


def clean_bullet(text: str, prefix: str = "•") -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[-–—*]\s*", "", text.strip())
    text = text.strip()
    if not text.startswith(prefix):
        text = f"{prefix} {text.lstrip('•·-– ')}"
    return text.strip()


# ── System prompt: the Content/Style Firewall ─────────────────────────────────
ALCHEMIST_SYSTEM = """\
You are CVonRAG's Bullet Alchemist — the world's most precise resume bullet-point writer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT / STYLE FIREWALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT (from User JSON ONLY):
  • Every number, %, financial figure — preserve EXACTLY
  • Every named tool, algorithm, library
  • The actual outcome the user achieved

STYLE (from Exemplars ONLY):
  • Separators  |  •  ;  and special chars  ↑  ↓  →  &  w/
  • Sentence architecture: VERB → TOOL → METRIC → IMPACT
  • Abbreviation style

ABSOLUTE PROHIBITIONS:
  ✗ NEVER copy company names or domain jargon from Exemplars
  ✗ NEVER alter any number. 87% stays 87%. 0.250 stays 0.250.
  ✗ NEVER invent steps or outcomes not in the JSON

OUTPUT: ONE plain-text bullet line. No markdown. No explanation."""

GENERATION_TEMPLATE = """\
Role Type   : {role_type}
JD Tone     : {tone}
JD Keywords : {keywords}

━━ USER JSON FACTS (CONTENT — all numbers/tools/outcomes from here ONLY) ━━
{facts}

━━ GOLD STANDARD EXEMPLARS (STYLE REFERENCE ONLY — do NOT copy content) ━━
{exemplars}

━━ FEW-SHOT EXAMPLES ━━
Fact   : "Built SARIMA(2,0,0)(1,0,0)[12] model using ACF/PACF analysis... reduced RMSE to 0.250"
Style  : "Enhanced forecast accuracy using ARIMAX | Reduced RMSE by 13.5%"
Output : • Built SARIMA(2,0,0)(1,0,0)[12] model via ACF/PACF analysis | Optimized ensemble weights via constrained SLSQP, ↑ predictive accuracy by reducing RMSE to 0.250

Fact   : "Architected multi-agent LLM system using LangChain + GPT-4... 87% faster (8-12 weeks to 5-10 days)"
Style  : "Handled class imbalance via undersampling; ↑ F1 score from 0.39 to 0.49"
Output : • Architected multi-agent LLM system using LangChain & GPT-4 | Orchestrated 6 specialized agents, ↑ evaluation speed by 87% (8-12 weeks to 5-10 days)

━━ TASK ━━
Bullet prefix : "{prefix}"
Target length : {target} chars  (acceptable: {lower}–{upper})

Write the bullet now (single line, no explanation):"""

EXPAND_TEMPLATE = """\
Current bullet ({cur} chars):
{bullet}

This is {delta} chars SHORT of the target {target} (range: {lower}–{upper}).

Expand by:
  • Replacing short words with longer technical synonyms already in the JSON
  • Adding a missing tool, step, or metric from the JSON that was omitted
  • Spelling out an abbreviation you used

FIREWALL: all new content must come from the original JSON facts.
Output only the revised bullet (single line):"""

COMPRESS_TEMPLATE = """\
Current bullet ({cur} chars):
{bullet}

This is {delta} chars OVER the target {target} (range: {lower}–{upper}).

Compress by:
  • & instead of "and", w/ instead of "with"
  • ↑ instead of "increased", ↓ instead of "decreased", → for "to"
  • ~ instead of "approximately", % stays, remove "successfully"/"effectively"
  • Remove filler articles ("the", "a") where meaning is preserved

FIREWALL: do NOT remove or alter any number, percentage, or named tool.
Output only the revised bullet (single line):"""


def generate_bullet(
    facts: list[dict],
    exemplars_hits,
    jd_analysis: dict,
    target: int = 130,
    tolerance: int = 2,
    max_iterations: int = 5,
    prefix: str = "•",
    verbose: bool = True,
) -> dict:
    """
    Full char-limit loop.
    Returns: {text, char_count, within_tolerance, iterations}
    """
    lower, upper = target - tolerance, target + tolerance
    keywords     = ", ".join(jd_analysis.get("required_skills", [])[:8])
    tone         = jd_analysis.get("tone", "balanced")

    initial_prompt = GENERATION_TEMPLATE.format(
        role_type="ml_engineering",
        tone=tone,
        keywords=keywords,
        facts=format_facts(facts),
        exemplars=format_exemplars(exemplars_hits),
        prefix=prefix,
        target=target,
        lower=lower,
        upper=upper,
    )

    history = [{"role": "user", "content": initial_prompt}]
    raw     = llm_multiturn(system=ALCHEMIST_SYSTEM, history=history)
    bullet  = clean_bullet(raw, prefix)
    history.append({"role": "assistant", "content": bullet})

    best = {"text": bullet, "char_count": len(bullet), "iterations": 1, "within_tolerance": False}

    for iteration in range(1, max_iterations + 1):
        length  = len(bullet)
        within  = lower <= length <= upper
        delta_b = abs(length - target)

        if verbose:
            status = "✅ HIT" if within else ("📏 SHORT" if length < lower else "📏 LONG")
            print(f"  iter {iteration}: {length} chars  {status}  delta={length-target:+d}")

        if delta_b < abs(best["char_count"] - target):
            best = {"text": bullet, "char_count": length, "iterations": iteration, "within_tolerance": within}

        if within:
            return best

        if iteration == max_iterations:
            if verbose:
                print(f"  ⚠️  Loop exhausted — returning closest ({best['char_count']} chars)")
            return best

        # Build correction message
        delta = abs(length - target)
        correction = (
            EXPAND_TEMPLATE if length < lower else COMPRESS_TEMPLATE
        ).format(
            cur=length, bullet=bullet, delta=delta,
            target=target, lower=lower, upper=upper,
        )
        history.append({"role": "user",      "content": correction})
        raw    = llm_multiturn(system=ALCHEMIST_SYSTEM, history=history)
        bullet = clean_bullet(raw, prefix)
        history.append({"role": "assistant", "content": bullet})

    return best


# %% [markdown]
# ## Cell 11 — Run Full Pipeline

# %%
print("=" * 70)
print(f"CVonRAG Pipeline Test  |  Model: {LLM_MODEL.split('/')[-1]}")
print("=" * 70)

bullets_out = []

for fact in sorted(USER_FACTS, key=lambda f: score_map.get(f["fact_id"], {}).get("relevance_score", 0), reverse=True):
    fact_score = score_map.get(fact["fact_id"], {})
    print(f"\n📌 [{fact['fact_id']}]  relevance={fact_score.get('relevance_score', '?'):.2f}")
    print(f"   Input: {fact['text'][:80]}...")

    result = generate_bullet(
        facts=[fact],
        exemplars_hits=exemplars,
        jd_analysis=jd_analysis,
        target=130,
        tolerance=2,
        max_iterations=5,
        verbose=True,
    )
    bullets_out.append({**result, "fact_id": fact["fact_id"]})

    print(f"\n   ✨ OUTPUT : {result['text']}")
    print(f"      chars  : {result['char_count']} / 130  "
          f"({'✅ within ±2' if result['within_tolerance'] else '⚠️ outside tolerance'})")


# %% [markdown]
# ## Cell 12 — Firewall Audit
# Verify every user metric and tool is preserved in the output — no hallucinated numbers.

# %%
print("\n" + "=" * 70)
print("CONTENT FIREWALL AUDIT")
print("=" * 70)

all_pass = True
for item in bullets_out:
    fact = next(f for f in USER_FACTS if f["fact_id"] == item["fact_id"])
    print(f"\n[{item['fact_id']}] {item['text']}")

    for metric in fact.get("metrics", []):
        present = metric in item["text"]
        icon    = "✅" if present else "❌ VIOLATION"
        if not present:
            all_pass = False
        print(f"  metric '{metric}': {icon}")

    for tool in fact.get("tools", []):
        present = tool in item["text"]
        icon    = "✅" if present else "⚠️  omitted (may be OK if irrelevant)"
        print(f"  tool   '{tool}': {icon}")

print(f"\n{'✅ ALL METRICS PRESERVED' if all_pass else '❌ METRIC VIOLATIONS DETECTED'}")


# %% [markdown]
# ## Cell 13 — Export Results

# %%
import pandas as pd

df = pd.DataFrame([
    {
        "fact_id"          : b["fact_id"],
        "bullet"           : b["text"],
        "char_count"       : b["char_count"],
        "target"           : 130,
        "delta"            : b["char_count"] - 130,
        "within_tolerance" : b["within_tolerance"],
        "iterations"       : b["iterations"],
    }
    for b in bullets_out
])

output_path = "/kaggle/working/cvonrag_bullets.csv"
df.to_csv(output_path, index=False)

print("✅ Saved to", output_path)
print()
print(df.to_string(index=False))

# Summary stats
within = df["within_tolerance"].sum()
print(f"\nSummary: {within}/{len(df)} bullets within ±2 char tolerance")
print(f"Avg delta: {df['delta'].abs().mean():.1f} chars")
print(f"Avg iterations: {df['iterations'].mean():.1f}")
