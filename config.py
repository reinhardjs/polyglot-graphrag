"""
config.py — Central configuration for the GraphRAG Engineering Knowledge Base.

Production-grade, 100% LOCAL. ALL models run on GPU (RTX 3060, 12 GB):
  E2B extraction (:8082, systemd)   ≈ 1.5 GB
  E4B synthesis  (:8084, systemd)   ≈ 3.0 GB
  Auxiliary models (:8000, serve_gpu.py FastAPI daemon, fp16 preloaded at startup):
    Jina v3, BGE reranker-v2-m3 ≈ 4.0 GB (GLiNER lazy-loaded on demand)
  ────────────────────────────────────────────────────────────────────
  Total GPU                      ≈ 10.6 GB  (1.4 GB headroom)

Entity resolution is vector-driven via Jina v3's cross-lingual embeddings
stored in Neo4j. No hardcoded CANON_MAP — "Basis Data", "Database", and
"Base de Datos" all converge to the same entity automatically.

serve_cpu.py is a CPU-only fallback for machines without CUDA torch.
serve_gpu.py is the primary daemon (loaded at startup, GPU-resident).

The daemon exposes a unified /ask endpoint that runs the full pipeline
server-side (embed → Qdrant||Neo4j → rerank → E4B synthesis) in ONE HTTP call.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
# All runtime paths are anchored to the project root (this file's directory) so
# the project is portable and does not depend on a machine-specific absolute prefix.
# Override via environment variables if needed.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# Model directory: place GGUF files here (./models) or set MODELS_DIR.
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
# HuggingFace cache: local to the project so it travels with the repo layout.
HF_HOME    = os.environ.get("HF_HOME", os.path.join(BASE_DIR, ".cache", "hf"))
DATA_DIR   = os.path.join(BASE_DIR, "sample_data")

# ── Databases (Docker) ──────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL  = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLL_CHUNKS = "engineering_chunks"
COLL_CACHE  = "query_cache"
CACHE_THRESHOLD = 0.95   # cosine similarity above which we return cached answer

# ── Multi-Domain Collections ─────────────────────────────────────────────────
# Each domain gets its own Qdrant collection (namespace isolation, zero
# overhead). Add a new domain here + create the collection in ingest.py /
# serve_gpu.py on first use. Queries route to the collection by name.
QDRANT_COLLECTIONS = {
    "engineering":  "engineering_chunks",
    "legal":        "legal_chunks",
    "hospitality":  "hospitality_chunks",
    "accounting":   "accounting_chunks",
    "medical":      "medical_chunks",
}
QDRANT_COLLECTION_DEFAULT = "engineering_chunks"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "ragpassword123"

# ── GPU Auxiliary Daemon (FastAPI) ──────────────────────────────────────────
# serve_gpu.py preloads Jina v3 and BGE reranker on GPU at startup.
# GLiNER is lazy-loaded only on first /extract_graph call (ingest fallback).
# Override with RAG_DAEMON_URL env var for remote agents.
DAEMON_URL = os.environ.get("RAG_DAEMON_URL", "http://127.0.0.1:8000")
DAEMON_EMBED_LATE  = f"{DAEMON_URL}/embed_late"
DAEMON_EMBED_QUERY = f"{DAEMON_URL}/embed_query"
DAEMON_RERANK      = f"{DAEMON_URL}/rerank"
DAEMON_EXTRACT     = f"{DAEMON_URL}/extract_graph"
DAEMON_ASK         = f"{DAEMON_URL}/ask"

# ── GPU LLM Endpoints (llama.cpp / OpenAI-compatible, via systemd) ───────────
# OVERRIDE any of these to swap models without touching code.
# Extraction: smaller model for structured JSON entity/edge output.
EXTRACTION_LLM_BASE_URL = "http://localhost:8082/v1"
EXTRACTION_LLM_API_KEY  = "not-needed"
EXTRACTION_LLM_MODEL    = "gemma-4-E2B_q4_0-it.gguf"

# V3.0 extraction mode:
#   "llm"           — full-doc single-pass extraction with E2B (89% precision)
#   "index_routing" — Hybrid GLiNER (entities) → Qwen (relation classification, 20% precision)
#   "hybrid"        — GLiNER (entities) → E2B (relation class, 100% precision) [RECOMMENDED]
#   "sliding_window" — sentence-boundary chunked extraction with coref
#                     resolution for documents >4096 tokens
# Override at runtime: EXTRACTION_MODE=sliding_window (richer extraction for
# long docs) | hybrid | index_routing | llm (single-pass).
# Default is "sliding_window" — richest extraction (per-window LLM + GLiNER,
# parallelized). For speed on huge corpora, set EXTRACTION_MODE=llm.
EXTRACTION_MODE = os.environ.get("EXTRACTION_MODE", "sliding_window")

# Parallel sliding-window extraction: number of windows processed concurrently.
# Each window does GLiNER + E2B calls; E2B shares ONE KV cache across parallel
# slots, so this MUST stay within E2B's context budget. With E2B launched at
# --ctx-size 32768, 4 workers gives ~2x faster extraction than sequential without
# KV overflow (HTTP 500). Lower to 1 if you hit 500s; raise on bigger GPUs.
SW_EXTRACT_WORKERS = int(os.environ.get("SW_EXTRACT_WORKERS", "4"))

# Extraction context — how much document text to feed the LLM at once.
# The E2B model supports 128K context, so 32K chars (~8K tokens) is safe
# for even long post-mortems. The remaining ~120K tokens are overhead for
# the JSON output and concurrent slots via unified KV pool.
EXTRACTION_CHAR_LIMIT = int(os.environ.get("EXTRACTION_CHAR_LIMIT", "32000"))
EXTRACTION_MAX_TOKENS = int(os.environ.get("EXTRACTION_MAX_TOKENS", "4096"))

# Synthesis: larger reasoning model for answer generation from context.
SYNTHESIS_LLM_BASE_URL = "http://localhost:8084/v1"
SYNTHESIS_LLM_API_KEY  = "not-needed"
SYNTHESIS_LLM_MODEL    = "gemma-4-E4B-it-QAT-Q4_0.gguf"

# Legacy single-LLM alias (kept for compatibility)
LLM_BASE_URL = SYNTHESIS_LLM_BASE_URL
LLM_API_KEY  = SYNTHESIS_LLM_API_KEY
LLM_MODEL    = SYNTHESIS_LLM_MODEL

# ── Auxiliary Model Identities & Behavior Flags ────────────────────────────────
# Each model has an identity (HuggingFace name) AND behavior flags that tell
# the daemon HOW to load and use it. This is what makes the system truly
# model-agnostic — swap any model by changing these, no code changes needed.
#
# SUPPORTED EMBEDDING MODELS:
#   jina-embeddings-v3:  dim 1024, tasks: retrieval.passage/query, trust_remote
#   jina-embeddings-v4:  dim 2048, tasks: retrieval/text-matching/code, needs
#                        torch≥2.6 + transformers≥4.52 + peft≥0.15.2 + maybe flash_attn
#   embeddinggemma-300m: dim 768, vanilla SentenceTransformer (no task, no trust_remote)
#   Any SentenceTransformer-compatible model on HuggingFace
#
# To swap: change EMBED_MODEL_NAME + VECTOR_DIM + the flags below + re-ingest.

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = "jinaai/jina-embeddings-v3"

# Behavior flags: set these per-model. The daemon reads them at startup.
EMBED_TRUST_REMOTE    = True    # Jina needs this; most others (BGE, Gemma) don't
EMBED_USE_HALF        = True    # fp16 conversion — saves VRAM, boosts speed
EMBED_TASK_PASSAGE    = "retrieval.passage"  # Jina-specific task adapter (None = vanilla)
EMBED_TASK_QUERY      = "retrieval.query"    # Jina-specific; set None for non-Jina
EMBED_MAX_LENGTH      = 32768   # max input tokens (None = model default)
EMBED_MATRYOSHKA_DIM  = None    # for models supporting MRL (Jina v4: 128-2048)

# ── Reranker ───────────────────────────────────────────────────────────────────
# GPU daemon (serve_gpu.py): full model, no pool cap. CUDA tensor cores handle
# the full 568M params on 27+ docs at ~61ms — no compromise needed.
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_USE_HALF    = True    # fp16 conversion

# CPU fallback (serve_cpu.py): lighter model + capped pool. BGE-base at 278M
# params is 2.7× faster than v2-m3 on CPU. Capping the fused pool to 10 docs
# keeps rerank under 500ms so the i5 doesn't freeze.
RERANK_MODEL_NAME_CPU = "BAAI/bge-reranker-v2-m3"   # same as GPU, capped pool delivers speed
RERANK_CPU_POOL_CAP   = 15  # sweet spot: 4.5/5 avg overlap, 2.2s /ask on CPU

# ── GLiNER (NER fallback) ──────────────────────────────────────────────────────
GLINER_MODEL_NAME = "urchade/gliner_multi-v2.1"

# ── Extraction LLM Behavior ────────────────────────────────────────────────────
# Some LLMs (Gemma E4B, Gemma 12B) are "reasoning" models that put output in
# `reasoning_content` instead of `content`. Set True if your extraction model
# does this so the daemon reads the right field.
EXTRACTION_READS_REASONING = False

# ── Prompt Templates (per-model overridable) ──────────────────────────────────
# Extraction prompt: sent to the EXTRACTION_LLM. {doc_id} and {text} are
# formatted in. Change this if you swap the extraction model and it needs a
# different prompt style (e.g. some models prefer "system" + "user" roles).
EXTRACTION_PROMPT = (
    "Extract a knowledge graph from the engineering document below.\n"
    "Entities: extract names EXACTLY as they appear — do NOT translate.\n"
    "  (e.g. 'Basis Data' stays 'Basis Data', 'Database' stays 'Database').\n"
    "Relationships: use one of ASSOCIATED_WITH, DEPENDS_ON, IMPACTS,\n"
    "  AUTHORED, REFERENCES, FIXES.\n"
    "Return ONLY valid JSON, no prose, no markdown:\n"
    '{{"nodes":[{{"id":"ExactEntityName","type":"Microservice|Database|API|'
    'Metric|Developer|Framework|Component|Bug|PR|ADR"}}],'
    '"edges":[{{"source":"entity_a","target":"entity_b",'
    '"type":"ASSOCIATED_WITH|DEPENDS_ON|IMPACTS|AUTHORED|REFERENCES|FIXES"}}]}}\n'
    "Document ({doc_id}):\n{text}"
)

# ── Context / token budgeting ────────────────────────────────────────────────
MAX_TOKENS_CONTEXT = 4096
LLM_MAX_TOKENS_OUT = 1024

# ── Embeddings ───────────────────────────────────────────────────────────────
VECTOR_DIM = 1024          # jina-embeddings-v3 output dim
RERANK_TOP_K = 5           # keep top-5 contexts after rerank
QDRANT_SEARCH_TOP_K = 10   # vector candidates from Qdrant
GRAPH_HOPS = 2             # k-hop subgraph from Neo4j
GRAPH_PRUNE_TOP_N = 10     # Phase 2: cap subgraph to Top-N nodes (context window guard)
GRAPH_PRUNE_STRATEGY = "degree"  # "degree" | "pagerank" | "none" — neighbor ranking

# ── CRAG (Phase 3): Corrective RAG & adaptive routing ────────────────────────
CRAG_USE_LLM_ROUTER = False  # True → confirm route + rewrite with E4B (slower, smarter)

# ── GLiNER target labels (graph extraction schema) ────────────────────────────
GLINER_LABELS = [
    "Microservice", "Database", "API", "Metric",
    "Developer", "Framework", "Component", "Bug", "PR", "ADR",
]
GLINER_THRESHOLD = 0.4

# ── Entity Resolution (vector-driven, language-agnostic) ─────────────────────
# Replaces the hardcoded CANON_MAP. Entities extracted in their native language
# (e.g. "Basis Data", "Database", "Base de Datos") are matched via Jina v3's
# cross-lingual embeddings stored in Neo4j's vector index. The LLM is instructed
# to extract verbatim — no translation required. Zero hardcoded translations.
ENTITY_RESOLUTION_THRESHOLD = 0.88   # cosine similarity for entity merger
ENTITY_VECTOR_INDEX = "entity_vector_idx"

# ── Domain config (v0.x) ──────────────────────────────────────────────────────
# Domain schemas live in `domain_config.yaml` (YAML), consumed by
# `domain_loader.py`. All callers use `domain_loader.get_domain(name)`.

