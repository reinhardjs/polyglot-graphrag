"""
config.py — Central configuration for the GraphRAG Engineering Knowledge Base.

Production-grade, 100% LOCAL. ALL models run on GPU (RTX 3060, 12 GB):
  E2B extraction (:8082, systemd)   ≈ 1.5 GB
  E4B synthesis  (:8084, systemd)   ≈ 3.0 GB
  Auxiliary models (:8000, serve_gpu.py FastAPI daemon, fp16 preloaded at startup):
    Jina v3, MiniLM, BGE reranker-v2-m3 ≈ 4.1 GB (GLiNER lazy-loaded on demand)
  ────────────────────────────────────────────────────────────────────
  Total GPU                            ≈ 10.7 GB  (1.5 GB headroom)

serve_cpu.py is a CPU-only fallback for machines without CUDA torch.
serve_gpu.py is the primary daemon (loaded at startup, GPU-resident).

The daemon now exposes a unified /ask endpoint that runs the full pipeline
server-side (embed → Qdrant||Neo4j → rerank → E4B synthesis) in ONE HTTP call.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
# All heavy artifacts live on the 458 GB NVMe.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HF_HOME    = "/mnt/data-970-plus/hf_cache"
DATA_DIR   = os.path.join(BASE_DIR, "sample_data")

# ── Databases (Docker) ──────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL  = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLL_CHUNKS = "engineering_chunks"
COLL_CACHE  = "query_cache"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "ragpassword123"

# ── GPU Auxiliary Daemon (FastAPI) ──────────────────────────────────────────
# serve_gpu.py preloads Jina v3, MiniLM, BGE reranker on GPU at startup.
# GLiNER is lazy-loaded only on first /extract_graph call (ingest fallback).
# Override with RAG_DAEMON_URL env var for remote agents.
DAEMON_URL = os.environ.get("RAG_DAEMON_URL", "http://127.0.0.1:8000")
DAEMON_EMBED_LATE  = f"{DAEMON_URL}/embed_late"
DAEMON_EMBED_QUERY = f"{DAEMON_URL}/embed_query"
DAEMON_RERANK      = f"{DAEMON_URL}/rerank"
DAEMON_EXTRACT     = f"{DAEMON_URL}/extract_graph"
DAEMON_ASK         = f"{DAEMON_URL}/ask"

# ── GPU LLM Endpoints (Gemma 4 family, separate small models via llama.cpp) ──
# Extraction LLM  -> Gemma 4 E2B QAT (graph extraction via LLM) — on :8082
EXTRACTION_LLM_BASE_URL = "http://localhost:8082/v1"
EXTRACTION_LLM_API_KEY  = "not-needed"
EXTRACTION_LLM_MODEL    = "gemma-4-E2B-it-QAT-Q4_0.gguf"

# Synthesis LLM    -> Gemma 4 E4B QAT (answer generation) — on :8084
SYNTHESIS_LLM_BASE_URL = "http://localhost:8084/v1"
SYNTHESIS_LLM_API_KEY  = "not-needed"
SYNTHESIS_LLM_MODEL    = "gemma-4-E4B-it-QAT-Q4_0.gguf"

# Legacy single-LLM alias (kept for compatibility)
LLM_BASE_URL = SYNTHESIS_LLM_BASE_URL
LLM_API_KEY  = SYNTHESIS_LLM_API_KEY
LLM_MODEL    = SYNTHESIS_LLM_MODEL

# ── Context / token budgeting ────────────────────────────────────────────────
MAX_TOKENS_CONTEXT = 4096
LLM_MAX_TOKENS_OUT = 1024

# ── Embeddings ───────────────────────────────────────────────────────────────
VECTOR_DIM = 1024          # jina-embeddings-v3 output dim
RERANK_TOP_K = 5           # keep top-5 contexts after rerank
QDRANT_SEARCH_TOP_K = 10   # vector candidates from Qdrant
GRAPH_HOPS = 2             # k-hop subgraph from Neo4j

# ── GLiNER target labels (graph extraction schema) ────────────────────────────
GLINER_LABELS = [
    "Microservice", "Database", "API", "Metric",
    "Developer", "Framework", "Component", "Bug", "PR", "ADR",
]
GLINER_THRESHOLD = 0.4

# ── Multilingual canonicalisation (Indonesian -> canonical English) ───────────
CANON_MAP = {
    "basis data": "Database",
    "pangkalan data": "Database",
    "jaringan": "Network",
    "pengembang": "Developer",
    "pembuat": "Developer",
    "kerangka kerja": "Framework",
    "layanan mikro": "Microservice",
    "layanan": "Service",
    "api": "API",
    "metrik": "Metric",
    "komponen": "Component",
    "basisdata": "Database",
    "servis": "Service",
    "cerita": "Bug",
    "gangguan": "Bug",
    "kesalahan": "Bug",
    "permintaan tarik": "PR",
    "catatan desain": "ADR",
}
