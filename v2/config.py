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
CACHE_THRESHOLD = 0.95   # cosine similarity above which we return cached answer

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
EXTRACTION_LLM_MODEL    = "gemma-4-E2B-it-QAT-Q4_0.gguf"

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
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_USE_HALF    = True    # fp16 conversion

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
