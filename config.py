"""
config.py — Central configuration for the GraphRAG Engineering Knowledge Base.

Production-grade, 100% LOCAL. ALL models run on GPU (RTX 3060, 12 GB):
  E2B (:8082, user-systemd)  ≈ 1.5 GB  — serves BOTH extraction AND synthesis
  E4B (:8084) RETIRED — was synthesis (~22s p95); E2B replaced it (~2.2s) in v1.0.2
  Auxiliary models (:8000, serve_gpu.py FastAPI daemon, fp16 preloaded at startup):
    Jina v3, BGE reranker-v2-m3 ≈ 4.0 GB (GLiNER lazy-loaded on demand)
  ────────────────────────────────────────────────────────────────────
  Total GPU (E2B + daemon)     ≈ 8.5 GB  (4.1 GB headroom)

Entity resolution is vector-driven via Jina v3's cross-lingual embeddings
stored in Neo4j. No hardcoded CANON_MAP — "Basis Data", "Database", and
"Base de Datos" all converge to the same entity automatically.

serve_cpu.py is a CPU-only fallback for machines without CUDA torch.
serve_gpu.py is the primary daemon (loaded at startup, GPU-resident).

The daemon exposes a unified /ask endpoint that runs the full pipeline
server-side (embed → Qdrant||Neo4j → rerank → E2B synthesis) in ONE HTTP call.
"""
import os

# ── Version ─────────────────────────────────────────────────────────────────
# Single source of truth is the VERSION file at the project root.
# This is the v1.0.0 stable release (contract frozen). See VERSIONING.md.
def _read_version() -> str:
    vf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    try:
        with open(vf, encoding="utf-8") as _f:
            return _f.read().strip() or "0.0.0"
    except OSError:
        return "0.0.0"
__version__ = _read_version()
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
COLL_CHUNKS = "clinical_prose"
COLL_CACHE  = "query_cache"
CACHE_THRESHOLD = 0.95   # cosine similarity above which we return cached answer

# ── Feature Gates ─────────────────────────────────────────────────────────────
ENABLE_CRAG = False     # Phase 3: Corrective RAG + adaptive routing (experimental)
                        # When True, the `crag` field is exposed on /ask AskReq.
                        # Kept off by default to preserve the v1.0 explicit-
                        # retrieval-only contract. Toggle on for v2.0 CRAG testing.

# ── Multi-Domain Collections ─────────────────────────────────────────────────
# Each domain gets its own Qdrant collection (namespace isolation, zero
# overhead). After the engineering corpus purge, the only live Qdrant
# collection is `clinical_prose` (snomed's semantic companion). `snomed` itself
# is graph-only (collection: null → terminology graph in Neo4j, no prose
# collection). Add a new domain here + create the collection on first use.
QDRANT_COLLECTIONS = {
    "snomed":       None,            # terminology graph: no prose collection
    "clinical_prose": "clinical_prose",
}
QDRANT_COLLECTION_DEFAULT = "clinical_prose"

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
EXTRACTION_LLM_BASE_URL = "http://127.0.0.1:8082/v1"
EXTRACTION_LLM_API_KEY  = "not-needed"
EXTRACTION_LLM_MODEL    = "gemma-4-E2B_q4_0-it.gguf"

# V3.0 extraction mode:
#   "llm"           — full-doc single-pass extraction with E2B (89% precision)
#   "index_routing" — Hybrid GLiNER (entities) → Qwen (relation classification, 20% precision)
#   "hybrid"        — GLiNER (entities) → E2B (relation class, 100% precision) [RECOMMENDED]
#   "sliding_window" — sentence-boundary chunked extraction with coref
#   Extraction mode (how relations are classified between GLiNER entities):
#     - "sliding_window" — chunk long docs, extract per window (default, best recall)
#     - "hybrid"         — GLiNER entities → E2B relationship classification
#     - "llm"            — single-pass E2B extraction (fast on huge corpora)
#   (The old "index_routing" mode — GLiNER → Qwen relation classification at
#   20% precision — was deprecated and removed.)
# Override at runtime: EXTRACTION_MODE=sliding_window (richer extraction for
# long docs) | hybrid | llm (single-pass).
# Default is "sliding_window" — richest extraction (per-window LLM + GLiNER,
# parallelized). For speed on huge corpora, set EXTRACTION_MODE=llm.
EXTRACTION_MODE = os.environ.get("EXTRACTION_MODE", "llm")

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

# Synthesis: answer generation from retrieved context.
# DEFAULT = E2B (gemma-4-E2B Q4_0 on :8082) — measured p95 ~2.6s on the
# RTX 3060 12GB card with the synthesis-cost caps below. This is the fast,
# production default. The larger E4B (:8084) gives deeper answers but runs
# ~22s on this hardware; enable it via env override:
#   SYNTHESIS_LLM_BASE_URL=http://127.0.0.1:8084/v1 \
#   SYNTHESIS_LLM_MODEL=gemma-4-E4B-it-QAT-Q4_0.gguf python serve_gpu.py
SYNTHESIS_LLM_BASE_URL = os.environ.get("SYNTHESIS_LLM_BASE_URL", "http://127.0.0.1:8082/v1")
SYNTHESIS_LLM_API_KEY  = os.environ.get("SYNTHESIS_LLM_API_KEY", "not-needed")
SYNTHESIS_LLM_MODEL    = os.environ.get("SYNTHESIS_LLM_MODEL", "gemma-4-E2B_q4_0-it.gguf")

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
# Reranker device. Default "cuda" — with E4B retired there is ~7 GB free on
# the 12 GB card, so BGE (~1.0 GB) coexists with Jina + E2B comfortably
# and rerank runs on GPU (faster than CPU, ~10-20 ms vs ~50 ms). It was
# The reranker stays on GPU: a CPU BGE reranker regressed retrieval to
# ~22s p95 under load (rerank is on the hot path of every /ask). E2B synthesis
# is the dominant cost; tuning the reranker device does not move the synthesis
# p95 SLO. Keep cuda.
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cuda")

# CPU fallback (serve_cpu.py): lighter model + capped pool. BGE-base at 278M
# params is 2.7× faster than v2-m3 on CPU. Capping the fused pool to 10 docs
# keeps rerank under 500ms so the i5 doesn't freeze.
RERANK_MODEL_NAME_CPU = "BAAI/bge-reranker-v2-m3"   # same as GPU, capped pool delivers speed
RERANK_CPU_POOL_CAP   = 15  # sweet spot: 4.5/5 avg overlap, 2.2s /ask on CPU

# ── Rerank calibration knobs ──────────────────────────────────────────────────
# Confidence bands + dual-signal boost, tuned WITHOUT code edits. Calibrated on
# bge-reranker-v2-m3 over 12 clinical queries (docs/rerank-calibration.md):
#   noise/weak 0.00–0.15 | moderate 0.15–0.50 | strong dual >= 0.50
# Confidence is computed on the RAW (de-boosted) cross-encoder score so it
# reflects model relevance; dual_signal is a separate corroboration flag.
CONFIDENCE_HIGH_THRESHOLD   = 0.50   # raw score >= this -> "high"
CONFIDENCE_MEDIUM_THRESHOLD = 0.15   # raw score >= this -> "medium"; else "low"
DUAL_SIGNAL_BOOST           = 0.15   # additive rerank bump for candidates
                                    # confirmed by BOTH SNOMED + clinical prose

# ── GLiNER (NER fallback) ──────────────────────────────────────────────────────
GLINER_MODEL_NAME = "urchade/gliner_multi-v2.1"

# ── Extraction LLM Behavior ────────────────────────────────────────────────────
# Some LLMs (Gemma E4B, Gemma 12B) are "reasoning" models that put output in
# `reasoning_content` instead of `content`. Set True if your extraction model
# does this so the daemon reads the right field.
EXTRACTION_READS_REASONING = False

# ── Prompt Templates (per-model overridable) ──────────────────────────────────
# Extraction prompt: sent to the EXTRACTION_LLM. Substitution tokens:
#   {doc_id}, {text}          — document id / body (always)
#   {entity_types}            — vocab from profile['entity_types'] (YAML)
#   {relation_types}          — vocab from profile['relation_types'] (YAML)
# The entity/relation VOCABULARY is the single source of truth in
# domain_config.yaml (entity_types / relation_types). ingest.py substitutes it
# into the prompt at render time, so the LLM prompt and the structured
# extractor/validation can never drift apart. Edit the vocab in YAML only.
#
# The default lives in prompts/extraction.md (editable without touching Python).
# config.EXTRACTION_PROMPT loads it at import; if the file is missing, the
# inline string below is the fallback so the system never breaks.
DEFAULT_ENTITY_TYPES = "Microservice|Database|API|Metric|Developer|Framework|Component|Bug|PR|ADR"
DEFAULT_RELATION_TYPES = "ASSOCIATED_WITH|DEPENDS_ON|IMPACTS|AUTHORED|REFERENCES|FIXES"
_EXTRACTION_PROMPT_FALLBACK = (
    "Extract a knowledge graph from the document below.\\n"
    "Entities: extract names EXACTLY as they appear — do NOT translate.\\n"
    "Relationships: use one of {relation_types}.\\n"
    "Return ONLY valid JSON, no prose, no markdown:\\n"
    '{{\"nodes\":[{{\"id\":\"ExactEntityName\",\"type\":\"{entity_types}\"}}],'
    '\"edges\":[{{\"source\":\"entity_a\",\"target\":\"entity_b\",'
    '\"type\":\"{relation_types}\"}}]}}\\n'
    "Document ({doc_id}):\\n{text}"
)
def _load_extraction_prompt():
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "prompts", "extraction.md")
    if os.path.isfile(_path):
        with open(_path, encoding="utf-8") as _f:
            return _f.read()
    return _EXTRACTION_PROMPT_FALLBACK
EXTRACTION_PROMPT = _load_extraction_prompt()

# Cap the length of each retrieved context chunk fed to the synthesis LLM.
# Must be >= the typical chunk size (~1700 chars) or the answer-bearing text
# gets truncated away and the model wrongly abstains ("not in the context").
# A prior value of 350 literally cut chunks mid-sentence and broke factual
# answers whose key fact sat past char 350. 1800 keeps whole chunks while
# still bounding pathological giant contexts. Set 0 to disable capping.
MAX_SYNTH_CONTEXT_CHARS = int(os.environ.get("MAX_SYNTH_CONTEXT_CHARS", "1800"))
# Cap the NUMBER of contexts sent to synthesis. The LLM only needs the top few
# most-relevant chunks; sending all reranked candidates just inflates prefill.
MAX_SYNTH_CONTEXTS = int(os.environ.get("MAX_SYNTH_CONTEXTS", "4"))
# Max tokens the synthesis LLM may generate. Lower = faster answers (caps
# generation time, the dominant cost for small models like E2B). On RTX 3060
# 12GB the E2B GGUF decodes at ~103 tok/s, so wall ≈ tokens/103 + ~0.5s
# (retrieval+rerank). 250 tokens -> ~1.2-1.8s daemon /ask, comfortably under
# the 3s synthesis SLO, while still allowing a grounded cited answer (the model
# stops at EOS well before the cap for most queries). 400 was too high: it forced
# the model to pad to the cap (~3.9s wall) with boilerplate. 250 is the sweet
# spot. Override via SYNTH_MAX_TOKENS_OUT if you need longer answers.
SYNTH_MAX_TOKENS_OUT = int(os.environ.get("SYNTH_MAX_TOKENS_OUT", "250"))

# Synthesis sampling temperature. RAG synthesis is a faithful-extraction task,
# not creative writing: a non-zero temperature makes the small E2B model
# randomly abstain ("context does not contain…") even when the answer is
# present in the retrieved context. Pin to 0.0 for deterministic, grounded
# answers. Override via env only if you deliberately want more variation.
SYNTH_TEMPERATURE = float(os.environ.get("SYNTH_TEMPERATURE", "0.0"))

# ── Context / token budgeting ────────────────────────────────────────────────
MAX_TOKENS_CONTEXT = 4096
LLM_MAX_TOKENS_OUT = 1024

# ── Embeddings ───────────────────────────────────────────────────────────────
VECTOR_DIM = 1024          # jina-embeddings-v3 output dim
RERANK_TOP_K = 5           # keep top-5 contexts after rerank
QDRANT_SEARCH_TOP_K = 10   # vector candidates from Qdrant
GRAPH_HOPS = 2             # k-hop subgraph from Neo4j. The full A->B->C->D chain
                            # (bob OWNS BUG-204 CAUSED_BY CPU DEPENDS_ON GPU) is
                            # reachable in 2 hops from the MIDDLE node BUG-204
                            # (bob@1, GPU@2). Latency-safe; GRAPH_TRAVERSAL_LIMIT
                            # bounds a hub entity's neighbourhood.
GRAPH_TRAVERSAL_LIMIT = 50  # cap nodes/edges returned by the k-hop expansion.
                              # Only GRAPH_PRUNE_TOP_N (10) nodes are kept after
                              # Phase-2 pruning, so fetching 200 was wasteful and
                              # made the O(n^2) edge-expansion UNWIND on hub
                              # entities (e.g. shared "Database"/"incident" nodes)
                              # cost ~3.7s. 50 bounds the worst case ~16x with no
                              # quality loss (pruning still keeps the best 10).
GRAPH_PRUNE_TOP_N = 10     # Phase 2: cap subgraph to Top-N nodes (context window guard)
GRAPH_PRUNE_STRATEGY = "degree"  # "degree" | "pagerank" | "none" — neighbor ranking
# Similarity floor for the graph entry node. When the best keyword-overlap /
# vector-similarity match is below this, the query has no real entity in the
# graph, so the expensive k-hop traversal is skipped (degrades to vector-only).
# Without this gate, unrelated queries ("cassandra repair" in an ops corpus)
# still pick a low-similarity entry and run a ~3.7s variable-length path query
# for context that contributes nothing. 0.0 disables the gate (old behaviour).
GRAPH_ENTRY_MIN_SIM = float(os.environ.get("GRAPH_ENTRY_MIN_SIM", "0.30"))
# Additive rerank/order boost for explicit graph-edge-statement contexts
# (doc_type="graph_edge"). These are the direct A->B->C->D relationship
# evidence for multi-hop questions; without a boost they get sliced off by
# top_k (pool is Qdrant-first) and the LLM never sees the chain. Boosting
# them surfaces the chain for graph questions and is a no-op when no graph
# edges are present (normal RAG queries unchanged).
# Reduced from 1.0 to 0.001 so vector similarity dominates for dense_prose
# domains; graph edges still get a tiny tiebreaker preference.
GRAPH_EDGE_BOOST = float(os.environ.get("GRAPH_EDGE_BOOST", "0.001"))

# ── CRAG (Phase 3): Corrective RAG & adaptive routing ────────────────────────
CRAG_USE_LLM_ROUTER = False  # True → confirm route + rewrite with E4B (slower, smarter)

# ── GLiNER target labels (graph extraction schema) ────────────────────────────
GLINER_LABELS = [
    "Microservice", "Database", "API", "Metric",
    "Developer", "Framework", "Component", "Bug", "PR", "ADR",
]
GLINER_THRESHOLD = 0.4
# Max words sent to GLiNER in a single call. GLiNER cost grows superlinearly with
# input length and can hang (effectively never return) on large docs. Chunking the
# text to <= this many words keeps each call fast and bounded. ingest.write_graph
# uses the same 512-word window for profile context.
GLINER_CHUNK_WORDS = 512

# ── Entity Resolution (vector-driven, language-agnostic) ─────────────────────
# Replaces the hardcoded CANON_MAP. Entities extracted in their native language
# (e.g. "Basis Data", "Database", "Base de Datos") are matched via Jina v3's
# cross-lingual embeddings stored in Neo4j's vector index. The LLM is instructed
# to extract verbatim — no translation required. Zero hardcoded translations.
ENTITY_RESOLUTION_THRESHOLD = 0.88   # cosine similarity for entity merger
ENTITY_VECTOR_INDEX = "entity_vector_idx"

# ── Domain config (v0.x) ──────────────────────────────────────────────
# Domain schemas live in `domain_config.yaml` (YAML), consumed by
# `domain_loader.py`. All callers use `domain_loader.get_domain(name)`.

# Auto-seed demo corpora on daemon startup so a NON-technical user can query
# them AS-IS with zero setup. Each entry: domain name whose ingestor is run
# if its Qdrant collection is empty. `example_companion` is the bundled
# zero-tech demo (domains/example_companion). Add real domains here to have
# them auto-populate on boot. Set to [] to disable auto-seed entirely.
# Companion corpora auto-seeded on daemon startup (if their Qdrant collection is
# empty). Left empty after the engineering corpus purge — the only remaining
# Auto-seed on first startup: for each listed domain whose Qdrant collection
# is empty, run its ingester in a background thread so a fresh user gets a
# queryable KB immediately. `enterprise` seeds the system's OWN docs/ tree
# (self-docs) — zero external corpus required for a first-run smoke test.
SEED_ON_STARTUP = ["enterprise"]

