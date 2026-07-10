# Model switching guide — copy the relevant block into config.py, restart daemon.
#
# After switching embedding models:
#   1. Update VECTOR_DIM in config.py to match the model's output dim
#   2. Restart daemon: sudo systemctl restart rag-gpu-daemon
#   3. Re-ingest all documents (embeddings must match new dim):
#      python ingest.py --docs sample_data/
#   4. Verify: curl -s http://127.0.0.1:8000/models | jq .embedder.dim


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG A: Jina Embeddings v3 (CURRENT DEFAULT — 1024-dim)
# Status: WORKS ✓, used in production
# ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME     = "jinaai/jina-embeddings-v3"
VECTOR_DIM           = 1024
EMBED_TRUST_REMOTE   = True
EMBED_USE_HALF       = True
EMBED_TASK_PASSAGE   = "retrieval.passage"
EMBED_TASK_QUERY     = "retrieval.query"
EMBED_MAX_LENGTH     = 32768
EMBED_MATRYOSHKA_DIM = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG B: Jina Embeddings v4 (3.8B params, 2048-dim default)
# Status: BLOCKED — needs torch≥2.6 + transformers≥4.52 + peft≥0.15.2
# Our env: torch 2.3.1, transformers 4.49.0 → needs upgrade
# ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME     = "jinaai/jina-embeddings-v4"
VECTOR_DIM           = 2048
EMBED_TRUST_REMOTE   = True
EMBED_USE_HALF       = True
EMBED_TASK_PASSAGE   = "retrieval"      # v4 uses 'retrieval' not 'retrieval.passage'
EMBED_TASK_QUERY     = "retrieval"      # same task for both
EMBED_MAX_LENGTH     = 32768
EMBED_MATRYOSHKA_DIM = 1024            # optional: truncate to 1024 to reduce VRAM
# UPGRADE REQUIRED:
#   pip install torch>=2.6.0 transformers>=4.52.0 peft>=0.15.2
#   pip install flash-attention --no-build-isolation  # optional but recommended
# VRAM estimate: ~7.6 GB fp16 (vs 3.0 GB for v3)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG C: Google EmbeddingGemma 300M (768-dim, tiny)
# Status: BLOCKED — gated model, requires HF login + license acceptance
# ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME     = "google/embeddinggemma-300m"
VECTOR_DIM           = 768
EMBED_TRUST_REMOTE   = False            # no trust_remote needed
EMBED_USE_HALF       = False            # model is already small (300M)
EMBED_TASK_PASSAGE   = None             # vanilla — no task adapter
EMBED_TASK_QUERY     = None
EMBED_MAX_LENGTH     = 2048             # hard cap for this model
EMBED_MATRYOSHKA_DIM = None
# AUTH REQUIRED:
#   1. huggingface-cli login
#   2. Visit https://huggingface.co/google/embeddinggemma-300m — click "Agree"
#   3. Then restart daemon
# VRAM estimate: ~0.6 GB fp32 (vs 3.0 GB for Jina v3)
# Tradeoff: 1/5th the size, 768-dim vs 1024, max context 2K vs 32K tokens


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG D: BGE-M3 (1024-dim, multilingual)
# Status: SHOULD WORK — vanilla SentenceTransformer
# ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME     = "BAAI/bge-m3"
VECTOR_DIM           = 1024
EMBED_TRUST_REMOTE   = False
EMBED_USE_HALF       = True
EMBED_TASK_PASSAGE   = None             # BGE doesn't use tasks
EMBED_TASK_QUERY     = None
EMBED_MAX_LENGTH     = 8192
EMBED_MATRYOSHKA_DIM = None
# No special auth needed. Should load from HF cache.
# VRAM estimate: ~2.2 GB fp16 (smaller than Jina v3 at ~3.0 GB)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG E: Any vanilla SentenceTransformer model (e.g. all-MiniLM-L6-v2)
# ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM           = 384
EMBED_TRUST_REMOTE   = False
EMBED_USE_HALF       = True
EMBED_TASK_PASSAGE   = None
EMBED_TASK_QUERY     = None
EMBED_MAX_LENGTH     = None
EMBED_MATRYOSHKA_DIM = None
# VRAM estimate: <0.1 GB. Tradeoff: 384-dim vectors, lower quality retrieval.
