# GraphRAG Engineering Knowledge Base

Local-first GraphRAG system for engineering documents. Ingests ADRs, Jira
tickets, PRs, and wikis into Qdrant (vector) + Neo4j (graph), then answers
multi-hop questions via zero-shot routing, hybrid search, cross-encoder
reranking, and Gemma 4 12B synthesis.

**All data on:** `/mnt/data-970-plus` (458 GB NVMe)

---

## Quick Start

```bash
# 1. Start databases (Qdrant :6333 + Neo4j :7687)
cd /mnt/data-970-plus/rag-system
docker compose up -d

# 2. Make sure Gemma is running on :8083
sudo systemctl start gemma-4-12b.service

# 3. Ingest sample docs
bash run.sh ingest-folder /mnt/data-970-plus/rag-system/data

# 4. Ask questions
bash run.sh ask "who reported BUG-204 and what severity was it?"
bash run.sh ask "how are checkout and billing connected?"

# 5. Interactive session (models load once, queries < 5s)
bash run.sh serve
```

---

## Commands

| Command | Purpose |
|---------|---------|
| `bash run.sh ingest <file> [--type ADR] [--author alice]` | Ingest a single document |
| `bash run.sh ingest-folder <dir>` | Ingest all docs in a directory |
| `bash run.sh ask "<question>"` | Single question (cold-starts models ~20s) |
| `bash run.sh serve` | Interactive loop (models stay loaded) |

---

## Architecture (2-min overview)

```
            INGESTION                              RETRIEVAL
            ─────────                              ─────────
Raw doc ──→ Sentence split                        User query
            │                                         │
            ├──→ Jina v3 embed → Qdrant               ├──→ Jina v3 embed
            │    (1024-d, binary quant)                │    (query vector)
            │                                         │
            └──→ Regex extraction → Neo4j             ├──→ Semantic cache? → return
                 (ADR- / BUG- / PR- / components)     │       │ no
                 (KV profiles from source text)        │       ↓
                                                      ├──→ MiniLM router
                                                      │    /         \
                                                      │ vector      graph
                                                      │ search    traversal
                                                      │    \         /
                                                      │     reranker
                                                      │         │
                                                      └──→ Gemma 4 12B → answer
```

**Key insight:** Gemma 4 is a reasoning model and cannot produce clean JSON
output. Therefore graph extraction is done via regex (deterministic, reliable).
Gemma is only used for final answer generation during retrieval.

---

## Stack

| Component | Role | Model | Location |
|-----------|------|-------|----------|
| Embedding | Chunk docs + query | `jinaai/jina-embeddings-v3` (1024-d) | CPU, 5.4 GB |
| Routing | Pick graph vs vector path | `all-MiniLM-L6-v2` (384-d) | CPU, 881 MB |
| Reranker | Condense retrieved context | `BAAI/bge-reranker-base` | CPU, 3.2 GB |
| LLM | Answer generation only | `Gemma 4 12B QAT Q4_0` | GPU `:8083`, 9.9 GB |
| Graph extraction | Regex-based (not LLM) | ADR/BUG/PR patterns + known components | CPU, instant |

---

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Constants, DB init, OpenAI client setup |
| `ingest.py` | Document → Qdrant + Neo4j ingestion pipeline |
| `router.py` | Semantic cache + MiniLM zero-shot routing |
| `retrieve.py` | Hybrid search, graph traversal, rerank, Gemma synthesis |
| `main.py` | CLI entry point (ingest / ingest-folder / ask / serve) |
| `config.yaml` | Tuning knobs (keep_top, top_k) |
| `docker-compose.yml` | Qdrant + Neo4j containers |
| `ARCHITECTURE.md` | Full technical deep-dive |

---

## Data State (current)

| Store | Count |
|-------|-------|
| Qdrant chunks | 51 |
| Neo4j entities | 37 |
| Neo4j relationships | 13 |

---

## Notes

- **Gemma 4 is a reasoning model** — it streams thinking in `reasoning_content`.
  Final answers in `content` (or `reasoning_content` as fallback). Cannot produce
  clean JSON → graph extraction uses regex.
- **Gemma is NOT used for ingestion.** Graph extraction is entirely regex-based
  (scans for ADR-\d+, BUG-\d+, PR-\d+, known component names, author lines).
  KV profiles are extracted as sentence windows from the source document.
- **Jina native `late_chunking=True`** is not wired through the installed
  SentenceTransformer version. Chunks are per-sentence embedded (clean, complete
  text) rather than with global-context pooling.
- **Model cold-start** ~20s per `ask` invocation (loads Jina + MiniLM +
  reranker). Use `serve` for sub-5s queries.
- For non-reasoning models (e.g. Granite 8B), switch `llm_extract()` and
  `llm_profile()` in `ingest.py` back to OpenAI completions for LLM-based
  extraction.
