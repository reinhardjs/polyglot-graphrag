# GraphRAG Engineering Knowledge Base — Architecture Document (LEGACY v1)

> [!WARNING]
> **Legacy v1 document — superseded by `v2/`.** Live architecture docs:
> `docs/architecture/`. In v2 the synthesis LLM is **Gemma 4 E4B QAT Q4_0 on
> `:8084`** (not 12B on :8083) and extraction is **Gemma 4 E2B QAT Q4_0 on
> `:8082`** (LLM-based, not regex). Both are config-driven in `v2/config.py`.
> Kept for historical rationale.

## Overview

A local-first GraphRAG system for engineering documents (ADRs, Jira tickets,
PRs, Wiki pages). Answers multi-hop questions by combining vector search on
Qdrant with knowledge-graph traversal on Neo4j, using a zero-shot router to
select the retrieval path and a cross-encoder reranker to condense context
before final LLM synthesis.

**Key design constraint:** Gemma 4 is a reasoning model — it cannot produce
reliable JSON output. Therefore graph extraction during ingestion is done via
LLM (Gemma 4 E2B on :8082, with GLiNER regex fallback). Gemma is used ONLY for
answer generation during retrieval (Gemma 4 E4B on :8084).

All data lives on the 458 GB NVMe at `/mnt/data-970-plus`. CPU-based embedding/
reranking models coexist with two GPU-hosted Gemma instances on `:8082` (E2B)
and `:8084` (E4B).

---

## 1. Hardware

```
GPU:  NVIDIA RTX 3060 (12 GB VRAM)
      Gemma 4 E4B QAT Q4_0  →  ~3.0 GB  (:8084 synthesis)
      Gemma 4 E2B QAT Q4_0  →  ~1.5 GB  (:8082 extraction)
      128K–256K context

CPU:  Jina v3 embeddings    →  1,024-d vectors, ~1.7s load
      all-MiniLM-L6-v2      →  384-d routing, ~12s load
      bge-reranker-base     →  cross-encoder, ~3.5s load

Disk: /mnt/data-970-plus    →  458 GB NVMe (333 GB free)
```

**Model roles:**
- **Gemma 4 E2B** (`:8082`, GPU) → graph extraction (LLM primary, GLiNER fallback)
- **Gemma 4 E4B** (`:8084`, GPU) → answer generation ONLY
- **Jina v3** (CPU) → chunk + query embedding
- **MiniLM** (CPU) → zero-shot routing
- **bge-reranker** (CPU) → context condensation
- **Regex extractor** (CPU, instant) → graph extraction during ingestion

---

## 2. Technology Stack

| Component              | Technology                          | Role | Location |
|------------------------|-------------------------------------|------|----------|
| Vector Database        | Qdrant                              | Store + search chunks | Docker `:6333` |
| Graph Database         | Neo4j                               | Store + traverse entities | Docker `:7687` |
| Embedding (chunks)     | `jinaai/jina-embeddings-v3` (1024-d)| Encode doc sentences + queries | CPU |
| Routing                | `all-MiniLM-L6-v2`                  | Classify intent (graph vs vector) | CPU |
| Reranking              | `BAAI/bge-reranker-base`            | Score + condense retrieved context | CPU |
| LLM (generation)       | Gemma 4 E4B QAT Q4_0 (config-driven via `SYNTHESIS_LLM_MODEL`) | Answer synthesis from condensed context | `:8084` (GPU) |
| Graph extraction       | Regex (deterministic)               | Extract entities + relationships from docs | CPU, instant |
| KV profiles            | Sentence-window from source doc      | Dense 1-3 sentence summaries per entity | CPU, instant |
| LLM Client             | OpenAI Python SDK                    | Talk to Gemma (E4B) on `:8084` | — |
| Python Env             | Conda, `/mnt/data-970-plus/rag-env`  | Python 3.11 | — |

---

## 3. File Layout

```
/mnt/data-970-plus/rag-system/
├── config.py            # Constants, OpenAI client, Qdrant/Neo4j init
├── ingest.py            # Document → Qdrant + Neo4j (write mode)
├── router.py            # Semantic cache + MiniLM zero-shot routing
├── retrieve.py          # Hybrid search, graph traversal, rerank, synthesis
├── main.py              # CLI: ingest, ingest-folder, ask, serve
├── run.sh               # Convenience wrapper
├── requirements.txt
├── docker-compose.yml   # Qdrant + Neo4j containers
├── config.yaml          # Tuning knobs (keep_top, top_k)
├── data/                # Sample docs (4 files)
├── logs/                # Runtime logs
├── README.md            # Quickstart (this doc)
└── ARCHITECTURE.md      # This document
```

---

## 4. Ingestion Pipeline (Write Mode)

**Key fact: Gemma is NOT involved in ingestion.** Graph extraction is
entirely regex-based. The code attempts an LLM call first (`llm_extract()`),
but Gemma's reasoning model always returns empty `content`, so the regex
fallback runs every time.

```
Raw document (MD/TXT)
    │
    ├──→ [1] Sentence splitter (regex: . ! ?)
    │       → per-sentence chunks (clean, complete text)
    │
    ├──→ [2] Jina v3 encode(task='retrieval.passage')
    │       → 1,024-d vectors, one per sentence
    │
    ├──→ [3] Qdrant upsert
    │       • dense vector (BinaryQuantized, 1-bit)
    │       • named sparse "text" (BM25-style lexical)
    │       • metadata: doc_id, chunk_idx, text, doc_type
    │
    └──→ [4] Graph extraction (100% regex, no LLM)
            • Entity patterns: ADR-\d+, BUG-\d+, PR-\d+, TICKET-\d+
            • Component names: checkout, billing, storefront, payments-consumer,
              payment_gateway, checkout_database
            • Author detection: "Author: ..." → DEV node
            • KV profiles: 2-sentence window from source doc per entity
            • Relationship keywords: impacts, depends on, authored by,
              references, fixes → typed edges
            • Neo4j MERGE: Entity nodes + relationships (id-deduplicated)
            • Types: Microservice, ADR, Bug, Dev, Component, Ticket
              Rels: IMPACTS, DEPENDS_ON, AUTHORED, REFERENCES, FIXES
```

**Current state:** 51 chunks in Qdrant, 37 entities / 13 relationships in Neo4j.

**Jina note:** The SentenceTransformer wrapper does not support Jina's native
`late_chunking=True`. Documents are sentence-split at the text level and each
sentence embedded independently with `task='retrieval.passage'`. Clean, complete
chunks, but without global-context pooling.

**Extraction model is already LLM-based in v2** (Gemma 4 E2B on `:8082`, with
GLiNER regex fallback). To change it, edit `EXTRACTION_LLM_MODEL` in `v2/config.py`
or the `gemma-4-e2b.service` model path (e.g. point at Granite 8B for clean-JSON
extraction), then restart the service and re-ingest. No code changes needed.

---

## 5. Retrieval Pipeline (Read Mode)

**Gemma is used ONLY here,** for the final answer generation step.

```
User query
    │
    ├──→ [1] Embed (Jina v3, task='retrieval.query')
    │       → 1,024-d query vector
    │
    ├──→ [2] Semantic cache (Qdrant query_cache, cosine >0.95)
    │       • HIT → return cached answer (0 LLM tokens)
    │
    ├──→ [3] Zero-shot router (all-MiniLM-L6-v2)
    │       • Cosine vs VECTOR_INTENTS centroid
    │       • Cosine vs GRAPH_INTENTS centroid
    │       • Higher score → chosen path
    │
    ├──→ [4a] VECTOR path (Qdrant)
    │       • Hybrid: dense (BinaryQuantized) + sparse (BM25)
    │       • RRF fusion → top-8 → extract text
    │
    ├──→ [4b] GRAPH path (Neo4j)
    │       • Keyword-match entry node
    │       • 2-hop traversal (degree ≥ 1 filter)
    │       • Return KV profiles; <2 results → fall back to vector
    │
    ├──→ [5] Cross-encoder rerank (bge-reranker-base)
    │       • Score (query, passage) pairs
    │       • Keep top 50%, minimum 3
    │
    └──→ [6] Synthesis (Gemma 4 E4B on :8084)
            • Concatenate condensed context
            • Generate answer with source citations
            • Store query + answer in semantic cache
```

---

## 6. Component Details

### 6.1 Embeddings (Jina v3)
- **Model:** `jinaai/jina-embeddings-v3` (5.4 GB flat dir on NVMe)
- **Dimension:** 1,024
- **Quantization:** BinaryQuantization (1-bit) in Qdrant
- **Query:** `task='retrieval.query'`
- **Document:** `task='retrieval.passage'`
- **Load time:** ~1.7 s per process

### 6.2 Routing (MiniLM)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (881 MB)
- **Dimension:** 384
- **Intents:** 4 vector + 5 graph predefined queries
- **Decision:** Higher cosine similarity to either intent centroid

### 6.3 Reranker (bge-reranker)
- **Model:** `BAAI/bge-reranker-base` (3.2 GB)
- **Type:** Cross-encoder
- **Keep:** Top 50%, min 3 passages
- **Device:** CPU

### 6.4 Qdrant
- **Collection:** `engineering_chunks` — dense (BinaryQuantized) + sparse ("text")
- **Cache:** `query_cache` — dense only, no quantization
- **Hybrid fusion:** RRF (Reciprocal Rank Fusion)
- **Prefetch:** Dense top_k*2 + Sparse top_k*2 → fusion → top_k

### 6.5 Neo4j
- **Connection:** `bolt://localhost:7687` (neo4j / ragpassword123)
- **Schema:** `Entity` nodes with `id`, `name`, `type`, `profile`, `source_doc`
- **Relationships:** Typed edges (IMPACTS, DEPENDS_ON, AUTHORED, REFERENCES, FIXES)
- **Traversal:** 2-hop k-core (degree ≥ 1) from keyword-matched entry node

### 6.6 LLM (Gemma 4 E4B) — Answer generation only
- **Purpose:** Final answer synthesis from condensed context. NOT used for
  graph extraction (that uses Gemma 4 E2B + GLiNER).
- **Model:** `gemma-4-E4B-it-QAT-Q4_0.gguf` (config-driven via `SYNTHESIS_LLM_MODEL`
  in `v2/config.py`; swap to any OpenAI-compatible model without code changes)
- **Context:** 256K tokens
- **Port:** `:8084` (llama-server, OpenAI-compatible API)
- **Systemd:** `gemma-4-e4b.service` (Type=simple, Restart=no)
- **VRAM:** ~3.0 GB

**Reasoning model behavior:**
- Emits chain-of-thought in `reasoning_content`
- Final answer in `content` (often empty for short generation budgets;
  falls back to `reasoning_content`)
- `enable_thinking: False` is NOT honored by this llama-server build
- Cannot produce clean JSON → extraction relies on regex

### 6.7 Graph Extraction — Regex (100%, no LLM)
The `llm_extract()` function in `ingest.py` attempts to call Gemma with a JSON
extraction prompt. Because Gemma is a reasoning model, it always returns empty
`content` (the JSON never leaves `reasoning_content`). The `except`
block catches the parse failure and runs `_extract_rulebased()` instead.

Patterns matched:
- `ADR-\d+`, `BUG-\d+`, `PR-\d+`, `TICKET-\d+` → typed entity nodes
- Known component names: checkout, billing, storefront, payments-consumer,
  payment gateway, checkout database
- `Author: <name>` → DEV node
- Relationship keywords in sentences: "impacts", "depends on", "authored by",
  "references", "fixes" → typed edges

---

## 7. Discrepancies vs. Original Spec

| Spec Requirement          | Actual Implementation                        |
|---------------------------|----------------------------------------------|
| Jina native late_chunking | Sentence-split + per-sentence embed           |
| Ornith 9B for extraction  | Regex (Gemma is reasoning model → can't emit JSON) |
| GPT-5 / OpenAI cloud      | Local Gemma 4 E4B on :8084 (config-driven)    |
| LLM-generated KV profiles | Doc sentence-window extraction               |
| k-core depth 2 (k=2)      | k-core degree ≥ 1 (lowered for recall)       |
| Discard bottom 80%        | Keep top 50%, min 3 (higher recall)          |

---

## 8. Usage

```bash
# Start databases
cd /mnt/data-970-plus/rag-system && docker compose up -d

# Start Gemma (E2B extraction :8082 + E4B synthesis :8084)
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service

# Ingest (uses E2B LLM extraction; GLiNER fallback)
bash run.sh ingest /path/to/doc.md --type ADR --author alice
bash run.sh ingest-folder /mnt/data-970-plus/rag-system/data

# Query (needs Gemma for answer generation)
bash run.sh ask "who reported BUG-204 and what severity was it?"
bash run.sh serve   # interactive REPL (models load once)
```

---

## 9. Data State

| Store | Count |
|-------|-------|
| Qdrant chunks | 51 |
| Neo4j entities | 37 |
| Neo4j relationships | 13 |

---

## 10. Verified Working

| Component | Status | Details |
|-----------|--------|---------|
| Jina v3 embeddings (1,024-d) | ✓ | Loads ~1.7s, clean sentence chunks |
| Qdrant BinaryQuantization | ✓ | 51 chunks, 1-bit + sparse lexical |
| Hybrid search (dense + sparse) | ✓ | RRF fusion working |
| MiniLM zero-shot routing | ✓ | Graph vs vector path selection |
| bge-reranker condensation | ✓ | Top 50%, min 3 |
| Gemma synthesis on :8084 | ✓ | Sourced answers from condensed context |
| Semantic cache | ✓ | 0-token cache hits confirmed |
| Regex graph extraction | ✓ | 37 entities, 13 relationships reliably |
| 2-hop fallback to vector | ✓ | Auto-falls when graph <2 results |

---

## 11. Known Gaps

| Issue | Impact | Fix |
|-------|--------|-----|
| Graph path misses content on some queries | Router picks graph path where neighbor profiles don't contain the answer text, even though it exists in Qdrant vector store | Add vector results as complementary fallback when graph returns <3 passages |
| Jina native late_chunking | Per-sentence embed vs global-context pooling. Correct answers, marginal recall loss | Use Jina API with `JINA_API_KEY` or upgrade ST version |
| Cold-start model loading per query | ~20s first query (loads 3 models). Use `serve` for <5s | `bash run.sh serve` |

---

*Generated: July 2026 | Verified with 4 sample engineering docs, 3/3 test queries.*
