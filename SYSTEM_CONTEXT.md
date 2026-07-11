# System Context (LEGACY v1)

> [!WARNING]
> **Legacy v1 document — superseded by `v2/` + `docs/`.** In v2, synthesis is
> **Gemma 4 E4B QAT Q4_0 on `:8084`** (not 12B on :8083), extraction is **Gemma 4
> E2B QAT Q4_0 on `:8082`** (LLM-based, GLiNER fallback — not regex), and both
> are config-driven in `v2/config.py`. Live docs: `docs/README.md`. Kept for
> historical rationale.

Single reference document for the GraphRAG Engineering Knowledge Base.

---

## 1. What It Does

Ingests engineering documents (ADRs, Jira tickets, PRs, wikis) into a dual-store
system — Qdrant (vector) + Neo4j (graph) — then answers multi-hop questions
by routing through hybrid search or graph traversal, reranking the context,
and synthesizing a final answer with Gemma 4 E4B (`:8084`, config-driven).

All data on `/mnt/data-970-plus` (458 GB NVMe).

---

## 2. Architecture Overview

```
INGESTION                              RETRIEVAL
─────────                              ─────────
Raw doc ──→ Sentence split             User query
            │                              │
            ├──→ Jina v3 ──→ Qdrant        ├──→ Jina v3 embed
            │    (1024-d vectors)          │    (query vector)
            │                              │
            └──→ Regex ──→ Neo4j           ├──→ Semantic cache? → return
                 (entities + rels)         │       │ no
                                           │       ↓
                                           ├──→ MiniLM router
                                           │     /        \
                                           │  vector     graph
                                           │  search   traversal
                                           │     \        /
                                           │    bge-reranker
                                           │         │
                                           └──→ Gemma 4 E4B (:8084) → answer
```

---

## 3. All Components

| # | Component | Model | Role | Size | Device |
|---|-----------|-------|------|------|--------|
| 1 | Embedding | `jinaai/jina-embeddings-v3` | Convert sentences → vectors | 5.4 GB | CPU |
| 2 | Routing | `all-MiniLM-L6-v2` | Classify query intent (graph vs vector) | 881 MB | CPU |
| 3 | Reranker | `BAAI/bge-reranker-base` | Score + condense retrieved passages | 3.2 GB | CPU |
| 4 | LLM (synthesis) | `Gemma 4 E4B QAT Q4_0` (config-driven: `SYNTHESIS_LLM_MODEL`) | Generate final answer from context | ~3.0 GB | GPU `:8084` |
| 4b | LLM (extraction) | `Gemma 4 E2B QAT Q4_0` (config-driven: `EXTRACTION_LLM_MODEL`) | Graph extraction (LLM primary, GLiNER fallback) | ~1.5 GB | GPU `:8082` |
| 5 | Extraction | Regex (deterministic) | Parse entities + relationships from docs | — | CPU |
| 6 | Vector DB | Qdrant (Docker) | Store + search sentence vectors | — | `:6333` |
| 7 | Graph DB | Neo4j (Docker) | Store + traverse entity nodes | — | `:7687` |

---

## 4. The Reasoning-Model Limitation (Why Extraction Needs Care)

> **v2 note:** In v2, graph extraction uses **Gemma 4 E2B** (LLM, with GLiNER
> regex fallback) — not the old regex-only path. The system handles
> reasoning-model output (`reasoning_content` vs `content`) in code
> (`v2/ask.py` `synthesize()`, `v2/ingest.py` extraction), so E2B's chain-of-thought
> no longer breaks extraction.

**The problem (general to Gemma 4 family):** Gemma 4 is a reasoning model.
When asked to produce JSON:

1. It starts thinking in `reasoning_content` (its chain-of-thought)
2. It never finishes thinking before hitting the token cap
3. `content` (the final answer) comes back empty
4. The `enable_thinking: False` parameter is NOT honored by this llama-server
   build
5. Result: Gemma cannot produce any structured output

**Impact on Neo4j graph extraction:** The ingestion pipeline (`ingest.py`
→ `llm_extract()`) tries to call Gemma first. It always fails. The regex
fallback (`_extract_rulebased()`) runs every time, producing ~37 entities
from 4 sample docs.

**Impact on KV profiles:** Same issue. KV profiles are extracted as
sentence-windows from the source doc (`_doc_profile()`), not LLM-generated.

**Impact on answer generation:** This works fine. Gemma takes condensed
context and produces a natural-language answer. It doesn't need to output
JSON — just readable text.

---

## 5. Changing the Extraction / Synthesis Model (v2)

v2 already uses **LLM-based extraction** (Gemma 4 E2B on `:8082`, GLiNER
fallback) — no regex workaround needed. Both models are config-driven in
`v2/config.py` (`EXTRACTION_LLM_MODEL`, `SYNTHESIS_LLM_MODEL`); swap the
`.gguf` name (or point at any OpenAI-compatible endpoint) and restart the
daemon — no code changes.

**Models on disk that work as drop-in replacements:**

| Model | File | Size | Systemd Service |
|-------|------|------|-----------------|
| Granite 4.1 8B | `granite-4.1-8b-Q4_K_M.gguf` | ~4.5 GB | edit `gemma-4-e2b.service` model path |
| Gemma 4 E4B QAT | `gemma-4-E4B-it-QAT-Q4_0.gguf` | ~2.5 GB | `gemma-4-e4b.service` |
| Gemma 4 E2B QAT | `gemma-4-E2B-it-QAT-Q4_0.gguf` | ~1.5 GB | `gemma-4-e2b.service` |

**To switch the extraction model to Granite 8B** (non-reasoning, clean JSON):

```bash
# 1. Stop the E2B extraction service
sudo systemctl stop gemma-4-e2b.service

# 2. Edit the service to point at the new model
sudo sed -i 's|gemma-4-E2B-it-QAT-Q4_0.gguf|granite-4.1-8b-Q4_K_M.gguf|g' \
  /etc/systemd/system/gemma-4-e2b.service

# 3. Start the new model + reload
sudo systemctl daemon-reload
sudo systemctl start gemma-4-e2b.service

# 4. Re-ingest (now uses the new extraction model)
bash run.sh ingest-folder /mnt/data-970-plus/rag-system/data
```

**Code path:** `v2/ingest.py` → `extract_graph()` → calls E2B (`:8082`) via
OpenAI SDK, GLiNER fallback on timeout. Synthesis: `v2/ask.py` → `synthesize()`
→ E4B (`:8084`).

---

## 6. Current Data State

| Store | Count |
|-------|-------|
| Qdrant chunks | 51 (13+10+9+19) |
| Neo4j entities | 37 (10+9+9+9) |
| Neo4j relationships | ~13 |

---

## 7. Running Services

| Service | Port | Status | VRAM |
|---------|------|--------|------|
| Gemma 4 E2B (extraction) | `:8082` | boot | ~1.5 GB |
| Gemma 4 E4B (synthesis) | `:8084` | boot | ~3.0 GB |
| Qdrant | `:6333` | Docker | — |
| Neo4j | `:7687` | Docker | — |

**Start/Stop commands:**

```bash
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service
docker compose up -d                    # Qdrant + Neo4j
docker compose down                     # Stop DBs
```

---

## 8. CLI Commands

| Command | Purpose |
|---------|---------|
| `bash run.sh ingest <file> [--type ADR] [--author alice]` | Ingest one document |
| `bash run.sh ingest-folder <dir>` | Ingest all docs in folder |
| `bash run.sh ask "<question>"` | Answer one question |
| `bash run.sh serve` | Interactive REPL (models load once) |

---

## 9. Known Gaps

| Gap | Detail | Workaround |
|-----|--------|------------|
| LLM extraction depends on E2B | E2B (:8082) down → GLiNER fallback only (no LLM nodes) | Keep E2B service up; GLiNER covers edges |
| Graph path misses content | Some queries route to graph, neighbor KV profiles lack answer | Tune router intents or add vector fallback |
| No native Jina late_chunking | Per-sentence embed, not global-context | Use Jina API with `JINA_API_KEY` |
| Cold start per query | ~20s loading 3 models | Use `serve` REPL |

---

*Generated: July 2026*
