# polyglot-graphrag v3.0.8 — Neuro-Symbolic GraphRAG

**Status:** Production (2026-07-12)  
**Extraction:** GLiNER + Gemma-4-E2B Q4_0 (hybrid / sliding_window)  
**Infra:** Domain-agnostic YAML config, batch UNWIND Neo4j, hybrid vector+graph retrieval  
**Quality:** 100% edge precision (engineering), 65 edges (journal)  

---

## 1. Architecture

```
┌─ RTX 3060 12 GB ────────────────────────────────────────────────────┐
│                                                                       │
│  rag-gpu-daemon (:8000)                       ~5.7 GB                │
│  ├─ Jina v3 embed   (fp16)                                           │
│  ├─ BGE reranker    (fp16)                                           │
│  └─ GLiNER multi-v2.1 (fp16, lazy-loaded)  ← entity detection        │
│                                                                       │
│  Gemma-4-E2B Q4_0 (:8082, llama.cpp Option B)    ~2.3 GB             │
│    8192 ctx, f16 KV, relation classification                         │
│                                                                       │
│  Neo4j (:7687)  ─┐   Qdrant (:6333)  ─┐                              │
│    vector-resolved  │     dense + sparse   │   hybrid retrieval        │
│    entity graph     │     vectors           │                           │
│                     │                       │                          │
└─────────────────────┴───────────────────────┴──────────────────────────┘
                          Total VRAM: ~8.0 GB / 12 GB
```

**One model change from v2.x:** Gemma-4-E4B synthesis model (:8084) has been
deprecated. The architecture now uses E2B for extraction only. Synthesis and
query handling are delegated to the parent AI agent via the `rag_query` tool.

---

## 2. Extraction Pipeline — 4 Modes

| Mode | How | Precision | Latency | Status |
|------|-----|-----------|----------|--------|
| `index_routing` | GLiNER → Qwen 1.5B pair-classify | ~20% | 1.2s | ❌ deprecated |
| `llm` | E2B full-doc single-pass | 89% | 11.8s | fallback |
| **`hybrid`** | GLiNER entities → E2B relation classify | **100%** | 10-15s | ✅ **default (≤4K tok)** |
| **`sliding_window`** | sentence-chunk + coref summaries | 100% (short) / high-recall | 21-323s | ✅ **long docs** |

**Flow:** `ingest_text()` → late-chunk vectors → Qdrant + GLiNER/E2B graph →
Neo4j (batched UNWIND, vector-resolved dedup). `/ask` does hybrid
vector+graph retrieval.

### 2.1 Hybrid mode (production, ≤4K tokens)

1. GLiNER detects entities from domain `entity_types` list (labels sent to
   `/extract_entities` endpoint on :8000).
2. E2B receives entity list + source text in a single prompt; classifies ALL
   pairs of entities into typed relations.
3. `_parse_and_validate` strips E2B's `<|channel|>thought` block, parses the
   final JSON array, validates entities against GLiNER set, and normalizes
   trailing-punctuation mismatches.

**Engineering gold benchmark:** 5/5 edges correct, 100% precision, zero
`ASSOCIATED_WITH` fallbacks.

### 2.2 Sliding window mode (any doc length)

1. spaCy sentence-boundary chunking — no entity names split across boundaries.
2. Per-chunk: GLiNER → E2B with "PREVIOUS WINDOW SUMMARY" for coreference.
3. Chunk size capped at `max_words=1400` (~2000 tokens) to leave E2B's 8192-token
   context room for the thinking block + JSON answer.
4. Long-doc benchmark: 77K-char arxiv paper → 138 entities, 65 edges across
   7 relation types (323s ingest time).

### 2.3 Collection routing

Domain config's `collection:` field routes vectors to the correct Qdrant
collection. Neo4j nodes carry a secondary label (`:Journal`, `:Engineering`)
from the domain's `neo4j_label`. Both auto-create on first ingest — no
manual setup.

---

## 3. Domain Configuration

All schema lives in `domain_config.yaml`. 6 domains defined:

| Domain | Entities | Relations | Collection | Label |
|--------|----------|-----------|------------|-------|
| `engineering` | 10 (PR, Bug, ADR…) | 6 (FIXES, DEPENDS_ON…) | `engineering_chunks` | `Engineering` |
| `journal` | 11 (Person, Model, Dataset…) | 10 (CITES, EVALUATES…) | `journal_chunks` | `Journal` |
| `legal` | 9 | 8 | `legal_chunks` | `Legal` |
| `medical` | 9 | 7 | `medical_chunks` | `Medical` |
| `accounting` | 9 | 8 | `accounting_chunks` | `Accounting` |
| `hospitality` | 9 | 7 | `hospitality_chunks` | `Hospitality` |

Only `engineering` and `journal` are operational. The other 4 are stub
configurations — GLiNER + E2B haven't been validated on those domains yet.

Hot-reload via `POST /admin/reload` on the GPU daemon (no restart needed).

---

## 4. API Endpoints (serve_gpu.py :8000)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | Full RAG (retrieval + optional synthesis) |
| `POST` | `/embed_query` | Single query vector |
| `POST` | `/embed_late` | Full-doc late-chunked vectors |
| `POST` | `/rerank` | BGE rerank |
| `POST` | `/extract_entities` | GLiNER zero-shot NER |
| `POST` | `/ingest` | Single-doc ingest (non-blocking) |
| `GET`  | `/ingest/status/{id}` | Poll ingest task |
| `DELETE` | `/ingest/{doc_id}` | Remove doc |
| `GET`  | `/ingest` | List docs |
| `GET`  | `/collections` | List Qdrant collections |
| `GET`  | `/profiles` | List loaded domain profiles |
| `GET`  | `/health` | GPU status + VRAM |
| `POST` | `/admin/reload` | Hot-reload `domain_config.yaml` |

---

## 5. Retrieval Performance (benchmarked 2026-07-12)

| Stage | Time |
|-------|------|
| Embed (Jina v3, GPU) | 58 ms |
| Qdrant search | 67 ms |
| Neo4j subgraph | 82 ms |
| Rerank (BGE, GPU) | 1 ms |
| **Retrieval total** | **127 ms** |

Synthesis is delegated to the parent AI agent (not local).

---

## 6. Key Design Decisions

1. **Hybrid over pure LLM.** GLiNER's entity vocabulary is precise (only
   YAML-defined types). E2B classifies relations between those entities.
   This eliminates `ASSOCIATED_WITH` fallbacks and hallucinated entity names.
2. **E2B over Qwen/Phi-3.** Gemma-4-E2B (3.2 GB on disk, Q4_0) has 89-100%
   precision. Qwen 1.5B (20%) and Phi-3-mini (23%) are not viable.
3. **Vector-driven entity resolution.** Jina v3 embeds entity names into
   1024-dim space. Neo4j vector index matches by cosine ≥0.88 — no hardcoded
   canonicalization map needed. Cross-lingual by design ("Basis Data" ≡
   "Database").
4. **Sentence-boundary chunking.** spaCy prevents entity-name splitting at
   chunk edges — critical for long documents with repeated entity mentions.
5. **Batch UNWIND.** Edges are grouped by type and written in one Cypher
   statement per type — orders-of-magnitude fewer round-trips than per-edge
   writes.

---

## 7. Services

| Service | Port | Managed by | Auto-start |
|---------|------|-----------|------------|
| rag-gpu-daemon | 8000 | systemd | boot |
| llama-server (E2B) | 8082 | user process | manual |
| Neo4j | 7687 | systemd | boot |
| Qdrant | 6333 | systemd | boot |

No synthesis LLM is needed on this node — the parent AI agent handles query
synthesis via the `rag_query` Hermes tool.

---

## 8. Known Gaps

1. **GLiNER entity drift.** Unknown entity types (not in YAML) are silently
   dropped by `_parse_and_validate`. Dynamic Label Injection plan in
   `plans/dynamic-label-injection.md` addresses this.
2. **E2B thinking-block overhead.** Gemma-4-E2B ignores `--reasoning-format`
   and always emits a `<|channel|>thought` block. `_salvage_from_thinking`
   recovers relations from the thinking text as a fallback when the JSON
   answer is truncated.
3. **Non-engineering domains untested.** Only `engineering` and `journal` have
   been benchmarked. Legal/medical/accounting/hospitality are stub configs.
4. **No multi-turn conversation.** Each `/ask` is stateless.

---

## 9. Documentation Index

| Document | Purpose |
|----------|---------|
| `README.md` | Quick-start, usage, sample queries |
| `BENCHMARKS.md` | Full benchmark history (WS1-4 + journal) |
| `NEXT_PHASE_PLAN.md` | Completed phases + dynamic-label roadmap |
| `MODEL_SWITCHING.md` | How to swap any model component |
| `domain_config.yaml` | Domain schemas (single source of truth) |
| `plans/dynamic-label-injection.md` | Implementation plan for entity-drift fix |

---

## 10. Version History

| Version | Date | Key change |
|---------|------|------------|
| `v3.0.8` | 2026-07-12 | Chunk-size fix (1400 words) + max_tokens 4096 — journal extraction works |
| `v3.0.7` | 2026-07-12 | Remove stray files from v3.0.6 commit |
| `v3.0.6` | 2026-07-12 | Journal domain + 5 prod bug fixes (GLiNER labels, Qdrant batch, sparse dedup, YAML domain passthrough, collection routing) |
| `v3.0.5` | 2026-07-12 | Remove out-of-scope SNOMED artifact |
| `v3.0.3` | 2026-07-12 | WS3 sliding window + WS4 Phi-3 eval |
| `v3.0.2` | 2026-07-12 | WS2 E2B tuning — Option B wins (8192 ctx) |
| `v3.0.1` | 2026-07-12 | GLiNER+E2B hybrid extraction (100% precision) |
| `v3.0.0` | 2026-07-12 | E2B full-doc extraction — re-architecture |
| `v2.7.0` | 2026-07-10 | Neuro-Symbolic GraphRAG |
| `v2.6.0` | 2026-07-10 | Domain profile system |
