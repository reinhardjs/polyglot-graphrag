# Changelog

All notable changes to the GraphRAG Engineering Knowledge Base.
This project follows [Semantic Versioning](https://semver.org/).

---

## [2.3.0] — 2026-07-10 (Vector-Driven Entity Resolution)

### Added
- **Vector-driven entity resolution** — Neo4j native vector index (`entity_vector_idx`) for language-agnostic entity merging. Replaces all hardcoded CANON_MAP translations.
- `ENTITY_RESOLUTION_THRESHOLD = 0.88` in config.py — cosine similarity cutoff for merging entities (tunable).
- `ENTITY_VECTOR_INDEX = "entity_vector_idx"` in config.py — Neo4j vector index name.
- Neo4j vector index auto-created at daemon startup (`CREATE VECTOR INDEX IF NOT EXISTS`).
- `_embed_name()`, `_resolve_in_neo4j()`, `resolve_node_ids()` in ingest.py — pipeline for vector-based identity resolution.
- Entity nodes now store `name_vector` (1024-dim Jina v3 embedding), `aliases[]` (audit trail of all merged names), and `source_docs[]` (multi-document provenance).
- Verbatim extraction: E2B prompt instructs LLM to extract entity names exactly as they appear — no forced English translation.

### Changed
- `delete_doc_neo4j()` now safely removes entities from shared nodes (`source_docs` list) rather than DETACH DELETE everything. Orphans (entities with empty source_docs) are cleaned up.
- `write_graph()` now uses 3-phase pipeline: resolve nodes via vector matching → write with aliases → create edges with resolved IDs.
- Edge types are now dynamic (extracted from LLM output) rather than always `ASSOCIATED_WITH`.

### Removed
- **CANON_MAP** — hardcoded 18-entry Indonesian→English translation table deleted from config.py.
- `canonicalize()` function from ingest.py — replaced by `resolve_node_ids()`.

### Performance
- Vector resolution adds ~0.05s per entity at ingest time (one `/embed_query` call + one vector index lookup per entity).
- Query path (/ask) unaffected — resolves entities at ingest time, not query time.
- Cross-lingual convergence validated: "Basis Data" (ID), "checkout database", "Checkout microservice" all auto-merged with correct canonical entities (≥0.88 cosine via Jina v3).

---

## [2.2.0] — 2026-07-10 (Semantic Cache + Observability)

### Added
- **Semantic cache** — Qdrant `query_cache` collection with >0.95 cosine threshold. First query runs full pipeline and stores result; subsequent identical/similar queries return cached answer in <0.01s (0 LLM tokens). `skip_cache: true` flag to bypass. Cache failure is non-blocking.
- **Route labels** — every `/ask` response now includes `source` (llm|cache), `path` (qdrant|graph|hybrid|none), `qdrant_hits`, `graph_hits`, `rerank_scores`, and `cache_hit` fields for full retrieval observability.
- **Clean E4B output** — `synthesize()` separates reasoning_content (stdout debug trace) from content (clean answer returned via API). No more chain-of-thought in the `answer` field.
- `CACHE_THRESHOLD` in config.py.

### Changed
- `/ask` response shape expanded with metadata fields. Backward-compatible — old fields (`query`, `n_contexts`, `contexts`, `answer`) unchanged.
- Cache collection auto-created at daemon startup (`on_startup`).
- AskReq now accepts `skip_cache: bool = False`.

### Performance
- Cache hit: **0.13s** (vs 4.69s cold). **36× speedup** on repeated queries.

---

## [2.1.0] — 2026-07-10 (Quality Audit)

### Fixed
- **Sparse vector indexing (critical)** — `_sparse()` used `enumerate(Counter(toks))` producing per-chunk indices 0..N. Each chunk had its own vocabulary, so cross-chunk sparse matching silently returned noise. Replaced with `hash(token) % 65536` — same term maps to same index everywhere (ingest AND query side). Affected: `ingest.py`, `ask.py`.
- **Neo4j graph leg dead (critical)** — `write_graph()` never populated `n.profile`, so `neo4j_subgraph()` returned empty lists for every query. The entire graph-RAG path was non-functional. All answers came from pure Qdrant dense search. Added `_build_profile()` to extract entity source-text context windows. `write_graph()` now SETs `n.profile` on every node.
- **Query tokenizer mismatch** — `neo4j_subgraph()` used `query.lower().split()` for the query but `re.findall(r'\w+')` for entities. `"bug-204?"` stayed as one token `"bug-204?"` while entity token `"bug-204"` split to `{"bug","204"}`. No overlap → slow 2.8s vector fallback triggered on every call. Fixed by using `re.findall(r'\w+')` for both.

### Added
- `delete_doc_neo4j(doc_id)` — DETACH DELETE all nodes sourced from a doc.
- `delete_doc_qdrant(doc_id)` — filter-delete all Qdrant points for a doc.
- `ingest_file()` now calls delete-before-reingest, enabling clean incremental updates.

### Changed
- `neo4j_subgraph()` entry selection: keyword-overlap (fast path) with vector-similarity fallback (only when keyword overlap finds zero matches).
- Entry node's own profile now included first in graph results.
- `serve_gpu.py` `/ask` handler passes `req.query` to `qdrant_search` for the sparse leg.

### Performance
- Retrieval stable at **0.19-0.24s** for all queries (was 0.17-3.29s before fix).

---

## [2.0.0] — 2026-07-10 (GPU Migration)

### Added
- **`serve_gpu.py`** — GPU daemon that preloads all aux models (Jina v3, MiniLM, BGE-reranker-v2-m3, GLiNER lazy) on CUDA at startup. Replaces `serve_cpu.py` as the primary daemon.
- **`/ask` endpoint** — one-call full RAG: `POST /ask` with `synthesize:true/false`. Runs the complete pipeline (embed→Qdrant‖Neo4j→rerank→E4B) server-side. Retrieval-only mode returns in ~0.2s.
- **LLM-based extraction** — Gemma E2B (:8082) replaced deterministic regex+GLiNER as the primary extractor. E2B emits valid JSON reliably (unlike the 12B reasoning model). GLiNER remains as fallback.
- **Hermes rag plugin** — `~/.hermes/plugins/rag/` calls `POST RAG_DAEMON_URL/ask` over HTTP. Portable to any Hermes agent on any machine. `check_requirements()` pings `/health` so the tool auto-disables when the daemon is down.
- `retrieve_json.py` — headless retrieval bridge (used by Hermes before `/ask` was the primary path).
- `bench_rag.py` — per-stage latency benchmark (embed/qdrant/neo4j/rerank/synth/total).

### Changed
- Extraction LLM: Gemma 4 12B (:8083) → Gemma 4 E2B QAT Q4_0 (:8082, ~1.5 GB).
- Synthesis LLM: Gemma 4 12B (:8083) → Gemma 4 E4B QAT Q4_0 (:8084, ~3.0 GB).
- **VRAM strategy: SEQUENTIAL → CONCURRENT.** All models fit on one 12 GB GPU simultaneously (~10.7 GB). No more Ornith↔Gemma swap.
- `config.py`: `CPU_DAEMON_URL` → `DAEMON_URL` (supports `RAG_DAEMON_URL` env override).
- `run.sh`: serves `serve_gpu.py`, not `serve_cpu.py`. Added `retrieve`, `health` commands.
- Dependency pin: `transformers==4.49.0` (compatible with torch 2.3.1+cu121).
- `serve_cpu.py`: marked as **FALLBACK** daemon. Added `/ask` endpoint for API parity.

### Removed
- Ornith 9B (:8081) from the pipeline (no longer needed — E2B handles extraction).
- Deterministic regex extractor (rendered unnecessary by LLM extraction, though config labels + CANON_MAP retained for GLiNER fallback).

---

## [1.0.0] — 2026-07-09 (Initial Build)

### Architecture
- **CPU auxiliary-model daemon** (`serve_cpu.py`, :8000): Jina v3 embeddings, MiniLM router, BGE reranker, GLiNER extractor.
- **Gemma 4 12B QAT** (:8083, 262K ctx): synthesis LLM.
- **Ornith 9B Q4_K_M** (:8081, 32K ctx): extraction LLM (via systemd, forking).
- Sequential dual-model VRAM strategy: Ornith active during ingest, Gemma during queries.
- Qdrant BinaryQuantization + sparse/dense hybrid.
- Neo4j Docker with 2-hop k-core traversal.
- Multilingual support via CANON_MAP (ID↔EN).

### Files
- `config.py`, `serve_cpu.py`, `ingest.py`, `ask.py`, `run.sh`, `README.md`.
- `ARCHITECTURE.md`, `SYSTEM_CONTEXT.md`, `EXTERNAL_ACCESS.md`.
- 5 sample engineering docs: 2 ADRs, 1 bug report, 1 PR, 1 wiki.

### Known Limitations
- `late_chunking=True` not available through SentenceTransformer 5.6 — used text-level split workaround.
- Edge types mostly `ASSOCIATED_WITH` (deterministic extractor couldn't infer relationship semantics).
- `conda activate` broken — used prefix paths directly.
- Gemma 12B returns empty `content` (reasoning model) — used `reasoning_content` fallback.

---

## Dev log (chronological, 2026-07-09 to 2026-07-10)

| Time | Event |
|------|-------|
| Jul 09 | Stand up Qdrant + Neo4j Docker, init collections |
| Jul 09 | Pin deps: transformers 4.49.0, torch 2.3.1, qdrant-client 1.18 |
| Jul 09 | Build CPU daemon (serve_cpu.py) with lazy-loaded Jina/MiniLM/BGE/GLiNER |
| Jul 09 | Ingest sample docs → 52 chunks (Qdrant), 37 nodes (Neo4j) |
| Jul 09 | Smoke test: ask.py answers BUG-204 (bob) via Ornith synthesis |
| Jul 10 | v2 starts: create v2/ subdir, migrate to E2B+E4B |
| Jul 10 | Install CUDA torch 2.3.1+cu121 into rag-env |
| Jul 10 | Build serve_gpu.py (all aux models on GPU, fp16) |
| Jul 10 | LLM extraction via E2B :8082 (JSON output, regex parse) |
| Jul 10 | End-to-end: "who reported BUG-204?" → bob, SEV-2 ✓ |
| Jul 10 | Benchmark: retrieval 0.22s, full synth 6.61s |
| Jul 10 | Add `/ask` endpoint to serve_gpu.py (one-call RAG) |
| Jul 10 | Build Hermes rag plugin (retrieve_json.py → HTTP /ask) |
| Jul 10 | Switch Hermes plugin to pure HTTP (portable, no rag-env dep) |
| Jul 10 | Verify: Hermes auto-invokes rag_query, returns sourced answers |
| Jul 10 | **Quality audit #1**: fix sparse vectors (hash-based) |
| Jul 10 | **Quality audit #2**: resurrect Neo4j graph leg (add profiles) |
| Jul 10 | **Quality audit #3**: fix query tokenizer (re.findall vs .split) |
| Jul 10 | Add delete-before-reingest (clean incremental updates) |
| Jul 10 | Git init, baseline commit |
| Jul 10 | **v2.2.0**: add semantic cache (0.13s hit, 36× speedup) |
| Jul 10 | **v2.2.0**: add route labels (qdrant_hits, graph_hits, path, rerank_scores) |
| Jul 10 | **v2.2.0**: clean E4B output (separate reasoning from answer) |
| Jul 10 | Update docs/ + CHANGELOG, git commit v2.2.0 |
| Jul 10 | **v2.2.1**: remove MiniLM zero-shot routing — 50% accuracy, parallel retrieval is better |
| Jul 10 | **v2.3.0**: vector-driven entity resolution — CANON_MAP removed, Neo4j vector index, verbatim E2B extraction, aliases + source_docs, cross-lingual Jina v3 merging |
| Jul 10 | **v2.4.0**: fully modular — all model behavior in config flags (trust_remote, half, task, matryoshka). Add /models endpoint. Sync serve_cpu.py (remove MiniLM). MODEL_SWITCHING.md with 5 config snippets |
| Jul 10 | **v2.4.1**: dual-architecture reranking — BGE v2-m3 on both GPU (full pool) and CPU (cap=15). Fix fp16 poison on CPU (14s→3.8s). CPU /ask from 3.8s→2.2s (2.3×). Pool cap quality proven (5→27 doc sweep). Benchmark docs |
