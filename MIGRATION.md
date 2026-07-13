# MIGRATION.md — Enhance without resetting your data

**Principle:** Neo4j (graph) and Qdrant (vectors) hold your ingested knowledge.
Any future enhancement MUST be able to run **on top of existing data** — no
wipe, no re-ingest required — *unless* the change is a breaking schema change,
in which case it MUST be called out prominently here and in the PR/commit.

This document is the contract. If you change the data model, update this file
first (Conventional Commits: `docs(migration): ...` or `feat!: ...` for breaking).

---

## 1. Current data model (the contract)

### Qdrant (vectors)
- **One collection per domain**, keyed in `config.QDRANT_COLLECTIONS`:
  `engineering_chunks`, `legal_chunks`, `medical_chunks`, `hospitality_chunks`,
  `accounting_chunks`. Plus `query_cache` (answer cache).
- Each point payload: `{text, doc_id, domain, chunk_index, …}`.
- Vector: Jina v3, **1024-dim**, cosine. Dimension is part of the collection
  schema — **changing the embed model changes the dimension → breaks existing
  points** (see §4).

### Neo4j (graph)
- Nodes: label `:Entity` + optional domain label (`:Engineering`, `:Legal`, …)
  + optional `:Discovered` (LLM-recovered).
- Node properties:
  - `id` (resolved stable ID, string)
  - `name` (raw name), `type` (entity type)
  - `name_vector` (Jina 1024-d, used for cross-ingest resolution)
  - `aliases[]` (all raw names seen), `source_docs[]` (origins)
  - `profile` (surrounding-text window), `discovered_by` (strategy tag)
- Edges: typed relationships between resolved `id`s.
- **Resolution is vector-driven**: a new entity name is embedded with Jina and
  matched (≥0.88 cosine) to existing nodes *before* write. This is what makes
  re-ingestion idempotent and safe.

---

## 2. Safe enhancements (no migration needed)

These can ship as normal features — existing data keeps working:

| Enhancement | Why safe |
|-------------|----------|
| New domain (add to `QDRANT_COLLECTIONS` + `domain_config.yaml`) | Creates a *new* collection; old ones untouched |
| New entity `type` value | Appended to `type`; old nodes unaffected |
| New `alias` / `source_doc` | MERGEd into lists, never overwrites |
| Prompt/extraction improvements | Only affects *future* ingests; old graph stays |
| Reranker / embed *swap within same dimension* (e.g. Jina v3 → another 1024-d) | Vectors stay compatible |
| New API endpoint | No data impact |
| BGE reranker tuning | No data impact |

---

## 3. Forward-compatible patterns to follow

When adding features, prefer these so data survives:

1. **Add, don't rename node properties.** If you need `description`, ADD it;
   don't rename `profile` → old nodes without `description` just return null.
2. **New node labels are additive.** `:Discovered` was added without touching
   `:Entity`. Queries should match `:Entity` and optionally filter by extra label.
3. **Resolution is the safety net.** Because nodes merge by name-vector, a
   re-ingest after a prompt change *enriches* the graph instead of duplicating.
4. **Collection-per-domain isolation** means a bad domain can't corrupt others.
5. **`doc_id` is the delete key.** `DELETE /ingest/{doc_id}` removes that doc's
   vectors + graph edges safely.

---

## 4. Breaking changes (MUST be flagged here + in PR)

If you do any of these, you MUST:
- Add a section below with: **what changed, who is affected, the migration
  command, and how to verify**;
- Use a `!` in the commit (`feat!:`, `refactor!:`) so release-please marks a
  MAJOR bump;
- Provide a one-shot migration script under `migrations/`.

Breaking examples:
- **Changing the embedding model to a different dimension** (e.g. Jina 1024-d →
  a 768-d model). Existing Qdrant points become unqueryable, and Neo4j
  `name_vector`s no longer match. **Migration exists:** `migrations/reembed.py`
  re-embeds all Qdrant chunks + Neo4j names. Run after changing
  `EMBED_MODEL_NAME`/`VECTOR_DIM`.
- **Renaming/removing a Neo4j node property** that queries depend on. Add
  `COALESCE(..., '')` fallbacks in queries, or a Cypher migration in `migrations/`.
- **Changing the edge `type` vocabulary** in a way old edges can't be read.
- **Changing the entity resolution threshold** (e.g. 0.88 → 0.70) in a way that
  would re-merge previously-distinct nodes (data-dependent; test on a sample
  first).
- **Swapping the graph DB or vector DB engine** (Neo4j → …, Qdrant → …).

### Blast-radius matrix (what breaks which store)

| Change | Qdrant vectors | Neo4j graph | Mitigation |
|--------|---------------|-------------|------------|
| New domain | ✅ none (new collection) | ✅ none | — |
| New entity `type` value | ✅ none | ✅ additive | — |
| Prompt / extractor tweak | ✅ none | ✅ future-only | — |
| **Swap extraction LLM** (same JSON shape) | ✅ none | ✅ none | `normalize_graph` maps key variants |
| **Swap extraction LLM (different JSON keys)** | ✅ none | ⚠️ empty graph if unnormalized | `normalize_graph` (already applied in `extract_graph_llm`) |
| **Change embed model (same dim)** | ⚠️ stale vectors (lower quality) | ⚠️ stale `name_vector` | `migrations/reembed.py --apply` |
| **Change embed model (different dim)** | ❌ unqueryable | ❌ resolution broken | `migrations/reembed.py --apply` (recreates collections) |
| Rename Neo4j property | ✅ none | ⚠️ queries return null | `COALESCE` + Cypher migration |
| Change resolution threshold | ✅ none | ⚠️ possible re-merge | test on sample first |

Legend: ✅ safe · ⚠️ degraded but recoverable · ❌ broken until migrated.

---

## 5. Migration log

### 2026-07-13 — Added infra-robustness safety net (non-breaking)
- `ingest.normalize_graph()` normalizes extractor JSON to the canonical schema
  (`{id,type}` / `{source,target,type}`), mapping common variants
  (`name`/`entity`/`label`, `from`/`to`/`relation`, …). Applied inside
  `extract_graph_llm`, so swapping the extraction model can no longer silently
  zero the graph. No data change; additive.
- `migrations/reembed.py` added — the missing re-embed migration for embedding-
  model swaps (Qdrant chunks + Neo4j `name_vector`). Dry-run by default; pass
  `--apply` to write. This satisfies the MIGRATION.md contract for the
  "embedding model swap" breaking change.
- Tests: `tests/test_normalize_graph.py` (7 cases).

> When a future breaking change ships, append a section here, e.g.:
> ```bash
> ### 2026-MM-DD — Embedding dimension 1024 → 768 (MAJOR)
> Affected: all Qdrant collections + Neo4j name_vector.
> Migration: python migrations/reembed.py --apply
> Verify: pytest tests/test_e2e_chunking.py tests/test_normalize_graph.py
> ```

---

## 6. How to verify a non-breaking change didn't reset data


```bash
# Before & after an upgrade, node/point counts should be stable (or grow):
docker exec rag-system-neo4j-1 cypher-shell -a bolt://localhost:7687 \
  -u neo4j -p ragpassword123 "MATCH (n:Entity) RETURN count(n);"

# Qdrant point count per collection:
curl -s http://localhost:6333/collections/engineering_chunks | \
  python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"
```
