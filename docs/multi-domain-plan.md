# Multi-Domain Knowledge Base Plan

**Status:** planned
**Target:** v2.5.0 (alongside /ingest endpoint)
**Created:** 2026-07-10

---

## Problem

Current system is single-domain: `engineering_chunks` in Qdrant, undifferentiated
Neo4j graph. Need to ingest and query documents from different domains:
- Legal (laws, contracts, compliance)
- Hospitality (SOPs, training manuals, recipes)
- Accounting (GL entries, tax docs, audit trails)
- Engineering (current — bugs, ADRs, PRs)
- Any other domain

Each domain has different update cadences, query patterns, and entity types.

---

## Design: Namespace-Based Collections

### Qdrant — Collection per domain

```
Qdrant
├── engineering_chunks   (1024-d, current)
├── legal_chunks          (1024-d, new)
├── hospitality_chunks    (1024-d, new)
├── accounting_chunks     (1024-d, new)
└── ...
```

All collections share the same vector dimension (Jina v3 = 1024-d) and payload
schema. Segregation is purely by collection name — zero overhead, native Qdrant
isolation.

### Neo4j — Single graph, domain labels

```
Single Neo4j graph
├── (:Entity:Engineering {name, type, ...})
├── (:Entity:Legal        {name, type, ...})
├── (:Entity:Hospitality  {name, type, ...})
└── (:Entity:Accounting   {name, type, ...})
```

Cross-domain entity resolution still works (a "Payment" entity in both Engineering
and Accounting might merge if the vector similarity is high enough — and that's
usually correct). If cross-domain merging is undesirable, set `ENTITY_RESOLUTION_THRESHOLD`
higher.

---

## API Changes

### `POST /ask` — add `collection` field

```json
{
  "query": "what is the tax rate for service charges?",
  "collection": "accounting_chunks",
  "synthesize": true
}
```

Default: `engineering_chunks` (backward compatible).

Daemon routes Qdrant search to the specified collection. Neo4j graph query
remains single-graph (domain label filters optional).

### `GET /collections` — list domains

```json
{
  "collections": {
    "engineering":  {"chunks": 215, "entities": 34, "last_ingest": "2026-07-10T..."},
    "legal":         {"chunks": 0,   "entities": 0,  "last_ingest": null},
    "hospitality":   {"chunks": 0,   "entities": 0,  "last_ingest": null},
    "accounting":    {"chunks": 0,   "entities": 0,  "last_ingest": null}
  }
}
```

### `POST /ingest` — add `collection` field

```json
{
  "text": "# LAW-001: Tax Code §45...",
  "doc_id": "law-001",
  "collection": "legal_chunks",
  "doc_type": "Law",
  "author": "counsel",
  "extract_graph": true
}
```

### `DELETE /ingest/{doc_id}?collection=legal_chunks`

---

## Config Changes

```python
# ── Multi-Domain ────────────────────────────────────────────────────────────
QDRANT_COLLECTIONS = {
    "engineering":  "engineering_chunks",
    "legal":        "legal_chunks",
    "hospitality":  "hospitality_chunks",
    "accounting":   "accounting_chunks",
}
QDRANT_COLLECTION_DEFAULT = "engineering_chunks"

# Domain label added to Neo4j entities during ingest:
#   MERGE (e:Entity:Engineering {name: $name})
# Domain passed as `domain_label` in ingest req, derived from collection name:
#   engineering_chunks → Engineering
#   legal_chunks       → Legal
```

---

## Implementation (2 phases)

### Phase 1 — Create collections + routing (~30 min)

1. Add `QDRANT_COLLECTIONS` and `QDRANT_COLLECTION_DEFAULT` to config.py
2. Add `collection` field to `AskReq` model (optional, defaults to config)
3. In `/ask` handler: pass `collection` to `rag.qdrant_search()`
4. In `rag.qdrant_search()`: accept `collection_name` param
5. Add `GET /collections` — queries Qdrant for each collection's point count
6. Neo4j domain labels: add optional `domain_label` filter to `neo4j_subgraph()`

### Phase 2 — Multi-domain ingest (~1 hour, depends on /ingest endpoint)

1. Add `collection` field to `IngestReq` model
2. Create collection on first use (Qdrant auto-creates on insert, but with wrong config — needs explicit create with correct vector params)
3. `ingest_text()` writes to specified collection + attaches domain label to Neo4j nodes
4. Checksum-based incremental ingest works per-collection

---

## Resource Impact

| Aspect | Single Domain | Multi-Domain |
|--------|--------------|-------------|
| VRAM | 10.4 GB | 10.4 GB (no change — same models) |
| Qdrant storage | ~5 MB | ~5 MB × N domains |
| Neo4j storage | ~2 MB | ~2 MB × N domains |
| Query latency | 0.16s | 0.16s (collection name is a free filter) |
| Ingest latency | ~72s (5 docs) | ~72s per domain (independent) |

No VRAM impact — models are shared. Qdrant collection overhead is negligible
(~KB per empty collection). Query performance is identical — Qdrant routes
to a single collection at search time.

---

## Frequent Update Strategy per Domain

Each domain can have its own update pipeline:

```
Legal docs → CI pipeline → POST /ingest?collection=legal_chunks
Accounting → nightly cron → POST /ingest?collection=accounting_chunks
Engineering → file watcher → POST /ingest?collection=engineering_chunks
```

Independent cadences. Checksum caching prevents re-extraction of unchanged docs
per domain.
