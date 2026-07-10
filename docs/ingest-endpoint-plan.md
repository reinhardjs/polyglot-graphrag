# Plan: `/ingest` Endpoint — Incremental Knowledge Base Updates

**Status:** planned, prioritized for frequent doc updates  
**Target:** v2.5.0  
**Created:** 2026-07-10, updated 2026-07-10

---

## Why this matters now

Engineering docs change frequently — bugs get updated, ADRs get amended, PRs merge.
Current flow: `bash run.sh ingest sample_data` rebuilds everything (batch, 5 docs = ~72s).
For 50+ docs with hourly changes, this is unsustainable.

The `/ingest` endpoint enables:
- **Single-doc ingest** — only the changed document is re-extracted and re-embedded
- **Single-doc delete** — remove a doc from Qdrant + Neo4j in one call
- **CI/CD pipeline friendly** — `curl -X POST :8000/ingest` from any build step
- **File-watcher potential** — daemon or cron watches a directory, auto-ingests changes

## API contract

### `POST /ingest` — ingest a single document

**Request:**
```json
{
  "text": "# BUG-204: Checkout 5xx cascade...",
  "doc_id": "bug-204",
  "doc_type": "Bug",
  "author": "bob",
  "extract_graph": true       // run LLM extraction (E2B) + GLiNER fallback
}
```

**Response:**
```json
{
  "doc_id": "bug-204",
  "status": "ok",
  "chunks": 43,                // late-chunked sentences
  "entities": 7,               // Neo4j Entity nodes created/merged
  "edges": 11,                 // Neo4j edges created
  "vectors_upserted": 43,      // Qdrant chunks written
  "extraction_method": "llm",  // "llm" (E2B) or "gliner" (fallback)
  "timing": {
    "embed_late": 0.42,
    "extract_graph": 3.18,
    "write_qdrant": 0.11,
    "write_neo4j": 0.09,
    "total": 3.80
  }
}
```

### `DELETE /ingest/{doc_id}` — remove a document

**Request:** (none — doc_id in path)

**Response:**
```json
{
  "doc_id": "bug-204",
  "status": "ok",
  "vectors_deleted": 43,
  "nodes_deleted": 7,
  "edges_deleted": 11
}
```

### `GET /ingest` — list ingested documents

**Response:**
```json
{
  "documents": [
    {"doc_id": "bug-204", "doc_type": "Bug", "chunks": 43, "entities": 7},
    {"doc_id": "adr-014", "doc_type": "ADR",  "chunks": 68, "entities": 12},
    ...
  ]
}
```

## Implementation

### Phase 1 — Wire `ingest_file()` into the daemon (~1 hour)

1. In `serve_gpu.py`, add `IngestReq` model and `ingest(req: IngestReq)` function.
2. The function calls `ask._ingest_file()` directly (same code path as `ingest.py`).
3. E2B extraction runs first (primary); GLiNER is the lazy fallback.
4. Wrap in try/except — return status + error if E2B or GLiNER both fail.
5. Add `IngestDelete`, `IngestList` endpoints.

**Key code structure:**
```python
class IngestReq(BaseModel):
    text: str
    doc_id: str
    doc_type: str = "Document"
    author: str = ""
    extract_graph: bool = True

@app.post("/ingest")
def ingest(req: IngestReq):
    from ingest import ingest_text   # reuse ingest.py's logic
    result = ingest_text(
        text=req.text, doc_id=req.doc_id,
        doc_type=req.doc_type, author=req.author,
        extract_graph=req.extract_graph,
    )
    return {"doc_id": req.doc_id, "status": "ok", **result}
```

### Phase 2 — Refactor ingest.py to be callable (~30 min)

Currently `ingest.py` is a CLI script. Extract the core logic into:
- `ingest_text(text, doc_id, ...)` — reusable function
- `delete_doc(doc_id)` — wraps existing `delete_doc_qdrant` + `delete_doc_neo4j`
- `list_docs()` — queries Qdrant for doc_id payload summary

### Phase 3 — Error handling (~20 min)

- E2B timeout (30s) → fall back to GLiNER
- GLiNER first-load timeout (300s) → return HTTP 503 "GLiNER still loading"
- Qdrant/Neo4j unavailable → return HTTP 502 with error detail
- Empty text → HTTP 400
- Duplicate doc_id → upsert (existing behavior, same as `bash run.sh ingest`)

### Phase 4 — Tests (~30 min)

```bash
# Basic ingest
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"# TEST-001: Minimal doc","doc_id":"test-001","doc_type":"Bug"}'
# → {"doc_id":"test-001","status":"ok","chunks":1,...}

# Verify retrieval works
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what is TEST-001?","synthesize":false}'
# → contexts should include TEST-001

# Delete and verify gone
curl -X DELETE http://127.0.0.1:8000/ingest/test-001
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what is TEST-001?","synthesize":false}'
# → 0 contexts
```

## VRAM impact

No change. The `/ingest` endpoint reuses resident models already loaded:

| Model | Already loaded? | Used by /ingest? |
|-------|-----------------|-------------------|
| Jina v3 (embed) | ✓ always | ✓ embed_late |
| E2B (:8082) | ✓ systemd | ✓ extract_graph (LLM) |
| GLiNER | ✗ lazy | ✓ extract_graph (fallback, lazy-loads on first call) |

E2B is a separate systemd service. GLiNER lazy-loads only if LLM extraction fails.

**Tradeoff during ingest:** If `/ask` is called mid-ingest, Qdrant/Neo4j writes from
the ingest thread may cause brief query inconsistency (a chunk appearing while its
neighbor is still being written). Mitigation: Qdrant upserts are atomic per-point;
Neo4j MERGE is idempotent. The impact is ≤1 retrieval result being stale for <100ms.

## After `/ingest` exists — what changes

| Before | After |
|--------|-------|
| `bash run.sh ingest sample_data` | `curl -X POST :8000/ingest` |
| Need `rag-env` + `HF_HOME` on caller | Any HTTP client, any machine |
| Batch-only via folder | Single-doc ingest, ideal for incremental updates |
| `ingest.py` CLI script | Still useful for batch/folder import (reuses same functions) |
| `run.sh` orchestration | Deprecated for ingest; `run.sh serve` + pure API for everything else |

## Files to create/modify

| File | Action |
|------|--------|
| `v2/ingest.py` | Extract `ingest_text()`, `delete_doc()`, `list_docs()` functions |
| `v2/serve_gpu.py` | Add `IngestReq` model, `POST /ingest`, `DELETE /ingest/{id}`, `GET /ingest` |
| `v2/run.sh` | Optionally add `ingest_via_api` helper or deprecate |

## Dependencies

- `ingest.py` already imports `qdrant_client`, `neo4j`, `requests` (E2B) — all present in rag-env.
- No new packages needed.
- E2B must be running (:8082). If E2B is down, GLiNER fallback covers extraction (no LLM nodes, heuristic edges only).

## Frequent Update Strategy

### Problem

Engineering docs change frequently. Re-running full batch ingest for every
change is wasteful — E2B extraction is expensive (~12s/doc), and re-embedding
unchanged text is pointless.

### Strategy: checksum-based incremental ingest

```
                     ┌──────────────────┐
                     │  File watcher    │  (cron, inotify, CI webhook)
                     │  detects change  │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  Compute SHA-256 │
                     │  of doc content  │
                     └────────┬─────────┘
                              │
              ┌───────────────▼───────────────┐
              │  Compare with stored checksum │
              │  (Qdrant payload.checksum)    │
              └───────────────┬───────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Same checksum?     │
                   └──────┬──────┬───────┘
                     YES  │      │  NO
                          │      │
                   ┌──────▼─┐  ┌─▼──────────────────┐
                   │  SKIP  │  │  POST /ingest      │
                   │  (0s)  │  │  (re-extract +     │
                   └────────┘  │   re-embed + write)│
                               └────────────────────┘
```

1. **Store checksum with each doc** — `ingest.py` already writes to Qdrant with `payload.doc_id`. Add `payload.checksum = sha256(text)`.

2. **`GET /ingest/{doc_id}/checksum`** — returns stored checksum. File-watcher compares before deciding to POST.

3. **`POST /ingest` with `if_changed: true`** — daemon compares checksum before extraction. If unchanged, returns `{"status": "unchanged", "skipped": true}` in <10ms.

4. **Batch revalidate** — `POST /ingest/revalidate` sends a list of `{doc_id: checksum}` pairs. Daemon compares all, returns list of only the changed ones. Caller then ingests only those.

### Integration with CI/CD

```bash
# In CI pipeline, after markdown docs are updated:
for f in docs/engineering/*.md; do
  doc_id=$(basename "$f" .md)
  checksum=$(sha256sum "$f" | cut -d' ' -f1)
  curl -s -X POST http://rag:8000/ingest \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$(cat $f)\",\"doc_id\":\"$doc_id\",\"if_checksum\":\"$checksum\"}"
done
```

### File-watcher cron script

A Hermes cron job or a simple bash script that:
1. Watches `/mnt/data-970-plus/rag-system/v2/sample_data/` (or a configurable dir)
2. On file change → compute checksum → compare with stored → POST /ingest if changed
3. Runs every 5 minutes or on inotify trigger

This keeps the knowledge base in sync with the engineering docs without manual intervention.