# Plan: `/ingest` Endpoint — Unified RAG Daemon

**Status:** planned (not implemented)  
**Target:** v2.3.0  
**Created:** 2026-07-10  

---

## Goal

Add a `POST /ingest` endpoint to `serve_gpu.py` so the daemon handles the full
system — both read (`/ask`) and write (`/ingest`) paths — from a single process.
Eliminates the need for `ingest.py` + `run.sh` batch workflow in day-to-day use.

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