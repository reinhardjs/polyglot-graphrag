# Plan: `/ingest` Endpoint — Incremental Knowledge Base Updates

**Status:** ✅ SHIPPED in v2.5.0 (see CHANGELOG.md [2.5.0] and v2/README.md for the live API). This file is the original design doc — kept for rationale.
**Target:** v2.5.0
**Created:** 2026-07-10, updated 2026-07-11

> [!IMPORTANT]
> **Review amendments (2026-07-11):** This plan has been updated to address
> critical findings from codebase review. Key changes:
> - `POST /ingest` uses `BackgroundTasks` (non-blocking) instead of synchronous handler
> - Entity resolution cache with locking to prevent race conditions
> - Cache invalidation (`query_cache`) on re-ingest
> - `serve_cpu.py` parity required
> - `_sparse()` hash determinism fix (pre-requisite bug fix)
> - Fixed incorrect reference to `ask._ingest_file()` → `ingest.ingest_text()`
> - `delete_doc_neo4j` edge-deletion scoping fix

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

### Phase 0 — Pre-requisite bug fix: `_sparse()` hash determinism (~10 min)

Python's `hash()` is randomized across processes (`PYTHONHASHSEED`). The
`_sparse()` function in both `ingest.py` and `ask.py` uses `hash(tok) % 65536`.
If ingest and query run in different Python processes, the same token maps to
different sparse indices, making sparse search return noise.

**Fix:** Replace `hash(tok)` with a deterministic hash:
```python
import hashlib
def _sparse_idx(tok: str) -> int:
    return int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], 'little') % 65536
```

Apply in both `ingest.py:_sparse()` and `ask.py:_sparse()`.

---

### Phase 1 — Wire `ingest_text()` into the daemon (~1.5 hours)

1. In `serve_gpu.py`, add `IngestReq` model.
2. The handler calls `ingest.ingest_text()` (extracted in Phase 2) via
   `BackgroundTasks` — returns `202 Accepted` immediately so `/ask` is never blocked.
3. E2B extraction runs first (primary); GLiNER is the lazy fallback.
4. Wrap in try/except — return status + error if E2B or GLiNER both fail.
5. Add `IngestDelete`, `IngestList` endpoints.
6. **Mirror all new endpoints in `serve_cpu.py`** for CPU fallback parity.

> [!CAUTION]
> `serve_gpu.py` runs with `workers=1` (GPU memory safety). A synchronous
> 12-second E2B extraction call will **block all `/ask` queries**. The handler
> MUST be non-blocking.

**Key code structure:**
```python
from fastapi import BackgroundTasks
import asyncio

# In-memory task tracker
_ingest_tasks: dict[str, dict] = {}  # task_id → {status, result}

class IngestReq(BaseModel):
    text: str
    doc_id: str
    doc_type: str = "Document"
    author: str = ""
    extract_graph: bool = True
    collection: str = "engineering_chunks"  # v2.5.0 multi-domain

@app.post("/ingest", status_code=202)
async def ingest(req: IngestReq, bg: BackgroundTasks):
    task_id = f"{req.doc_id}_{int(time.time())}"
    _ingest_tasks[task_id] = {"status": "running"}
    bg.add_task(_run_ingest, task_id, req)
    return {"task_id": task_id, "doc_id": req.doc_id, "status": "accepted"}

def _run_ingest(task_id: str, req: IngestReq):
    from ingest import ingest_text
    try:
        result = ingest_text(
            text=req.text, doc_id=req.doc_id,
            doc_type=req.doc_type, author=req.author,
            extract_graph=req.extract_graph,
            collection=req.collection,
        )
        # Invalidate cached answers that may reference stale data
        _invalidate_cache_for_doc(req.doc_id)
        _ingest_tasks[task_id] = {"status": "done", "result": result}
    except Exception as e:
        _ingest_tasks[task_id] = {"status": "error", "error": str(e)}

@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str):
    return _ingest_tasks.get(task_id, {"status": "not_found"})
```

### Phase 2 — Refactor ingest.py to be callable (~30 min)

Currently `ingest.py` is a CLI script. Extract the core logic into:
- `ingest_text(text, doc_id, ...)` — reusable function
- `delete_doc(doc_id)` — wraps existing `delete_doc_qdrant` + `delete_doc_neo4j`
- `list_docs()` — queries Qdrant for doc_id payload summary

### Phase 3 — Error handling + safety fixes (~40 min)

- E2B timeout (30s) → fall back to GLiNER
- GLiNER first-load timeout (300s) → return HTTP 503 "GLiNER still loading"
- Qdrant/Neo4j unavailable → return HTTP 502 with error detail
- Empty text → HTTP 400
- Duplicate doc_id → upsert (existing behavior, same as `bash run.sh ingest`)

**Entity resolution race condition fix:**
With API-driven ingestion, concurrent requests mentioning the same novel entity
(e.g., "Kafka") will both query Neo4j's vector index, find no match, and create
duplicate nodes. Fix: add a process-level resolution cache with a lock in
`ingest.py:resolve_node_ids()`.

```python
# ingest.py — module level
_resolution_cache: dict[str, str] = {}  # raw_name → resolved_id
_resolution_lock = threading.Lock()
```

**`delete_doc_neo4j` edge over-deletion fix:**
Currently deletes ALL edges touching any node where `doc_id ∈ source_docs`,
even edges that came from other documents via shared nodes. Fix: only delete
edges where at least one endpoint will become an orphan, or track `source_docs`
on edges too.

**Cache invalidation on re-ingest:**
When a document is re-ingested, cached answers in `query_cache` may reference
stale data. After successful ingest, flush cache entries whose `contexts`
mention the re-ingested doc_id.

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

**Concurrency during ingest:** With `BackgroundTasks`, ingestion runs in a
background thread while `/ask` remains fully responsive. The E2B extraction
call is I/O-bound (HTTP to :8082), so it releases the GIL and does not starve
GPU inference. Qdrant upserts are atomic per-point; Neo4j MERGE is idempotent.
The impact is ≤1 retrieval result being stale for <100ms during the write window.

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