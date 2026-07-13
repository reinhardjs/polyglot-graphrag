# Task: Build a git-style file-change sync daemon for the GraphRAG system

## Context (what you're integrating with)

You're working on the polyglot-graphrag project at `/mnt/data-970-plus/rag-system/`.
It ingests documents into Neo4j (knowledge graph) + Qdrant (vector chunks) and
serves RAG queries via a GPU daemon (`serve_gpu.py`, port 8000).

The pipeline is:
  File → POST /ingest {doc_id, text, domain, if_checksum} → chunks embedded
  → Qdrant + entities/edges extracted → Neo4j

The system's re-ingest semantics are:
  - Sending the SAME `doc_id` triggers clean replacement: old chunks deleted,
    old Neo4j nodes/edges scoped to that doc_id cleaned up, new data written.
  - `if_checksum` (optional): if passed and matches the stored Qdrant checksum,
    the server returns `{"status":"unchanged","skipped":true}` instantly — no
    re-extraction cost. This is how CHANGE DETECTION works: same checksum = skip.
  - DELETE /ingest/{doc_id} removes a document entirely from both stores.

Key API contract (verified in serve_gpu.py):
  POST /ingest
    Required:  {"text": "...", "doc_id": "some-stable-id"}
    Optional:  {"domain": "engineering|journal|...", "if_checksum": "<sha256>"}
    Response:  {"task_id": "...", "status": "accepted"} ... poll for completion
               or {"status": "unchanged", "skipped": true} if if_checksum matches

  DELETE /ingest/{doc_id}
    Response: {"status": "deleted"}

  GET  /ingest/status/{task_id}
    Returns progress + final result with checksum, chunks, entities, edges.

DOC_ID IS MANDATORY — the system does NOT auto-resolve document identity.
You must supply a stable doc_id. The simplest scheme: use the relative file
path as doc_id (e.g. "eng/runbooks/service-catalog.md").

No "diff / append" endpoint exists. You always send the FULL current text.
The if_checksum guard makes this cheap for unchanged files.

## Requirements

Build a document sync tool (`sync_docs.py`) that keeps Neo4j and Qdrant
consistent with a directory of source documents, tracking changes the way
git tracks file changes — content-hash (SHA256) based, no timestamps.

### R1 — File → doc_id mapping
- By default, `doc_id = relative path from the watched root`.
  Example: watching `docs/`, file `docs/eng/adr-014.md` → doc_id `eng/adr-014.md`.
- Support an optional overrides map (JSON/YAML): `{"docs/eng/adr-014.md": "custom-id"}`.

### R2 — Git-style change detection
- Maintain a state file (JSON, e.g. `.sync_state.json`) mapping:
    `{ "relative_path": {"checksum": "<sha256>", "doc_id": "...", "domain": "..."} }`
- On each run, compute SHA256 of every file under the watched root (recursive).
- Compare current checksums to stored checksums:
  - **New** (path in filesystem, not in state) → POST /ingest (create)
  - **Unchanged** (same checksum) → skip (or POST with if_checksum for safety)
  - **Changed** (different checksum) → POST /ingest with if_checksum=<old> (update)
  - **Deleted** (path in state, not in filesystem) → DELETE /ingest/{doc_id}
- After each successful ingest, update the state file with the new checksum
  (returned by the ingest status endpoint). Do NOT update on failure.

### R3 — Domain inference
- Infer `domain` from the file path or file extension:
  - `eng/`, `engineering/` → domain = "engineering"
  - `journal/`, `papers/`, `*.pdf` → domain = "journal"
  - `legal/` → domain = "legal"
  - Else → domain = "engineering" (default)
- Support a domain-overrides map (same YAML/JSON as overrides): `path → domain`.

### R4 — Idempotent and safe
- Re-running `sync_docs.py` on an already-synced directory does nothing (all
  unchanged → skipped via if_checksum). No duplicate ingestion.
- If the daemon is down, fail clearly with an error message (don't silently
  skip or mark as synced).
- Don't delete the state file or its content on failure — only update after
  confirmed successful sync.

### R5 — Daemon mode (optional/stretch)
- `--watch` mode: use inotify (Linux) to detect file changes in real time
  and sync immediately. Without --watch, do a one-shot sync and exit.
- This is the "it must follow any changes happen" part — the daemon watches
  and reacts.

### R6 — Ignore patterns
- Support `.syncignore` (gitignore syntax) to skip files (e.g. `.git/`,
  `*.tmp`, `_drafts/`).

## Integration constraints
- Use the existing `/ingest` endpoint (port 8000 by default, configurable).
- For text extraction: plain-text files read directly; PDFs use `pdftotext`
  (already available on the system).
- File encoding: always UTF-8. Skip binary files.
- Handle large files: files > EXTRACTION_CHAR_LIMIT (32K chars) are fine —
  the server truncates the extraction window but embeds all chunks.

## File placement
- Script: `/mnt/data-970-plus/rag-system/sync_docs.py`
- State file: `/mnt/data-970-plus/rag-system/.sync_state.json` (gitignore it)
- Config (optional): `/mnt/data-970-plus/rag-system/sync_config.yaml`

## Deliverables
1. `sync_docs.py` — the script described above.
2. A brief `SYNC.md` — usage docs with copy-pasteable commands.
3. A manual validation run: place 2-3 sample .md files in a test directory,
   run sync_docs.py, verify they appear in Neo4j+Qdrant, modify one file and
   rerun, verify update. Call this the proof.

## Edge cases to handle
- Empty files (0 bytes): skip with a warning, don't ingest.
- Binary files: skip, log.
- File moved/renamed: by default this is a delete (old path gone) + create
  (new path appears). If you want move detection (content-hash match),
  optionally scan for the old checksum in new files — but keep it simple
  unless you have time.
- Daemon returns error: don't update state, retry or fail.
- Concurrent edits to state file: use file locking (fcntl).

## Environment
- The GPU daemon runs at localhost:8000 (or `RAG_DAEMON_URL` env).
- The venv is at `/mnt/data-970-plus/rag-system/venv` (symlinked to ./venv).
- Python 3.12. pip install any new deps into that venv (watchdog for inotify).
- Neo4j + Qdrant are managed via docker-compose (already running).

## Verification criteria (what success looks like)
After implementation, we should be able to:
```
# Create test docs
mkdir -p /tmp/testdocs/eng
echo "Service-X depends on Database-Y." > /tmp/testdocs/eng/svc1.md
echo "ADR-001: use Redis for caching." > /tmp/testdocs/eng/adr001.md

# One-shot sync
python sync_docs.py /tmp/testdocs

# Verify in Neo4j
docker exec rag-system-neo4j-1 cypher-shell ... "MATCH (n:Entity) RETURN n.name, n.source_docs"

# Edit a file
echo "Service-X no longer needs Database-Y. Uses Postgres instead." > /tmp/testdocs/eng/svc1.md

# Re-sync — should update svc1.md, skip adr001.md (unchanged)
python sync_docs.py /tmp/testdocs

# Delete a file
rm /tmp/testdocs/eng/adr001.md

# Re-sync — should DELETE adr001.md from both stores
python sync_docs.py /tmp/testdocs
```

The state file should reflect accurate checksums after each sync.
Neo4j + Qdrant should match the current filesystem state exactly
(no stale entities from deleted files, no stale edges from updated files).
