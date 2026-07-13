# SYNC.md — `sync_docs.py`: git-style file → GraphRAG sync

`sync_docs.py` keeps the Neo4j knowledge graph + Qdrant vector store in sync
with a folder of source documents. It detects changes the same way git tracks
files: by **SHA256 content hash**, not mtime. Re-running on an already-synced
folder is a no-op (cheap server-side checksum skip).

```
file on disk ── rel path ──▶ doc_id ──POST /ingest──▶ Qdrant + Neo4j
                                   DELETE /ingest/{doc_id} on removal
```

## Quick start

```bash
cd /mnt/data-970-plus/rag-system

# One-shot sync of a docs folder (defaults: daemon http://localhost:8000)
./venv/bin/python sync_docs.py /tmp/testdocs

# Or use the venv's python directly
./venv/bin/python sync_docs.py /path/to/your/docs

# Real-time watch mode (reacts to edits/deletes via inotify)
./venv/bin/python sync_docs.py /path/to/your/docs --watch

# Preview what would change without touching the daemon
./venv/bin/python sync_docs.py /path/to/your/docs --dry-run

# Force a full re-ingest of every file (ignores the checksum fast-path)
./venv/bin/python sync_docs.py /path/to/your/docs --force
```

The state file `.sync_state.json` is written **next to the watched root**
(e.g. `/tmp/testdocs/.sync_state.json`). It records each file's SHA256, the
`doc_id`, domain, and Qdrant collection. It is only updated *after* the server
confirms success, so a crash mid-sync never desyncs state from reality.

## doc_id scheme (important)

`doc_id` defaults to the file's **path relative to the watched root**
(`eng/runbooks/svc1.md`). It is stable across edits, and globally unique
because it includes the path — the server does **not** namespace `doc_id` by
domain, so two folders with the same bare filename would otherwise collide.

## Domain inference (and overrides)

| Path / extension            | Domain      |
|-----------------------------|-------------|
| `eng/`, `engineering/`      | engineering |
| `journal/`, `papers/`, *.pdf| journal     |
| `legal/`                    | legal       |
| anything else               | engineering |

Override per-file via `sync_config.yaml` (see that file).

## Multi-domain deletes

The server's `DELETE /ingest/{doc_id}` defaults its collection to
`engineering_chunks`. For `journal`/`legal` docs, `sync_docs.py` remembers the
doc's collection in state and passes `?collection=` so the delete actually
lands. You don't need to do anything — it's automatic.

## Ignoring files (`.syncignore`)

Drop a `.syncignore` (gitignore syntax) in the watched root:

```
.git/
*.tmp
_drafts/
build/*
```

## Config (`sync_config.yaml`)

```yaml
daemon_url: http://127.0.0.1:8000
doc_id_map:
  eng/runbooks/svc1.md: custom-stable-id
domain_map:
  eng/runbooks/svc1.md: legal
```

Point at a different config with `--config /path/to.yaml`. All keys optional.

## Environment

- Daemon URL: `--daemon-url` or `RAG_DAEMON_URL` env (default `http://127.0.0.1:8000`).
- The GPU daemon (`serve_gpu.py`) and Neo4j/Qdrant must be running. If the
  daemon is down, `sync_docs.py` **fails closed** (exit 2) rather than marking
  files as synced.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | success (or `--watch` running) |
| 1    | usage / config error |
| 2    | daemon unreachable (fail-closed) |
| 3    | partial failure (some docs failed; re-run to retry only the failures) |

## Verifying the result

```bash
# Neo4j: entities and which docs they came from
docker exec rag-system-neo4j-1 cypher-shell \
  -u neo4j -p ragpassword123 \
  "MATCH (n:Entity) RETURN n.name, n.source_docs ORDER BY n.name"

# Qdrant: how many chunks per collection
curl -s http://localhost:6333/collections | jq

# List ingested docs via the daemon
curl -s http://localhost:8000/ingest | jq
```
